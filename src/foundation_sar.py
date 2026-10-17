# Foundation SAR - Core Data Schemas and Utilities

"""
This module contains the foundational components for SAR processing:

1. Pydantic Data Schemas:
   - CustomerData: Customer profile information
   - AccountData: Account details and balances
   - TransactionData: Individual transaction records
   - CaseData: Unified case combining all data sources
   - RiskAnalystOutput: Risk analysis results
   - ComplianceOfficerOutput: Compliance narrative results

2. Utility Classes:
   - ExplainabilityLogger: Audit trail logging
   - DataLoader: Combines fragmented data into case objects

The schemas accept records straight from ``pandas.DataFrame.to_dict('records')``:
empty CSV cells (NaN) become ``None`` for optional fields, and numeric columns
that pandas infers for string fields (e.g. ``ssn_last_4``) are coerced back to
strings before validation.
"""

import json
import numbers
import re
import time
import pandas as pd
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
import uuid
import os
from collections import Counter

# ===== SHARED VALIDATION HELPERS =====

DATE_FORMAT = "%Y-%m-%d"
RISK_RATINGS = ('Low', 'Medium', 'High')         # Accepted CustomerData.risk_rating values
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_missing(value: Any) -> bool:
    """Return True for None, NaN/NA values, or blank strings.

    pandas represents empty CSV cells as NaN, which Pydantic would otherwise
    reject for Optional[str] fields.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # pd.isna on list-like values returns an array - those are not "missing"
        return False


def _validate_date_string(value: Any) -> str:
    """Validate a date and normalise it to a 'YYYY-MM-DD' string.

    Accepts ISO date strings as well as date/datetime objects (including
    pandas Timestamps, which subclass datetime).
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError(
            f"expected a date string in YYYY-MM-DD format, got {type(value).__name__}"
        )

    value = value.strip()
    try:
        if not _DATE_PATTERN.match(value):
            raise ValueError
        datetime.strptime(value, DATE_FORMAT)
    except ValueError:
        raise ValueError(f"'{value}' is not a valid date in YYYY-MM-DD format") from None
    return value


# ===== PYDANTIC SCHEMAS =====

class CustomerData(BaseModel):
    """Customer information schema with validation

    Mirrors data/customers.csv. Required identity fields must be present;
    phone, occupation and annual_income are optional and default to None.

    Validation rules:
    - Dates must be valid YYYY-MM-DD values, and customer_since cannot
      precede date_of_birth
    - ssn_last_4 must be exactly 4 digits (integers are zero-padded)
    - risk_rating must be one of 'Low', 'Medium', 'High'
    - annual_income cannot be negative
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: str = Field(..., min_length=1, description="Unique customer identifier, e.g. 'CUST_0001'")
    name: str = Field(..., min_length=1, description="Full customer name")
    date_of_birth: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    ssn_last_4: str = Field(..., pattern=r"^\d{4}$", description="Last 4 digits of the SSN")
    address: str = Field(..., min_length=1, description="Full mailing address")
    customer_since: str = Field(..., description="Relationship start date in YYYY-MM-DD format")
    risk_rating: Literal['Low', 'Medium', 'High'] = Field(..., description="Customer risk assessment")
    phone: Optional[str] = Field(None, description="Contact phone number")
    occupation: Optional[str] = Field(None, description="Customer occupation or job title")
    annual_income: Optional[int] = Field(None, ge=0, description="Reported annual income in USD")

    @field_validator('phone', 'occupation', 'annual_income', mode='before')
    @classmethod
    def missing_to_none(cls, v: Any) -> Any:
        """Treat empty CSV cells (NaN/blank) as missing optional values."""
        return None if _is_missing(v) else v

    @field_validator('ssn_last_4', mode='before')
    @classmethod
    def normalize_ssn_last_4(cls, v: Any) -> Any:
        """pandas reads this column as int64; restore the 4-digit string form."""
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        # numbers.Integral also covers numpy integer types
        if isinstance(v, numbers.Integral) and not isinstance(v, bool):
            return str(int(v)).zfill(4)
        return v

    @field_validator('date_of_birth', 'customer_since', mode='before')
    @classmethod
    def validate_dates(cls, v: Any) -> str:
        """Require real calendar dates in YYYY-MM-DD format."""
        return _validate_date_string(v)

    @field_validator('risk_rating', mode='before')
    @classmethod
    def normalize_risk_rating(cls, v: Any) -> Any:
        """Accept case variants such as 'high' or ' HIGH '; reject anything else by name."""
        if not isinstance(v, str):
            return v  # Non-strings fall through to the Literal check
        normalized = v.strip().capitalize()
        if normalized not in RISK_RATINGS:
            raise ValueError(f"Invalid risk_rating '{v}': must be one of {', '.join(repr(r) for r in RISK_RATINGS)}")
        return normalized

    @model_validator(mode='after')
    def check_relationship_dates(self) -> 'CustomerData':
        """Reject relationships that start before the customer was born."""
        # ISO date strings compare chronologically
        if self.customer_since < self.date_of_birth:
            raise ValueError(
                f"customer_since ({self.customer_since}) cannot be before "
                f"date_of_birth ({self.date_of_birth})"
            )
        return self


class AccountData(BaseModel):
    """Account information schema with validation

    Mirrors data/accounts.csv. All fields are required.

    Validation rules:
    - opening_date must be a valid YYYY-MM-DD value
    - Balances must be finite numbers (NaN/inf rejected); current_balance
      may be negative to allow for overdrafts
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    account_id: str = Field(..., min_length=1, description="Unique account identifier, e.g. 'CUST_0001_ACC_1'")
    customer_id: str = Field(..., min_length=1, description="Owning customer; must match CustomerData.customer_id")
    account_type: str = Field(..., min_length=1, description="Account type, e.g. 'Checking', 'Savings', 'Money_Market'")
    opening_date: str = Field(..., description="Account opening date in YYYY-MM-DD format")
    current_balance: float = Field(..., allow_inf_nan=False, description="Current balance in USD (negative for overdrafts)")
    average_monthly_balance: float = Field(..., allow_inf_nan=False, description="Average monthly balance in USD")
    status: str = Field(..., min_length=1, description="Account status, e.g. 'Active', 'Closed', 'Suspended'")

    @field_validator('opening_date', mode='before')
    @classmethod
    def validate_opening_date(cls, v: Any) -> str:
        """Require a real calendar date in YYYY-MM-DD format."""
        return _validate_date_string(v)


class TransactionData(BaseModel):
    """Transaction information schema with validation

    Mirrors data/transactions.csv. counterparty and location are optional
    (most rows leave them empty).

    Validation rules:
    - transaction_date must be a valid YYYY-MM-DD value
    - amount must be a finite, non-zero number; negative values represent
      debits/withdrawals
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    transaction_id: str = Field(..., min_length=1, description="Unique transaction identifier, e.g. 'TXN_B24455F3'")
    account_id: str = Field(..., min_length=1, description="Account the transaction posted to; must match AccountData.account_id")
    transaction_date: str = Field(..., description="Posting date in YYYY-MM-DD format")
    transaction_type: str = Field(..., min_length=1, description="Transaction type, e.g. 'Cash_Deposit', 'Wire_Transfer'")
    amount: float = Field(..., allow_inf_nan=False, description="Amount in USD (negative for debits/withdrawals)")
    description: str = Field(..., min_length=1, description="Transaction description")
    method: str = Field(..., min_length=1, description="Channel, e.g. 'Wire', 'ACH', 'ATM', 'Cash'")
    counterparty: Optional[str] = Field(None, description="Other party in the transaction")
    location: Optional[str] = Field(None, description="Branch or location of the transaction")

    @field_validator('counterparty', 'location', mode='before')
    @classmethod
    def missing_to_none(cls, v: Any) -> Any:
        """Treat empty CSV cells (NaN/blank) as missing optional values."""
        return None if _is_missing(v) else v

    @field_validator('transaction_date', mode='before')
    @classmethod
    def validate_transaction_date(cls, v: Any) -> str:
        """Require a real calendar date in YYYY-MM-DD format."""
        return _validate_date_string(v)

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v: float) -> float:
        """Reject zero-value transactions (negative amounts are debits)."""
        if v == 0:
            raise ValueError("transaction amount cannot be zero")
        return v


class CaseData(BaseModel):
    """Unified case object combining all data sources

    Validation rules:
    - transactions cannot be empty - a case needs activity to analyze
    - case_created_at must be an ISO-8601 timestamp
    - All accounts must belong to the case customer
    - When account data is present, every transaction must post to one of
      the case accounts. Cases built from a transaction-only alert (no
      account records available) are allowed with an empty accounts list.
    - Transaction IDs must be unique within the case
    """
    case_id: str = Field(..., min_length=1, description="Unique case identifier (UUID)")
    customer: CustomerData = Field(..., description="Subject customer of the case")
    accounts: List[AccountData] = Field(..., description="Accounts held by the customer")
    transactions: List[TransactionData] = Field(..., description="Transactions under review")
    case_created_at: str = Field(..., description="ISO-8601 timestamp when the case was created")
    data_sources: Dict[str, str] = Field(..., description="Source tracking, e.g. {'customer_source': 'csv_extract_20241219'}")

    @field_validator('transactions')
    @classmethod
    def validate_transactions_not_empty(cls, v: List[TransactionData]) -> List[TransactionData]:
        """A case needs at least one transaction to analyze."""
        if not v:
            raise ValueError("Case must contain at least one transaction")
        return v

    @field_validator('case_created_at', mode='before')
    @classmethod
    def validate_case_created_at(cls, v: Any) -> str:
        """Accept ISO-8601 strings or datetimes, stored as an ISO-8601 string."""
        if isinstance(v, datetime):
            return v.isoformat()
        if not isinstance(v, str):
            raise ValueError(f"case_created_at must be an ISO-8601 string, got {type(v).__name__}")
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"case_created_at '{v}' is not a valid ISO-8601 timestamp") from None
        return v

    @model_validator(mode='after')
    def check_case_consistency(self) -> 'CaseData':
        """Enforce account ownership, transaction-account links and unique transaction IDs."""
        customer_id = self.customer.customer_id

        foreign_accounts = [acc.account_id for acc in self.accounts if acc.customer_id != customer_id]
        if foreign_accounts:
            raise ValueError(
                f"Accounts {foreign_accounts} do not belong to customer {customer_id}"
            )

        if self.accounts:
            account_ids = {acc.account_id for acc in self.accounts}
            orphaned = [txn.transaction_id for txn in self.transactions if txn.account_id not in account_ids]
            if orphaned:
                raise ValueError(
                    f"Transactions {orphaned} do not belong to any account in the case"
                )

        id_counts = Counter(txn.transaction_id for txn in self.transactions)
        duplicates = sorted(tid for tid, count in id_counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate transaction IDs in case: {duplicates}")

        return self


class RiskAnalystOutput(BaseModel):
    """Risk Analyst agent structured output (Chain-of-Thought result)"""
    classification: Literal['Structuring', 'Sanctions', 'Fraud', 'Money_Laundering', 'Other'] = Field(
        ..., description="Suspicious activity typology"
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the classification (0.0-1.0)")
    reasoning: str = Field(..., min_length=1, max_length=500, description="Step-by-step analysis reasoning")
    key_indicators: List[str] = Field(..., description="Suspicious indicators identified in the case")
    risk_level: Literal['Low', 'Medium', 'High', 'Critical'] = Field(..., description="Overall risk assessment")


class ComplianceOfficerOutput(BaseModel):
    """Compliance Officer agent structured output (ReACT result)"""
    narrative: str = Field(..., min_length=1, max_length=1000, description="SAR narrative text for FinCEN filing")
    narrative_reasoning: str = Field(..., max_length=500, description="Reasoning behind the narrative construction")
    regulatory_citations: List[str] = Field(..., description="Relevant regulations, e.g. '31 CFR 1020.320 (BSA)'")
    completeness_check: bool = Field(..., description="Whether the narrative meets all filing requirements")


# ===== AUDIT LOGGING =====

class ExplainabilityLogger:
    """Audit logging for compliance trails

    Every agent action is kept in memory (``entries``) and appended to a
    JSONL file, one JSON object per line, so the trail survives the process
    and can be handed to examiners as-is.

    Attributes:
        log_file: Path to the JSONL log file
        entries: In-memory list of log entries written by this instance
    """

    def __init__(self, log_file: str = "sar_audit.jsonl") -> None:
        """Create the logger, creating the log file's directory if needed.

        Args:
            log_file: Path to the JSONL audit log (appended to, never truncated)
        """
        self.log_file = log_file
        self.entries: List[Dict[str, Any]] = []

        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log_agent_action(self, agent_type: str, action: str, case_id: str,
                        input_data: Dict, output_data: Dict, reasoning: str,
                        execution_time_ms: float, success: bool = True,
                        error_message: Optional[str] = None) -> Dict[str, Any]:
        """Log an agent action with essential context

        Args:
            agent_type: Component that acted - "DataLoader", "RiskAnalyst", "ComplianceOfficer"
            action: What it did - "create_case", "analyze_case", "generate_narrative"
            case_id: Case the action relates to
            input_data: Summary of the inputs (stored as a JSON object)
            output_data: Summary of the outputs (stored as a JSON object)
            reasoning: Explanation of the decision or failure
            execution_time_ms: Duration of the action in milliseconds
            success: Whether the action succeeded
            error_message: Error details when success is False

        Returns:
            The log entry that was recorded

        The summaries are stored as JSON objects, not strings, so each line can be
        queried directly (e.g. ``jq '.output_summary.classification'``). They are
        copied through a JSON round trip, so the in-memory entry matches the file
        exactly and later changes to the caller's dicts cannot alter the record.
        """
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'case_id': case_id,
            'agent_type': agent_type,
            'action': action,
            'input_summary': _to_json_object(input_data),
            'output_summary': _to_json_object(output_data),
            'reasoning': reasoning,
            'execution_time_ms': execution_time_ms,
            'success': success,
            'error_message': error_message
        }

        self.entries.append(entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, default=str) + '\n')

        return entry


def _to_json_object(data: Any) -> Any:
    """JSON-compatible copy of data; values json cannot encode are stored as strings"""
    return json.loads(json.dumps(data, default=str))


# ===== DATA LOADER =====

class DataLoader:
    """Creates unified case objects from fragmented CSV records

    Attributes:
        logger: ExplainabilityLogger used to audit every case creation
    """

    def __init__(self, explainability_logger: ExplainabilityLogger) -> None:
        """Create the loader.

        Args:
            explainability_logger: Logger that receives one entry per case creation attempt
        """
        self.logger = explainability_logger

    def create_case_from_data(self,
                            customer_data: Dict,
                            account_data: List[Dict],
                            transaction_data: List[Dict]) -> CaseData:
        """Create a unified case object from fragmented AML data

        The account and transaction lists may cover many customers (e.g. a
        full CSV extract); only records belonging to this customer are kept.

        Args:
            customer_data: Single customer record
            account_data: Account records (filtered to the customer's accounts)
            transaction_data: Transaction records (filtered to the customer's accounts)

        Returns:
            Validated CaseData with transactions sorted by date

        Raises:
            pydantic.ValidationError: If any record fails schema validation or
                the customer has no transactions. Failures are logged before
                being re-raised.
        """
        start_time = time.perf_counter()
        case_id = str(uuid.uuid4())
        input_summary = {
            'customer_id': customer_data.get('customer_id') if isinstance(customer_data, dict) else None,
            'accounts_provided': len(account_data),
            'transactions_provided': len(transaction_data)
        }

        try:
            customer = CustomerData(**customer_data)

            accounts = [
                AccountData(**acc) for acc in account_data
                if str(acc.get('customer_id', '')).strip() == customer.customer_id
            ]
            account_ids = {acc.account_id for acc in accounts}

            transactions = [
                TransactionData(**txn) for txn in transaction_data
                if str(txn.get('account_id', '')).strip() in account_ids
            ]
            transactions.sort(key=lambda txn: txn.transaction_date)

            extract_label = f"csv_extract_{datetime.now().strftime('%Y%m%d')}"
            case = CaseData(
                case_id=case_id,
                customer=customer,
                accounts=accounts,
                transactions=transactions,
                case_created_at=datetime.now(timezone.utc).isoformat(),
                data_sources={
                    'customer_source': extract_label,
                    'account_source': extract_label,
                    'transaction_source': extract_label
                }
            )

            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_case",
                case_id=case_id,
                input_data=input_summary,
                output_data={
                    'customer_id': customer.customer_id,
                    'num_accounts': len(accounts),
                    'num_transactions': len(transactions)
                },
                reasoning=(
                    f"Created case for {customer.customer_id} with {len(accounts)} account(s) "
                    f"and {len(transactions)} transaction(s)"
                ),
                execution_time_ms=execution_time_ms,
                success=True
            )
            return case

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_case",
                case_id=case_id,
                input_data=input_summary,
                output_data={},
                reasoning=f"Case creation failed: {type(e).__name__}",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )
            raise

# ===== HELPER FUNCTIONS (PROVIDED) =====

def load_csv_data(data_dir: str = "data/") -> tuple:
    """Helper function to load all CSV files

    Returns:
        tuple: (customers_df, accounts_df, transactions_df)
    """
    try:
        customers_df = pd.read_csv(f"{data_dir}/customers.csv")
        accounts_df = pd.read_csv(f"{data_dir}/accounts.csv")
        transactions_df = pd.read_csv(f"{data_dir}/transactions.csv")
        return customers_df, accounts_df, transactions_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading CSV data: {e}")

if __name__ == "__main__":
    # Quick validation: build a case for every customer in the provided CSV data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    customers_df, accounts_df, transactions_df = load_csv_data(os.path.join(project_root, "data"))

    logger = ExplainabilityLogger(os.path.join(project_root, "outputs", "audit_logs", "foundation_selfcheck.jsonl"))
    loader = DataLoader(logger)
    accounts = accounts_df.to_dict('records')
    transactions = transactions_df.to_dict('records')

    created, failures = 0, Counter()
    for customer in customers_df.to_dict('records'):
        try:
            loader.create_case_from_data(customer, accounts, transactions)
            created += 1
        except ValidationError as e:
            failures[e.errors()[0]['msg']] += 1

    print("🏗️  Foundation SAR Module - Data Validation")
    print(f"✅ Cases created: {created}")
    for reason, count in failures.items():
        print(f"⏭️  No case for {count} customer(s): {reason}")
    print(f"📝 Audit log: {logger.log_file}")
