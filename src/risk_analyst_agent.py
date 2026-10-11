# Risk Analyst Agent - Chain-of-Thought Implementation

"""
Risk Analyst Agent Module

This agent performs suspicious activity classification using Chain-of-Thought reasoning.
It analyzes customer profiles, account behavior, and transaction patterns to identify
potential financial crimes.

How it works:
1. The case is rendered into a structured prompt: customer profile, accounts,
   every transaction, and pre-computed financial summary statistics.
2. The LLM reasons through a 5-step Chain-of-Thought framework, writing each
   step out, then emits the final decision as a JSON object.
3. The JSON is extracted, normalised (e.g. "money laundering" -> "Money_Laundering",
   85 -> 0.85) and validated against RiskAnalystOutput. Malformed output triggers
   one corrective retry that tells the model what went wrong.
4. Every analysis - successful or not - is written to the audit trail together
   with the full reasoning trace and token usage.
"""

import re
import time
import openai
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Relative import when loaded as part of the `src` package (pytest), plain import
# when `src/` itself is on sys.path (notebooks). Either way the schemas come from
# the same module object as the caller's, so isinstance checks hold.
try:
    from .foundation_sar import (
        RiskAnalystOutput,
        ExplainabilityLogger,
        CaseData,
        AccountData,
        TransactionData
    )
except ImportError:
    from foundation_sar import (
        RiskAnalystOutput,
        ExplainabilityLogger,
        CaseData,
        AccountData,
        TransactionData
    )

try:
    from .agent_utils import (
        extract_json_object, parse_json_lenient, strip_json_blocks, truncate_text,
        new_usage_counter, record_usage, summarize_case_activity, years_between
    )
except ImportError:
    from agent_utils import (
        extract_json_object, parse_json_lenient, strip_json_blocks, truncate_text,
        new_usage_counter, record_usage, summarize_case_activity, years_between
    )

from pydantic import ValidationError

# Load environment variables
load_dotenv()

REASONING_MAX_CHARS = 500           # RiskAnalystOutput.reasoning limit

VALID_CLASSIFICATIONS = ['Structuring', 'Sanctions', 'Fraud', 'Money_Laundering', 'Other']
VALID_RISK_LEVELS = ['Low', 'Medium', 'High', 'Critical']

RISK_ANALYST_SYSTEM_PROMPT = """You are a Senior Financial Crime Risk Analyst in a U.S. bank's Financial Intelligence Unit, with 15 years of experience in BSA/AML transaction monitoring, alert investigation and SAR decisioning. You analyze cases objectively, cite specific evidence (dates, amounts, transaction types, locations) and never speculate beyond the data provided.

TASK
Classify the suspicious activity in the case into exactly one category, quantify the risk, and explain your decision using the Chain-of-Thought framework below. Your classification decides whether the case goes to human review for SAR filing, so your reasoning must be transparent and auditable.

REGULATORY CONTEXT
- Currency Transaction Report (CTR): cash transactions over $10,000 must be reported (31 CFR 1010.311).
- Structuring: breaking up cash transactions to evade the CTR threshold is a federal crime (31 U.S.C. 5324), even when the funds are legitimate.
- SAR filing: banks must report suspicious transactions aggregating $5,000 or more (31 CFR 1020.320).
- Sanctions: dealings with OFAC-sanctioned parties or embargoed jurisdictions are prohibited (31 CFR Part 501).
- Money laundering: transactions designed to conceal the source of illicit funds (18 U.S.C. 1956/1957).

DATA NOTES
- Amounts are recorded as absolute values. Infer direction from the transaction type (e.g. Wire_Transfer_Credit = incoming, Wire_Transfer_Debit = outgoing).
- Treat the transaction type as authoritative. Descriptions can contradict it - flag contradictions as data-quality anomalies, not as evidence.
- Cases also contain routine activity (bills, salary, purchases). Focus on activity that deviates from the customer's profile.

CLASSIFICATION CATEGORIES
- Structuring: cash kept below the $10,000 CTR threshold - repeated cash amounts just under $10,000, or multiple smaller cash transactions split across days or branches that together exceed $10,000. Individual cash transactions of $10,000 or more are NOT structuring: they trigger a CTR rather than evade one.
- Sanctions: any counterparty, beneficiary or location identified as sanctioned, OFAC-listed, embargoed or a restricted jurisdiction. Sanctions exposure overrides every other category.
- Fraud: account takeover, identity theft, scams or elder financial exploitation - sudden high-value outflows to unknown or new recipients that are inconsistent with the customer's history and profile.
- Money_Laundering: placement or layering that obscures the source of funds - large incoming wires rapidly passed through to multiple counterparties, transfers to offshore or high-risk jurisdictions, or large cash transactions (including single amounts of $10,000 or more) inconsistent with the customer's income and occupation.
- Other: suspicious activity that fits none of the above, OR no material suspicious activity (use risk_level "Low" and say so explicitly).

CHAIN-OF-THOUGHT ANALYSIS FRAMEWORK (think step-by-step)
Step 1 - Data Review: Summarize the customer profile (occupation, income, tenure, risk rating), the accounts, and the overall activity (period, volume, transaction mix).
Step 2 - Pattern Recognition: Identify specific red flags with evidence - threshold avoidance, velocity, multi-branch cash, pass-through wires, unusual counterparties or jurisdictions, activity on closed accounts, volume inconsistent with income.
Step 3 - Regulatory Mapping: Map the red flags to the categories and regulations above. State which typology the evidence best supports and which you ruled out, and why.
Step 4 - Risk Quantification: Weigh severity (amounts relative to income, number and consistency of indicators, time span), then set the risk level using the RISK LEVELS definitions and the confidence using the CONFIDENCE SCORE bands below.
Step 5 - Classification Decision: Choose exactly one category - the typology most specifically supported by the evidence. Sanctions always takes precedence.

RISK LEVELS
- Critical: sanctions exposure; laundering or fraud where the suspicious amounts total more than $100,000 or involve offshore/high-risk jurisdictions; or several typologies combined.
- High: a clear, repeated pattern of one typology involving material amounts.
- Medium: red flags present, but a legitimate explanation is plausible.
- Low: no material red flags.

CONFIDENCE SCORE (0.0-1.0)
- 0.90-1.00: multiple strong, mutually reinforcing indicators.
- 0.70-0.89: clear pattern with minor ambiguity.
- 0.50-0.69: indicators present, but alternative explanations are plausible.
- Below 0.50: weak or insufficient evidence.

OUTPUT FORMAT
First write your analysis under the heading "CHAIN-OF-THOUGHT ANALYSIS", with 1-3 sentences per step labelled "Step 1 - Data Review:" through "Step 5 - Classification Decision:". Then give the final result as a single JSON object in a ```json code block with exactly these fields:

```json
{
  "classification": "Structuring | Sanctions | Fraud | Money_Laundering | Other",
  "confidence_score": 0.0,
  "reasoning": "Condensed step-by-step summary of Steps 1-5",
  "key_indicators": ["specific indicator citing evidence"],
  "risk_level": "Low | Medium | High | Critical"
}
```

RULES
- "reasoning" must be at most 500 characters and follow the step sequence, e.g. "Profile: ... Pattern: ... Mapping: ... Risk: ... Decision: ...".
- "key_indicators" must contain 2-6 short phrases, each naming a red-flag pattern with its evidence (e.g. "7 cash deposits of $9.2k-$9.9k across 5 branches in 7 days") - not a single routine transaction or the overall account volume. If there are no red flags, list the observations that support that conclusion.
- Use the exact category and risk level spellings shown above.
- Write nothing after the JSON block."""


class RiskAnalystAgent:
    """
    Risk Analyst agent using Chain-of-Thought reasoning.

    Attributes:
        client: OpenAI client (Vocareum-routed in the classroom)
        logger: ExplainabilityLogger receiving one audit entry per analysis
        model: Chat model used for analysis
        temperature: Sampling temperature - low for consistent, auditable output
        max_tokens: Completion budget covering the reasoning steps plus the JSON
        max_attempts: Total LLM attempts per case (1 initial + corrective retries)
        system_prompt: Chain-of-Thought system prompt
        last_raw_response: Raw text of the most recent LLM response
        last_reasoning_trace: Step-by-step analysis text preceding the JSON
        usage_totals: Cumulative token usage and call counts across all analyses
    """

    def __init__(self, openai_client, explainability_logger: ExplainabilityLogger,
                 model: str = "gpt-4o-mini", temperature: float = 0.3,
                 max_tokens: int = 1000, max_attempts: int = 2):
        """Initialize the Risk Analyst Agent

        Args:
            openai_client: OpenAI client instance
            explainability_logger: Logger for audit trails
            model: OpenAI model to use
            temperature: Sampling temperature (0.3 balances consistency and nuance)
            max_tokens: Maximum completion tokens per call
            max_attempts: Attempts per case when the output cannot be parsed
        """
        self.client = openai_client
        self.logger = explainability_logger
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max(1, max_attempts)

        self.system_prompt = RISK_ANALYST_SYSTEM_PROMPT

        self.last_raw_response: Optional[str] = None
        self.last_reasoning_trace: Optional[str] = None
        self.usage_totals: Dict[str, int] = {
            'api_calls': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }

    def analyze_case(self, case_data: CaseData) -> RiskAnalystOutput:
        """
        Perform risk analysis on a case using Chain-of-Thought reasoning.

        Args:
            case_data: Validated case to analyze

        Returns:
            Validated RiskAnalystOutput

        Raises:
            ValueError: If case_data is missing, or the LLM output cannot be parsed
                or validated after all attempts
            RuntimeError: If the OpenAI API call fails
        """
        if case_data is None:
            raise ValueError("case_data is required for risk analysis")

        start_time = time.perf_counter()
        input_summary = {
            'customer_id': case_data.customer.customer_id,
            'num_accounts': len(case_data.accounts),
            'num_transactions': len(case_data.transactions),
            'model': self.model
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._format_case_for_prompt(case_data)}
        ]
        call_usage = new_usage_counter()
        last_error: Optional[Exception] = None
        reasoning_trace = ""

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._call_llm(messages)
            except Exception as e:
                self._log_failure(case_data.case_id, input_summary, start_time,
                                  reasoning=f"OpenAI API call failed on attempt {attempt}: {type(e).__name__}",
                                  error=e, attempts=attempt, usage=call_usage)
                raise RuntimeError(f"Risk Analyst API call failed: {e}") from e

            record_usage(response, call_usage, self.usage_totals)
            choice = response.choices[0]
            content = choice.message.content
            self.last_raw_response = content
            # A corrective retry usually returns bare JSON - keep the original step-by-step trace
            reasoning_trace = self._extract_reasoning_trace(content) or reasoning_trace

            try:
                parsed = self._parse_json(self._extract_json_from_response(content))
                normalized, adjustments = self._normalize_output(parsed)
                result = RiskAnalystOutput(**normalized)
            except (ValueError, TypeError) as e:
                # ValidationError subclasses ValueError, so schema failures land here too
                last_error = e
                if getattr(choice, 'finish_reason', None) == 'length':
                    last_error = ValueError(f"Response truncated at max_tokens={self.max_tokens}: {e}")
                if attempt < self.max_attempts:
                    messages = messages[:2] + [
                        {"role": "assistant", "content": content or ""},
                        {"role": "user", "content": self._build_correction_message(last_error)}
                    ]
                continue

            self.last_reasoning_trace = reasoning_trace
            self.logger.log_agent_action(
                agent_type="RiskAnalyst",
                action="analyze_case",
                case_id=case_data.case_id,
                input_data=input_summary,
                output_data={
                    **result.model_dump(),
                    'chain_of_thought': self.last_reasoning_trace,
                    'normalizations': adjustments,
                    'attempts': attempt,
                    'token_usage': call_usage
                },
                reasoning=result.reasoning,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                success=True
            )
            return result

        # All attempts exhausted - classify the failure for the audit trail
        if isinstance(last_error, ValidationError):
            self._log_failure(case_data.case_id, input_summary, start_time,
                              reasoning=f"Schema validation failed after {self.max_attempts} attempt(s)",
                              error=last_error, attempts=self.max_attempts, usage=call_usage)
            raise ValueError(f"Risk Analyst output failed schema validation: {last_error}") from last_error

        self._log_failure(case_data.case_id, input_summary, start_time,
                          reasoning=f"JSON parsing failed after {self.max_attempts} attempt(s)",
                          error=last_error, attempts=self.max_attempts, usage=call_usage)
        raise ValueError(f"Failed to parse Risk Analyst JSON output: {last_error}") from last_error

    def _call_llm(self, messages: List[Dict[str, str]]):
        """Send the conversation to the chat completions API"""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def _extract_json_from_response(self, response_content: str) -> str:
        """Extract JSON content from LLM response

        Handles JSON in ```json code blocks, JSON in plain text, malformed JSON
        (returned as-is so the parser can repair or report it) and empty
        responses. See agent_utils.extract_json_object for the search order.

        Raises:
            ValueError: If the response is empty or contains no JSON object
        """
        return extract_json_object(response_content)

    def _parse_json(self, json_text: str) -> Any:
        """Parse JSON, repairing trailing commas - a common LLM formatting slip"""
        return parse_json_lenient(json_text)

    def _normalize_output(self, data: Any) -> tuple:
        """Coerce near-miss LLM output into the RiskAnalystOutput contract

        Fixes formatting slips without changing the substance of the decision,
        and applies one policy rule: a Sanctions classification is always
        Critical risk. Values that cannot be mapped are left untouched so
        schema validation rejects them.

        Returns:
            (normalized_dict, list of human-readable adjustments for the audit trail)
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

        data = dict(data)
        adjustments: List[str] = []

        classification = data.get('classification')
        if isinstance(classification, str) and classification not in VALID_CLASSIFICATIONS:
            key = re.sub(r"[\s\-]+", "_", classification.strip()).lower()
            mapped = {c.lower(): c for c in VALID_CLASSIFICATIONS}.get(key)
            if mapped:
                data['classification'] = mapped
                adjustments.append(f"classification '{classification}' -> '{mapped}'")

        risk_level = data.get('risk_level')
        if isinstance(risk_level, str) and risk_level not in VALID_RISK_LEVELS:
            mapped = risk_level.strip().capitalize()
            if mapped in VALID_RISK_LEVELS:
                data['risk_level'] = mapped
                adjustments.append(f"risk_level '{risk_level}' -> '{mapped}'")

        confidence = data.get('confidence_score')
        if isinstance(confidence, str):
            try:
                confidence_value = float(confidence.strip().rstrip('%'))
                data['confidence_score'] = confidence_value
                adjustments.append(f"confidence_score '{confidence}' parsed as number")
                confidence = confidence_value
            except ValueError:
                pass
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 1 < confidence <= 100:
            data['confidence_score'] = confidence / 100
            adjustments.append(f"confidence_score {confidence} rescaled to {confidence / 100}")

        indicators = data.get('key_indicators')
        if isinstance(indicators, str):
            data['key_indicators'] = [indicators]
            adjustments.append("key_indicators string wrapped in a list")
        elif isinstance(indicators, list):
            data['key_indicators'] = [str(item) for item in indicators]

        # Policy floor: sanctions exposure is always Critical (31 CFR Part 501 - prohibited, not just suspicious)
        if data.get('classification') == 'Sanctions' and data.get('risk_level') in VALID_RISK_LEVELS \
                and data['risk_level'] != 'Critical':
            adjustments.append(f"risk_level '{data['risk_level']}' raised to 'Critical' by sanctions policy floor")
            data['risk_level'] = 'Critical'

        reasoning = data.get('reasoning')
        if isinstance(reasoning, str) and len(reasoning) > REASONING_MAX_CHARS:
            truncated = truncate_text(reasoning, REASONING_MAX_CHARS)
            data['reasoning'] = truncated
            adjustments.append(f"reasoning truncated from {len(reasoning)} to {len(truncated)} characters")

        return data, adjustments

    def _build_correction_message(self, error: Exception) -> str:
        """Feedback for the corrective retry, telling the model exactly what failed"""
        return (
            "Your previous response could not be processed: "
            f"{str(error)[:300]}\n\n"
            "Respond again with ONLY the final JSON object in a ```json code block, using exactly the fields "
            "classification, confidence_score, reasoning, key_indicators and risk_level. "
            f"classification must be one of {VALID_CLASSIFICATIONS}, risk_level one of {VALID_RISK_LEVELS}, "
            "confidence_score a number between 0.0 and 1.0, and reasoning at most 500 characters."
        )

    def _extract_reasoning_trace(self, response_content: Optional[str]) -> str:
        """Return the Chain-of-Thought text with the JSON block removed"""
        return strip_json_blocks(response_content)

    def _log_failure(self, case_id: str, input_summary: Dict[str, Any], start_time: float,
                     reasoning: str, error: Optional[Exception], attempts: int,
                     usage: Dict[str, int]) -> None:
        """Write a failed analysis to the audit trail"""
        self.logger.log_agent_action(
            agent_type="RiskAnalyst",
            action="analyze_case",
            case_id=case_id,
            input_data=input_summary,
            output_data={
                'raw_response': self.last_raw_response,
                'attempts': attempts,
                'token_usage': usage
            },
            reasoning=reasoning,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            success=False,
            error_message=str(error) if error else None
        )

    def _format_case_for_prompt(self, case_data: CaseData) -> str:
        """Format case data for the analysis prompt

        Includes the customer profile (SSN fragment and address are omitted -
        they are not needed for classification), accounts, every transaction,
        and summary statistics that surface the patterns the framework asks about.
        """
        customer = case_data.customer
        transactions = case_data.transactions
        last_activity = max(txn.transaction_date for txn in transactions)

        income = f"${customer.annual_income:,}" if customer.annual_income is not None else "Not reported"
        tenure = years_between(customer.customer_since, last_activity)
        profile_lines = [
            f"- Name: {customer.name} ({customer.customer_id})",
            f"- Age: {years_between(customer.date_of_birth, last_activity)} (born {customer.date_of_birth})",
            f"- Occupation: {customer.occupation or 'Not reported'}",
            f"- Annual income: {income}",
            f"- Customer since: {customer.customer_since} ({tenure} year{'s' if tenure != 1 else ''})",
            f"- Internal risk rating: {customer.risk_rating}"
        ]

        return "\n".join([
            f"SUSPICIOUS ACTIVITY CASE {case_data.case_id}",
            "",
            "CUSTOMER PROFILE",
            *profile_lines,
            "",
            f"ACCOUNTS ({len(case_data.accounts)})",
            self._format_accounts(case_data.accounts),
            "",
            "FINANCIAL SUMMARY",
            self._format_financial_summary(case_data),
            "",
            f"TRANSACTIONS ({len(transactions)}, chronological)",
            self._format_transactions(transactions),
            "",
            "Analyze this case using the 5-step Chain-of-Thought framework, then provide the JSON result."
        ])

    def _format_accounts(self, accounts: List[AccountData]) -> str:
        """One line per account with type, status, balances and opening date"""
        if not accounts:
            return "- No account records available"
        return "\n".join(
            f"- {acc.account_id} ({acc.account_type}, {acc.status}): "
            f"current balance ${acc.current_balance:,.2f}, "
            f"average monthly balance ${acc.average_monthly_balance:,.2f}, "
            f"opened {acc.opening_date}"
            for acc in accounts
        )

    def _format_transactions(self, transactions: List[TransactionData]) -> str:
        """Numbered transaction list: date, type and amount first, then the details"""
        if not transactions:
            return "- No transactions"

        lines = []
        for i, txn in enumerate(transactions, 1):
            line = f"{i}. {txn.transaction_date}: {txn.transaction_type} ${txn.amount:,.2f} via {txn.method}"
            if txn.location:
                line += f" at {txn.location}"
            if txn.counterparty:
                line += f" | counterparty: {txn.counterparty}"
            line += f" | {txn.description} [{txn.account_id}]"
            lines.append(line)
        return "\n".join(lines)

    def _format_financial_summary(self, case_data: CaseData) -> str:
        """Pre-computed statistics that make the key patterns explicit to the model"""
        return summarize_case_activity(case_data)

# ===== PROMPT ENGINEERING HELPERS =====

def create_chain_of_thought_framework():
    """Helper function showing Chain-of-Thought structure

    The agent's system prompt expands each of these steps with the guiding
    questions an AML analyst works through:

    **Analysis Framework** (Think step-by-step):
    1. **Data Review**: What does the data tell us?
    2. **Pattern Recognition**: What patterns are suspicious?
    3. **Regulatory Mapping**: Which regulations apply?
    4. **Risk Quantification**: How severe is the risk?
    5. **Classification Decision**: What category fits best?
    """
    return {
        "step_1": "Data Review - Examine all available information",
        "step_2": "Pattern Recognition - Identify suspicious indicators",
        "step_3": "Regulatory Mapping - Connect to known typologies",
        "step_4": "Risk Quantification - Assess severity level",
        "step_5": "Classification Decision - Determine final category"
    }

def get_classification_categories():
    """Standard SAR classification categories used in the agent's prompt"""
    return {
        "Structuring": "Transactions designed to avoid reporting thresholds",
        "Sanctions": "Potential sanctions violations or prohibited parties",
        "Fraud": "Fraudulent transactions or identity-related crimes",
        "Money_Laundering": "Complex schemes to obscure illicit fund sources",
        "Other": "Suspicious patterns not fitting standard categories"
    }

# ===== TESTING UTILITIES =====

def test_agent_with_sample_case(openai_client=None, model: str = "gpt-4o-mini"):
    """Test the agent with a sample structuring case

    Builds a three-day run of cash deposits just under $10,000, analyzes it
    and prints the result. Builds a client from OPENAI_API_KEY when none is
    passed (Vocareum endpoint for 'voc-' keys, standard OpenAI otherwise).

    Returns:
        RiskAnalystOutput for the sample case
    """
    import os

    print("🧪 Testing Risk Analyst Agent")
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL") or ("https://openai.vocareum.com/v1" if api_key.startswith("voc-") else None)
        openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)

    try:
        from .foundation_sar import DataLoader
    except ImportError:
        from foundation_sar import DataLoader

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger = ExplainabilityLogger(os.path.join(project_root, "outputs", "audit_logs", "risk_analyst_sample_test.jsonl"))
    case = DataLoader(logger).create_case_from_data(
        customer_data={
            "customer_id": "CUST_SAMPLE", "name": "Sample Customer", "date_of_birth": "1985-03-15",
            "ssn_last_4": "4567", "address": "456 Oak Avenue, Springfield, IL 62701",
            "customer_since": "2018-06-01", "risk_rating": "Medium",
            "occupation": "Retail Clerk", "annual_income": 42000
        },
        account_data=[{
            "account_id": "ACC_SAMPLE", "customer_id": "CUST_SAMPLE", "account_type": "Checking",
            "opening_date": "2018-06-01", "current_balance": 31200.0,
            "average_monthly_balance": 4800.0, "status": "Active"
        }],
        transaction_data=[
            {"transaction_id": f"TXN_SAMPLE_{i}", "account_id": "ACC_SAMPLE",
             "transaction_date": f"2025-07-{14 + i:02d}", "transaction_type": "Cash_Deposit",
             "amount": amount, "description": "Cash deposit", "method": "Cash", "location": branch}
            for i, (amount, branch) in enumerate([(9800.0, "Branch_Downtown_Main"),
                                                  (9750.0, "Branch_Eastside_Center"),
                                                  (9900.0, "Branch_Westside_Plaza")])
        ]
    )

    agent = RiskAnalystAgent(openai_client, logger, model=model)
    result = agent.analyze_case(case)
    print(f"✅ Classification: {result.classification} (confidence {result.confidence_score:.2f}, "
          f"risk {result.risk_level})")
    print(f"🔑 Indicators: {result.key_indicators}")
    return result

if __name__ == "__main__":
    import os

    print("🔍 Risk Analyst Agent Module")
    print("Chain-of-Thought reasoning for suspicious activity classification")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("voc-0000"):  # skip the .env.template placeholder
        test_agent_with_sample_case()
    else:
        print("ℹ️ Set a real OPENAI_API_KEY (Vocareum or OpenAI) in .env to run the live sample test")
