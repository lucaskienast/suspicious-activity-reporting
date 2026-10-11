# SAR Workflow - Two-Stage Processing with Human-in-the-Loop Gates

"""
SAR Workflow Module

Orchestrates the end-to-end SAR process:

    CSV data -> screening -> case creation -> Risk Analyst (CoT)
             -> GATE 1: human approves SAR drafting
             -> Compliance Officer (ReACT)
             -> GATE 2: human approves SAR filing
             -> SAR document (outputs/filed_sars/) + audit trail (outputs/audit_logs/)

Cost optimisation comes from two filters in front of the expensive steps:
rule-based screening decides which customers reach the Risk Analyst, and the
human drafting gate decides which analyses reach the Compliance Officer.

Human review gates run in one of two modes:
- 'interactive': the reviewer sees a review card and answers approve/reject
  with a rationale (required for rejections - "no SAR" decisions must be
  documented for examiners)
- 'simulated_policy': a documented reviewer policy decides, for unattended
  runs (e.g. automated notebook execution). Every decision records its
  source, so simulated decisions are never mistaken for human ones.
"""

import ast
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

try:
    from .foundation_sar import (
        CaseData, RiskAnalystOutput, ComplianceOfficerOutput, ExplainabilityLogger, DataLoader
    )
    from .agent_utils import CTR_THRESHOLD, NEAR_THRESHOLD_FLOOR
except ImportError:
    from foundation_sar import (
        CaseData, RiskAnalystOutput, ComplianceOfficerOutput, ExplainabilityLogger, DataLoader
    )
    from agent_utils import CTR_THRESHOLD, NEAR_THRESHOLD_FLOOR

try:
    from IPython.core.error import StdinNotImplementedError
    _NO_STDIN_ERRORS: Tuple[type, ...] = (StdinNotImplementedError, EOFError, OSError)
except ImportError:
    _NO_STDIN_ERRORS = (EOFError, OSError)

SAR_FILING_DEADLINE_DAYS = 30       # SARs are due 30 days after initial detection (31 CFR 1020.320(b)(3))

# ===== SCREENING =====

# Points per screening signal. Typology rules are the strong signals validated in Phase 1;
# profile signals alone can raise a (weak) alert that the human drafting gate filters out.
SCREENING_WEIGHTS = {
    'structuring': 40,               # 3+ cash deposits between $9,000 and $10,000
    'rapid_movement': 40,            # incoming wire fanned out as 3+ outgoing wires within 3 days
    'large_cash': 30,                # cash transactions of $10,000 or more
    'cash_velocity': 25,             # cash/ATM at 2+ locations within 3 days totalling >= $10,000
    'high_risk_rating': 15,
    'medium_risk_rating': 5,
    'volume_vs_income': 10,          # total activity above 2x reported annual income
    'closed_account_activity': 10,
}
TYPOLOGY_SIGNALS = ('structuring', 'rapid_movement', 'large_cash', 'cash_velocity')
ALERT_THRESHOLD = 15

# Typology a screening signal points to - used to compare AI classifications with the rules
SIGNAL_TO_CLASSIFICATION = {
    'structuring': 'Structuring',
    'cash_velocity': 'Structuring',
    'rapid_movement': 'Money_Laundering',
    'large_cash': 'Money_Laundering',
}


def _detect_typology_patterns(txns: pd.DataFrame) -> Dict[str, List[str]]:
    """Return {signal: [transaction_ids]} for the typology rules a customer's transactions trigger"""
    patterns: Dict[str, List[str]] = {}
    cash = txns[txns['transaction_type'].str.contains('Cash|ATM')]

    near_threshold = cash[(cash['transaction_type'] == 'Cash_Deposit')
                          & (cash['amount'] >= NEAR_THRESHOLD_FLOOR) & (cash['amount'] < CTR_THRESHOLD)]
    if len(near_threshold) >= 3:
        patterns['structuring'] = near_threshold['transaction_id'].tolist()

    wires_in = txns[txns['transaction_type'] == 'Wire_Transfer_Credit']
    wires_out = txns[txns['transaction_type'] == 'Wire_Transfer_Debit']
    if len(wires_in) and len(wires_out) >= 3 and \
            (wires_out['date'].max() - wires_in['date'].min()).days <= 3:
        patterns['rapid_movement'] = wires_in['transaction_id'].tolist() + wires_out['transaction_id'].tolist()

    large = cash[cash['amount'] >= CTR_THRESHOLD]
    if len(large):
        patterns['large_cash'] = large['transaction_id'].tolist()

    if 'structuring' not in patterns:  # structuring cases trip this rule too
        located = cash.dropna(subset=['location']).sort_values('date')
        for start in located['date']:
            window = located[(located['date'] >= start) & (located['date'] < start + pd.Timedelta(days=3))]
            if window['location'].nunique() >= 2 and window['amount'].sum() >= CTR_THRESHOLD:
                patterns['cash_velocity'] = window['transaction_id'].tolist()
                break

    return patterns


def screen_customers(customers_df: pd.DataFrame, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame,
                     alert_threshold: int = ALERT_THRESHOLD) -> pd.DataFrame:
    """Score every customer with activity and flag alerts for AI analysis

    Args:
        customers_df, accounts_df, transactions_df: Raw CSV extracts
        alert_threshold: Minimum screening score that raises an alert

    Returns:
        One row per customer with transactions, sorted by score: customer_id, name,
        risk_rating, screening_score, signals, typology_signals, flagged_transaction_ids,
        expected_classification and alert
    """
    txns = transactions_df.merge(accounts_df[['account_id', 'customer_id', 'status']], on='account_id')
    txns = txns.assign(date=pd.to_datetime(txns['transaction_date']))
    customers = customers_df.set_index('customer_id')

    rows = []
    for customer_id, customer_txns in txns.groupby('customer_id'):
        customer = customers.loc[customer_id]
        patterns = _detect_typology_patterns(customer_txns)
        signals = list(patterns)

        if customer['risk_rating'] == 'High':
            signals.append('high_risk_rating')
        elif customer['risk_rating'] == 'Medium':
            signals.append('medium_risk_rating')
        income = customer.get('annual_income')
        if pd.notna(income) and income > 0 and customer_txns['amount'].sum() > 2 * income:
            signals.append('volume_vs_income')
        if (customer_txns['status'] == 'Closed').any():
            signals.append('closed_account_activity')

        score = sum(SCREENING_WEIGHTS[signal] for signal in signals)
        typologies = [signal for signal in signals if signal in TYPOLOGY_SIGNALS]
        flagged_ids = sorted({txn_id for ids in patterns.values() for txn_id in ids})
        rows.append({
            'customer_id': customer_id,
            'name': customer['name'],
            'risk_rating': customer['risk_rating'],
            'screening_score': score,
            'signals': signals,
            'typology_signals': typologies,
            'flagged_transaction_ids': flagged_ids,
            'expected_classification': SIGNAL_TO_CLASSIFICATION[typologies[0]] if typologies else None,
            'alert': score >= alert_threshold
        })

    return pd.DataFrame(rows).sort_values(['screening_score', 'customer_id'], ascending=[False, True]) \
        .reset_index(drop=True)


# ===== HUMAN REVIEW GATES =====

GateName = Literal['sar_drafting', 'sar_filing']
GATE_LABELS = {
    'sar_drafting': 'GATE 1 - Approve SAR drafting',
    'sar_filing': 'GATE 2 - Approve SAR filing',
}


class ReviewDecision(BaseModel):
    """A documented human (or simulated-policy) decision at a review gate"""
    gate: GateName
    decision: Literal['approve', 'reject']
    reviewer_id: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=3)
    decision_source: Literal['interactive', 'simulated_policy']
    decided_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def approved(self) -> bool:
        return self.decision == 'approve'


class HumanReviewGate:
    """Human-in-the-loop decision points between the agents

    Gate 1 (after the Risk Analyst) decides whether a SAR narrative is drafted;
    gate 2 (after the Compliance Officer) decides whether the SAR is filed.
    Every decision is written to the audit trail as a "HumanReviewer" entry.

    Attributes:
        logger: ExplainabilityLogger for decision records
        reviewer_id: Identifier recorded with each decision
        mode: 'interactive' (prompt the reviewer) or 'simulated_policy'
        min_confidence: Policy threshold for escalating High/Critical findings
        decisions: Every ReviewDecision made by this gate
    """

    def __init__(self, logger: ExplainabilityLogger, reviewer_id: str,
                 mode: Literal['interactive', 'simulated_policy'] = 'interactive',
                 input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print,
                 min_confidence: float = 0.7):
        if mode not in ('interactive', 'simulated_policy'):
            raise ValueError(f"Unknown review mode '{mode}' - use 'interactive' or 'simulated_policy'")
        self.logger = logger
        self.reviewer_id = reviewer_id
        self.mode = mode
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.min_confidence = min_confidence
        self.decisions: List[ReviewDecision] = []

    # ----- gate entry points -----

    def review_risk_analysis(self, case_data: CaseData, risk_analysis: RiskAnalystOutput) -> ReviewDecision:
        """Gate 1: approve or reject drafting a SAR narrative for this analysis"""
        return self._decide(
            gate='sar_drafting',
            case_id=case_data.case_id,
            card=self.format_risk_review_card(case_data, risk_analysis),
            policy=self.policy_drafting_decision(risk_analysis),
            context={
                'customer_id': case_data.customer.customer_id,
                'classification': risk_analysis.classification,
                'risk_level': risk_analysis.risk_level,
                'confidence_score': risk_analysis.confidence_score
            }
        )

    def review_narrative(self, case_data: CaseData, risk_analysis: RiskAnalystOutput,
                         compliance_output: ComplianceOfficerOutput,
                         validation: Optional[Dict[str, Any]]) -> ReviewDecision:
        """Gate 2: approve or reject filing the SAR with this narrative"""
        validation = validation or {}
        return self._decide(
            gate='sar_filing',
            case_id=case_data.case_id,
            card=self.format_narrative_review_card(case_data, risk_analysis, compliance_output, validation),
            policy=self.policy_filing_decision(compliance_output, validation),
            context={
                'customer_id': case_data.customer.customer_id,
                'classification': risk_analysis.classification,
                'word_count': validation.get('word_count'),
                'completeness_check': compliance_output.completeness_check,
                'warnings': validation.get('warnings', [])
            }
        )

    # ----- documented reviewer policy (simulated mode and reviewer guidance) -----

    def policy_drafting_decision(self, risk_analysis: RiskAnalystOutput) -> Tuple[str, str]:
        """Escalate sanctions always, and High/Critical findings with sufficient confidence"""
        if risk_analysis.classification == 'Sanctions':
            return 'approve', "Sanctions exposure is always escalated for SAR drafting"
        if risk_analysis.risk_level in ('High', 'Critical') and risk_analysis.confidence_score >= self.min_confidence:
            return 'approve', (f"{risk_analysis.risk_level} risk {risk_analysis.classification} finding with "
                               f"confidence {risk_analysis.confidence_score:.2f} meets escalation criteria")
        return 'reject', (f"No SAR: {risk_analysis.risk_level} risk / confidence {risk_analysis.confidence_score:.2f} "
                          f"is below escalation criteria (High or Critical with confidence >= {self.min_confidence:.2f}); "
                          f"alert closed with documented rationale")

    def policy_filing_decision(self, compliance_output: ComplianceOfficerOutput,
                               validation: Dict[str, Any]) -> Tuple[str, str]:
        """File only complete narratives without accuracy or tone concerns"""
        blocking = []
        if not compliance_output.completeness_check:
            blocking.append("narrative incomplete: " + "; ".join(validation.get('issues', []) or ["model reported incomplete"]))
        if validation.get('ungrounded_amounts'):
            blocking.append(f"amounts not traceable to case facts {validation['ungrounded_amounts']}")
        if validation.get('other_typologies'):
            blocking.append(f"mentions other typologies {validation['other_typologies']}")
        if validation.get('speculative_terms'):
            blocking.append(f"speculative language {validation['speculative_terms']}")
        if blocking:
            return 'reject', "Returned for manual revision - " + "; ".join(blocking)
        return 'approve', (f"Narrative complete ({validation.get('word_count')} words, "
                           f"{len(compliance_output.regulatory_citations)} citations) with no accuracy concerns")

    # ----- review cards -----

    def format_risk_review_card(self, case_data: CaseData, risk_analysis: RiskAnalystOutput) -> str:
        customer = case_data.customer
        dates = [txn.transaction_date for txn in case_data.transactions]
        recommendation, reason = self.policy_drafting_decision(risk_analysis)
        income = f"${customer.annual_income:,}" if customer.annual_income is not None else "not reported"
        return "\n".join([
            "═" * 78,
            f"👤 HUMAN REVIEW · {GATE_LABELS['sar_drafting']}",
            "═" * 78,
            f"Case:        {case_data.case_id}",
            f"Subject:     {customer.name} ({customer.customer_id}) · {customer.occupation or 'occupation n/a'} · "
            f"income {income} · internal rating {customer.risk_rating}",
            f"Activity:    {len(case_data.transactions)} transactions, {min(dates)} to {max(dates)}, "
            f"{len(case_data.accounts)} account(s)",
            f"AI finding:  {risk_analysis.classification} · {risk_analysis.risk_level} risk · "
            f"confidence {risk_analysis.confidence_score:.2f}",
            "Indicators:  " + "\n             ".join(f"- {i}" for i in risk_analysis.key_indicators),
            f"Reasoning:   {risk_analysis.reasoning}",
            f"Policy guidance: {recommendation.upper()} - {reason}",
            "─" * 78,
        ])

    def format_narrative_review_card(self, case_data: CaseData, risk_analysis: RiskAnalystOutput,
                                     compliance_output: ComplianceOfficerOutput, validation: Dict[str, Any]) -> str:
        recommendation, reason = self.policy_filing_decision(compliance_output, validation)
        return "\n".join([
            "═" * 78,
            f"👤 HUMAN REVIEW · {GATE_LABELS['sar_filing']}",
            "═" * 78,
            f"Case:        {case_data.case_id} · {case_data.customer.name} ({case_data.customer.customer_id}) · "
            f"{risk_analysis.classification}",
            f"Narrative ({validation.get('word_count', '?')} words):",
            f"  {compliance_output.narrative}",
            f"Citations:   {'; '.join(compliance_output.regulatory_citations)}",
            f"Complete:    {compliance_output.completeness_check} · issues: {validation.get('issues') or 'none'}",
            f"Warnings:    {validation.get('warnings') or 'none'}",
            f"Policy guidance: {recommendation.upper()} - {reason}",
            "─" * 78,
        ])

    # ----- decision mechanics -----

    def _decide(self, gate: GateName, case_id: str, card: str, policy: Tuple[str, str],
                context: Dict[str, Any]) -> ReviewDecision:
        start_time = time.perf_counter()
        source = self.mode
        if self.mode == 'interactive':
            self.output_fn(card)
            try:
                decision, rationale = self._prompt_reviewer(gate)
            except _NO_STDIN_ERRORS as e:
                # No stdin (e.g. unattended notebook execution): fall back to the documented policy
                self.output_fn(f"⚠️ Interactive input unavailable ({type(e).__name__}) - applying simulated reviewer policy")
                decision, rationale = policy
                source = 'simulated_policy'
        else:
            decision, rationale = policy

        review = ReviewDecision(gate=gate, decision=decision, reviewer_id=self.reviewer_id,
                                rationale=rationale, decision_source=source)
        self.decisions.append(review)
        self.logger.log_agent_action(
            agent_type="HumanReviewer",
            action=f"review_{gate}",
            case_id=case_id,
            input_data=context,
            output_data=review.model_dump(),
            reasoning=rationale,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            success=True
        )
        return review

    def _prompt_reviewer(self, gate: GateName, max_tries: int = 5) -> Tuple[str, str]:
        """Ask for approve/reject and a rationale; rejections must be justified"""
        answers = {'a': 'approve', 'approve': 'approve', 'y': 'approve', 'yes': 'approve',
                   'r': 'reject', 'reject': 'reject', 'n': 'reject', 'no': 'reject'}
        for _ in range(max_tries):
            raw = self.input_fn(f"{GATE_LABELS[gate]} - approve or reject? [a/r]: ").strip().lower()
            if raw in answers:
                decision = answers[raw]
                break
            self.output_fn("   Please answer 'a' (approve) or 'r' (reject).")
        else:
            raise ValueError(f"No valid decision entered after {max_tries} attempts")

        for _ in range(max_tries):
            rationale = self.input_fn("Rationale (required for rejections): ").strip()
            if rationale and len(rationale) >= 3:
                return decision, rationale
            if decision == 'approve' and not rationale:
                return decision, "Approved by reviewer"
            self.output_fn("   A rationale of at least 3 characters is required to document this decision.")
        raise ValueError(f"No rationale entered after {max_tries} attempts")


# ===== CASE STATE =====

class CaseStatus(str, Enum):
    """Final state of a case in the workflow"""
    REJECTED_AT_DRAFTING = 'rejected_at_drafting_gate'
    NARRATIVE_FAILED = 'narrative_failed'
    REJECTED_AT_FILING = 'rejected_at_filing_gate'
    FILED = 'sar_filed'
    ERROR = 'error'


class CaseRecord(BaseModel):
    """Everything the workflow decided about one alerted customer"""
    customer_id: str
    customer_name: str
    screening_score: int
    screening_signals: List[str]
    expected_classification: Optional[str] = None
    case_id: Optional[str] = None
    status: Optional[CaseStatus] = None
    risk_analysis: Optional[RiskAnalystOutput] = None
    drafting_decision: Optional[ReviewDecision] = None
    compliance_output: Optional[ComplianceOfficerOutput] = None
    narrative_validation: Optional[Dict[str, Any]] = None
    filing_decision: Optional[ReviewDecision] = None
    sar_id: Optional[str] = None
    sar_path: Optional[str] = None
    error: Optional[str] = None
    stage_times_ms: Dict[str, float] = Field(default_factory=dict)


# ===== SAR DOCUMENTS =====

DEFAULT_INSTITUTION = {
    'name': 'Demo Community Bank (simulated institution)',
    'institution_type': 'Depository institution',
    'primary_regulator': 'OCC',
    'filing_office': 'BSA/AML Compliance Department',
}


def _document_checksum(document: Dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of the document, excluding the checksum field itself"""
    content = json.loads(json.dumps(document, default=str))
    content.get('sar_metadata', {}).pop('document_checksum', None)
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode('utf-8')).hexdigest()


def create_sar_document(record: CaseRecord, case_data: CaseData, run_id: str, audit_log_file: str,
                        flagged_transaction_ids: Optional[List[str]] = None,
                        institution: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Assemble a complete SAR document for a case approved at both gates

    Args:
        record: CaseRecord with risk analysis, narrative and both review decisions
        case_data: The case the SAR is filed on
        run_id: Workflow run identifier
        audit_log_file: Audit trail holding every decision for this case
        flagged_transaction_ids: Transactions identified by screening (defaults to all)
        institution: Filing institution details

    Returns:
        SAR document dictionary including a SHA-256 integrity checksum
    """
    risk, narrative = record.risk_analysis, record.compliance_output
    if risk is None or narrative is None or record.drafting_decision is None or record.filing_decision is None:
        raise ValueError("A SAR document requires a risk analysis, a narrative and both review decisions")
    if not (record.drafting_decision.approved and record.filing_decision.approved):
        raise ValueError("A SAR document can only be created for cases approved at both review gates")

    now = datetime.now(timezone.utc)
    customer = case_data.customer
    flagged_ids = set(flagged_transaction_ids or [])
    flagged = [txn for txn in case_data.transactions if txn.transaction_id in flagged_ids] or case_data.transactions
    detected_on = datetime.fromisoformat(case_data.case_created_at)

    document = {
        'sar_metadata': {
            'sar_id': f"SAR-{now.strftime('%Y%m%d')}-{customer.customer_id}-{uuid.uuid4().hex[:8]}",
            'filing_type': 'Initial report',
            'form': 'FinCEN Suspicious Activity Report (simulated)',
            'prepared_at': now.isoformat(),
            'detection_date': detected_on.date().isoformat(),
            'filing_deadline': (detected_on + timedelta(days=SAR_FILING_DEADLINE_DAYS)).date().isoformat(),
            'status': 'approved_for_filing',
            'case_id': case_data.case_id,
            'workflow_run_id': run_id,
            'ai_generated': True,
            'schema_version': '1.0',
        },
        'filing_institution': institution or DEFAULT_INSTITUTION,
        'subject_information': {
            'name': customer.name,
            'customer_id': customer.customer_id,
            'date_of_birth': customer.date_of_birth,
            'tin_last_4_masked': f"XXX-XX-{customer.ssn_last_4}",
            'address': customer.address,
            'occupation': customer.occupation,
            'annual_income': customer.annual_income,
            'customer_since': customer.customer_since,
            'internal_risk_rating': customer.risk_rating,
            'accounts': [acc.model_dump() for acc in case_data.accounts],
        },
        'suspicious_activity': {
            'classification': risk.classification,
            'risk_level': risk.risk_level,
            'confidence_score': risk.confidence_score,
            'activity_period': {
                'start': min(txn.transaction_date for txn in flagged),
                'end': max(txn.transaction_date for txn in flagged),
            },
            'flagged_transaction_count': len(flagged),
            'flagged_amount_total': round(sum(txn.amount for txn in flagged), 2),
            'flagged_transactions': [txn.model_dump() for txn in flagged],
            'key_indicators': risk.key_indicators,
            'screening_signals': record.screening_signals,
        },
        'narrative': {
            'text': narrative.narrative,
            'word_count': len(narrative.narrative.split()),
            'regulatory_citations': narrative.regulatory_citations,
            'completeness_check': narrative.completeness_check,
            'validation_warnings': (record.narrative_validation or {}).get('warnings', []),
        },
        'ai_analysis': {
            'risk_analyst': {'method': 'Chain-of-Thought', 'reasoning': risk.reasoning},
            'compliance_officer': {'method': 'ReACT', 'narrative_reasoning': narrative.narrative_reasoning},
        },
        'human_review': [record.drafting_decision.model_dump(), record.filing_decision.model_dump()],
        'audit_trail': {
            'audit_log_file': audit_log_file,
            'case_id': case_data.case_id,
            'data_sources': case_data.data_sources,
            'agents': ['DataLoader', 'RiskAnalyst', 'HumanReviewer', 'ComplianceOfficer', 'HumanReviewer'],
        },
    }
    document['sar_metadata']['document_checksum'] = _document_checksum(document)
    return document


def save_sar_document(document: Dict[str, Any], output_dir: str) -> str:
    """Write a SAR document to <output_dir>/<sar_id>.json and return the path"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{document['sar_metadata']['sar_id']}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(document, f, indent=2, default=str)
    return path


def verify_sar_document(path: str) -> bool:
    """Recompute a saved SAR document's checksum to confirm it has not been altered"""
    with open(path, encoding='utf-8') as f:
        document = json.load(f)
    return document['sar_metadata'].get('document_checksum') == _document_checksum(document)


# ===== WORKFLOW ORCHESTRATION =====

class SARWorkflow:
    """Two-stage SAR workflow: Risk Analyst -> gate 1 -> Compliance Officer -> gate 2 -> filing

    Attributes:
        run_id: Identifier recorded on every SAR document and the run summary
        records: CaseRecord for every processed alert
    """

    def __init__(self, risk_agent, compliance_agent, review_gate: HumanReviewGate,
                 data_loader: DataLoader, logger: ExplainabilityLogger, sar_output_dir: str,
                 run_id: Optional[str] = None, institution: Optional[Dict[str, str]] = None):
        self.risk_agent = risk_agent
        self.compliance_agent = compliance_agent
        self.review_gate = review_gate
        self.data_loader = data_loader
        self.logger = logger
        self.sar_output_dir = sar_output_dir
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.institution = institution
        self.records: List[CaseRecord] = []

    def process_case(self, customer_record: Dict, account_records: List[Dict],
                     transaction_records: List[Dict], screening: Dict[str, Any]) -> CaseRecord:
        """Run one alerted customer through the full workflow

        The Compliance Officer only runs after gate 1 approval, and a SAR is only
        filed after gate 2 approval. Any exception stops this case (status ERROR)
        without stopping the batch.
        """
        record = CaseRecord(
            customer_id=screening['customer_id'],
            customer_name=screening['name'],
            screening_score=int(screening['screening_score']),
            screening_signals=list(screening['signals']),
            # DataFrame rows turn missing values into NaN - only keep a real typology name
            expected_classification=screening.get('expected_classification')
            if isinstance(screening.get('expected_classification'), str) else None,
        )
        stage = 'case_creation'
        try:
            started = time.perf_counter()
            case = self.data_loader.create_case_from_data(customer_record, account_records, transaction_records)
            record.case_id = case.case_id
            record.stage_times_ms['case_creation'] = (time.perf_counter() - started) * 1000

            stage = 'risk_analysis'
            started = time.perf_counter()
            record.risk_analysis = self.risk_agent.analyze_case(case)
            record.stage_times_ms['risk_analysis'] = (time.perf_counter() - started) * 1000

            stage = 'drafting_gate'
            record.drafting_decision = self.review_gate.review_risk_analysis(case, record.risk_analysis)
            if not record.drafting_decision.approved:
                record.status = CaseStatus.REJECTED_AT_DRAFTING
                return record

            stage = 'narrative_generation'
            started = time.perf_counter()
            try:
                record.compliance_output = self.compliance_agent.generate_compliance_narrative(case, record.risk_analysis)
            except ValueError as e:
                # Logged by the agent; the case goes to manual narrative drafting
                record.status, record.error = CaseStatus.NARRATIVE_FAILED, str(e)
                return record
            finally:
                record.stage_times_ms['narrative_generation'] = (time.perf_counter() - started) * 1000
            record.narrative_validation = getattr(self.compliance_agent, 'last_validation', None)

            stage = 'filing_gate'
            record.filing_decision = self.review_gate.review_narrative(
                case, record.risk_analysis, record.compliance_output, record.narrative_validation)
            if not record.filing_decision.approved:
                record.status = CaseStatus.REJECTED_AT_FILING
                return record

            stage = 'sar_filing'
            started = time.perf_counter()
            document = create_sar_document(record, case, self.run_id, self.logger.log_file,
                                           screening.get('flagged_transaction_ids'), self.institution)
            record.sar_path = save_sar_document(document, self.sar_output_dir)
            record.sar_id = document['sar_metadata']['sar_id']
            record.status = CaseStatus.FILED
            self.logger.log_agent_action(
                agent_type="SARFiling",
                action="file_sar",
                case_id=case.case_id,
                input_data={'customer_id': record.customer_id, 'classification': record.risk_analysis.classification},
                output_data={'sar_id': record.sar_id, 'path': record.sar_path,
                             'checksum': document['sar_metadata']['document_checksum'],
                             'filing_deadline': document['sar_metadata']['filing_deadline']},
                reasoning="SAR approved at both review gates and written to the filing directory",
                execution_time_ms=(time.perf_counter() - started) * 1000,
                success=True
            )
            record.stage_times_ms['sar_filing'] = (time.perf_counter() - started) * 1000
            return record

        except Exception as e:
            record.status, record.error = CaseStatus.ERROR, f"{stage}: {e}"
            self.logger.log_agent_action(
                agent_type="Workflow",
                action="process_case",
                case_id=record.case_id or f"UNASSIGNED_{record.customer_id}",
                input_data={'customer_id': record.customer_id, 'stage': stage},
                output_data={},
                reasoning=f"Workflow stopped at stage '{stage}' for this case",
                execution_time_ms=sum(record.stage_times_ms.values()),
                success=False,
                error_message=str(e)
            )
            return record

    def run(self, alert_queue: List[Dict[str, Any]], customer_records: Dict[str, Dict],
            account_records: List[Dict], transaction_records: List[Dict],
            progress: Optional[Callable[[str], None]] = print) -> List[CaseRecord]:
        """Process every alert in the queue and log a run summary

        Args:
            alert_queue: Screening rows (dicts) for the customers to process
            customer_records: customer_id -> raw customer record
            account_records, transaction_records: Full raw extracts
            progress: Callback for one status line per case (None for silence)
        """
        started = time.perf_counter()
        for position, screening in enumerate(alert_queue, 1):
            record = self.process_case(customer_records[screening['customer_id']], account_records,
                                       transaction_records, screening)
            self.records.append(record)
            if progress:
                finding = (f"{record.risk_analysis.classification}/{record.risk_analysis.risk_level}"
                           if record.risk_analysis else "-")
                progress(f"[{position:>2}/{len(alert_queue)}] {record.customer_id} {record.customer_name:<22} "
                         f"{finding:<24} -> {record.status.value if record.status else 'unknown'}")

        status_counts = {status.value: sum(r.status == status for r in self.records) for status in CaseStatus}
        self.logger.log_agent_action(
            agent_type="Workflow",
            action="run_summary",
            case_id=f"RUN_{self.run_id}",
            input_data={'alerts_processed': len(alert_queue)},
            output_data=status_counts,
            reasoning="Two-stage SAR workflow run completed",
            execution_time_ms=(time.perf_counter() - started) * 1000,
            success=True
        )
        return self.records


# ===== EFFICIENCY METRICS =====

# gpt-4o-mini list prices in USD per 1M tokens - update if the model or pricing changes
DEFAULT_PRICING = {'prompt_tokens': 0.15, 'completion_tokens': 0.60}


def _parse_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Audit entries store output data as a Python-literal string"""
    try:
        value = ast.literal_eval(entry['output_summary'])
        return value if isinstance(value, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def compute_workflow_metrics(records: List[CaseRecord], audit_entries: List[Dict[str, Any]],
                             customers_with_activity: int, alerts_raised: int,
                             pricing: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Funnel, cost, timing, human-decision and quality metrics for a workflow run

    Cost avoided is estimated from the measured average cost of each agent run:
    - screening: customers with activity that never reached the Risk Analyst
    - drafting gate: analyses that never reached the Compliance Officer
    """
    pricing = pricing or DEFAULT_PRICING

    def count(status: CaseStatus) -> int:
        return sum(record.status == status for record in records)

    # ----- per-agent cost and latency from the audit trail -----
    stages: Dict[str, Dict[str, float]] = {}
    for entry in audit_entries:
        if entry['agent_type'] not in ('RiskAnalyst', 'ComplianceOfficer'):
            continue
        details = _parse_summary(entry)
        usage = details.get('token_usage', {})
        stage = stages.setdefault(entry['agent_type'], {'runs': 0, 'api_calls': 0, 'prompt_tokens': 0,
                                                        'completion_tokens': 0, 'latency_ms': 0.0})
        stage['runs'] += 1
        stage['api_calls'] += details.get('attempts', 1)
        stage['prompt_tokens'] += usage.get('prompt_tokens', 0)
        stage['completion_tokens'] += usage.get('completion_tokens', 0)
        stage['latency_ms'] += entry['execution_time_ms']
    for stage in stages.values():
        stage['cost_usd'] = sum(stage[k] / 1_000_000 * price for k, price in pricing.items())
        stage['avg_cost_usd'] = stage['cost_usd'] / stage['runs'] if stage['runs'] else 0.0
        stage['avg_latency_s'] = stage['latency_ms'] / stage['runs'] / 1000 if stage['runs'] else 0.0

    risk_stage = stages.get('RiskAnalyst', {'runs': 0, 'cost_usd': 0.0, 'avg_cost_usd': 0.0})
    compliance_stage = stages.get('ComplianceOfficer', {'runs': 0, 'cost_usd': 0.0, 'avg_cost_usd': 0.0})
    analysed = sum(record.risk_analysis is not None for record in records)
    actual_cost = risk_stage['cost_usd'] + compliance_stage['cost_usd']

    # ----- counterfactual costs -----
    single_stage_cost = risk_stage['cost_usd'] + compliance_stage['avg_cost_usd'] * analysed
    no_screening_cost = (risk_stage['avg_cost_usd'] + compliance_stage['avg_cost_usd']) * customers_with_activity

    # ----- human decisions -----
    decisions = [d for record in records for d in (record.drafting_decision, record.filing_decision) if d]
    gate_stats = {}
    for gate in ('sar_drafting', 'sar_filing'):
        gate_decisions = [d for d in decisions if d.gate == gate]
        gate_stats[gate] = {
            'reviewed': len(gate_decisions),
            'approved': sum(d.approved for d in gate_decisions),
            'rejected': sum(not d.approved for d in gate_decisions),
            'approval_rate': (sum(d.approved for d in gate_decisions) / len(gate_decisions)) if gate_decisions else 0.0,
            'decision_sources': sorted({d.decision_source for d in gate_decisions}),
        }

    # ----- AI quality -----
    analysed_records = [r for r in records if r.risk_analysis]
    with_expectation = [r for r in analysed_records if r.expected_classification]
    agreement = sum(r.risk_analysis.classification == r.expected_classification for r in with_expectation)
    confidences = [r.risk_analysis.confidence_score for r in analysed_records]
    narratives = [r for r in records if r.compliance_output]
    validations = [r.narrative_validation or {} for r in narratives]

    case_times = [sum(r.stage_times_ms.values()) for r in records if r.stage_times_ms]
    return {
        'funnel': {
            'customers_with_activity': customers_with_activity,
            'alerts_raised': alerts_raised,
            'alerts_processed': len(records),
            'risk_analyses': analysed,
            'rejected_at_drafting_gate': count(CaseStatus.REJECTED_AT_DRAFTING),
            'narratives_generated': len(narratives),
            'narrative_failures': count(CaseStatus.NARRATIVE_FAILED),
            'rejected_at_filing_gate': count(CaseStatus.REJECTED_AT_FILING),
            'sars_filed': count(CaseStatus.FILED),
            'errors': count(CaseStatus.ERROR),
        },
        'stage_costs': stages,
        'cost_optimization': {
            'actual_cost_usd': actual_cost,
            'single_stage_cost_usd': single_stage_cost,
            'no_screening_cost_usd': no_screening_cost,
            'compliance_runs_avoided_by_gate': count(CaseStatus.REJECTED_AT_DRAFTING),
            'risk_analyses_avoided_by_screening': customers_with_activity - analysed,
            'savings_vs_single_stage_pct': (1 - actual_cost / single_stage_cost) * 100 if single_stage_cost else 0.0,
            'savings_vs_no_screening_pct': (1 - actual_cost / no_screening_cost) * 100 if no_screening_cost else 0.0,
            'cost_per_filed_sar_usd': actual_cost / count(CaseStatus.FILED) if count(CaseStatus.FILED) else None,
        },
        'timing': {
            'avg_case_time_s': sum(case_times) / len(case_times) / 1000 if case_times else 0.0,
            'total_processing_time_s': sum(case_times) / 1000,
        },
        'human_decisions': gate_stats,
        'ai_quality': {
            'classification_agreement_with_screening': agreement / len(with_expectation) if with_expectation else None,
            'cases_with_screening_expectation': len(with_expectation),
            'confidence_mean': sum(confidences) / len(confidences) if confidences else None,
            'confidence_min': min(confidences) if confidences else None,
            'confidence_max': max(confidences) if confidences else None,
            'narratives_complete_pct': (sum(r.compliance_output.completeness_check for r in narratives)
                                        / len(narratives) * 100) if narratives else None,
            'narratives_within_word_limit_pct': (sum(v.get('within_word_limit', False) for v in validations)
                                                 / len(validations) * 100) if validations else None,
            'narratives_with_warnings': sum(bool(v.get('warnings')) for v in validations),
            'avg_narrative_words': (sum(v.get('word_count', 0) for v in validations) / len(validations))
            if validations else None,
        },
    }
