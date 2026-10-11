# Compliance Officer Agent - ReACT Implementation

"""
Compliance Officer Agent Module

This agent generates regulatory-compliant SAR narratives using ReACT prompting.
It takes risk analysis results and creates structured documentation for
FinCEN submission.

How it works:
1. The prompt combines the Risk Analyst's findings with the case facts the
   narrative needs: subject, accounts, activity statistics and the relevant
   transactions (routine activity is left out).
2. The LLM follows a ReACT framework: a REASONING phase (review findings,
   identify regulatory requirements, plan WHO/WHAT/WHEN/WHERE/WHY) and an
   ACTION phase (draft, verify with an OBSERVATION, finalize as JSON).
3. The code then acts as the external observer: it validates the word limit,
   the five narrative elements and the citations. A narrative over 120 words,
   or one with speculative language or other typologies, is sent back with
   the observation for revision. A narrative still over the limit after all
   attempts raises a ValueError; a flagged but valid draft is kept as fallback.
4. Every narrative - successful or not - is written to the audit trail with
   the ReACT trace, the validation report and token usage.
"""

import re
import time
from collections import Counter
import openai
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

try:
    from .foundation_sar import (
        ComplianceOfficerOutput,
        ExplainabilityLogger,
        CaseData,
        RiskAnalystOutput,
        TransactionData
    )
    from .agent_utils import (
        extract_json_object, parse_json_lenient, strip_json_blocks, truncate_text,
        new_usage_counter, record_usage, summarize_case_activity, years_between,
        is_cash_transaction
    )
except ImportError:
    from foundation_sar import (
        ComplianceOfficerOutput,
        ExplainabilityLogger,
        CaseData,
        RiskAnalystOutput,
        TransactionData
    )
    from agent_utils import (
        extract_json_object, parse_json_lenient, strip_json_blocks, truncate_text,
        new_usage_counter, record_usage, summarize_case_activity, years_between,
        is_cash_transaction
    )

from pydantic import ValidationError

# Load environment variables
load_dotenv()

WORD_LIMIT = 120                    # SAR narrative limit for this program
NARRATIVE_REASONING_MAX_CHARS = 500 # ComplianceOfficerOutput.narrative_reasoning limit
SAR_REVIEW_THRESHOLD = 5_000        # SAR filing threshold for banks (31 CFR 1020.320)
MAX_PROMPT_TRANSACTIONS = 30        # Relevant transactions shown to the model

MANDATORY_CITATION = "31 CFR 1020.320 (SAR filing requirement)"

# Citation guidance per typology - the prompt restricts the model to these sources
TYPOLOGY_CITATIONS = {
    'Structuring': ["31 USC 5324 (Structuring)", "31 CFR 1010.311 (CTR requirement)"],
    'Money_Laundering': ["18 USC 1956 (Money laundering)", "18 USC 1957 (Transactions in criminally derived property)"],
    'Fraud': ["18 USC 1344 (Bank fraud)", "18 USC 1343 (Wire fraud)"],
    'Sanctions': ["31 CFR Part 501 (OFAC sanctions)", "50 USC 1705 (IEEPA penalties)"],
    'Other': ["31 USC 5318(g) (BSA suspicious activity reporting)"],
}

# Patterns identifying the regulations the prompt allows; anything else is flagged for review
RECOGNIZED_CITATION_PATTERNS = [
    r"1020\.320", r"5318", r"21\.11", r"5324", r"1010\.311", r"1956", r"1957",
    r"1344", r"1343", r"\bpart 501\b|\b501\b", r"1705|ieepa", r"fincen", r"\bbsa\b|bank secrecy act", r"ofac",
]

_DATE_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}", re.IGNORECASE)
_TIMEFRAME_PATTERN = re.compile(
    r"\b(consecutive|between|from .{1,30} to|within|over (a |the )?\w+ (days|weeks|months)|"
    r"during|period|daily|weekly)\b", re.IGNORECASE)
_AMOUNT_PATTERN = re.compile(r"\$\s?\d")
_ACTIVITY_PATTERN = re.compile(r"deposit|withdraw|transfer|wire|transaction|payment|purchase", re.IGNORECASE)
# WHERE must be named: generic phrases such as "across three locations" or "multiple branches" do not count.
# A specific channel is a valid WHERE (FinCEN narrative guidance: branch, channel, account or jurisdiction).
_CHANNEL_PATTERN = re.compile(r"\b(atm|online|internet banking|mobile|wire|ach|teller)\b", re.IGNORECASE)
# Fallback when no case data is available: a branch/account identifier or a capitalised branch name
_NAMED_LOCATION_PATTERN = re.compile(
    r"\bBranch_\w+|\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s[Bb]ranch(?:es)?\b|\b[A-Z]+_\w*ACC\w*\b")
# Location values too generic to count as a name on their own
_GENERIC_LOCATION_NAMES = {'local', 'online', 'ach', 'atm', 'branch', 'unknown', 'main'}
_SUSPICION_PATTERN = re.compile(
    r"suspicious|structur|launder|layering|fraud|sanction|ofac|evade|avoid|inconsistent|"
    r"unusual|indicat|red flag|threshold", re.IGNORECASE)

# Mentions of a typology other than the analyst's classification are flagged as inconsistent
TYPOLOGY_PATTERNS = {
    'Structuring': r"structur",
    'Money_Laundering': r"launder|layering",
    'Fraud': r"\bfraud",
    'Sanctions': r"sanction|ofac",
}
# Statements about intent, guilt or the source of funds go beyond what a SAR narrative should assert
_SPECULATION_PATTERN = re.compile(
    r"legitima\w*|source of (the |these |his |her |their )?funds|illicit|criminal\w*|guilt\w*|intent(ion)?\b",
    re.IGNORECASE)
_DOLLAR_AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")
_STANDARD_THRESHOLDS = {3_000.0, 5_000.0, 10_000.0}   # wire monitoring, SAR and CTR thresholds

REGULATORY_TERMS = [
    "suspicious activity", "Bank Secrecy Act", "BSA", "Currency Transaction Report", "CTR",
    "reporting threshold", "structuring", "money laundering", "layering", "OFAC", "sanctions", "fraud",
]

COMPLIANCE_OFFICER_SYSTEM_PROMPT = """You are a Senior BSA/AML Compliance Officer at a U.S. bank, with 15 years of experience writing and quality-reviewing Suspicious Activity Report (SAR) narratives for FinCEN. You turn analyst findings into clear, factual, examiner-ready narratives.

TASK
Write the SAR narrative for the case, based on the Risk Analyst's findings and the case facts provided. The narrative is filed with FinCEN, so it must be accurate, complete, objective and no longer than 120 words.

ReACT FRAMEWORK - work in two phases, keeping each step to 1-2 sentences:

REASONING PHASE (think before writing)
THOUGHT 1 - Review findings: What did the Risk Analyst conclude, and which case facts support it?
THOUGHT 2 - Regulatory requirements: Which statutes and filing obligations apply to this typology?
THOUGHT 3 - Narrative plan: The facts for WHO (subject name with customer ID, occupation), WHAT (activity type, amounts, count), WHEN (dates or date range), WHERE (the named branches, account IDs, channels such as ATM or wire, or jurisdictions - never just "several locations") and WHY (red flags and why the activity is suspicious).

ACTION PHASE (produce the filing)
ACTION 1 - Draft: Write the narrative using only the facts from your plan.
ACTION 2 - Verify: Check the draft against the narrative requirements and the 120-word limit, tighten it if needed, and record the result as "OBSERVATION: <word count> words; elements covered: ...".
ACTION 3 - Finalize: Output the JSON result.

NARRATIVE REQUIREMENTS
- Word limit: at most 120 words (aim for 80-110), written as one paragraph of plain prose - no bullet points or headings.
- Cover all five elements: WHO, WHAT, WHEN, WHERE, WHY.
- Name the WHERE explicitly: the specific branches (e.g. "at the Northgate Mall and Westside Plaza branches"), the account ID (e.g. "CUST_0053_ACC_3"), the channel (ATM, wire, ACH, online) or the jurisdiction. "Across several locations" or "multiple branches" is not sufficient.
- Identify the subject by name with customer ID in parentheses, e.g. "Tracy Lewis (CUST_0053)".
- Quote figures and dates exactly as given in the ACTIVITY SUMMARY, the analyst's key indicators or the transaction list (e.g. "$67,569.06", "between 2025-07-25 and 2025-07-31"). Do not calculate new totals or ranges, distinguish individual amounts from totals, and never invent facts.
- Describe only the typology the Risk Analyst classified - do not suggest additional typologies.
- Objective, professional regulatory tone: third person, past tense. State what was observed and why it is suspicious, describing it as "consistent with" a typology - never assert guilt, and do not comment on the subject's intent or on the legitimacy or source of the funds (avoid words such as "intent", "legitimacy", "illicit" and "criminal").
- Description/type contradictions are data-quality issues, not evidence; activity on closed accounts is a valid red flag.
- Use BSA/AML terminology (e.g. Bank Secrecy Act, Currency Transaction Report threshold, structuring, layering, OFAC sanctions).
- Name the primary statute once in the narrative (e.g. "potential structuring under 31 USC 5324"), and cite each statute only for the conduct it covers - 31 CFR 1020.320 is the filing requirement, not an offence.
- Do not include the SSN or identifiers other than the subject's name, customer ID and account IDs.

REGULATORY CITATIONS (cite only from this list)
- 31 CFR 1020.320 - SAR filing requirement for banks (always include)
- 31 USC 5318(g) - BSA authority for suspicious activity reporting
- 12 CFR 21.11 - SAR filing requirement for national banks (OCC)
- 31 USC 5324 - Structuring transactions to evade reporting requirements
- 31 CFR 1010.311 - Currency Transaction Report requirement
- 18 USC 1956 / 18 USC 1957 - Money laundering
- 18 USC 1344 - Bank fraud; 18 USC 1343 - Wire fraud
- 31 CFR Part 501 - OFAC sanctions reporting; 50 USC 1705 - IEEPA penalties
- FinCEN SAR Narrative Guidance

OUTPUT FORMAT
Write the REASONING PHASE (THOUGHT 1-3) and ACTION 1-2, then give ACTION 3 as a single JSON object in a ```json code block with exactly these fields:

```json
{
  "narrative": "SAR narrative text, at most 120 words",
  "narrative_reasoning": "How the narrative was constructed from the findings, at most 500 characters",
  "regulatory_citations": ["31 CFR 1020.320 (SAR filing requirement)", "..."],
  "completeness_check": true
}
```

RULES
- "regulatory_citations": 2-5 citations from the list above, always including 31 CFR 1020.320.
- "completeness_check": true only if the narrative covers WHO, WHAT, WHEN, WHERE and WHY within the 120-word limit.
- Write nothing after the JSON block."""


class NarrativeWordLimitError(ValueError):
    """Raised internally when a narrative exceeds the word limit"""


class ComplianceOfficerAgent:
    """
    Compliance Officer agent using ReACT prompting framework.

    Attributes:
        client: OpenAI client (Vocareum-routed in the classroom)
        logger: ExplainabilityLogger receiving one audit entry per narrative
        model: Chat model used for narrative generation
        temperature: Sampling temperature - lower than the analyst for consistent regulatory language
        max_tokens: Completion budget covering the ReACT trace plus the JSON
        max_attempts: Total LLM attempts per narrative (word-limit and parse failures are retried)
        system_prompt: ReACT system prompt
        last_raw_response: Raw text of the most recent LLM response
        last_react_trace: REASONING/ACTION text preceding the JSON
        last_validation: Compliance validation report for the most recent narrative
        usage_totals: Cumulative token usage and call counts across all narratives
    """

    def __init__(self, openai_client: openai.OpenAI, explainability_logger: ExplainabilityLogger,
                 model: str = "gpt-4o-mini", temperature: float = 0.2,
                 max_tokens: int = 800, max_attempts: int = 3) -> None:
        """Initialize the Compliance Officer Agent

        Args:
            openai_client: OpenAI client instance
            explainability_logger: Logger for audit trails
            model: OpenAI model to use
            temperature: Sampling temperature (0.2 for consistent regulatory language)
            max_tokens: Maximum completion tokens per call
            max_attempts: Attempts per narrative when the output fails the word limit or cannot be parsed
        """
        self.client = openai_client
        self.logger = explainability_logger
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max(1, max_attempts)

        self.system_prompt = COMPLIANCE_OFFICER_SYSTEM_PROMPT

        self.last_raw_response: Optional[str] = None
        self.last_react_trace: Optional[str] = None
        self.last_validation: Optional[Dict[str, Any]] = None
        self.usage_totals: Dict[str, int] = {'api_calls': 0, **new_usage_counter()}

    def generate_compliance_narrative(self, case_data: CaseData,
                                      risk_analysis: RiskAnalystOutput) -> ComplianceOfficerOutput:
        """
        Generate regulatory-compliant SAR narrative using ReACT framework.

        Args:
            case_data: Validated case the narrative describes
            risk_analysis: Risk Analyst findings for the case

        Returns:
            Validated ComplianceOfficerOutput. completeness_check is True only if
            both the model and the compliance validation confirm a complete narrative.

        Raises:
            ValueError: If inputs are missing, the narrative exceeds the 120 word limit,
                or the output cannot be parsed/validated after all attempts
            RuntimeError: If the OpenAI API call fails
        """
        if case_data is None or risk_analysis is None:
            raise ValueError("case_data and risk_analysis are required to generate a narrative")

        start_time = time.perf_counter()
        input_summary = {
            'customer_id': case_data.customer.customer_id,
            'classification': risk_analysis.classification,
            'risk_level': risk_analysis.risk_level,
            'confidence_score': risk_analysis.confidence_score,
            'model': self.model
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._format_case_for_prompt(case_data, risk_analysis)}
        ]
        call_usage = new_usage_counter()
        last_error: Optional[Exception] = None
        react_trace = ""
        fallback: Optional[tuple] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            except Exception as e:
                self._log_failure(case_data.case_id, input_summary, start_time,
                                  reasoning=f"OpenAI API call failed on attempt {attempt}: {type(e).__name__}",
                                  error=e, attempts=attempt, usage=call_usage)
                raise RuntimeError(f"Compliance Officer API call failed: {e}") from e

            record_usage(response, call_usage, self.usage_totals)
            choice = response.choices[0]
            content = choice.message.content
            self.last_raw_response = content
            # A revision after an OBSERVATION may skip the reasoning phase - keep the original trace
            react_trace = strip_json_blocks(content) or react_trace

            try:
                parsed = parse_json_lenient(self._extract_json_from_response(content))
                normalized, adjustments = self._normalize_output(parsed)
                word_count = len(str(normalized.get('narrative', '')).split())
                if word_count > WORD_LIMIT:
                    raise NarrativeWordLimitError(
                        f"Narrative exceeds {WORD_LIMIT} word limit ({word_count} words)")
                result = ComplianceOfficerOutput(**normalized)
            except (ValueError, TypeError) as e:
                # ValidationError and NarrativeWordLimitError subclass ValueError
                last_error = e
                if getattr(choice, 'finish_reason', None) == 'length':
                    last_error = ValueError(f"Response truncated at max_tokens={self.max_tokens}: {e}")
                if attempt < self.max_attempts:
                    messages = messages[:2] + [
                        {"role": "assistant", "content": content or ""},
                        {"role": "user", "content": self._build_observation_message(last_error)}
                    ]
                continue

            # External observation: validate the narrative before it can be filed
            validation = self._validate_narrative_compliance(
                result.narrative, case_data=case_data, citations=result.regulatory_citations,
                risk_analysis=risk_analysis)
            if result.completeness_check and not validation['is_complete']:
                adjustments.append(
                    "completeness_check overridden to False: " + "; ".join(validation['issues']))
                result = result.model_copy(update={'completeness_check': False})

            # Tone and typology problems are revisable: send the observation back while attempts remain,
            # keeping this valid draft in case the revision comes back worse
            revision_reasons = self._revision_reasons(validation, risk_analysis)
            if revision_reasons and attempt < self.max_attempts:
                fallback = (result, validation, adjustments + ["kept as fallback: " + "; ".join(revision_reasons)],
                            react_trace, attempt)
                messages = messages[:2] + [
                    {"role": "assistant", "content": content or ""},
                    {"role": "user", "content": self._build_revision_message(revision_reasons, risk_analysis)}
                ]
                continue
            if revision_reasons:  # final attempt and still flagged - the reviewer sees the warnings
                adjustments.append("still flagged after revision: " + "; ".join(revision_reasons))
            elif fallback is not None:
                adjustments.append(f"narrative revised after OBSERVATION: {fallback[2][-1].split(': ', 1)[-1]}")

            return self._log_success(case_data, input_summary, start_time, result, validation,
                                     adjustments, react_trace, attempt, call_usage)

        # A valid but flagged draft beats no narrative - the reviewer sees its warnings
        if fallback is not None:
            result, validation, adjustments, trace, _ = fallback
            return self._log_success(case_data, input_summary, start_time, result, validation,
                                     adjustments, trace, self.max_attempts, call_usage)

        # All attempts exhausted - classify the failure for the audit trail
        if isinstance(last_error, NarrativeWordLimitError):
            self._log_failure(case_data.case_id, input_summary, start_time,
                              reasoning=f"Word limit validation failed after {self.max_attempts} attempt(s)",
                              error=last_error, attempts=self.max_attempts, usage=call_usage)
            raise ValueError(f"Compliance narrative exceeds {WORD_LIMIT} word limit after "
                             f"{self.max_attempts} attempt(s): {last_error}") from last_error

        if isinstance(last_error, ValidationError):
            self._log_failure(case_data.case_id, input_summary, start_time,
                              reasoning=f"Schema validation failed after {self.max_attempts} attempt(s)",
                              error=last_error, attempts=self.max_attempts, usage=call_usage)
            raise ValueError(f"Compliance Officer output failed schema validation: {last_error}") from last_error

        self._log_failure(case_data.case_id, input_summary, start_time,
                          reasoning=f"JSON parsing failed after {self.max_attempts} attempt(s)",
                          error=last_error, attempts=self.max_attempts, usage=call_usage)
        raise ValueError(f"Failed to parse Compliance Officer JSON output: {last_error}") from last_error

    def _log_success(self, case_data: CaseData, input_summary: Dict[str, Any], start_time: float,
                     result: ComplianceOfficerOutput, validation: Dict[str, Any], adjustments: List[str],
                     react_trace: str, attempts: int, call_usage: Dict[str, int]) -> ComplianceOfficerOutput:
        """Record the accepted narrative on the agent and in the audit trail"""
        self.last_validation = validation
        self.last_react_trace = react_trace
        self.logger.log_agent_action(
            agent_type="ComplianceOfficer",
            action="generate_narrative",
            case_id=case_data.case_id,
            input_data=input_summary,
            output_data={
                **result.model_dump(),
                'word_count': validation['word_count'],
                'validation': validation,
                'react_trace': react_trace,
                'normalizations': adjustments,
                'attempts': attempts,
                'token_usage': call_usage
            },
            reasoning=result.narrative_reasoning,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            success=True
        )
        return result

    @staticmethod
    def _revision_reasons(validation: Dict[str, Any], risk_analysis: RiskAnalystOutput) -> List[str]:
        """Validation findings the model can fix by rewording with facts it already has"""
        reasons = []
        if not validation.get('elements', {}).get('where', True):
            examples = validation.get('where_candidates', [])[:4]
            reasons.append("missing WHERE - no specific branch, account ID, channel or jurisdiction named"
                           + (f" (case facts include: {', '.join(examples)})" if examples else ""))
        if validation.get('speculative_terms'):
            reasons.append(f"speculative language {validation['speculative_terms']}")
        if validation.get('other_typologies'):
            reasons.append(f"mentions typologies other than {risk_analysis.classification}: "
                           f"{validation['other_typologies']}")
        return reasons

    def _build_revision_message(self, reasons: List[str], risk_analysis: RiskAnalystOutput) -> str:
        """ReACT observation asking for a reworded narrative"""
        return (
            f"OBSERVATION: compliance validation flagged the narrative - {'; '.join(reasons)}.\n\n"
            f"Revise the narrative: name the WHERE explicitly (the specific branches, account ID or channel from "
            f"the case facts, not \"several locations\"), describe only the observed activity and why it is "
            f"consistent with {risk_analysis.classification}, remove any statement about intent, legitimacy, "
            "illicit or criminal conduct, or the source of funds, and do not mention other typologies. Keep every figure, date, "
            "the subject identification and the citations. Reply with a one-line THOUGHT, then the corrected "
            "JSON in a ```json code block with the fields narrative, narrative_reasoning, "
            "regulatory_citations and completeness_check."
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

    def _normalize_output(self, data: Any) -> tuple:
        """Coerce near-miss LLM output into the ComplianceOfficerOutput contract

        Fixes formatting slips and applies one policy rule: the SAR filing
        requirement (31 CFR 1020.320) is always cited. The narrative text is
        never rewritten.

        Returns:
            (normalized_dict, list of human-readable adjustments for the audit trail)
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

        data = dict(data)
        adjustments: List[str] = []

        if isinstance(data.get('narrative'), str):
            narrative = " ".join(data['narrative'].split())  # collapse line breaks / double spaces
            if narrative != data['narrative']:
                adjustments.append("narrative whitespace collapsed to a single paragraph")
            data['narrative'] = narrative

        reasoning = data.get('narrative_reasoning')
        if isinstance(reasoning, str) and len(reasoning) > NARRATIVE_REASONING_MAX_CHARS:
            truncated = truncate_text(reasoning, NARRATIVE_REASONING_MAX_CHARS)
            data['narrative_reasoning'] = truncated
            adjustments.append(f"narrative_reasoning truncated from {len(reasoning)} to {len(truncated)} characters")

        citations = data.get('regulatory_citations')
        if isinstance(citations, str):
            citations = [citations]
            adjustments.append("regulatory_citations string wrapped in a list")
        if isinstance(citations, list):
            cleaned: List[str] = []
            for citation in citations:
                citation = str(citation).strip()
                if citation and citation.lower() not in {c.lower() for c in cleaned}:
                    cleaned.append(citation)
            if not any(re.search(r"1020\.320", c) for c in cleaned):
                cleaned.insert(0, MANDATORY_CITATION)
                adjustments.append(f"added mandatory citation '{MANDATORY_CITATION}'")
            data['regulatory_citations'] = cleaned

        completeness = data.get('completeness_check')
        if isinstance(completeness, str) and completeness.strip().lower() in ('true', 'false'):
            data['completeness_check'] = completeness.strip().lower() == 'true'
            adjustments.append(f"completeness_check '{completeness}' parsed as boolean")

        return data, adjustments

    def _build_observation_message(self, error: Exception) -> str:
        """ReACT observation fed back to the model when its narrative fails validation"""
        if isinstance(error, NarrativeWordLimitError):
            guidance = (f"Revise the narrative to at most {WORD_LIMIT} words (target 100) by removing "
                        "secondary detail - keep WHO, WHAT, WHEN, WHERE and WHY, the exact key figures "
                        "and the primary statute.")
        else:
            guidance = "Return a valid JSON object with exactly the required fields."
        return (
            f"OBSERVATION: your output failed compliance validation - {str(error)[:300]}\n\n"
            f"{guidance} Reply with a one-line THOUGHT, then the corrected JSON in a ```json code block "
            "with the fields narrative, narrative_reasoning, regulatory_citations and completeness_check."
        )

    def _format_case_for_prompt(self, case_data: CaseData, risk_analysis: RiskAnalystOutput) -> str:
        """User prompt: subject, accounts, analyst findings, activity facts and citation guidance"""
        customer = case_data.customer
        last_activity = max(txn.transaction_date for txn in case_data.transactions)
        tenure = years_between(customer.customer_since, last_activity)
        income = f"${customer.annual_income:,}" if customer.annual_income is not None else "Not reported"

        relevant = self._select_relevant_transactions(case_data)
        omitted = len(case_data.transactions) - len(relevant)
        accounts = "\n".join(
            f"- {acc.account_id} ({acc.account_type}, {acc.status}), opened {acc.opening_date}"
            for acc in case_data.accounts
        ) or "- No account records available"

        citations = [MANDATORY_CITATION] + TYPOLOGY_CITATIONS.get(risk_analysis.classification, [])

        return "\n".join([
            f"SAR NARRATIVE REQUEST - CASE {case_data.case_id}",
            "",
            "SUBJECT",
            f"- Name: {customer.name}",
            f"- Customer ID: {customer.customer_id}",
            f"- Occupation: {customer.occupation or 'Not reported'}; annual income: {income}",
            f"- Customer since: {customer.customer_since} ({tenure} year{'s' if tenure != 1 else ''}); "
            f"internal risk rating: {customer.risk_rating}",
            "",
            "ACCOUNTS",
            accounts,
            "",
            "RISK ANALYST FINDINGS (Chain-of-Thought analysis)",
            self._format_risk_analysis_for_prompt(risk_analysis),
            "",
            "ACTIVITY SUMMARY",
            summarize_case_activity(case_data),
            "",
            f"RELEVANT TRANSACTIONS ({len(relevant)} shown"
            + (f"; {omitted} routine transactions omitted" if omitted else "") + ")",
            self._format_transactions_for_compliance(relevant),
            "",
            "CITATION GUIDANCE",
            f"- Recommended for {risk_analysis.classification}: " + "; ".join(citations),
            "",
            "Follow the ReACT framework: REASONING PHASE (THOUGHT 1-3), then ACTION PHASE ending with the JSON."
        ])

    def _select_relevant_transactions(self, case_data: CaseData) -> List[TransactionData]:
        """Transactions the narrative may need: cash, wires and amounts at the SAR review threshold

        Small cases are shown in full. Larger cases keep the largest relevant
        transactions (capped at MAX_PROMPT_TRANSACTIONS), in date order.
        """
        transactions = case_data.transactions
        if len(transactions) <= MAX_PROMPT_TRANSACTIONS:
            return list(transactions)

        relevant = [txn for txn in transactions
                    if is_cash_transaction(txn) or 'Wire' in txn.transaction_type
                    or txn.amount >= SAR_REVIEW_THRESHOLD]
        if not relevant:
            relevant = list(transactions)
        if len(relevant) > MAX_PROMPT_TRANSACTIONS:
            relevant = sorted(relevant, key=lambda txn: txn.amount, reverse=True)[:MAX_PROMPT_TRANSACTIONS]
        return sorted(relevant, key=lambda txn: txn.transaction_date)

    def _format_transactions_for_compliance(self, transactions: List[TransactionData]) -> str:
        """Numbered list leading with date and amount, as narrative facts are quoted"""
        if not transactions:
            return "- No transactions"

        lines = []
        for i, txn in enumerate(transactions, 1):
            line = f"{i}. {txn.transaction_date}: ${txn.amount:,.2f} {txn.transaction_type} via {txn.method}"
            if txn.location:
                line += f" at {txn.location}"
            if txn.counterparty:
                line += f" (counterparty: {txn.counterparty})"
            line += f" - {txn.description} [{txn.account_id}]"
            lines.append(line)
        return "\n".join(lines)

    def _format_risk_analysis_for_prompt(self, risk_analysis: RiskAnalystOutput) -> str:
        """Format risk analysis results for compliance prompt

        Includes classification and confidence, key suspicious indicators,
        risk level assessment and the analyst's reasoning.
        """
        indicators = "\n".join(f"  - {indicator}" for indicator in risk_analysis.key_indicators) or "  - None listed"
        return "\n".join([
            f"- Classification: {risk_analysis.classification} (confidence {risk_analysis.confidence_score:.2f})",
            f"- Risk level: {risk_analysis.risk_level}",
            "- Key indicators:",
            indicators,
            f"- Analyst reasoning: {risk_analysis.reasoning}"
        ])

    def _validate_narrative_compliance(self, narrative: str, case_data: Optional[CaseData] = None,
                                       citations: Optional[List[str]] = None,
                                       risk_analysis: Optional[RiskAnalystOutput] = None) -> Dict[str, Any]:
        """Validate narrative meets regulatory requirements

        Hard requirements (all must hold for is_complete): word count within the
        limit, WHO, WHAT, WHEN, WHERE and WHY present, at least one citation, and
        no SSN fragment.

        WHERE must be named, as FinCEN narrative guidance requires: a branch or
        jurisdiction recorded on the case's transactions (e.g. "Branch_Westside_Plaza"
        or "Westside Plaza branch"), an account ID, or a specific channel (ATM, wire,
        ACH, online, mobile, teller). Phrases like "across three locations" fail.
        When the case records no transaction locations, the transaction method
        (e.g. "cash") is the most specific WHERE available and is accepted.

        Warnings for the human reviewer (accuracy and tone, not completeness):
        citations outside the approved list, no BSA/AML terminology, dollar amounts
        that cannot be traced to the case facts, typologies other than the analyst's
        classification, and speculative language about intent or source of funds.

        Args:
            narrative: SAR narrative text
            case_data: Case the narrative describes (enables subject, SSN and amount checks)
            citations: Regulatory citations accompanying the narrative
            risk_analysis: Analyst findings (enables typology and amount checks)

        Returns:
            Validation report with word count, element checks, issues and warnings
        """
        word_count = len(narrative.split())
        text = narrative.lower()

        if case_data is not None:
            subject_terms = [case_data.customer.name, case_data.customer.customer_id]
            who = any(term.lower() in text for term in subject_terms)
        else:
            who = bool(re.search(r"\b(customer|subject|account holder|client)\b", text))
        where, where_candidates = self._check_where(narrative, case_data)

        elements = {
            'who': who,
            'what': bool(_AMOUNT_PATTERN.search(narrative)) and bool(_ACTIVITY_PATTERN.search(text)),
            'when': bool(_DATE_PATTERN.search(narrative)) or bool(_TIMEFRAME_PATTERN.search(text)),
            'where': where,
            'why': bool(_SUSPICION_PATTERN.search(text)),
        }

        citations = citations or []
        unrecognized = [c for c in citations
                        if not any(re.search(pattern, c, re.IGNORECASE) for pattern in RECOGNIZED_CITATION_PATTERNS)]
        terminology = [term for term in REGULATORY_TERMS
                       if re.search(rf"\b{re.escape(term.lower())}\b", text)]

        issues: List[str] = []
        if word_count > WORD_LIMIT:
            issues.append(f"narrative exceeds {WORD_LIMIT} word limit ({word_count} words)")
        for element in ('who', 'what', 'when', 'where', 'why'):
            if not elements[element]:
                issues.append(f"missing {element.upper()} element")
        if not citations:
            issues.append("no regulatory citations")
        if case_data is not None and re.search(rf"\b{re.escape(case_data.customer.ssn_last_4)}\b", narrative):
            issues.append("narrative contains the SSN fragment")

        warnings: List[str] = []
        if unrecognized:
            warnings.append(f"citations outside the approved list: {unrecognized}")
        if not terminology:
            warnings.append("no BSA/AML terminology found")

        ungrounded = self._find_ungrounded_amounts(narrative, case_data, risk_analysis) if case_data else []
        if ungrounded:
            warnings.append(f"amounts not traceable to the case facts: {ungrounded}")

        other_typologies = []
        if risk_analysis is not None:
            other_typologies = [typology for typology, pattern in TYPOLOGY_PATTERNS.items()
                                if typology != risk_analysis.classification and re.search(pattern, text)]
            if other_typologies:
                warnings.append(f"mentions typologies other than the {risk_analysis.classification} "
                                f"classification: {other_typologies}")

        speculation = sorted({match.group(0).lower() for match in _SPECULATION_PATTERN.finditer(narrative)})
        if speculation:
            warnings.append(f"speculative language about intent or source of funds: {speculation}")

        return {
            'word_count': word_count,
            'within_word_limit': word_count <= WORD_LIMIT,
            'elements': elements,
            'regulatory_terms': terminology,
            'citation_count': len(citations),
            'unrecognized_citations': unrecognized,
            'where_candidates': where_candidates,
            'ungrounded_amounts': ungrounded,
            'other_typologies': other_typologies,
            'speculative_terms': speculation,
            'issues': issues,
            'warnings': warnings,
            'is_complete': not issues
        }

    @staticmethod
    def _location_names(location: str) -> List[str]:
        """Ways a recorded location can be named, e.g. 'Branch_Westside_Plaza' -> 'Westside Plaza'"""
        names = [location]
        readable = re.sub(r"(?i)^branch[_\s]+|[_\s]+branch$", "", location).replace('_', ' ').strip()
        if readable and readable != location and len(readable) >= 3 \
                and readable.lower() not in _GENERIC_LOCATION_NAMES:
            names.append(readable)
        return names

    def _check_where(self, narrative: str, case_data: Optional[CaseData]) -> tuple:
        """Whether the narrative names a specific WHERE, plus names it could use

        Returns:
            (where_present, candidate names from the case facts for revision guidance)
        """
        text = narrative.lower()
        if case_data is None:
            return bool(_NAMED_LOCATION_PATTERN.search(narrative) or _CHANNEL_PATTERN.search(text)), []

        recorded = sorted({txn.location for txn in case_data.transactions if txn.location})
        names = [name for location in recorded for name in self._location_names(location)]
        account_ids = [acc.account_id for acc in case_data.accounts]
        named = (any(re.search(rf"\b{re.escape(name.lower())}\b", text) for name in names)
                 or any(acc_id.lower() in text for acc_id in account_ids)
                 or bool(_CHANNEL_PATTERN.search(text)))
        if not named and not recorded:
            # No location recorded in the data: the transaction method is the most specific WHERE available
            methods = {txn.method.lower() for txn in case_data.transactions}
            named = any(re.search(rf"\b{re.escape(method)}\b", text) for method in methods)

        # Suggest the locations of cash/wire activity first - that is where the suspicious activity happened
        activity = Counter(txn.location for txn in case_data.transactions
                           if txn.location and (is_cash_transaction(txn) or 'Wire' in txn.transaction_type))
        ranked = sorted(recorded, key=lambda location: (-activity[location], location))
        readable = [self._location_names(location)[-1] for location in ranked]
        return named, readable + account_ids

    def _find_ungrounded_amounts(self, narrative: str, case_data: CaseData,
                                 risk_analysis: Optional[RiskAnalystOutput] = None) -> List[str]:
        """Dollar amounts in the narrative that do not appear in the case facts

        Known figures are every transaction and balance, reported income, the
        activity-summary statistics, the analyst's indicators and the standard
        regulatory thresholds. A narrative amount matches when it equals a known
        figure to the cent, or a known figure rounded to the stated precision
        (e.g. "$67,569" or "$67.6k"-style rounding to whole dollars).
        """
        known = set(_STANDARD_THRESHOLDS)
        known.update(txn.amount for txn in case_data.transactions)
        for acc in case_data.accounts:
            known.update((acc.current_balance, acc.average_monthly_balance))
        if case_data.customer.annual_income is not None:
            known.add(float(case_data.customer.annual_income))

        fact_text = summarize_case_activity(case_data)
        if risk_analysis is not None:
            fact_text += " " + " ".join(risk_analysis.key_indicators) + " " + risk_analysis.reasoning
        known.update(float(value.replace(',', '')) for value in _DOLLAR_AMOUNT.findall(fact_text))

        ungrounded = []
        for raw in _DOLLAR_AMOUNT.findall(narrative):
            amount = float(raw.replace(',', ''))
            has_cents = '.' in raw
            if not any(abs(amount - figure) < 0.01 or (not has_cents and abs(amount - round(figure)) < 1)
                       for figure in known):
                ungrounded.append(f"${raw}")
        return ungrounded

    def _log_failure(self, case_id: str, input_summary: Dict[str, Any], start_time: float,
                     reasoning: str, error: Optional[Exception], attempts: int,
                     usage: Dict[str, int]) -> None:
        """Write a failed narrative generation to the audit trail"""
        self.logger.log_agent_action(
            agent_type="ComplianceOfficer",
            action="generate_narrative",
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

# ===== REACT PROMPTING HELPERS =====

def create_react_framework() -> Dict[str, List[str]]:
    """Helper function showing ReACT structure

    The agent's system prompt expands these phases into THOUGHT 1-3 and
    ACTION 1-3, with the code-level validator acting as the external
    OBSERVATION when a narrative fails its checks:

    **REASONING Phase:**
    1. Review the risk analyst's findings
    2. Assess regulatory narrative requirements
    3. Identify key compliance elements
    4. Consider narrative structure

    **ACTION Phase:**
    1. Draft concise narrative (≤120 words)
    2. Include specific details and amounts
    3. Reference suspicious activity pattern
    4. Ensure regulatory language
    """
    return {
        "reasoning_phase": [
            "Review risk analysis findings",
            "Assess regulatory requirements",
            "Identify compliance elements",
            "Plan narrative structure"
        ],
        "action_phase": [
            "Draft concise narrative",
            "Include specific details",
            "Reference activity patterns",
            "Use regulatory language"
        ]
    }

def get_regulatory_requirements() -> Dict[str, Any]:
    """Key regulatory requirements for SAR narratives used in the agent's prompt"""
    return {
        "word_limit": WORD_LIMIT,
        "required_elements": [
            "Customer identification",
            "Suspicious activity description",
            "Transaction amounts and dates",
            "Why activity is suspicious"
        ],
        "terminology": [
            "Suspicious activity",
            "Regulatory threshold",
            "Financial institution",
            "Money laundering",
            "Bank Secrecy Act"
        ],
        "citations": [
            "31 CFR 1020.320 (BSA)",
            "12 CFR 21.11 (SAR Filing)",
            "FinCEN SAR Instructions"
        ]
    }

# ===== TESTING UTILITIES =====

def test_narrative_generation(openai_client: Optional[openai.OpenAI] = None,
                              model: str = "gpt-4o-mini") -> ComplianceOfficerOutput:
    """Test the agent with a sample structuring case and risk analysis

    Builds a client from OPENAI_API_KEY when none is passed (Vocareum
    endpoint for 'voc-' keys, standard OpenAI otherwise), generates a
    narrative and prints the validation report.

    Returns:
        ComplianceOfficerOutput for the sample case
    """
    import os

    print("🧪 Testing Compliance Officer Agent")
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL") or ("https://openai.vocareum.com/v1" if api_key.startswith("voc-") else None)
        openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)

    try:
        from .test_scenarios import RiskAnalystScenarios, ComplianceOfficerScenarios
    except ImportError:
        from test_scenarios import RiskAnalystScenarios, ComplianceOfficerScenarios

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger = ExplainabilityLogger(os.path.join(project_root, "outputs", "audit_logs", "compliance_officer_sample_test.jsonl"))
    case = RiskAnalystScenarios().create_case_data("structuring_classic")
    risk_analysis = ComplianceOfficerScenarios().create_risk_analysis_output("structuring_narrative")

    agent = ComplianceOfficerAgent(openai_client, logger, model=model)
    result = agent.generate_compliance_narrative(case, risk_analysis)
    print(f"📝 Narrative ({len(result.narrative.split())} words): {result.narrative}")
    print(f"📚 Citations: {result.regulatory_citations}")
    print(f"✅ Complete: {result.completeness_check} | Issues: {agent.last_validation['issues']}")
    return result

def validate_word_count(text: str, max_words: int = WORD_LIMIT) -> bool:
    """Helper to validate word count"""
    word_count = len(text.split())
    return word_count <= max_words

if __name__ == "__main__":
    import os

    print("✅ Compliance Officer Agent Module")
    print("ReACT prompting for regulatory narrative generation")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("voc-0000"):  # skip the .env.template placeholder
        test_narrative_generation()
    else:
        print("ℹ️ Set a real OPENAI_API_KEY (Vocareum or OpenAI) in .env to run the live sample test")
