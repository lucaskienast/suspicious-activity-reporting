# Agent Utilities - helpers shared by the Risk Analyst and Compliance Officer agents

"""
Shared helpers for the LLM agents:

- JSON handling: extract the final JSON object from free-form LLM output and
  repair common formatting slips
- Reasoning traces: strip the JSON block to keep the step-by-step reasoning
- Token accounting: per-call and cumulative usage for cost tracking
- Case statistics: pre-computed activity signals (near-threshold cash,
  CTR-reportable cash, busiest cash day, wire pass-through) that both agents
  include in their prompts
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from .foundation_sar import CaseData, TransactionData
except ImportError:
    from foundation_sar import CaseData, TransactionData

CTR_THRESHOLD = 10_000              # Currency Transaction Report threshold (31 CFR 1010.311)
NEAR_THRESHOLD_FLOOR = 9_000        # Lower bound of the "just under CTR" structuring band
TOKEN_FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens')

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


# ===== JSON HANDLING =====

def extract_json_object(response_content: Optional[str]) -> str:
    """Extract the JSON object from an LLM response

    Search order:
    1. ```json fenced blocks (the last one wins - it holds the final answer)
    2. Any other fenced block containing an object
    3. The first decodable {...} object anywhere in the text
    4. The outermost {...} span, even if malformed, so the caller's parser
       can attempt a repair or report the exact decode error

    Raises:
        ValueError: If the response is empty or contains no JSON object
    """
    if not response_content or not response_content.strip():
        raise ValueError("No JSON content found: empty response")

    fenced_blocks = _FENCED_JSON.findall(response_content)
    if fenced_blocks:
        return fenced_blocks[-1].strip()

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response_content):
        try:
            obj, end = decoder.raw_decode(response_content, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return response_content[match.start():end]

    start, end = response_content.find("{"), response_content.rfind("}")
    if start != -1 and end > start:
        return response_content[start:end + 1]

    raise ValueError("No JSON content found in response")


def parse_json_lenient(json_text: str) -> Any:
    """Parse JSON, repairing trailing commas - a common LLM formatting slip

    Raises:
        json.JSONDecodeError: If the text is still invalid after repair
    """
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", json_text))


def strip_json_blocks(response_content: Optional[str]) -> str:
    """Return the reasoning text of a response with fenced JSON blocks removed"""
    if not response_content:
        return ""
    return _FENCED_JSON.sub("", response_content).strip()


def truncate_text(text: str, max_chars: int) -> str:
    """Shorten text to max_chars at a word boundary, ending with '...'"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rsplit(' ', 1)[0] + "..."


# ===== TOKEN ACCOUNTING =====

def new_usage_counter() -> Dict[str, int]:
    """Zeroed token counter for one agent call"""
    return {field: 0 for field in TOKEN_FIELDS}


def record_usage(response: Any, call_usage: Dict[str, int], usage_totals: Dict[str, int]) -> None:
    """Add a response's token usage to the per-call and cumulative counters

    Responses without integer usage figures (e.g. mocks) only count as an API call.
    """
    usage_totals['api_calls'] = usage_totals.get('api_calls', 0) + 1
    usage = getattr(response, 'usage', None)
    for field in TOKEN_FIELDS:
        value = getattr(usage, field, None)
        if isinstance(value, int):
            call_usage[field] += value
            usage_totals[field] = usage_totals.get(field, 0) + value


# ===== CASE STATISTICS =====

def days_between(start: str, end: str) -> int:
    """Days between two YYYY-MM-DD dates"""
    return (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days


def years_between(start: str, end: str) -> int:
    """Whole years between two YYYY-MM-DD dates"""
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    return end_date.year - start_date.year - ((end_date.month, end_date.day) < (start_date.month, start_date.day))


def is_cash_transaction(txn: TransactionData) -> bool:
    """Cash/ATM activity, keyed on transaction type - the method column is noisy"""
    return 'Cash' in txn.transaction_type or 'ATM' in txn.transaction_type


def summarize_case_activity(case_data: CaseData) -> str:
    """Pre-computed statistics that make the key patterns explicit to the model

    Amounts are unsigned in this data, so direction comes from the
    transaction type. Lines only appear when the pattern is present.
    """
    transactions = case_data.transactions
    customer = case_data.customer
    dates = [txn.transaction_date for txn in transactions]
    total_volume = sum(txn.amount for txn in transactions)

    lines = [
        f"- Period: {min(dates)} to {max(dates)}, {len(transactions)} transactions, "
        f"total volume ${total_volume:,.2f}"
    ]
    if customer.annual_income:
        lines.append(f"- Total volume is {total_volume / customer.annual_income:.1f}x reported annual income")

    type_totals: Dict[str, List[float]] = {}
    for txn in transactions:
        type_totals.setdefault(txn.transaction_type, []).append(txn.amount)
    top_types = sorted(type_totals.items(), key=lambda item: sum(item[1]), reverse=True)[:6]
    lines.append("- Volume by type: " + "; ".join(
        f"{txn_type} {len(amounts)} txns ${sum(amounts):,.2f}" for txn_type, amounts in top_types
    ))

    cash = [txn for txn in transactions if is_cash_transaction(txn)]
    if cash:
        branches = {txn.location for txn in cash if txn.location}
        lines.append(f"- Cash/ATM activity: {len(cash)} txns, ${sum(txn.amount for txn in cash):,.2f}, "
                     f"{len(branches)} distinct locations")

        near_threshold = [txn for txn in cash if 'Deposit' in txn.transaction_type
                          and NEAR_THRESHOLD_FLOOR <= txn.amount < CTR_THRESHOLD]
        if near_threshold:
            lines.append(f"- Cash deposits between $9,000 and $10,000: {len(near_threshold)} "
                         f"totalling ${sum(txn.amount for txn in near_threshold):,.2f} "
                         f"({min(t.transaction_date for t in near_threshold)} to "
                         f"{max(t.transaction_date for t in near_threshold)})")

        ctr_reportable = [txn for txn in cash if txn.amount >= CTR_THRESHOLD]
        if ctr_reportable:
            lines.append(f"- Cash transactions of $10,000 or more: {len(ctr_reportable)} "
                         f"totalling ${sum(txn.amount for txn in ctr_reportable):,.2f} "
                         f"(above the CTR threshold - reportable, not threshold avoidance)")

        # Same-day cash spread across transactions/branches - the aggregate is what CTR rules look at
        cash_by_day: Dict[str, List[TransactionData]] = {}
        for txn in cash:
            cash_by_day.setdefault(txn.transaction_date, []).append(txn)
        busiest_day, day_txns = max(cash_by_day.items(), key=lambda item: sum(t.amount for t in item[1]))
        if len(day_txns) >= 2:
            day_total = sum(t.amount for t in day_txns)
            day_branches = {t.location for t in day_txns if t.location}
            lines.append(f"- Busiest cash day: {busiest_day} with {len(day_txns)} cash transactions totalling "
                         f"${day_total:,.2f} at {len(day_branches)} locations"
                         + (" (aggregate exceeds $10,000)" if day_total > CTR_THRESHOLD else ""))

    wires_in = [txn for txn in transactions if txn.transaction_type == 'Wire_Transfer_Credit']
    wires_out = [txn for txn in transactions if txn.transaction_type == 'Wire_Transfer_Debit']
    if wires_in or wires_out:
        counterparties = {txn.counterparty for txn in wires_out if txn.counterparty}
        lines.append(f"- Incoming wires: {len(wires_in)} totalling ${sum(t.amount for t in wires_in):,.2f}; "
                     f"outgoing wires: {len(wires_out)} totalling ${sum(t.amount for t in wires_out):,.2f} "
                     f"to {len(counterparties)} distinct counterparties")
        if wires_in and wires_out:
            first_in = min(t.transaction_date for t in wires_in)
            passed_on = [t for t in wires_out if t.transaction_date >= first_in]
            if passed_on:
                days = days_between(first_in, max(t.transaction_date for t in passed_on))
                share = sum(t.amount for t in passed_on) / sum(t.amount for t in wires_in)
                lines.append(f"- Pass-through: {share:.0%} of incoming wire value sent out by wire within "
                             f"{days} day{'s' if days != 1 else ''} of the first incoming wire")

    largest = max(transactions, key=lambda txn: txn.amount)
    lines.append(f"- Largest transaction: ${largest.amount:,.2f} {largest.transaction_type} "
                 f"on {largest.transaction_date}")

    closed_ids = {acc.account_id for acc in case_data.accounts if acc.status == 'Closed'}
    closed_activity = [txn for txn in transactions if txn.account_id in closed_ids]
    if closed_activity:
        lines.append(f"- Transactions posted to CLOSED accounts: {len(closed_activity)}")

    contradictions = sum(
        1 for txn in transactions
        if ('Deposit' in txn.transaction_type and 'withdrawal' in txn.description.lower())
        or ('Withdrawal' in txn.transaction_type and 'deposit' in txn.description.lower())
    )
    if contradictions:
        lines.append(f"- Description contradicts transaction type: {contradictions} txns")

    return "\n".join(lines)
