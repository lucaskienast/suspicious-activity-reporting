"""
Tests for the Compliance Officer's OBSERVATION revision loop

Speculative language and off-typology mentions are sent back to the model
for rewording; a valid draft is kept as fallback if the revisions fail.
"""

import json
import os
from unittest.mock import Mock

import pytest

from src.compliance_officer_agent import ComplianceOfficerAgent
from src.foundation_sar import ExplainabilityLogger, DataLoader, RiskAnalystOutput, load_csv_data

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CLEAN = ("Tracy Lewis (CUST_0053) made seven cash deposits totaling $67,569.06 between 2025-07-25 and "
         "2025-07-31 at the Southpark Ave and Westside Plaza branches, consistent with structuring under 31 USC 5324.")
SPECULATIVE = CLEAN + " This raises concerns about the legitimacy of the funds."


@pytest.fixture(scope='module')
def case(tmp_path_factory):
    customers, accounts, transactions = load_csv_data(DATA_DIR)
    logger = ExplainabilityLogger(str(tmp_path_factory.mktemp('logs') / 'case.jsonl'))
    return DataLoader(logger).create_case_from_data(
        customers[customers['customer_id'] == 'CUST_0053'].iloc[0].to_dict(),
        accounts.to_dict('records'), transactions.to_dict('records'))


@pytest.fixture
def risk_analysis():
    return RiskAnalystOutput(classification='Structuring', confidence_score=0.9, reasoning='r',
                             key_indicators=['7 near-threshold deposits'], risk_level='High')


def response(narrative):
    mock = Mock()
    mock.choices = [Mock()]
    mock.choices[0].finish_reason = 'stop'
    mock.usage = None
    mock.choices[0].message.content = narrative if narrative.startswith('no json') else (
        "THOUGHT 1: ...\n```json\n" + json.dumps({
            "narrative": narrative, "narrative_reasoning": "r", "completeness_check": True,
            "regulatory_citations": ["31 CFR 1020.320 (SAR filing requirement)", "31 USC 5324 (Structuring)"]
        }) + "\n```")
    return mock


def run_agent(tmp_path, case, risk_analysis, narratives):
    client = Mock()
    client.chat.completions.create.side_effect = [response(n) for n in narratives]
    agent = ComplianceOfficerAgent(client, ExplainabilityLogger(str(tmp_path / 'audit.jsonl')))
    return agent, client, agent.generate_compliance_narrative(case, risk_analysis)


def test_speculative_narrative_is_revised(tmp_path, case, risk_analysis):
    """The OBSERVATION names the flagged term and the clean revision is accepted"""
    agent, client, result = run_agent(tmp_path, case, risk_analysis, [SPECULATIVE, CLEAN])
    assert result.narrative == CLEAN
    assert client.chat.completions.create.call_count == 2
    observation = client.chat.completions.create.call_args_list[1].kwargs['messages'][-1]['content']
    assert observation.startswith("OBSERVATION") and "legitimacy" in observation
    assert agent.last_validation['speculative_terms'] == []


def test_persistent_speculation_is_kept_with_warning(tmp_path, case, risk_analysis):
    """After all attempts the flagged draft is returned with its warnings for the reviewer"""
    agent, client, result = run_agent(tmp_path, case, risk_analysis, [SPECULATIVE] * 3)
    assert client.chat.completions.create.call_count == 3
    assert agent.last_validation['speculative_terms'] == ['legitimacy']
    normalizations = agent.logger.entries[-1]['output_summary']['normalizations']
    assert any("still flagged after revision" in note for note in normalizations)


def test_failed_revision_falls_back_to_valid_draft(tmp_path, case, risk_analysis):
    """A revision that breaks the JSON never loses the earlier valid narrative"""
    agent, _, result = run_agent(tmp_path, case, risk_analysis, [SPECULATIVE, "no json", "no json"])
    assert result.narrative == SPECULATIVE
    assert agent.logger.entries[-1]['success'] is True
