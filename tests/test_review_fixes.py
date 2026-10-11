"""
Tests for the review feedback fixes

1. Audit log summaries are native JSON objects (queryable with jq / SIEM tools)
2. WHERE is a hard narrative requirement and must name a branch, account, channel or jurisdiction
3. Invalid risk ratings fail with a message naming the value and the accepted options
"""

import json
import os
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.compliance_officer_agent import ComplianceOfficerAgent
from src.foundation_sar import (
    CustomerData, ExplainabilityLogger, DataLoader, RiskAnalystOutput, load_csv_data
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CUSTOMER = dict(customer_id="CUST_T", name="Test", date_of_birth="1980-01-01", ssn_last_4="1234",
                address="1 Test St", customer_since="2020-01-01")


@pytest.fixture(scope='module')
def case_0143(tmp_path_factory):
    customers, accounts, transactions = load_csv_data(DATA_DIR)
    logger = ExplainabilityLogger(str(tmp_path_factory.mktemp('logs') / 'case.jsonl'))
    return DataLoader(logger).create_case_from_data(
        customers[customers['customer_id'] == 'CUST_0143'].iloc[0].to_dict(),
        accounts.to_dict('records'), transactions.to_dict('records'))


@pytest.fixture
def agent():
    return ComplianceOfficerAgent(Mock(), Mock())


class TestAuditLogJson:
    def test_summaries_are_json_objects_in_file_and_memory(self, tmp_path):
        """Each JSONL line holds the summaries as objects, not Python-repr strings"""
        log_file = tmp_path / 'audit.jsonl'
        logger = ExplainabilityLogger(str(log_file))
        logger.log_agent_action("RiskAnalyst", "analyze_case", "CASE_1",
                                input_data={'customer_id': 'CUST_1', 'num_transactions': 3},
                                output_data={'classification': 'Structuring', 'key_indicators': ['a', 'b']},
                                reasoning="r", execution_time_ms=1.0)
        line = json.loads(log_file.read_text().splitlines()[0])
        assert line['input_summary'] == {'customer_id': 'CUST_1', 'num_transactions': 3}
        assert line['output_summary']['classification'] == 'Structuring'
        assert logger.entries[0]['output_summary'] == line['output_summary']

    def test_entry_is_a_snapshot_of_the_callers_data(self, tmp_path):
        """Mutating the caller's dict after logging does not change the record"""
        logger = ExplainabilityLogger(str(tmp_path / 'audit.jsonl'))
        output = {'indicators': ['a']}
        logger.log_agent_action("T", "a", "C", {}, output, "r", 1.0)
        output['indicators'].append('b')
        assert logger.entries[0]['output_summary'] == {'indicators': ['a']}

    def test_non_json_values_are_stringified(self, tmp_path):
        """Values json cannot encode (e.g. dates) are stored as strings, matching the file"""
        from datetime import date
        logger = ExplainabilityLogger(str(tmp_path / 'audit.jsonl'))
        logger.log_agent_action("T", "a", "C", {'day': date(2025, 1, 2)}, {}, "r", 1.0)
        assert logger.entries[0]['input_summary'] == {'day': '2025-01-02'}


class TestRiskRatingValidation:
    @pytest.mark.parametrize("value, expected", [("high", "High"), (" HIGH ", "High"), ("Medium", "Medium")])
    def test_case_variants_are_normalised(self, value, expected):
        assert CustomerData(**CUSTOMER, risk_rating=value).risk_rating == expected

    @pytest.mark.parametrize("value", ["HIGHX", "Critical", ""])
    def test_invalid_rating_names_value_and_options(self, value):
        with pytest.raises(ValidationError) as error:
            CustomerData(**CUSTOMER, risk_rating=value)
        message = error.value.errors()[0]['msg']
        assert f"'{value}'" in message
        assert "'Low', 'Medium', 'High'" in message


class TestNamedWhere:
    @pytest.mark.parametrize("phrase", [
        "across three locations", "at multiple branches", "at several branch locations"])
    def test_unnamed_locations_fail(self, agent, case_0143, phrase):
        """Counting locations without naming them is not a WHERE"""
        narrative = (f"Andre Rodriguez (CUST_0143) made cash deposits totaling $17,038.47 {phrase} "
                     "between 2025-08-01 and 2025-08-03, consistent with structuring under 31 USC 5324.")
        report = agent._validate_narrative_compliance(narrative, case_data=case_0143, citations=["31 CFR 1020.320"])
        assert report['elements']['where'] is False
        assert "missing WHERE element" in report['issues']
        assert report['is_complete'] is False

    @pytest.mark.parametrize("phrase", [
        "at the Northgate Mall branch", "at Branch_Westside_Plaza", "into account CUST_0143_ACC_1",
        "through ATM transactions", "by outgoing wire"])
    def test_named_branch_account_or_channel_passes(self, agent, case_0143, phrase):
        narrative = (f"Andre Rodriguez (CUST_0143) made cash deposits totaling $17,038.47 {phrase} "
                     "between 2025-08-01 and 2025-08-03, consistent with structuring under 31 USC 5324.")
        report = agent._validate_narrative_compliance(narrative, case_data=case_0143, citations=["31 CFR 1020.320"])
        assert report['elements']['where'] is True
        assert report['is_complete'] is True

    def test_missing_where_triggers_revision_with_case_locations(self, tmp_path, case_0143):
        """The OBSERVATION names branches from the case so the model can fix the narrative"""
        risk = RiskAnalystOutput(classification='Structuring', confidence_score=0.85, reasoning='r',
                                 key_indicators=['cash'], risk_level='High')

        def response(narrative):
            mock = Mock()
            mock.choices = [Mock()]
            mock.choices[0].finish_reason = 'stop'
            mock.usage = None
            mock.choices[0].message.content = "```json\n" + json.dumps({
                "narrative": narrative, "narrative_reasoning": "r", "completeness_check": True,
                "regulatory_citations": ["31 CFR 1020.320 (SAR filing requirement)"]}) + "\n```"
            return mock

        base = ("Andre Rodriguez (CUST_0143) made cash deposits totaling $17,038.47 {} between "
                "2025-08-01 and 2025-08-03, consistent with structuring under 31 USC 5324.")
        client = Mock()
        client.chat.completions.create.side_effect = [response(base.format("across three locations")),
                                                      response(base.format("at the Westside Plaza branch"))]
        agent = ComplianceOfficerAgent(client, ExplainabilityLogger(str(tmp_path / 'audit.jsonl')))
        result = agent.generate_compliance_narrative(case_0143, risk)

        observation = client.chat.completions.create.call_args_list[1].kwargs['messages'][-1]['content']
        assert "missing WHERE" in observation and "Westside Plaza" in observation
        assert "Westside Plaza branch" in result.narrative
        assert result.completeness_check is True
