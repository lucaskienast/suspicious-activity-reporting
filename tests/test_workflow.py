"""
Tests for the two-stage SAR workflow (Phase 4)

Covers screening, the human review gates, cost optimisation (the Compliance
Officer only runs after drafting approval), SAR document generation and
integrity, audit trail completeness, error containment and metrics.
The agents are mocked, so no API calls are made.
"""

import json
import os
from unittest.mock import Mock

import pytest

from src.foundation_sar import (
    ExplainabilityLogger, DataLoader, RiskAnalystOutput, ComplianceOfficerOutput, load_csv_data
)
from src.sar_workflow import (
    screen_customers, HumanReviewGate, SARWorkflow, CaseStatus,
    verify_sar_document, compute_workflow_metrics
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CLEAN_VALIDATION = {'word_count': 40, 'within_word_limit': True, 'issues': [], 'warnings': [],
                    'ungrounded_amounts': [], 'other_typologies': [], 'speculative_terms': [], 'is_complete': True}


@pytest.fixture(scope='module')
def data():
    return load_csv_data(DATA_DIR)


@pytest.fixture(scope='module')
def screening(data):
    return screen_customers(*data)


@pytest.fixture
def logger(tmp_path):
    return ExplainabilityLogger(str(tmp_path / 'audit_logs' / 'workflow_test.jsonl'))


def risk_output(classification='Structuring', risk_level='High', confidence=0.9):
    return RiskAnalystOutput(classification=classification, confidence_score=confidence,
                             reasoning="Profile: ... Decision: test", key_indicators=["7 near-threshold deposits"],
                             risk_level=risk_level)


def narrative_output(complete=True):
    return ComplianceOfficerOutput(
        narrative="Tracy Lewis (CUST_0053) made seven cash deposits totaling $67,569.06 between 2025-07-25 "
                  "and 2025-07-31 at five branches, consistent with structuring under 31 USC 5324.",
        narrative_reasoning="Built from the analyst findings.",
        regulatory_citations=["31 CFR 1020.320 (SAR filing requirement)", "31 USC 5324 (Structuring)"],
        completeness_check=complete)


def build_workflow(tmp_path, logger, risk, narrative=None, validation=None, risk_error=None):
    risk_agent = Mock()
    if risk_error:
        risk_agent.analyze_case.side_effect = risk_error
    else:
        risk_agent.analyze_case.return_value = risk
    compliance_agent = Mock()
    compliance_agent.generate_compliance_narrative.return_value = narrative or narrative_output()
    compliance_agent.last_validation = validation or CLEAN_VALIDATION
    gate = HumanReviewGate(logger, 'test_reviewer', mode='simulated_policy', output_fn=lambda *_: None)
    workflow = SARWorkflow(risk_agent, compliance_agent, gate, DataLoader(logger), logger,
                           str(tmp_path / 'filed_sars'), run_id='TEST_RUN')
    return workflow, risk_agent, compliance_agent


def process(workflow, data, screening, customer_id):
    customers, accounts, transactions = data
    customer = customers[customers['customer_id'] == customer_id].iloc[0].to_dict()
    row = screening[screening['customer_id'] == customer_id].iloc[0].to_dict()
    return workflow.process_case(customer, accounts.to_dict('records'), transactions.to_dict('records'), row)


class TestScreening:
    def test_screening_flags_injected_typologies(self, screening):
        """Typology rules flag the injected scenarios and leave benign customers out"""
        rows = screening.set_index('customer_id')
        assert 'structuring' in rows.loc['CUST_0053', 'signals'] and rows.loc['CUST_0053', 'alert']
        assert 'rapid_movement' in rows.loc['CUST_0062', 'signals']
        assert 'large_cash' in rows.loc['CUST_0125', 'signals']
        assert not rows.loc['CUST_0002', 'typology_signals'] and not rows.loc['CUST_0002', 'alert']
        assert len(rows.loc['CUST_0053', 'flagged_transaction_ids']) == 7

    def test_alert_queue_mixes_strong_and_weak_alerts(self, screening):
        """All 21 typology customers alert, plus profile-only alerts for the gates to filter"""
        alerts = screening[screening['alert']]
        with_typology = alerts['typology_signals'].apply(bool)
        assert with_typology.sum() == 21
        assert (~with_typology).sum() > 0


class TestReviewGates:
    def test_policy_drafting_decisions(self, logger):
        """Escalate High/Critical with confidence >= 0.7 and all sanctions; close the rest"""
        gate = HumanReviewGate(logger, 'reviewer', mode='simulated_policy')
        assert gate.policy_drafting_decision(risk_output(risk_level='High', confidence=0.9))[0] == 'approve'
        assert gate.policy_drafting_decision(risk_output(risk_level='Medium', confidence=0.9))[0] == 'reject'
        assert gate.policy_drafting_decision(risk_output(risk_level='High', confidence=0.5))[0] == 'reject'
        assert gate.policy_drafting_decision(risk_output('Sanctions', 'Medium', 0.6))[0] == 'approve'

    def test_interactive_gate_validates_input(self, logger, data, screening, tmp_path):
        """Invalid answers are re-prompted and a rejection requires a rationale"""
        answers = iter(["maybe", "r", "", "Activity explained by documented payroll business"])
        gate = HumanReviewGate(logger, 'analyst_jane', mode='interactive',
                               input_fn=lambda _: next(answers), output_fn=lambda *_: None)
        case = DataLoader(logger).create_case_from_data(
            data[0][data[0]['customer_id'] == 'CUST_0053'].iloc[0].to_dict(),
            data[1].to_dict('records'), data[2].to_dict('records'))
        decision = gate.review_risk_analysis(case, risk_output())
        assert decision.decision == 'reject'
        assert decision.decision_source == 'interactive'
        assert decision.rationale == "Activity explained by documented payroll business"

    def test_interactive_gate_falls_back_without_stdin(self, logger, data):
        """Unattended runs apply the documented policy and record it as simulated"""
        def no_stdin(_):
            raise EOFError("stdin unavailable")
        gate = HumanReviewGate(logger, 'reviewer', mode='interactive', input_fn=no_stdin, output_fn=lambda *_: None)
        case = DataLoader(logger).create_case_from_data(
            data[0][data[0]['customer_id'] == 'CUST_0053'].iloc[0].to_dict(),
            data[1].to_dict('records'), data[2].to_dict('records'))
        decision = gate.review_risk_analysis(case, risk_output())
        assert decision.decision == 'approve' and decision.decision_source == 'simulated_policy'


class TestWorkflow:
    def test_drafting_rejection_skips_compliance_agent(self, tmp_path, logger, data, screening):
        """Cost optimisation: no narrative is generated for cases rejected at gate 1"""
        workflow, _, compliance_agent = build_workflow(tmp_path, logger, risk_output(risk_level='Medium'))
        record = process(workflow, data, screening, 'CUST_0053')
        assert record.status == CaseStatus.REJECTED_AT_DRAFTING
        compliance_agent.generate_compliance_narrative.assert_not_called()
        assert not os.path.exists(tmp_path / 'filed_sars')

    def test_approved_case_files_valid_sar_document(self, tmp_path, logger, data, screening):
        """Approval at both gates writes a complete, checksummed SAR document"""
        workflow, _, compliance_agent = build_workflow(tmp_path, logger, risk_output())
        record = process(workflow, data, screening, 'CUST_0053')
        assert record.status == CaseStatus.FILED
        compliance_agent.generate_compliance_narrative.assert_called_once()
        assert verify_sar_document(record.sar_path)

        with open(record.sar_path) as f:
            document = json.load(f)
        for section in ('sar_metadata', 'filing_institution', 'subject_information', 'suspicious_activity',
                        'narrative', 'ai_analysis', 'human_review', 'audit_trail'):
            assert section in document
        assert document['sar_metadata']['case_id'] == record.case_id
        assert document['suspicious_activity']['flagged_transaction_count'] == 7
        assert [d['gate'] for d in document['human_review']] == ['sar_drafting', 'sar_filing']
        assert document['subject_information']['tin_last_4_masked'].startswith('XXX-XX-')

    def test_filing_gate_rejects_inaccurate_narrative(self, tmp_path, logger, data, screening):
        """Narratives with untraceable amounts are returned for revision, not filed"""
        validation = {**CLEAN_VALIDATION, 'warnings': ["amounts not traceable"], 'ungrounded_amounts': ['$15,642.78']}
        workflow, _, _ = build_workflow(tmp_path, logger, risk_output(), validation=validation)
        record = process(workflow, data, screening, 'CUST_0053')
        assert record.status == CaseStatus.REJECTED_AT_FILING
        assert record.sar_path is None
        assert 'not traceable' in record.filing_decision.rationale

    def test_audit_trail_records_every_decision(self, tmp_path, logger, data, screening):
        """Case creation, both human decisions and the filing are all audited"""
        workflow, _, _ = build_workflow(tmp_path, logger, risk_output())
        record = process(workflow, data, screening, 'CUST_0053')
        entries = [e for e in logger.entries if e['case_id'] == record.case_id]
        assert [e['agent_type'] for e in entries] == ['DataLoader', 'HumanReviewer', 'HumanReviewer', 'SARFiling']
        assert [e['action'] for e in entries[1:3]] == ['review_sar_drafting', 'review_sar_filing']

    def test_profile_only_alert_without_typology(self, tmp_path, logger, data, screening):
        """Alerts without a typology signal (NaN expectation in DataFrame rows) process cleanly"""
        workflow, _, _ = build_workflow(tmp_path, logger, risk_output('Other', 'Low', 0.6))
        profile_only = screening[screening['alert'] & ~screening['typology_signals'].apply(bool)].iloc[0]
        record = process(workflow, data, screening, profile_only['customer_id'])
        assert record.expected_classification is None
        assert record.status == CaseStatus.REJECTED_AT_DRAFTING

    def test_agent_error_is_contained(self, tmp_path, logger, data, screening):
        """An agent failure stops that case only and is logged"""
        workflow, _, _ = build_workflow(tmp_path, logger, None, risk_error=RuntimeError("API down"))
        record = process(workflow, data, screening, 'CUST_0053')
        assert record.status == CaseStatus.ERROR and 'risk_analysis' in record.error
        assert logger.entries[-1]['agent_type'] == 'Workflow' and logger.entries[-1]['success'] is False

    def test_tampered_sar_fails_verification(self, tmp_path, logger, data, screening):
        """The checksum detects edits to a filed SAR"""
        workflow, _, _ = build_workflow(tmp_path, logger, risk_output())
        record = process(workflow, data, screening, 'CUST_0053')
        with open(record.sar_path) as f:
            document = json.load(f)
        document['narrative']['text'] += " Edited after approval."
        with open(record.sar_path, 'w') as f:
            json.dump(document, f)
        assert not verify_sar_document(record.sar_path)


class TestMetrics:
    def test_metrics_report_cost_avoided(self, tmp_path, logger, data, screening):
        """Rejected drafting decisions show up as avoided compliance runs and savings"""
        filed_flow, _, _ = build_workflow(tmp_path, logger, risk_output())
        rejected_flow, _, _ = build_workflow(tmp_path, logger, risk_output(risk_level='Low'))
        records = [process(filed_flow, data, screening, 'CUST_0053'),
                   process(rejected_flow, data, screening, 'CUST_0062')]
        usage = "{'attempts': 1, 'token_usage': {'prompt_tokens': 3000, 'completion_tokens': 500}}"
        audit = [{'agent_type': 'RiskAnalyst', 'output_summary': usage, 'execution_time_ms': 4000.0}] * 2 + \
                [{'agent_type': 'ComplianceOfficer', 'output_summary': usage, 'execution_time_ms': 5000.0}]

        metrics = compute_workflow_metrics(records, audit, customers_with_activity=119, alerts_raised=25)
        assert metrics['funnel']['sars_filed'] == 1
        assert metrics['funnel']['rejected_at_drafting_gate'] == 1
        assert metrics['cost_optimization']['compliance_runs_avoided_by_gate'] == 1
        assert metrics['cost_optimization']['actual_cost_usd'] < metrics['cost_optimization']['single_stage_cost_usd']
        assert metrics['human_decisions']['sar_drafting']['approval_rate'] == 0.5
