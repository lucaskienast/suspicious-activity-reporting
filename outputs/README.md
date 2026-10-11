# Output Directories

This folder contains generated outputs from the SAR processing system:

## 📄 filed_sars/
Generated SAR documents in JSON format. Each SAR includes:
- Customer information
- Transaction details
- Risk analysis results
- Compliance narrative
- Regulatory citations
- Audit trail references

## 📊 audit_logs/
Comprehensive audit trails in JSONL format capturing:
- Agent decisions and reasoning
- Processing timestamps
- Input/output data summaries
- Success/failure status
- Error messages and debugging info

These outputs demonstrate the explainability and auditability required for regulatory compliance in financial crime detection systems.

## 🏷️ File Naming Conventions

| File | Contents |
|---|---|
| `filed_sars/SAR-<YYYYMMDD>-<customer_id>-<8-hex>.json` | One SAR approved at both review gates. Verify integrity with `sar_workflow.verify_sar_document(path)` (SHA-256 checksum in `sar_metadata`) |
| `audit_logs/sar_workflow_<run_id>.jsonl` | Complete Phase 4 audit trail for one workflow run: DataLoader, RiskAnalyst, HumanReviewer, ComplianceOfficer, SARFiling and Workflow entries |
| `audit_logs/workflow_metrics_<run_id>.json` | Funnel, cost-optimisation, timing, human-decision and AI-quality metrics plus per-case outcomes for the same run |
| `audit_logs/agent_development_<run_id>.jsonl` | Agent development and evaluation runs (`02_agent_development.ipynb`) |
| `audit_logs/phase1_case_creation_<run_id>.jsonl` | Phase 1 validation of case creation across the full CSV extract |

`<run_id>` is the run's start time (`YYYYMMDD_HHMMSS`), so every SAR document's `workflow_run_id` points to its audit log.
