Use this project rubric to understand and assess the project criteria.

Foundation & Data Modeling
Criteria	Submission Requirements
Implement Pydantic data schemas for financial crime detection with proper validation

The foundation_sar.py file contains all required schemas: CustomerData, AccountData, TransactionData, CaseData, RiskAnalystOutput, and ComplianceOfficerOutput.
Each schema includes proper field types matching the CSV data structure (e.g., customer_id as string, balance as float).
Validation rules are implemented for critical fields including date formats, amount ranges, and risk rating values.
Optional fields are properly marked and handle missing values from CSV data.
Schema validation succeeds with the provided sample data files (customers.csv, accounts.csv, transactions.csv).
Implement data loading and audit logging systems for compliance tracking

The DataLoader class successfully creates unified CaseData objects from separate CSV files.
The ExplainabilityLogger class writes structured audit entries in JSONL format.
Data loading handles missing or malformed data gracefully with appropriate error messages.
The audit logging captures all agent decisions and reasoning with timestamps and unique identifiers.
Integration with the provided CSV data demonstrates proper data flow from raw files to structured case objects.
Risk Analyst Agent Implementation
Criteria	Submission Requirements
Implement Chain-of-Thought reasoning for suspicious activity classification

The RiskAnalystAgent class uses a systematic prompt with clear reasoning steps for financial crime analysis.
The system prompt establishes appropriate financial analyst persona and expertise.
Chain-of-Thought methodology is evident in agent outputs showing step-by-step analysis.
Agent correctly classifies all 5 suspicious activity types: Structuring, Sanctions, Fraud, Money_Laundering, Other.
Confidence scores (0.0-1.0) and risk levels (Low, Medium, High, Critical) are generated appropriately.
Implement robust output processing and error handling for LLM responses

JSON parsing extracts structured data from LLM responses and validates against RiskAnalystOutput schema.
Error handling manages malformed responses, API failures, and unexpected output formats.
Agent gracefully recovers from parsing errors with appropriate fallback mechanisms.
Clear error messages and logging facilitate debugging and system monitoring.
Testing demonstrates reliable operation across different case types and edge cases.
Compliance Officer Agent Implementation
Criteria	Submission Requirements
Implement ReACT framework for regulatory narrative generation

The ComplianceOfficerAgent class uses ReACT (Reasoning + Action) methodology with clear separation of reasoning and action phases.
System prompt establishes appropriate compliance officer persona with regulatory expertise.
Narrative generation follows structured approach including required elements: who, what, when, where, why.
Generated narratives use proper regulatory terminology and professional tone suitable for FinCEN submission.
Integration with Risk Analyst outputs demonstrates proper data flow and decision-making process.
Generate regulatory-compliant SAR narratives with proper formatting and citations

Generated narratives consistently adhere to 120-word limit as specified in project requirements.
Narratives include all required elements for suspicious activity reporting (subject, activity, timeframe, amounts, indicators).
Proper regulatory citations reference relevant BSA/AML regulations and guidelines.
Narrative quality meets professional standards with factual, objective tone.
Output validation ensures completeness and regulatory compliance before finalization.
System Integration & Workflow
Criteria	Submission Requirements
Implement complete two-stage workflow with human-in-the-loop decision gates

The 03_workflow_integration.ipynb notebook demonstrates end-to-end processing from CSV data to complete SAR documents.
Two-stage workflow implements Risk Analyst analysis followed by human review gates before Compliance Officer narrative generation.
Human decision points allow interactive approval/rejection of cases with proper user interface.
Integration between agents maintains data integrity and proper state management throughout the workflow.
Cost optimization is demonstrated by only running expensive compliance generation on human-approved cases.
Generate complete audit trails and comprehensive system documentation

Complete SAR documents are generated with all required metadata including case IDs, timestamps, and agent decisions.
Comprehensive audit logs capture all decisions, reasoning, and human interactions in structured format suitable for regulatory examination.
Efficiency metrics demonstrate cost optimization and processing performance of the two-stage approach.
Generated outputs are properly organized in designated directories (filed_sars/, audit_logs/) with clear file naming conventions.
System documentation includes implementation notes and any customizations beyond starter code requirements.
Suggestions to Make Your Project Stand Out
Advanced Implementation Features
Enhanced Error Handling: Implement sophisticated retry mechanisms, graceful degradation, and comprehensive logging for production-level reliability.
Performance Optimization: Add caching, batch processing, or parallel execution for large-scale data processing.
Advanced Validation: Implement custom Pydantic validators for complex business rules and data quality checks.
Interactive Dashboards: Create visualizations for audit trails, efficiency metrics, or case processing workflows.
AI/ML Enhancements
Prompt Engineering: Develop sophisticated prompting strategies, few-shot learning examples, or dynamic prompt adaptation.
Model Comparison: Implement testing across different LLM models with performance benchmarking.
Confidence Calibration: Add sophisticated confidence scoring and uncertainty quantification.
Agent Specialization: Create specialized sub-agents for different crime types or analysis phases.
Domain Expertise
Regulatory Compliance: Implement additional BSA/AML requirements, real regulatory citation databases, or compliance scoring.
Financial Crime Patterns: Add detection for additional suspicious activity patterns or industry-specific scenarios.
Risk Assessment: Implement sophisticated risk scoring models or customer profiling algorithms.
Audit Features: Create comprehensive audit dashboards or regulatory examination preparation tools.