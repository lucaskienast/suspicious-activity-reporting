Getting Started
This project is divided into four distinct phases. You will begin by building the foundational data components, then implement the two AI agents, and finally integrate everything into a single, cohesive workflow.

Your primary working directory is the starter/ folder. All the code you need to write will be within the starter/src/ directory. You will use the three Jupyter notebooks in starter/notebooks/ to guide your development, test your components, and integrate the final system.

Starter Folder Structure
Before you begin, familiarize yourself with the key files and directories:

data/: Contains the sample financial data (customers.csv, accounts.csv, transactions.csv).
notebooks/: Contains the Jupyter notebooks that will guide you through each major part of the project. Start with 01_data_exploration.ipynb.
src/: This is where you will write your Python code. You will implement the core logic in these files:
foundation_sar.py: For all data schemas and the data loader.
risk_analyst_agent.py: For the first AI agent that detects suspicious activity.
compliance_officer_agent.py: For the second AI agent that writes the SAR narrative.
tests/: Contains unit tests to help you validate your implementation.
Phase 1: Foundation & Data Modeling
Goal: Understand the financial data and build the robust data structures that will serve as the foundation for the entire system.

Tasks:

Explore the customer, account, and transaction data to identify patterns.
Implement all required Pydantic schemas for data validation.
Create the data loading and audit logging systems.
Deliverables:

A fully implemented src/foundation_sar.py file.
A completed notebooks/01_data_exploration.ipynb that validates your foundation components.
Phase 2: Risk Analyst Agent
Goal: Build an AI agent that can reason through a case, classify suspicious activity, and provide a detailed explanation for its decision, using the Chain-of-Thought methodology.

Tasks:

Develop a Chain-of-Thought system prompt for financial crime analysis.
Implement the agent class, including the method to process a case and return a structured output.
Incorporate robust JSON parsing and error handling for the LLM's responses.
Deliverables:

A fully implemented src/risk_analyst_agent.py file.
The relevant sections of notebooks/02_agent_development.ipynb completed to test and validate the agent.
Phase 3: Compliance Officer Agent
Goal: Build an AI agent that can take the structured output from the Risk Analyst and generate a concise, regulatory-compliant SAR narrative using the ReACT framework.

Tasks:

Learn and implement the ReACT (Reasoning + Action) prompting technique.
Implement the agent class that generates the SAR narrative.
Ensure the narrative meets strict word count limits and includes appropriate regulatory citations.
Deliverables:

A fully implemented src/compliance_officer_agent.py file.
The relevant sections of notebooks/02_agent_development.ipynb completed to test and validate the agent.
Phase 4: System Integration
Goal: Combine the individual components into a complete, end-to-end SAR processing workflow, including a human-in-the-loop for final oversight.

Tasks:

Integrate both the Risk Analyst and Compliance Officer agents into a unified workflow.
Implement interactive human-in-the-loop decision gates for approval or rejection.
Generate the final SAR documents and comprehensive audit trails.
Develop efficiency metrics to demonstrate the system's performance and cost-effectiveness.
Deliverables:

A completed notebooks/03_workflow_integration.ipynb that successfully runs the entire process.
Generated SAR documents in JSON format, located in the starter/outputs/filed_sars/ directory.
Generated audit logs in JSONL format, located in the starter/outputs/audit_logs/ directory.
Final Submission Checklist
Before you submit your project, ensure your system meets all the following criteria:

All foundation schemas in foundation_sar.py are implemented and validated.
The RiskAnalystAgent in risk_analyst_agent.py correctly classifies 5 types of suspicious activity.
The ComplianceOfficerAgent in compliance_officer_agent.py generates narratives that are 120 words or less.
The final notebook 03_workflow_integration.ipynb runs end-to-end, processing data from CSV to a final SAR document.
The workflow includes human decision gates for compliance oversight.
The system generates complete audit trails in the starter/outputs/audit_logs/ directory.
The system generates complete SAR documents in the starter/outputs/filed_sars/ directory.
The final notebook includes a section on cost optimization and efficiency metrics.
All code is well-documented with docstrings, type hints, and comments.