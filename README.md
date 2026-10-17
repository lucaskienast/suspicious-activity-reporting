# 🏦 Financial Services Agentic AI Project - SAR Processing System

## 📋 Project Overview

**The Challenge:** As the newest AI Engineer on the financial crimes task force, you are at the forefront of the bank's defense against a rising tide of illicit transactions. The institution is grappling with increasingly complex schemes, from intricate money laundering operations to sophisticated fraud networks, and the pressure from regulators is mounting. 

Your mission is to architect and build an intelligent, two-pronged AI system that can not only think like a seasoned Risk Analyst to detect suspicious activities but also write like a Compliance Officer to articulate these findings in perfectly crafted Suspicious Activity Reports. This system will be the critical tool that empowers the compliance team to move faster, act with greater precision, and ultimately protect the integrity of the financial system, all while under the watchful eye of regulatory examiners who will scrutinize your work for its accuracy, efficiency, and transparency.

You will build an **AI-powered Suspicious Activity Report (SAR) processing system** that automates financial crime detection using a multi-agent architecture. This project simulates real-world regulatory requirements that financial institutions face when detecting and reporting suspicious activities to authorities like FinCEN.

### 🎯 Learning Objectives

By completing this project, you will:

1. **Design Multi-Agent Systems**: Build cooperating AI agents with distinct responsibilities
2. **Implement Prompting Strategies**: Apply Chain-of-Thought and ReACT prompting frameworks
3. **Handle Structured Data**: Work with Pydantic schemas for data validation and type safety
4. **Build Compliance Workflows**: Create audit trails and regulatory reporting systems
5. **Optimize AI Costs**: Implement efficient two-stage processing to minimize API calls

### 🏗️ System Architecture

Your system will consist of **two specialized AI agents**:

```
📊 Data Processing → 🔍 Risk Analyst Agent → 👤 Human Review → ✅ Compliance Officer Agent → 📄 SAR Filing
```

- **Risk Analyst Agent**: Uses Chain-of-Thought reasoning to classify suspicious activities
- **Compliance Officer Agent**: Uses ReACT prompting to generate regulatory narratives
- **Human-in-the-Loop**: Critical decision gates for regulatory compliance

### 🎯 Business Context: Why SAR Processing Matters

**Regulatory Requirements:**
- Financial institutions **must** file SARs within 30 days of detecting suspicious activity
- Failure to file can result in **criminal penalties** and fines exceeding $1 billion
- Average investigation costs $500-2,000 per case
- Large banks file 15,000-50,000 SARs annually

**Your AI Solution Addresses:**
- **Volume**: Process millions of transactions efficiently
- **Quality**: Consistent analytical frameworks reduce false positives
- **Cost**: Automated screening and documentation
- **Risk**: Systematic detection prevents regulatory violations

## ✅ Implementation Notes

This section documents the implemented system and every customisation beyond the starter code.

### Components

| Module | Responsibility |
|---|---|
| `src/foundation_sar.py` | Pydantic schemas with business-rule validators, `ExplainabilityLogger` (JSONL audit trail with input/output summaries stored as JSON objects) and `DataLoader` (per-customer cases from full CSV extracts) |
| `src/risk_analyst_agent.py` | Chain-of-Thought Risk Analyst. A 5-step framework produces a validated `RiskAnalystOutput` |
| `src/compliance_officer_agent.py` | ReACT Compliance Officer. A REASONING phase (THOUGHT 1-3) and an ACTION phase (draft → OBSERVATION → JSON) produce a ≤120-word SAR narrative |
| `src/agent_utils.py` | Shared helpers: JSON extraction/repair, token accounting, and the pre-computed case activity summary used by both agents |
| `src/sar_workflow.py` | Screening, human review gates, the two-stage workflow, SAR document generation (SHA-256 checksummed) and efficiency metrics |

### Design decisions and customisations
- **Schemas accept raw pandas records.** Empty CSV cells become `None`, and the int64 `ssn_last_4` column becomes a zero-padded string. Dates are strictly `YYYY-MM-DD`, `customer_since` cannot precede `date_of_birth`, and cases need at least one transaction and consistent account ownership.
- **Grounded prompts.** Both agents receive pre-computed signals (near-threshold cash, CTR-reportable cash, busiest cash day, wire pass-through rate, closed-account activity), so the model interprets facts instead of doing arithmetic.
- **LLM output is untrusted input.** JSON is extracted from prose or code fences, trailing commas are repaired, and formatting slips are normalised. Failures trigger a corrective retry that feeds the exact error back to the model. Every adjustment is written to the audit trail.
- **Queryable audit trail.** `input_summary` and `output_summary` are native JSON objects, so every log works directly with `jq` or a SIEM, e.g. `jq -c 'select(.agent_type=="RiskAnalyst") | .output_summary | {classification, risk_level}' outputs/audit_logs/sar_workflow_*.jsonl`.
- **Clear validation errors.** An invalid `risk_rating` (e.g. `"HIGHX"`) fails with a message naming the value and the accepted options, rather than being silently case-folded first.
- **Policy rules in code.** Sanctions findings are always `Critical`. Every narrative cites 31 CFR 1020.320 (the SAR filing requirement).
- **Narrative validation is the authoritative observation.** The model's own word counts proved unreliable, so the code enforces:
  - **Word limit:** narratives over 120 words are revised, then rejected.
  - **Completeness:** the WHO/WHAT/WHEN/WHERE/WHY elements, citations and no SSN, which can override `completeness_check`. WHERE must be *named*: a branch or jurisdiction recorded on the case's transactions, an account ID, or a specific channel (ATM, wire, ACH, online). "Across three locations" fails, and the OBSERVATION sent back lists the case's branches and accounts so the model can name them.
  - **Accuracy:** it flags dollar amounts not traceable to the case data, off-typology mentions and speculative language. Speculation and off-typology mentions are sent back for revision.
- **Human-in-the-loop.**
  - **Gate 1** approves SAR drafting, so the Compliance Officer only runs on approved cases.
  - **Gate 2** approves filing.
  - **Modes:** `SAR_REVIEW_MODE=interactive` (the default) prompts the reviewer, and rejections require a rationale. `simulated_policy` applies a documented policy for unattended runs.
  - Every decision records its `decision_source`.
- **API keys.** `create_openai_client()` routes `voc-` keys through Vocareum and other keys to the standard OpenAI endpoint, without printing key material. The default model is `gpt-4o-mini` (override with `OPENAI_MODEL`).
- **`src/test_scenarios.py` fixed.** It now matches the schemas (`transaction_date`, `method`, `case_created_at`/`data_sources`), and the compliance scenarios use real scenario cases.

### Results (latest live run, `gpt-4o-mini`)
- **Risk Analyst:** 5/5 synthetic scenarios correct across all five categories. In the workflow, classifications agree 95% with the screening typology.
- **Compliance Officer:** 4/4 compliance scenarios pass. Workflow narratives are 100% complete, 100% within 120 words (64 on average), every filed narrative names its WHERE, and figures are checked against the case data.
- **Workflow:**
  - 119 customers screened → 25 alerts → 20 approved for drafting → 19 SARs filed.
  - 5 alerts were closed at Gate 1, and 1 narrative was returned at Gate 2 (an amount not traceable to the case facts), each with a documented rationale.
  - Cost was about $0.0020 per filed SAR, 81% less than running both agents on every customer.
- **Tests:** `python -m pytest tests/` gives **64 passed**: the 30 provided tests, 13 workflow tests, 3 revision-loop tests and 18 review-fix tests.

### Running the workflow
```bash
# Interactive review (you decide at both gates; limited to the top 5 alerts)
jupyter notebook notebooks/03_workflow_integration.ipynb

# Unattended run of all alerts with the documented reviewer policy
cd notebooks && SAR_REVIEW_MODE=simulated_policy jupyter nbconvert --to notebook --execute --inplace 03_workflow_integration.ipynb
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Vocareum OpenAI API key (from your Udacity workspace "Cloud Resources")
- VS Code with Jupyter extension (recommended)

### 1. Environment Setup

```bash
# Clone the project (if not already done)
cd project/starter

# Install dependencies
pip install -r requirements.txt

# Set up your Vocareum OpenAI API key
cp .env.template .env
# Edit .env and replace the placeholder with your actual Vocareum API key
```

### 2. Vocareum OpenAI API Key Setup

**Important:** This project uses Vocareum OpenAI API keys, not direct OpenAI keys.

**Getting Your API Key:**
1. In your Udacity workspace, click "Cloud Resources" in the navigation pane
2. Copy the provided OpenAI API key (starts with `voc-`)
3. Paste it into your `.env` file as `OPENAI_API_KEY=voc-your-actual-key-here`

**Key Differences:**
- Vocareum keys start with `voc-` instead of `sk-`
- API calls are routed through `https://openai.vocareum.com/v1`
- Budget and usage are managed by Udacity through Vocareum

### 3. Project Structure

```
starter/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.template               # Environment variables template
├── data/                       # Sample financial data
│   ├── customers.csv           # Customer profiles
│   ├── accounts.csv            # Account information
│   └── transactions.csv        # Transaction records
├── notebooks/                  # Jupyter notebooks for development
│   ├── 01_data_exploration.ipynb
│   ├── 02_agent_development.ipynb
│   └── 03_workflow_integration.ipynb
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── foundation_sar.py       # Core data schemas (TO IMPLEMENT)
│   ├── risk_analyst_agent.py   # Risk analysis agent (TO IMPLEMENT)
│   └── compliance_officer_agent.py  # Compliance agent (TO IMPLEMENT)
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_foundation.py     # Foundation tests (10) - Run to validate Phase 1
│   ├── test_risk_analyst.py   # Risk Analyst tests (10) - Run to validate Phase 2  
│   └── test_compliance_officer.py # Compliance tests (10) - Run to validate Phase 3
├── outputs/                    # Generated files
│   ├── filed_sars/            # SAR documents
│   └── audit_logs/            # Decision audit trails
└── docs/                      # Additional documentation
    ├── system_architecture.md    # 🏗️ SYSTEM OVERVIEW (Read FIRST!)
    ├── prompting_guide.md
    ├── regulatory_context.md
    └── troubleshooting.md
```

## 📚 Project Phases

### 🏗️ **PREREQUISITE: System Architecture Overview**
**📖 Before Starting Phase 1, Read:** `docs/system_architecture.md`

**Understanding Required:**
- Complete system data flow from CSV files to SAR documents
- Role of each Pydantic schema (Customer, Account, Transaction, Case)
- Purpose of RiskAnalystOutput and ComplianceOfficerOutput schemas
- ExplainabilityLogger and audit trail requirements
- DataLoader responsibilities and error handling
- Human-in-the-loop decision gates

**Why This Matters:** Phase 1 requires implementing foundation components (schemas, logger, dataloader) that you haven't encountered in the data exploration notebook. The system architecture explains what each component does and how they work together.

### Phase 1: Foundation & Data Modeling
**Notebook: `01_data_exploration.ipynb`**
**📖 Required Reading FIRST:** `docs/system_architecture.md`

**Learning Focus:** Pydantic schemas, data validation, type safety

**Tasks:**
1. **Explore the Dataset**: Understand customer, account, and transaction data
2. **Design Data Schemas**: Create Pydantic models for type safety
3. **Build DataLoader**: Combine fragmented data into unified cases
4. **Implement Logging**: Create audit trail system

**Key Files to Implement:**
- `src/foundation_sar.py` - Core data schemas and utilities

**Success Criteria:**
- [ ] All data schemas validate correctly
- [ ] DataLoader creates unified case objects
- [ ] Audit logging captures all operations
- [ ] **Unit tests pass**: `python -m pytest tests/test_foundation.py -v` (10/10 tests should pass)

### Phase 2: Risk Analyst Agent
**Notebook: `02_agent_development.ipynb`**

**Learning Focus:** Chain-of-Thought prompting, financial crime detection

**Tasks:**
1. **Study Chain-of-Thought Prompting**: Learn systematic reasoning frameworks
2. **Implement Risk Classification**: Build agent to categorize suspicious activities
3. **Handle Structured Output**: Parse and validate AI responses
4. **Test with Real Data**: Validate against sample cases

**Key Files to Implement:**
- `src/risk_analyst_agent.py` - Risk analysis with Chain-of-Thought reasoning

**Success Criteria:**
- [ ] Agent classifies 5 activity types: Structuring, Sanctions, Fraud, Money_Laundering, Other
- [ ] Structured JSON output with confidence scores
- [ ] Chain-of-Thought reasoning visible in responses
- [ ] Handles edge cases and parsing errors
- [ ] **Unit tests pass**: `python -m pytest tests/test_risk_analyst.py -v` (10/10 tests should pass)

### Phase 3: Compliance Officer Agent
**Notebook: `02_agent_development.ipynb` (continued)**

**Learning Focus:** ReACT prompting, regulatory narrative generation

**Tasks:**
1. **Learn ReACT Framework**: Reasoning + Action structured prompting
2. **Generate SAR Narratives**: Create regulatory-compliant documentation
3. **Enforce Constraints**: 120-word limits, specific terminology
4. **Regulatory Citations**: Include relevant BSA/AML references

**Key Files to Implement:**
- `src/compliance_officer_agent.py` - Compliance narrative generation

**Success Criteria:**
- [ ] Generates compliant SAR narratives ≤120 words
- [ ] Includes regulatory citations
- [ ] ReACT reasoning framework visible
- [ ] Validates narrative completeness
- [ ] **Unit tests pass**: `python -m pytest tests/test_compliance_officer.py -v` (10/10 tests should pass)

### Phase 4: Workflow Integration
**Notebook: `03_workflow_integration.ipynb`**

**Learning Focus:** Multi-agent coordination, human-in-the-loop systems

**Tasks:**
1. **Build Two-Stage Workflow**: Risk analysis → Human review → Compliance generation
2. **Implement Human Gates**: Decision points for proceeding with SAR filing
3. **Generate SAR Documents**: Complete regulatory forms
4. **Create Efficiency Metrics**: Track cost savings and processing times

**Success Criteria:**
- [ ] Complete workflow processes real cases
- [ ] Human review points function correctly
- [ ] SAR documents generated with all required fields
- [ ] Audit trails capture all decisions
- [ ] Efficiency metrics show cost optimization

## 💡 Key Implementation Hints

### 1. Pydantic Schema Design
```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional

class CustomerData(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    # Add more fields based on data/customers.csv
    
    @field_validator('customer_id')
    @classmethod
    def validate_customer_id(cls, v):
        # Add your validation logic
        return v
```

### 2. Chain-of-Thought Prompting Structure
```python
system_prompt = """You are a Senior Financial Crime Risk Analyst...

**Analysis Framework** (Think step-by-step):
1. **Data Review**: Examine customer profile...
2. **Pattern Recognition**: Identify indicators...
3. **Regulatory Mapping**: Connect to typologies...
4. **Risk Quantification**: Assess severity...
5. **Classification Decision**: Determine category...
"""
```

### 3. ReACT Framework Implementation
```python
react_prompt = """**ReACT Framework**: Follow this approach:

**REASONING Phase:**
1. Review the risk analyst's findings...
2. Assess regulatory requirements...

**ACTION Phase:**
1. Draft concise narrative...
2. Include specific details...
"""
```

### 4. Error Handling & Validation
```python
try:
    # Parse AI response
    result = json.loads(response_content)
    validated_output = YourPydanticModel(**result)
except json.JSONDecodeError:
    # Handle parsing errors
except ValidationError:
    # Handle schema validation errors
```

## 📊 Sample Data Overview

Your dataset includes:
- **150 customers** with varying risk ratings
- **200+ accounts** across different types
- **500+ transactions** with suspicious patterns

**Key Suspicious Patterns to Detect:**
- **Structuring**: Multiple transactions just under $10,000 reporting threshold
- **Money Laundering**: Complex transaction chains obscuring fund sources
- **Sanctions**: Transactions involving prohibited parties or countries
- **Fraud**: Irregular patterns suggesting fraudulent activity

## 🧪 Testing Your Implementation

### **Progressive Testing Strategy**

Each project phase includes comprehensive unit tests that validate your implementation. The tests are designed to:
- **Skip automatically** when modules aren't implemented yet
- **Provide clear feedback** on what needs to be fixed
- **Validate production readiness** when implementation is complete

### **Running Tests by Phase**

```bash
# Phase 1: Foundation (after implementing foundation_sar.py)
python -m pytest tests/test_foundation.py -v
# Expected: 10 tests pass (or skip if not implemented)

# Phase 2: Risk Analyst (after implementing risk_analyst_agent.py)  
python -m pytest tests/test_risk_analyst.py -v
# Expected: 10 tests pass (or skip if not implemented)

# Phase 3: Compliance Officer (after implementing compliance_officer_agent.py)
python -m pytest tests/test_compliance_officer.py -v  
# Expected: 10 tests pass (or skip if not implemented)

# All phases complete - Full system validation
python -m pytest tests/ -v
# Expected: 30 tests pass (100% success rate)
```

### **Test Categories**

**Foundation Tests (10 tests)**:
- Data schema validation (Pydantic models)
- CSV loading and data aggregation
- Audit logging functionality
- Error handling and edge cases

**Risk Analyst Tests (10 tests)**:
- Agent initialization and configuration
- Chain-of-Thought analysis workflow
- OpenAI API integration and JSON parsing
- Error handling for malformed responses

**Compliance Officer Tests (10 tests)**:
- ReACT framework implementation
- Regulatory narrative generation (≤120 words)
- Multi-format response parsing
- Compliance validation and citations

### **Understanding Test Results**

**✅ PASSED**: Your implementation works correctly  
**⏭️ SKIPPED**: Module not implemented yet (expected during development)  
**❌ FAILED**: Implementation needs fixes - check error messages for guidance

### **Test-Driven Development Tips**

1. **Start with failing tests**: Run tests before implementing to understand requirements
2. **Implement incrementally**: Focus on making one test pass at a time
3. **Use test errors as guides**: Error messages tell you exactly what to fix
4. **Validate frequently**: Run tests after each major change

```bash
# Quick validation during development
python -m pytest tests/test_foundation.py::TestCustomerData::test_valid_customer_data -v
# Run specific test to debug individual functions
```

## 🎯 Assessment Criteria

Your project will be evaluated on:

1. **Technical Implementation (40%)**
   - Correct Pydantic schema design
   - Proper error handling and validation
   - Clean, well-structured code

2. **AI Agent Design (30%)**
   - Effective prompting strategies
   - Structured output parsing
   - Agent coordination

3. **Regulatory Compliance (20%)**
   - Accurate SAR narrative generation
   - Complete audit trails
   - Proper regulatory citations

4. **System Integration (10%)**
   - End-to-end workflow functionality
   - Human-in-the-loop implementation
   - Efficiency optimizations

## 🔧 Development Tips

### Debugging AI Responses
- Always print raw LLM responses before parsing
- Use structured logging to track decision flows
- Test with edge cases and malformed inputs

### Prompt Engineering
- Start with simple prompts and iterate
- Use examples in prompts for better performance
- Test prompts with different transaction patterns

### Performance Optimization
- Implement the two-stage workflow to minimize API calls
- Cache expensive operations where possible
- Use appropriate temperature settings (0.2-0.3 for structured tasks)

## 📚 Additional Resources

- **🏗️ System Architecture**: `docs/system_architecture.md` - **READ FIRST!** Complete system overview with data flow diagrams
- **Regulatory Context**: `docs/regulatory_context.md`
- **Prompting Guide**: `docs/prompting_guide.md`
- **Troubleshooting**: `docs/troubleshooting.md`
- **BSA/AML Guidelines**: [FinCEN Official Resources](https://www.fincen.gov/)

## 🆘 Getting Help

1. **🏗️ Start with System Architecture** - Read `docs/system_architecture.md` for complete system understanding
2. **Review the documentation** in the `docs/` folder
3. **Check the test files** for expected behavior
4. **Use the notebooks** for interactive development
5. **Study the sample data** to understand patterns

Remember: This project simulates real regulatory requirements. Focus on building systems that are **explainable**, **auditable**, and **compliant** with financial regulations.

---

**Ready to build the future of financial crime detection? Let's get started! 🚀**
