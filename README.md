# Agentic AI System for Automated Discharge Summaries

### Role: AI Engineer | Take-Home Assignment Delivery

This repository contains an autonomous, zero-fabrication agentic AI system designed to ingest unstructured, multi-document clinical note PDFs and compile a structured, clinically safe discharge summary draft for physician review.

The core architecture uses a custom-built **ReAct (Reasoning and Acting) Agent Loop** written from scratch in native Python, utilizing structured outputs via OpenAI's GPT-4o and dynamic schema enforcement via Pydantic v2. This eliminates framework overhead while guaranteeing strict clinical safety bounds, full execution visibility, and robust error recovery.

---

## 1. Core Architectural Design & Agent Loop

Rather than using rigid, hardcoded chains or fragile third-party orchestration frameworks, this system employs a state-controlled ReAct loop. The agent is exposed to available patient files and dynamically plans its execution track.

### The Execution State Machine Pipeline

The system reads state iteratively, tracking what documents have been evaluated and what gaps remain:

1. **Plan & Reason (`Thought`):** The model analyzes current case state (e.g., *"I have processed the ER Chart but have not yet inspected the clinical notes or lab sheets to check for resolving trends"*).
2. **Action Dispatch (`Tool`):** The model dynamically decides when to run specialized tools (`read_pdf`, `lookup_drugs`, `escalate`, or `finalize`).
3. **Observation Feedback (`Result`):** The tool returns its output wrapper (or a controlled error payload) directly to the conversational log context, updating the agent's knowledge graph for the next iteration.

---

## 2. Strict Clinical Guardrails & Enforcement

### Crucial Safety Gateways (Part 1 Hard Requirements)

* **Zero-Fabrication Mandate (Req 3 & 4):** The agent prompt completely forbids guessing missing clinical elements. If required data values are absent, the system explicitly structures the fields as `[[MISSING - ESCALATED FOR REVIEW]]`.
* **Medication Reconciliation (Req 5):** The system tracks inpatient administration protocols (e.g., IV Meropenem and Subcutaneous Insulin) against written outpatient advice, highlighting abrupt drop-offs or unverified alternative compound transitions.
* **Contradiction Management (Req 6):** If conflicting data surfaces (e.g., Page 1 lists *Acute Gastroenteritis* while Page 3 records *DKA*), the engine logs a high-priority structural flag in the output schema rather than picking a value arbitrarily.
* **Deterministic Loop Controls (Req 9):** To prevent execution context drift or runaway runtime charges, a hard limit of `12` iterations is strictly enforced. If reached, the state controller intercepts execution and generates a safe emergency fallback template marking all unresolved spaces for immediate human validation.

---

## 3. Part 2: Learning from Doctor Edits (Feedback Simulation Loop)

To create a continuous improvement pipeline, the repository includes a simulated programmatic clinician loop that evaluates and learns from updates over time.

### 1. Reward & Accuracy Evaluation Function

The reward signal relies on **Character and Token-level Normalized Levenshtein Edit Distance**:

$$\text{Reward} = 1.0 - \min\left(1.0, \frac{\text{LevenshteinDistance}(\text{Draft}, \text{Edited})}{\max(\text{len}(\text{Draft}), \text{len}(\text{Edited}))}\right)$$

A higher reward indicates less manual modification was required by the physician, indicating superior structural and stylistic alignment.

### 2. Simulated Doctor Critic Policy

The engine builds a `SimulatedDoctorReviewer` executing a hidden, pedantic formatting preference framework:

* Explodes clinical short-hand expressions (e.g., `1-0-1` $\rightarrow$ `twice daily`).
* Mandates structural casing preferences across primary discharge diagnostics fields.
* Force-bulletizes extended, continuous paragraph narratives inside the `hospital_course` block.

### 3. Contextual In-Context Memory Loop Mechanism

Instead of complex fine-tuning pipelines that can alter underlying model alignments and degrade clinical safety balances, this architecture saves the `(draft, edit, reward)` triplets into a local appending `.jsonl` database cache. Before launching a patient evaluation run, the system retrieves historical corrections and appends them directly into the prompt context buffer as few-shot constraints.

---

## 4. Engineering Limitations & Mitigation Strategies

* **The Sycophancy / Vagueness Game Risk:** An optimization engine focused purely on lowering edit distance risks developing a "vague agent"—one that writes minimal or overly non-committal descriptions to avoid being corrected on specific formatting details.
* **Safety Isolation Mitigation:** We mitigate this risk by completely separating the clinical data extraction layer from stylistic formatting updates. Pydantic strictly requires all clinical data attributes (such as dates, dosages, and demographics), preventing the model from omitting medical facts in an attempt to optimize its alignment reward score.

---

## 5. Quick-Start Local Installation & Environment Execution

### 1. Installation Environment Setup

```bash
# Clone the repository and navigate to its root directory
cd medical-discharge-agent

# Build and activate a clean isolated Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install strictly locked structural dependencies
pip install -r requirements.txt

```

### 2. Secret Configuration

Create a `.env` file directly at the root of the project directory and insert your operational tokens safely:

```env
OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_OPENAI_API_KEY_HERE

```

### 3. Running the Live Pipeline Test

Place your unstructured patient documentation notes directly inside `data/source_pdfs/patient_2.pdf` (or provide a matching `.txt` configuration file fallback) and execute the test harness script:

```bash
python run_pipeline.py

```

### 4. Code Delivery Structure Review

Upon successful compilation execution, your compiled artifacts land safely inside the generated `/output` directory:

* **`output/patient_2_discharge_draft.json`:** The validated summary JSON artifact matching your required schema.
* **`output/patient_2_step_trace.json`:** The step-by-step observability log tracing the agent's real thoughts, actions, and internal decision paths.
