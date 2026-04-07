# Data Quality & Validation OpenEnv Environment 🧹

A real-world OpenEnv environment for training and evaluating AI agents on  
data cleaning, format normalisation, and cross-table integrity tasks.

---

## Why this environment?

Data quality work is one of the most time-consuming activities in every organisation that runs data pipelines. Engineers spend **30–40% of their time** fixing nulls, deduplicating records, standardising formats, and enforcing referential integrity.

This environment allows agents to learn and automate these processes using:
- deterministic evaluation
- structured observations
- meaningful partial rewards

---

## Why this is challenging for AI agents

This environment requires more than simple rule execution:

- **Multi-step reasoning** — correct sequencing of actions is necessary  
- **Constraint awareness** — actions can introduce new violations  
- **Trade-offs** — delete vs fix vs modify decisions  
- **Irreversibility** — wrong deletions cannot be undone  
- **Implicit strategy** — optimal sequence is not explicitly given  

This makes it suitable for evaluating **true decision-making capability**, not just pattern matching.

---

## Deterministic Evaluation Advantage

Unlike LLM-judged environments, this system uses fully programmatic grading:

- exact constraint checking (nulls, duplicates, formats, integrity)
- reproducible scores across runs
- no hallucination or subjective evaluation

---

## Tasks

| ID | Name | Difficulty | Issues | Max Steps |
|----|------|-----------|--------|-----------|
| clean_nulls | Clean Nulls & Duplicates | ⭐ Easy | 8 | 15 |
| normalize_formats | Normalize Data Formats | ⭐⭐ Medium | 18 | 20 |
| reconcile_tables | Reconcile Related Tables | ⭐⭐⭐ Hard | 7 invalid orders | 20 |

---

### Task 1 — Clean Nulls & Duplicates

- 10-row dataset  
- 6 null values (email, age, city)  
- 2 duplicate rows (name + email)  

Goal: eliminate all nulls and duplicates.

---

### Task 2 — Normalize Data Formats

- 7-row dataset  
- 18 format inconsistencies  

Fix:
- dates → `YYYY-MM-DD`
- amounts → numeric
- phone → E.164 format

---

### Task 3 — Reconcile Related Tables

Orders must satisfy constraints against customers:

- FK violations (non-existent customer_id)
- credit limit exceeded
- non-positive amounts
- invalid delivery dates

---

## Action Space

Agents interact using JSON actions via `POST /step`:

```json
{"operation": "fill_null", "column": "email", "strategy": "mode"}
{"operation": "drop_duplicates", "subset": ["name", "email"]}
{"operation": "normalize_column", "column": "transaction_date"}
{"operation": "delete_row", "row_id": 3}
{"operation": "set_value", "row_id": 5, "column": "delivery_date", "value": "2024-03-01"}
{"operation": "submit"}
Observation Space

Each step returns:

dataset snapshot
detected issues
remaining issue count
available actions
Reward Function
Score = 1 − (current_issues / initial_issues)
Smooth partial reward signal
Each fix increases score incrementally
Episode ends on:
submit
all issues resolved
max steps reached
Baseline Performance (gpt-4o-mini)
Task	Score
clean_nulls	1.00
normalize_formats	1.00
reconcile_tables	1.00
Average	1.00

The baseline achieves perfect performance due to:

structured observations exposing issues
deterministic action space
reward shaping guiding corrections

However, correct multi-step reasoning is still required, especially in the hard task.

Setup & Usage
Local
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
Docker
docker build -t data-quality-env .
docker run -p 7860:7860 data-quality-env
Run Inference
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-key"
export ENV_URL="http://localhost:7860"

python inference.py
Project Structure
data-quality-env/
├── inference.py
├── openenv.yaml
├── server.py
├── Dockerfile
├── requirements.txt
├── README.md
└── src/
    ├── models.py
    ├── tasks.py
    └── environment.py

Validation
pip install openenv-core
openenv validate
