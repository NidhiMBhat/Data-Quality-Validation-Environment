Data Quality & Validation OpenEnv Environment🧹
A real-world OpenEnv environment for training and evaluating AI agents on data cleaning, format normalisation, and cross-table integrity tasks.

Why this environment?
Data quality work is one of the most time-consuming activities in every organisation that runs data pipelines. Engineers spend 30–40 % of their time fixing nulls, deduplicating records, standardising formats, and enforcing referential integrity. This environment lets agents learn to do exactly that, with deterministic, fully programmatic graders and rich partial-progress rewards.

Why this is challenging for AI agents
This environment is not a simple rule-based transformation task. It requires:

Multi-step reasoning — agents must plan sequences of actions (e.g., deduplicate before fixing nulls)
Constraint awareness — actions can introduce new violations if applied incorrectly
Trade-offs — especially in the hard task, agents must decide whether to delete, repair, or modify rows
Irreversible decisions — incorrect deletions cannot be undone
Partial observability of optimal strategy — the best sequence is not explicitly given
These aspects make the environment suitable for evaluating real decision-making capabilities rather than simple pattern matching.

Deterministic Evaluation Advantage
Unlike many environments that rely on LLM-based grading, this environment uses fully deterministic, programmatic evaluation:

Exact constraint checking (nulls, duplicates, formats, integrity)
Reproducible scores across runs
No hallucination or subjective scoring
This ensures fair and consistent benchmarking of agent performance.

Tasks
ID	Name	Difficulty	Issues	Max Steps
clean_nulls	Clean Nulls & Duplicates	⭐ Easy	8	15
normalize_formats	Normalize Data Formats	⭐⭐ Medium	18	20
reconcile_tables	Reconcile Related Tables	⭐⭐⭐ Hard	7 invalid orders	20
Task 1 — Clean Nulls & Duplicates (Easy)
A 10-row customer records dataset contains:

6 null values spread across email, age, and city columns
2 duplicate rows (matched on name + email)
The agent must fill all nulls and remove all duplicates to score 1.0.

Task 2 — Normalize Data Formats (Medium)
A 7-row transactions dataset has 18 format violations across three columns:

transaction_date — various formats (MM/DD/YYYY, DD-Mon-YYYY, YYYYMMDD…) → must be YYYY-MM-DD
amount — dollar signs, commas mixed in ($1,234.56) → must be plain decimal (1234.56)
phone — many dialects (+1-NXX-NXX-XXXX, (NXX) NXX-XXXX…) → must be E164 (+1XXXXXXXXXX)
Task 3 — Reconcile Related Tables (Hard)
A 10-row orders table must be validated against a 4-row customers reference. 7 orders contain violations across 4 constraint types:

FK violation — customer_id references a non-existent customer (×2)
Credit exceeded — amount > customer's credit_limit (×2)
Non-positive amount — amount ≤ 0 (×2)
Bad delivery date — delivery_date ≤ order_date (×1)
Action Space
All actions are JSON objects sent to POST /step:

{"operation": "fill_null",        "column": "email",    "strategy": "mode"}
{"operation": "drop_duplicates",  "subset": ["name", "email"]}
{"operation": "normalize_column", "column": "transaction_date"}
{"operation": "delete_row",       "row_id": 3}
{"operation": "set_value",        "row_id": 5, "column": "delivery_date", "value": "2024-03-01"}
{"operation": "submit"}
Field	Type	Description
operation	string	One of: fill_null · drop_duplicates · normalize_column · set_value · delete_row · submit
column	string?	Target column name
row_id	int?	Row id to target
value	any?	Value to write
strategy	string?	fill_null strategy: mean·median·mode·<constant>
subset	list?	Columns to check for duplicates
Observation Space
Each POST /step and POST /reset returns:

{
  "task_id": "clean_nulls",
  "task_description": "...",
  "data": [{...}, ...],
  "schema_info": {"email": "string — required", ...},
  "constraints": ["No nulls in email, age, city", ...],
  "issues_found": ["Row id=2: 'email' is null", ...],
  "issues_remaining": 8,
  "step_count": 1,
  "max_steps": 15,
  "available_actions": ["fill_null(column, strategy) — ...", ...]
}
Reward Function
Per-step reward = current grade score (0.0–1.0), computed after every action
Score = 1 − (current_issues / initial_issues), providing a smooth partial-progress signal
Each issue fixed → +1/N to score (N = initial issue count)
Harmful actions (e.g. deleting valid orders) receive no bonus
Episode ends on submit, all-issues-fixed, or max_steps reached
HTTP API
Method	Endpoint	Body / Params	Description
GET	/health	—	Liveness probe
GET	/tasks	—	List tasks with metadata
POST	/reset	{"task_id": "clean_nulls"}	Start episode, returns Observation
POST	/step	Action JSON	Take action, returns StepResult
GET	/state	—	Inspect current state
Setup & Usage
Local (Python)
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
Docker
docker build -t data-quality-env .
docker run -p 7860:7860 data-quality-env
Run the inference baseline
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
export ENV_URL="http://localhost:7860"   # or your HF Space URL

python inference.py
Baseline Scores (gpt-4o-mini)
Task	Score	Notes
clean_nulls	~1.00	Occasionally misses one null fill
normalize_formats	~1.00	May struggle with ambiguous date formats
reconcile_tables	~1.00	FK reasoning sometimes requires explicit prompting
Average	~1.00	
Baseline Performance (gpt-4o-mini)
The baseline agent achieves a perfect score (1.0) across all tasks.

This is possible because:

The environment provides structured observations (issues explicitly listed)
Actions are atomic and deterministic
Reward shaping guides the agent toward correct sequences
However, achieving this score still requires correct multi-step decision-making, especially in the hard task involving referential integrity and business rules.

Project Structure
data-quality-env/
├── inference.py        ← mandatory baseline script
├── openenv.yaml        ← OpenEnv metadata
├── server.py           ← FastAPI HTTP server
├── Dockerfile          ← container definition
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── models.py       ← Pydantic Observation / Action / Reward models
    ├── tasks.py        ← task data, constraints, graders
    └── environment.py  ← DataQualityEnvironment (step / reset / state)
OpenEnv Validation
pip install openenv-core
openenv validate
All three gates must pass before submission:

HF Space responds to POST /reset with HTTP 200
docker build succeeds
openenv validate passes
