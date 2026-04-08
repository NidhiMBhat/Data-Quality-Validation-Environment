"""
Task definitions for the Data Quality OpenEnv environment.

Task 1 — clean_nulls        (easy)   : Fix null values and remove duplicate rows
Task 2 — normalize_formats  (medium) : Standardise date, amount, and phone formats
Task 3 — reconcile_tables   (hard)   : Fix FK violations and business-rule violations
"""
from __future__ import annotations

import re
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Clean Nulls and Duplicates  (easy)
# ─────────────────────────────────────────────────────────────────────────────

TASK1_DESCRIPTION = """You are a data quality agent. Clean a customer records dataset.

Issues to fix:
1. NULL/missing values in required columns: email, age, city
2. Duplicate rows — same person appearing more than once (matched on name + email)

Available actions:
  fill_null(column, strategy)         — strategy: 'mean' | 'median' | 'mode' | '<constant>'
  drop_duplicates(subset?)            — subset defaults to ['name', 'email']
  set_value(row_id, column, value)    — manually override a single cell
  submit                              — finalise and receive your score

Goal: zero nulls in [email, age, city] AND zero duplicate (name, email) pairs."""

TASK1_INITIAL_DATA: List[Dict[str, Any]] = [
    {"id": 1,  "name": "Alice Johnson", "email": "alice@example.com",   "age": 30,   "city": "New York"},
    {"id": 2,  "name": "Bob Smith",     "email": None,                  "age": 25,   "city": "Los Angeles"},
    {"id": 3,  "name": "Charlie Brown", "email": "charlie@example.com", "age": None, "city": "Chicago"},
    {"id": 4,  "name": "Alice Johnson", "email": "alice@example.com",   "age": 30,   "city": "New York"},   # duplicate of id=1
    {"id": 5,  "name": "Diana Prince",  "email": "diana@example.com",   "age": 35,   "city": None},
    {"id": 6,  "name": "Eve Adams",     "email": None,                  "age": 28,   "city": "Boston"},
    {"id": 7,  "name": "Frank Miller",  "email": "frank@example.com",   "age": 45,   "city": "Seattle"},
    {"id": 8,  "name": "Bob Smith",     "email": None,                  "age": 25,   "city": "Los Angeles"}, # duplicate of id=2
    {"id": 9,  "name": "Grace Lee",     "email": "grace@example.com",   "age": None, "city": "Austin"},
    {"id": 10, "name": "Henry Ford",    "email": "henry@example.com",   "age": 60,   "city": "Detroit"},
]

TASK1_SCHEMA: Dict[str, str] = {
    "id":    "integer — unique row identifier",
    "name":  "string  — full name (required)",
    "email": "string  — valid email address (required, no nulls)",
    "age":   "integer — 18–100 inclusive (required, no nulls)",
    "city":  "string  — city name (required, no nulls)",
}

TASK1_CONSTRAINTS: List[str] = [
    "No null values allowed in: email, age, city",
    "No duplicate rows based on (name, email) combination",
]


def _task1_get_issues(data: List[Dict]) -> List[str]:
    issues: List[str] = []
    seen: Dict[tuple, int] = {}
    for row in data:
        for col in ("email", "age", "city"):
            if row.get(col) is None:
                issues.append(f"Row id={row['id']}: '{col}' is null")
        key = (row.get("name"), row.get("email"))
        if key in seen:
            issues.append(f"Row id={row['id']}: duplicate of row id={seen[key]}")
        else:
            seen[key] = row["id"]
    return issues


TASK1_INITIAL_ISSUES: int = len(_task1_get_issues(TASK1_INITIAL_DATA))  # 8


def grade_task1(data: List[Dict]) -> float:
    current = len(_task1_get_issues(data))
    raw = max(0.0, 1.0 - current / TASK1_INITIAL_ISSUES)
    return round(max(0.001, min(0.999, raw)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Normalize Data Formats  (medium)
# ─────────────────────────────────────────────────────────────────────────────

TASK2_DESCRIPTION = """You are a data quality agent. Normalise a transactions dataset.

All columns must match exactly these formats:
  transaction_date  →  YYYY-MM-DD               (e.g. "2024-01-15")
  amount            →  plain decimal, no symbols  (e.g. "1234.56")
  phone             →  E164: +1XXXXXXXXXX         (e.g. "+12025550100")

Available actions:
  normalize_column(column)             — auto-normalise every value in that column
  set_value(row_id, column, value)     — manually correct a single cell
  submit                               — finalise and receive your score

Goal: every cell in [transaction_date, amount, phone] must pass format validation."""

TASK2_INITIAL_DATA: List[Dict[str, Any]] = [
    {"id": 1, "transaction_date": "2024-01-15",   "amount": "1234.56",   "phone": "+12025550100",    "status": "completed"},
    {"id": 2, "transaction_date": "01/20/2024",   "amount": "$2,345.67", "phone": "(202) 555-0101",  "status": "pending"},
    {"id": 3, "transaction_date": "15-Feb-2024",  "amount": "3,456.00",  "phone": "2025550102",      "status": "completed"},
    {"id": 4, "transaction_date": "2024/03/10",   "amount": "$4,567.89", "phone": "+1-202-555-0103", "status": "failed"},
    {"id": 5, "transaction_date": "Apr 5 2024",   "amount": "5,678.90",  "phone": "1-202-555-0104",  "status": "completed"},
    {"id": 6, "transaction_date": "20240601",     "amount": "$6789.00",  "phone": "202.555.0105",    "status": "pending"},
    {"id": 7, "transaction_date": "July 4, 2024", "amount": "7,890.12",  "phone": "12025550106",     "status": "completed"},
]

TASK2_SCHEMA: Dict[str, str] = {
    "id":               "integer — unique identifier",
    "transaction_date": "string  — YYYY-MM-DD format only",
    "amount":           "string  — plain decimal e.g. '1234.56' (no $, no commas)",
    "phone":            "string  — E164: +1 followed by exactly 10 digits",
    "status":           "string  — completed|pending|failed  (DO NOT change)",
}

TASK2_CONSTRAINTS: List[str] = [
    r"transaction_date must match regex: ^\d{4}-\d{2}-\d{2}$",
    r"amount must match regex: ^\d+(\.\d{1,2})?$  (digits, optional dot + 1-2 decimals)",
    r"phone must match regex: ^\+1\d{10}$  (+1 then exactly 10 digits)",
    "status column: leave unchanged",
]

_DATE_RE   = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_AMOUNT_RE = re.compile(r"^\d+(\.\d{1,2})?$")
_PHONE_RE  = re.compile(r"^\+1\d{10}$")


def _task2_get_issues(data: List[Dict]) -> List[str]:
    issues: List[str] = []
    for row in data:
        rid = row["id"]
        if not _DATE_RE.match(str(row.get("transaction_date", ""))):
            issues.append(f"Row id={rid}: transaction_date '{row['transaction_date']}' ≠ YYYY-MM-DD")
        if not _AMOUNT_RE.match(str(row.get("amount", ""))):
            issues.append(f"Row id={rid}: amount '{row['amount']}' is not a plain decimal")
        if not _PHONE_RE.match(str(row.get("phone", ""))):
            issues.append(f"Row id={rid}: phone '{row['phone']}' is not +1XXXXXXXXXX")
    return issues


TASK2_INITIAL_ISSUES: int = len(_task2_get_issues(TASK2_INITIAL_DATA))  # 18


def grade_task2(data: List[Dict]) -> float:
    current = len(_task2_get_issues(data))
    raw = max(0.0, 1.0 - current / TASK2_INITIAL_ISSUES)
    return round(max(0.001, min(0.999, raw)), 4)


# ── normalisation helpers used by the environment ────────────────────────────

_DATE_FORMATS = (
    "%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%Y/%m/%d",
    "%b %d %Y", "%B %d, %Y", "%Y%m%d", "%d %b %Y",
)


def _normalize_date(val: Any) -> str:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(str(val).strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return str(val)


def _normalize_amount(val: Any) -> str:
    try:
        cleaned = re.sub(r"[^\d.]", "", str(val))
        return f"{float(cleaned):.2f}"
    except (ValueError, TypeError):
        return str(val)


def _normalize_phone(val: Any) -> str:
    digits = re.sub(r"\D", "", str(val))
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return str(val)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Reconcile Related Tables  (hard)
# ─────────────────────────────────────────────────────────────────────────────

TASK3_DESCRIPTION = """You are a data quality agent. Reconcile an orders dataset against a customers reference table.

CUSTOMERS TABLE (reference — read-only):
| id | name         | credit_limit |
|----|--------------|-------------|
|  1 | Alice Corp   | 10000.00    |
|  2 | Bob LLC      |  5000.00    |
|  3 | Charlie Inc  | 20000.00    |
|  4 | Diana Co     |  8000.00    |

ORDERS TABLE violations to fix:
  1. FK violation      — customer_id references a non-existent customer
  2. Credit exceeded   — order amount > customer's credit_limit
  3. Non-positive amt  — amount must be > 0
  4. Bad delivery date — delivery_date must be strictly AFTER order_date

Available actions:
  delete_row(row_id)                  — remove an invalid order
  set_value(row_id, column, value)    — correct a specific field value
  submit                              — finalise and receive your score

Goal: every remaining order must satisfy all four constraints."""

TASK3_CUSTOMERS: List[Dict[str, Any]] = [
    {"id": 1, "name": "Alice Corp",  "credit_limit": 10000.0},
    {"id": 2, "name": "Bob LLC",     "credit_limit":  5000.0},
    {"id": 3, "name": "Charlie Inc", "credit_limit": 20000.0},
    {"id": 4, "name": "Diana Co",    "credit_limit":  8000.0},
]

TASK3_INITIAL_DATA: List[Dict[str, Any]] = [
    {"id": 1,  "customer_id": 1,  "amount":  5000.0, "order_date": "2024-01-10", "delivery_date": "2024-01-20"},  # ✓ valid
    {"id": 2,  "customer_id": 2,  "amount":  6000.0, "order_date": "2024-01-15", "delivery_date": "2024-01-25"},  # ✗ exceeds Bob LLC limit $5000
    {"id": 3,  "customer_id": 5,  "amount":  1000.0, "order_date": "2024-01-20", "delivery_date": "2024-01-30"},  # ✗ customer_id=5 not found
    {"id": 4,  "customer_id": 3,  "amount":  -500.0, "order_date": "2024-02-01", "delivery_date": "2024-02-10"},  # ✗ negative amount
    {"id": 5,  "customer_id": 1,  "amount":  3000.0, "order_date": "2024-02-15", "delivery_date": "2024-02-10"},  # ✗ delivery before order
    {"id": 6,  "customer_id": 99, "amount":  2000.0, "order_date": "2024-02-20", "delivery_date": "2024-03-01"},  # ✗ customer_id=99 not found
    {"id": 7,  "customer_id": 4,  "amount":  9000.0, "order_date": "2024-03-01", "delivery_date": "2024-03-15"},  # ✗ exceeds Diana Co limit $8000
    {"id": 8,  "customer_id": 2,  "amount":  4000.0, "order_date": "2024-03-10", "delivery_date": "2024-03-20"},  # ✓ valid
    {"id": 9,  "customer_id": 3,  "amount": 15000.0, "order_date": "2024-03-15", "delivery_date": "2024-03-25"},  # ✓ valid
    {"id": 10, "customer_id": 1,  "amount":     0.0, "order_date": "2024-03-20", "delivery_date": "2024-03-25"},  # ✗ zero amount
]

TASK3_SCHEMA: Dict[str, str] = {
    "id":            "integer — unique order identifier",
    "customer_id":   "integer — FK → customers.id  (must be 1, 2, 3, or 4)",
    "amount":        "float   — must be > 0 and ≤ customer credit_limit",
    "order_date":    "string  — YYYY-MM-DD",
    "delivery_date": "string  — YYYY-MM-DD, must be strictly after order_date",
}

TASK3_CONSTRAINTS: List[str] = [
    "customer_id must reference an existing customer (valid ids: 1, 2, 3, 4)",
    "amount must be strictly positive (> 0)",
    "amount must not exceed the customer's credit_limit",
    "delivery_date must be strictly after order_date",
]


def _task3_get_invalid_order_ids(
    data: List[Dict],
    customers: List[Dict] = TASK3_CUSTOMERS,
) -> Dict[int, List[str]]:
    """Returns {order_id: [reason, ...]} for every invalid order."""
    cmap = {c["id"]: c for c in customers}
    invalid: Dict[int, List[str]] = {}

    for order in data:
        oid = order["id"]
        cid = order["customer_id"]
        reasons: List[str] = []

        if cid not in cmap:
            reasons.append(f"customer_id={cid} not found (FK violation)")
        else:
            customer = cmap[cid]
            if order["amount"] <= 0:
                reasons.append(f"amount={order['amount']} is not positive")
            elif order["amount"] > customer["credit_limit"]:
                reasons.append(
                    f"amount={order['amount']} exceeds {customer['name']} "
                    f"credit_limit={customer['credit_limit']}"
                )
        try:
            od = datetime.strptime(order["order_date"], "%Y-%m-%d")
            dd = datetime.strptime(order["delivery_date"], "%Y-%m-%d")
            if dd <= od:
                reasons.append(
                    f"delivery_date={order['delivery_date']} must be "
                    f"after order_date={order['order_date']}"
                )
        except ValueError:
            reasons.append("invalid date format in order_date or delivery_date")

        if reasons:
            invalid[oid] = reasons
    return invalid


def _task3_get_issues(data: List[Dict], customers: List[Dict] = TASK3_CUSTOMERS) -> List[str]:
    invalid = _task3_get_invalid_order_ids(data, customers)
    issues: List[str] = []
    for oid, reasons in invalid.items():
        for r in reasons:
            issues.append(f"Order id={oid}: {r}")
    return issues


TASK3_INITIAL_INVALID: int = len(_task3_get_invalid_order_ids(TASK3_INITIAL_DATA))  # 7


def grade_task3(data: List[Dict], customers: List[Dict] = TASK3_CUSTOMERS) -> float:
    current_invalid = len(_task3_get_invalid_order_ids(data, customers))
    fixed = max(0, TASK3_INITIAL_INVALID - current_invalid)
    raw = fixed / TASK3_INITIAL_INVALID
    return round(max(0.001, min(0.999, raw)), 4)
