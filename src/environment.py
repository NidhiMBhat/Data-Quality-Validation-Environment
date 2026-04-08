"""
DataQualityEnvironment — implements the OpenEnv step()/reset()/state() interface.
"""
from __future__ import annotations

import copy
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import Action, Observation, Reward, StateInfo
from .tasks import (
    # Task 1
    TASK1_DESCRIPTION, TASK1_INITIAL_DATA, TASK1_SCHEMA, TASK1_CONSTRAINTS,
    TASK1_INITIAL_ISSUES, _task1_get_issues, grade_task1,
    # Task 2
    TASK2_DESCRIPTION, TASK2_INITIAL_DATA, TASK2_SCHEMA, TASK2_CONSTRAINTS,
    TASK2_INITIAL_ISSUES, _task2_get_issues, grade_task2,
    _normalize_date, _normalize_amount, _normalize_phone,
    # Task 3
    TASK3_DESCRIPTION, TASK3_INITIAL_DATA, TASK3_SCHEMA, TASK3_CONSTRAINTS,
    TASK3_INITIAL_INVALID, TASK3_CUSTOMERS, _task3_get_issues, grade_task3,
)

# ── task registry ─────────────────────────────────────────────────────────────

TASKS: Dict[str, Dict[str, Any]] = {
    "clean_nulls": {
        "description":       TASK1_DESCRIPTION,
        "initial_data":      TASK1_INITIAL_DATA,
        "schema":            TASK1_SCHEMA,
        "constraints":       TASK1_CONSTRAINTS,
        "initial_issues":    TASK1_INITIAL_ISSUES,
        "get_issues":        _task1_get_issues,
        "grade":             grade_task1,
        "max_steps":         15,
        "available_actions": [
            "fill_null(column, strategy)         — fill nulls in a column; strategy: mean|median|mode|<constant>",
            "drop_duplicates(subset?)            — remove duplicate rows; subset defaults to ['name','email']",
            "set_value(row_id, column, value)    — set a specific cell",
            "submit                              — finalise and get your score",
        ],
    },
    "normalize_formats": {
        "description":       TASK2_DESCRIPTION,
        "initial_data":      TASK2_INITIAL_DATA,
        "schema":            TASK2_SCHEMA,
        "constraints":       TASK2_CONSTRAINTS,
        "initial_issues":    TASK2_INITIAL_ISSUES,
        "get_issues":        _task2_get_issues,
        "grade":             grade_task2,
        "max_steps":         20,
        "available_actions": [
            "normalize_column(column)            — auto-normalise all values in column (transaction_date|amount|phone)",
            "set_value(row_id, column, value)    — manually correct a single cell",
            "submit                              — finalise and get your score",
        ],
    },
    "reconcile_tables": {
        "description":       TASK3_DESCRIPTION,
        "initial_data":      TASK3_INITIAL_DATA,
        "schema":            TASK3_SCHEMA,
        "constraints":       TASK3_CONSTRAINTS,
        "initial_issues":    TASK3_INITIAL_INVALID,
        "get_issues":        lambda d: _task3_get_issues(d, TASK3_CUSTOMERS),
        "grade":             lambda d: grade_task3(d, TASK3_CUSTOMERS),
        "max_steps":         20,
        "available_actions": [
            "delete_row(row_id)                  — delete an invalid order by its id",
            "set_value(row_id, column, value)    — correct a specific field in an order",
            "submit                              — finalise and get your score",
        ],
    },
}


class DataQualityEnvironment:
    """
    OpenEnv-compliant environment for real-world data quality tasks.

    Usage:
        env = DataQualityEnvironment()
        obs = env.reset("clean_nulls")
        obs, reward, done, info = env.step(Action(operation="fill_null", column="email", strategy="mode"))
        state = env.state()
    """

    def __init__(self) -> None:
        self.task_id: Optional[str] = None
        self.data: List[Dict[str, Any]] = []
        self.step_count: int = 0
        self.max_steps: int = 20
        self.done: bool = False
        self.current_score: float = 0.0
        self._cfg: Dict[str, Any] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "clean_nulls") -> Observation:
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}"
            )
        cfg = TASKS[task_id]
        self.task_id      = task_id
        self.data         = copy.deepcopy(cfg["initial_data"])
        self.step_count   = 0
        self.max_steps    = cfg["max_steps"]
        self.done         = False
        self.current_score = 0.0
        self._cfg         = cfg
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.task_id is None:
            raise RuntimeError("Call reset() before step()")

        if self.done:
            obs = self._make_observation()
            reward = Reward(
                score=self.current_score,
                issues_fixed_this_step=0,
                total_issues_fixed=self._issues_fixed_count(),
                issues_remaining=len(self._get_issues()),
                message="Episode already finished.",
            )
            return obs, reward, True, {"error": "Episode already done"}

        self.step_count += 1
        old_score = self.current_score

        message, error = self._apply_action(action)
        info: Dict[str, Any] = {}
        if error:
            info["error"] = error

        raw_score = self._cfg["grade"](self.data)

# Clamp score into (0,1)
        if raw_score <= 0.0:
             new_score = 0.01
        elif raw_score >= 1.0:
             new_score = 0.99
        else:
             new_score = raw_score

        self.current_score = new_score

        issues       = self._get_issues()
        n_remaining  = len(issues)
        n_fixed_step = max(0, round((new_score - old_score) * self._cfg["initial_issues"]))

        done_conditions = (
            action.operation.lower() == "submit"
            or n_remaining == 0
            or self.step_count >= self.max_steps
        )
        if done_conditions:
            self.done = True

        reward = Reward(
            score=new_score,
            issues_fixed_this_step=n_fixed_step,
            total_issues_fixed=self._issues_fixed_count(),
            issues_remaining=n_remaining,
            message=message,
        )
        return self._make_observation(), reward, self.done, info

    def state(self) -> StateInfo:
        return StateInfo(
            task_id=self.task_id,
            step_count=self.step_count,
            max_steps=self.max_steps,
            issues_remaining=len(self._get_issues()) if self._cfg else 0,
            done=self.done,
            current_score=self.current_score,
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_issues(self) -> List[str]:
        if not self._cfg:
            return []
        return self._cfg["get_issues"](self.data)

    def _issues_fixed_count(self) -> int:
        return round(self.current_score * self._cfg.get("initial_issues", 1))

    def _make_observation(self) -> Observation:
        issues = self._get_issues()
        return Observation(
            task_id=self.task_id or "",
            task_description=self._cfg.get("description", ""),
            data=copy.deepcopy(self.data),
            schema_info=self._cfg.get("schema", {}),
            constraints=self._cfg.get("constraints", []),
            issues_found=issues,
            issues_remaining=len(issues),
            step_count=self.step_count,
            max_steps=self.max_steps,
            available_actions=self._cfg.get("available_actions", []),
        )

    def _apply_action(self, action: Action) -> Tuple[str, Optional[str]]:
        """Apply *action* to self.data.  Returns (human message, error|None)."""
        op = action.operation.lower().strip()

        # ── submit ────────────────────────────────────────────────────────────
        if op == "submit":
            return "Submitted — episode ending.", None

        # ── set_value ─────────────────────────────────────────────────────────
        if op == "set_value":
            if action.row_id is None or action.column is None:
                return "", "set_value requires both row_id and column"
            for row in self.data:
                if row.get("id") == action.row_id:
                    old = row.get(action.column, "N/A")
                    row[action.column] = action.value
                    return (
                        f"Set row id={action.row_id} '{action.column}': "
                        f"{old!r} → {action.value!r}"
                    ), None
            return "", f"Row id={action.row_id} not found"

        # ── delete_row ────────────────────────────────────────────────────────
        if op == "delete_row":
            if action.row_id is None:
                return "", "delete_row requires row_id"
            before = len(self.data)
            self.data = [r for r in self.data if r.get("id") != action.row_id]
            if len(self.data) < before:
                return f"Deleted order id={action.row_id}", None
            return "", f"Row id={action.row_id} not found"

        # ── fill_null ─────────────────────────────────────────────────────────
        if op == "fill_null":
            if not action.column:
                return "", "fill_null requires column"
            col      = action.column
            strategy = action.strategy or "mode"
            null_rows = [r for r in self.data if r.get(col) is None]
            if not null_rows:
                return f"No nulls in '{col}'", None

            non_null = [r[col] for r in self.data if r.get(col) is not None]
            fill_val: Any
            if strategy == "mean":
                try:
                    fill_val = round(statistics.mean(float(v) for v in non_null), 2)
                except Exception:
                    fill_val = non_null[0] if non_null else None
            elif strategy == "median":
                try:
                    fill_val = statistics.median(float(v) for v in non_null)
                except Exception:
                    fill_val = non_null[0] if non_null else None
            elif strategy == "mode":
                try:
                    fill_val = statistics.mode(non_null)
                except Exception:
                    fill_val = non_null[0] if non_null else None
            else:
                # Treat strategy as a constant; try numeric coercions
                try:
                    fill_val = int(strategy)
                except (ValueError, TypeError):
                    try:
                        fill_val = float(strategy)
                    except (ValueError, TypeError):
                        fill_val = strategy

            for row in self.data:
                if row.get(col) is None:
                    row[col] = fill_val

            return (
                f"Filled {len(null_rows)} null(s) in '{col}' "
                f"with {fill_val!r} (strategy={strategy})"
            ), None

        # ── drop_duplicates ───────────────────────────────────────────────────
        if op == "drop_duplicates":
            subset   = action.subset or ["name", "email"]
            seen: Dict[tuple, int] = {}
            new_data: List[Dict] = []
            dropped  = 0
            for row in self.data:
                key = tuple(row.get(c) for c in subset)
                if key not in seen:
                    seen[key] = row["id"]
                    new_data.append(row)
                else:
                    dropped += 1
            self.data = new_data
            return f"Dropped {dropped} duplicate row(s) on columns {subset}", None

        # ── normalize_column ──────────────────────────────────────────────────
        if op == "normalize_column":
            if not action.column:
                return "", "normalize_column requires column"
            col = action.column
            _normalizers = {
                "transaction_date": _normalize_date,
                "amount":           _normalize_amount,
                "phone":            _normalize_phone,
            }
            if col not in _normalizers:
                return "", (
                    f"normalize_column not supported for '{col}'. "
                    f"Valid columns: {list(_normalizers)}"
                )
            norm_fn = _normalizers[col]
            fixed = 0
            for row in self.data:
                old = row.get(col)
                new = norm_fn(old)
                if new != old:
                    row[col] = new
                    fixed += 1
            return f"Normalised {fixed} value(s) in '{col}'", None

        # ── unknown ───────────────────────────────────────────────────────────
        return "", (
            f"Unknown operation '{op}'. "
            "Valid: fill_null | drop_duplicates | normalize_column | set_value | delete_row | submit"
        )
