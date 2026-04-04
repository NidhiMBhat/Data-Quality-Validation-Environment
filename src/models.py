"""
Typed Pydantic models for the Data Quality OpenEnv environment.
Implements the full OpenEnv spec: Observation, Action, Reward, StepResult, StateInfo.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str = Field(..., description="Active task identifier")
    task_description: str = Field(..., description="Full task description with instructions")
    data: List[Dict[str, Any]] = Field(..., description="Current state of the dataset")
    schema_info: Dict[str, str] = Field(..., description="Column name → expected type/format description")
    constraints: List[str] = Field(..., description="Human-readable data quality constraints")
    issues_found: List[str] = Field(..., description="List of specific issues currently present")
    issues_remaining: int = Field(..., description="Total count of remaining issues")
    step_count: int = Field(..., description="Steps taken so far this episode")
    max_steps: int = Field(..., description="Maximum allowed steps per episode")
    available_actions: List[str] = Field(..., description="Available operations with usage descriptions")


class Action(BaseModel):
    """An action the agent can take to modify the dataset."""
    operation: str = Field(
        ...,
        description=(
            "Operation to perform. One of: "
            "fill_null | drop_duplicates | normalize_column | set_value | delete_row | submit"
        ),
    )
    column: Optional[str] = Field(None, description="Target column name (fill_null, normalize_column, set_value)")
    row_id: Optional[int] = Field(None, description="Row ID to operate on (set_value, delete_row)")
    value: Optional[Any] = Field(None, description="Value to write (set_value)")
    strategy: Optional[str] = Field(
        None,
        description="fill_null strategy: 'mean' | 'median' | 'mode' | '<constant_value>'",
    )
    subset: Optional[List[str]] = Field(
        None,
        description="Columns to consider for drop_duplicates (defaults to ['name','email'])",
    )


class Reward(BaseModel):
    """Per-step reward signal."""
    score: float = Field(..., ge=0.0, le=1.0, description="Cumulative task completion score (0.0–1.0)")
    issues_fixed_this_step: int = Field(..., description="Number of issues resolved by this action")
    total_issues_fixed: int = Field(..., description="Total issues fixed across the episode so far")
    issues_remaining: int = Field(..., description="Issues still present after this action")
    message: str = Field(..., description="Human-readable description of what happened")


class StepResult(BaseModel):
    """Return value of POST /step."""
    observation: Observation
    reward: Reward
    done: bool = Field(..., description="True when the episode has ended")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra debugging info")


class StateInfo(BaseModel):
    """Return value of GET /state."""
    task_id: Optional[str] = None
    step_count: int = 0
    max_steps: int = 20
    issues_remaining: int = 0
    done: bool = False
    current_score: float = 0.0
