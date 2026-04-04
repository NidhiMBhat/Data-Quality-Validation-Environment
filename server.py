"""
FastAPI server — exposes the DataQualityEnvironment via the OpenEnv HTTP API.

Endpoints:
  GET  /health         — liveness probe
  GET  /tasks          — list available tasks
  POST /reset          — start / restart an episode  { "task_id": "clean_nulls" }
  POST /step           — take one action             { "operation": "...", ... }
  GET  /state          — inspect current state
"""
from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from src.environment import DataQualityEnvironment, TASKS
from src.models import Action, StepResult, StateInfo
from pydantic import BaseModel


app = FastAPI(
    title="Data Quality OpenEnv",
    description=(
        "OpenEnv environment for training AI agents on real-world "
        "data cleaning and validation tasks."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (stateful, single-session)
env = DataQualityEnvironment()


# ── models ────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "clean_nulls"


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "data-quality-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    difficulty_map = {
        "clean_nulls":        "easy",
        "normalize_formats":  "medium",
        "reconcile_tables":   "hard",
    }
    return {
        "tasks": [
            {
                "id":           tid,
                "difficulty":   difficulty_map.get(tid, "unknown"),
                "max_steps":    cfg["max_steps"],
                "initial_issues": cfg["initial_issues"],
            }
            for tid, cfg in TASKS.items()
        ]
    }


@app.post("/reset")
def reset(request: ResetRequest = Body(default=ResetRequest())):
    """
    Reset the environment for the given task.
    Body: { "task_id": "clean_nulls" | "normalize_formats" | "reconcile_tables" }
    Returns an Observation.
    """
    try:
        obs = env.reset(request.task_id)
        return obs.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(action: Action):
    """
    Take one action.
    Body: { "operation": "...", "column": "...", ... }
    Returns a StepResult { observation, reward, done, info }.
    """
    if env.task_id is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")
    obs, reward, done, info = env.step(action)
    return StepResult(observation=obs, reward=reward, done=done, info=info).model_dump()


@app.get("/state")
def state():
    """Return current environment state (task, step count, score, …)."""
    return env.state().model_dump()


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
