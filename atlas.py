from __future__ import annotations

import json
import os
import statistics
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

try:
    from openai import OpenAI
except ImportError:  # optional until a key is configured
    OpenAI = None

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./atlas.db")
DEFAULT_JWT_SECRET = "dev_secret_change_in_production_32_chars_min"
JWT_SECRET = os.getenv("JWT_SECRET", DEFAULT_JWT_SECRET)
JWT_ALGORITHM = "HS256"
TOKEN_MINUTES = 30
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CLIENT_FACTORY = OpenAI
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
]
DEMO_ORG_ID = "demo-org"
DEMO_USER_ID = "demo-user"
DEMO_USER_EMAIL = "owner@atlas.local"

if APP_ENV != "development":
    if JWT_SECRET == DEFAULT_JWT_SECRET:
        raise RuntimeError("JWT_SECRET must be overridden outside development")
    if len(JWT_SECRET.encode("utf-8")) < 32:
        raise RuntimeError("JWT_SECRET must be at least 32 bytes outside development")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), index=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    actor: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ProcessDefinition(Base):
    __tablename__ = "process_definitions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(String(500))
    automation_score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="proposed")
    observed_facts: Mapped[dict] = mapped_column(JSON)
    ai_interpretation: Mapped[str] = mapped_column(String(2000))
    workflow_spec: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class AutomationExecution(Base):
    __tablename__ = "automation_executions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), index=True)
    automation_id: Mapped[str] = mapped_column(
        ForeignKey("process_definitions.id"), index=True
    )
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32))
    context: Mapped[dict] = mapped_column(JSON)
    current_step_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    awaiting_approval: Mapped[bool] = mapped_column(Boolean, default=False)
    approval_step_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    approver_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    approval_decision: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    active_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)


class StepExecution(Base):
    __tablename__ = "step_executions"
    __table_args__ = (
        UniqueConstraint("execution_id", "sequence", name="uq_execution_sequence"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    execution_id: Mapped[str] = mapped_column(
        ForeignKey("automation_executions.id"), index=True
    )
    organization_id: Mapped[str] = mapped_column(String(64), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    step_id: Mapped[str] = mapped_column(String(64))
    step_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    input_data: Mapped[dict] = mapped_column(JSON, default=dict)
    output_data: Mapped[dict] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)


class Lead(Base):
    __tablename__ = "leads"
    __table_args__ = (
        UniqueConstraint("organization_id", "entity_id", name="uq_org_lead_entity"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), index=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    email: Mapped[str] = mapped_column(String(256))
    assigned_to: Mapped[str] = mapped_column(String(128))
    last_assignment_execution_id: Mapped[str] = mapped_column(String(64), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


Base.metadata.create_all(bind=engine)


class Actor(BaseModel):
    org_id: str
    user_id: str


class ConditionStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["condition"]
    label: str
    field: str
    operator: Literal["=="]
    value: str
    goto_true: str
    goto_false: str


class ApprovalStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["human_approval"]
    label: str
    goto_approved: str
    goto_rejected: str


class ActionStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["api_call"]
    label: str
    action: Literal["assign_lead", "send_notification"]
    payload: dict[str, Any] = Field(default_factory=dict)
    next_step: str | None = None


Step = ConditionStep | ApprovalStep | ActionStep


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trigger: str
    steps: list[Step]


class ExecuteRequest(BaseModel):
    entity_id: str
    email: str = "lead@example.com"


class ROIRequest(BaseModel):
    hourly_labor_cost: float = Field(ge=0)
    automation_runs_per_month: int = Field(ge=0)
    implementation_cost: float = Field(ge=0)
    api_cost_per_run: float = Field(ge=0)
    avg_human_action_minutes: float = Field(default=5.0, ge=0)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def issue_demo_token() -> str:
    payload = {
        "org_id": DEMO_ORG_ID,
        "user_id": DEMO_USER_ID,
        "email": DEMO_USER_EMAIL,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def current_actor(authorization: str = Header(default="")) -> Actor:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        payload = jwt.decode(
            authorization[7:],
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "org_id", "user_id"]},
        )
        return Actor(org_id=str(payload["org_id"]), user_id=str(payload["user_id"]))
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="invalid token") from exc


class ProcessMiner:
    @staticmethod
    def discover_processes(db: Session, org_id: str) -> tuple[list[dict], dict[str, list[str]]]:
        rows = (
            db.query(Event)
            .filter(Event.organization_id == org_id)
            .order_by(Event.entity_id, Event.timestamp, Event.id)
            .all()
        )
        journeys: dict[str, list[Event]] = {}
        for row in rows:
            journeys.setdefault(row.entity_id, []).append(row)
        journeys = {k: v for k, v in journeys.items() if len(v) >= 2}
        if not journeys:
            return [], {}
        stats: dict[str, dict[str, Any]] = {}
        path_map: dict[str, list[str]] = {}
        for entity_id, events in journeys.items():
            event_types = [e.event_type for e in events]
            path_map[entity_id] = event_types
            key = " -> ".join(event_types)
            entry = stats.setdefault(
                key,
                {
                    "path": event_types,
                    "frequency": 0,
                    "durations": [],
                    "human_steps": [],
                    "anomalies": 0,
                },
            )
            entry["frequency"] += 1
            duration = max(
                0, (events[-1].timestamp - events[0].timestamp).total_seconds()
            )
            entry["durations"].append(duration)
            human = sum(1 for e in events if e.actor == "human")
            entry["human_steps"].append(human)
            if len(set(event_types)) < len(event_types):
                entry["anomalies"] += 1
        out = []
        for row in stats.values():
            out.append(
                {
                    "path": row["path"],
                    "frequency": row["frequency"],
                    "avg_duration_seconds": round(statistics.mean(row["durations"]), 2),
                    "avg_human_steps": round(statistics.mean(row["human_steps"]), 2),
                    "anomaly_rate": round(row["anomalies"] / row["frequency"], 4),
                }
            )
        out.sort(key=lambda x: x["frequency"], reverse=True)
        return out, path_map


class ROIEngine:
    @staticmethod
    def calculate(
        discovered: list[dict],
        execution_rows: list[AutomationExecution],
        req: ROIRequest,
    ) -> dict:
        if not discovered:
            raise ValueError("no mined process data")
        weighted_human_steps = sum(
            p["avg_human_steps"] * p["frequency"] for p in discovered
        )
        total_frequency = sum(p["frequency"] for p in discovered)
        avg_manual_steps = weighted_human_steps / max(total_frequency, 1)
        manual_minutes_per_month = (
            avg_manual_steps * req.avg_human_action_minutes * req.automation_runs_per_month
        )
        completed_rows = [r for r in execution_rows if r.status == "completed"]
        total_active_seconds = sum(max(0.0, r.active_seconds or 0.0) for r in completed_rows)
        completed_count = len(completed_rows)
        completed_approvals = sum(1 for r in completed_rows if r.approval_decision is not None)
        avg_active_seconds = total_active_seconds / completed_count if completed_count else 0.0
        approval_rate = completed_approvals / completed_count if completed_count else 0.0
        automation_minutes_per_run = avg_active_seconds / 60.0 + approval_rate * req.avg_human_action_minutes
        automation_minutes_per_month = (
            automation_minutes_per_run * req.automation_runs_per_month
        )
        manual_cost = manual_minutes_per_month / 60.0 * req.hourly_labor_cost
        automation_labor_cost = (
            automation_minutes_per_month / 60.0 * req.hourly_labor_cost
        )
        api_cost = req.api_cost_per_run * req.automation_runs_per_month
        net_monthly_savings = manual_cost - automation_labor_cost - api_cost
        if req.implementation_cost <= 0:
            roi_percentage = 0.0
        else:
            roi_percentage = net_monthly_savings / req.implementation_cost * 100
        payback = (
            req.implementation_cost / net_monthly_savings
            if net_monthly_savings > 0
            else None
        )
        return {
            "manual_minutes_per_month": round(manual_minutes_per_month, 2),
            "automation_minutes_per_month": round(automation_minutes_per_month, 2),
            "completed_execution_count": completed_count,
            "completed_approval_count": completed_approvals,
            "projected_approvals_per_month": round(
                approval_rate * req.automation_runs_per_month, 2
            ),
            "api_cost_per_month": round(api_cost, 2),
            "net_monthly_savings": round(net_monthly_savings, 2),
            "roi_percentage": round(roi_percentage, 2),
            "payback_period_months": round(payback, 2) if payback else None,
            "roi_note": (
                "Projection uses observed completed executions where available. "
                "With no completed executions, approval rate and active execution time are assumed to be zero, which is not evidence of zero effort. "
                "Use one currency consistently for all cost inputs."
            ),
        }


class AutomationDesigner:
    @staticmethod
    def _fallback_interpretation(observed: dict) -> str:
        return (
            f"Observed {observed['journey_count']} multi-event journeys. The selected path "
            f"occurred {observed['frequency']} times and averaged {observed['avg_human_steps']} "
            f"human steps. Manual work is concentrated enough to test a guarded routing automation."
        )

    @classmethod
    def design(cls, db: Session, org_id: str) -> ProcessDefinition:
        mined, paths = ProcessMiner.discover_processes(db, org_id)
        if not mined:
            raise ValueError("not enough events")
        best = max(mined, key=lambda p: p["frequency"] * p["avg_human_steps"])
        observed = {
            "journey_count": len(paths),
            "path": best["path"],
            "frequency": best["frequency"],
            "avg_duration_seconds": best["avg_duration_seconds"],
            "avg_human_steps": best["avg_human_steps"],
            "anomaly_rate": best["anomaly_rate"],
        }
        interpretation = cls._fallback_interpretation(observed)
        if OPENAI_API_KEY and OPENAI_CLIENT_FACTORY is not None:
            try:
                ai = OPENAI_CLIENT_FACTORY(api_key=OPENAI_API_KEY)
                response = ai.chat.completions.create(
                    model="gpt-4o",
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": "You explain process-mining facts. Never invent measurements.",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(observed),
                        },
                    ],
                )
                candidate = None
                if getattr(response, "choices", None):
                    candidate = getattr(response.choices[0].message, "content", None)
                if candidate:
                    interpretation = str(candidate)[:2000]
            except Exception:
                interpretation = cls._fallback_interpretation(observed)
        workflow = {
            "trigger": "lead_created",
            "steps": [
                {
                    "id": "s1",
                    "type": "condition",
                    "label": "Classify lead",
                    "field": "classification",
                    "operator": "==",
                    "value": "high_risk",
                    "goto_true": "s3",
                    "goto_false": "s2",
                },
                {
                    "id": "s2",
                    "type": "api_call",
                    "label": "Assign lead",
                    "action": "assign_lead",
                    "payload": {"owner": "bot_rep_1"},
                    "next_step": "s4",
                },
                {
                    "id": "s3",
                    "type": "human_approval",
                    "label": "Review risky lead",
                    "goto_approved": "s4",
                    "goto_rejected": "s5",
                },
                {
                    "id": "s4",
                    "type": "api_call",
                    "label": "Assign lead",
                    "action": "assign_lead",
                    "payload": {"owner": "bot_rep_1"},
                    "next_step": None,
                },
                {
                    "id": "s5",
                    "type": "api_call",
                    "label": "Record rejection notice",
                    "action": "send_notification",
                    "payload": {"template": "not_a_fit"},
                    "next_step": None,
                },
            ],
        }
        WorkflowInterpreter._validate_spec(workflow)
        score = min(100.0, max(0.0, best["avg_human_steps"] * 24.0 + best["frequency"] / 10))
        confidence = min(0.95, 0.5 + best["frequency"] / 500)
        row = ProcessDefinition(
            id=str(uuid.uuid4()),
            organization_id=org_id,
            name="Lead Routing Automation",
            description="A fixed, validated lead-routing workflow proposed from observed manual volume.",
            automation_score=round(score, 2),
            confidence=round(confidence, 2),
            status="proposed",
            observed_facts=observed,
            ai_interpretation=interpretation,
            workflow_spec=workflow,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row


class WorkflowInterpreter:
    @staticmethod
    def _validate_spec(spec: dict) -> WorkflowSpec:
        parsed = WorkflowSpec.model_validate(spec)
        if not parsed.steps:
            raise ValueError("workflow has no steps")
        ids = [step.id for step in parsed.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step id")
        known = set(ids)
        edges: dict[str, list[str]] = {sid: [] for sid in ids}
        for step in parsed.steps:
            targets: list[str] = []
            if isinstance(step, ConditionStep):
                targets = [step.goto_true, step.goto_false]
            elif isinstance(step, ApprovalStep):
                targets = [step.goto_approved, step.goto_rejected]
            elif isinstance(step, ActionStep) and step.next_step:
                targets = [step.next_step]
            for target in targets:
                if target not in known:
                    raise ValueError(f"unknown target: {target}")
            edges[step.id] = targets
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str):
            if node in visiting:
                raise ValueError("workflow graph contains a cycle")
            if node in visited:
                return
            visiting.add(node)
            for target in edges[node]:
                visit(target)
            visiting.remove(node)
            visited.add(node)

        visit(ids[0])
        if visited != known:
            raise ValueError("workflow contains unreachable steps")
        return parsed

    @classmethod
    def execute(
        cls,
        db: Session,
        definition: ProcessDefinition,
        entity_id: str,
        context: dict,
    ) -> AutomationExecution:
        spec = cls._validate_spec(definition.workflow_spec)
        if definition.status != "deployed":
            raise ValueError("automation is not deployed")
        now = datetime.now(timezone.utc)
        execution = AutomationExecution(
            id=str(uuid.uuid4()),
            organization_id=definition.organization_id,
            automation_id=definition.id,
            entity_id=entity_id,
            status="running",
            context=context,
            current_step_id=spec.steps[0].id,
            started_at=now,
            active_seconds=0.0,
        )
        db.add(execution)
        db.commit()
        return cls._run(db, definition, execution, decision=None)

    @classmethod
    def resume(
        cls,
        db: Session,
        definition: ProcessDefinition,
        execution: AutomationExecution,
        approved: bool,
        approver_id: str,
    ) -> AutomationExecution:
        if not execution.awaiting_approval or not execution.approval_step_id:
            raise ValueError("execution is not awaiting approval")
        if execution.current_step_id != execution.approval_step_id:
            raise ValueError("approval pointer is inconsistent")
        execution.approver_id = approver_id
        execution.approval_decision = approved
        db.commit()
        return cls._run(db, definition, execution, decision=approved)

    @classmethod
    def _run(
        cls,
        db: Session,
        definition: ProcessDefinition,
        execution: AutomationExecution,
        decision: bool | None,
    ) -> AutomationExecution:
        spec = cls._validate_spec(definition.workflow_spec)
        by_id = {s.id: s for s in spec.steps}
        sequence = (
            db.query(StepExecution)
            .filter(StepExecution.execution_id == execution.id)
            .count()
        )
        started = time.perf_counter()
        approval_to_finish = execution.approval_step_id if decision is not None else None
        try:
            while execution.current_step_id:
                step = by_id.get(execution.current_step_id)
                if step is None:
                    raise ValueError("current step is invalid")
                if approval_to_finish and step.id == approval_to_finish:
                    next_id = step.goto_approved if decision else step.goto_rejected
                    sequence += 1
                    history = StepExecution(
                        execution_id=execution.id,
                        organization_id=execution.organization_id,
                        sequence=sequence,
                        step_id=step.id,
                        step_type=step.type,
                        status="completed",
                        started_at=datetime.now(timezone.utc),
                        ended_at=datetime.now(timezone.utc),
                        input_data={"approved": decision},
                        output_data={"approved": decision, "approver_id": execution.approver_id},
                    )
                    db.add(history)
                    execution.awaiting_approval = False
                    execution.approval_step_id = None
                    execution.current_step_id = next_id
                    db.commit()
                    approval_to_finish = None
                    decision = None
                    continue
                if isinstance(step, ApprovalStep):
                    execution.status = "awaiting_approval"
                    execution.awaiting_approval = True
                    execution.approval_step_id = step.id
                    execution.current_step_id = step.id
                    db.commit()
                    return execution
                sequence += 1
                history = StepExecution(
                    execution_id=execution.id,
                    organization_id=execution.organization_id,
                    sequence=sequence,
                    step_id=step.id,
                    step_type=step.type,
                    status="running",
                    started_at=datetime.now(timezone.utc),
                    input_data=execution.context,
                    output_data={},
                )
                db.add(history)
                db.flush()
                try:
                    if isinstance(step, ConditionStep):
                        actual = str(execution.context.get(step.field, ""))
                        next_id = step.goto_true if actual == step.value else step.goto_false
                        output = {"actual": actual, "matched": actual == step.value}
                    else:
                        output = {"action_result": cls._action(db, step.action, step.payload, execution)}
                        next_id = step.next_step
                    history.status = "completed"
                    history.ended_at = datetime.now(timezone.utc)
                    history.output_data = output
                    execution.current_step_id = next_id
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    failed = StepExecution(
                        execution_id=execution.id,
                        organization_id=execution.organization_id,
                        sequence=sequence,
                        step_id=step.id,
                        step_type=step.type,
                        status="failed",
                        started_at=datetime.now(timezone.utc),
                        ended_at=datetime.now(timezone.utc),
                        input_data=execution.context,
                        output_data={},
                        error_message=str(exc)[:1000],
                    )
                    db.add(failed)
                    execution.status = "failed"
                    execution.current_step_id = step.id
                    execution.error_message = str(exc)[:1000]
                    execution.end_time = datetime.now(timezone.utc)
                    db.commit()
                    raise
            execution.status = "completed"
            execution.end_time = datetime.now(timezone.utc)
            execution.current_step_id = None
            db.commit()
            return execution
        finally:
            execution.active_seconds = (execution.active_seconds or 0.0) + max(
                0.0, time.perf_counter() - started
            )
            db.commit()

    @staticmethod
    def _action(db: Session, action: str, payload: dict, execution: AutomationExecution) -> dict:
        if action == "assign_lead":
            email = str(execution.context.get("email", "lead@example.com"))
            owner = str(payload.get("owner", "bot_rep_1"))
            lead = (
                db.query(Lead)
                .filter(
                    Lead.organization_id == execution.organization_id,
                    Lead.entity_id == execution.entity_id,
                )
                .one_or_none()
            )
            if lead is None:
                lead = Lead(
                    organization_id=execution.organization_id,
                    entity_id=execution.entity_id,
                    email=email,
                    assigned_to=owner,
                    last_assignment_execution_id=execution.id,
                    updated_at=datetime.now(timezone.utc),
                )
                db.add(lead)
            else:
                lead.email = email
                lead.assigned_to = owner
                lead.last_assignment_execution_id = execution.id
                lead.updated_at = datetime.now(timezone.utc)
            return {"status": "assigned", "owner": owner, "entity_id": execution.entity_id}
        if action == "send_notification":
            return {
                "status": "simulated",
                "template": str(payload.get("template", "default")),
                "entity_id": execution.entity_id,
            }
        raise ValueError("unsupported action")


app = FastAPI(title="Atlas", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/auth/login")
def login():
    if APP_ENV != "development":
        raise HTTPException(status_code=403, detail="demo login disabled")
    return {"access_token": issue_demo_token(), "token_type": "bearer"}


@app.post("/api/setup")
def setup_demo_data(db: Session = Depends(get_db)):
    if APP_ENV != "development":
        raise HTTPException(status_code=403, detail="demo setup disabled")
    # Setup is deliberately destructive for the demo organization so repeated setup
    # produces a known, reproducible fixture rather than accumulating stale runs.
    execution_ids = [
        row[0]
        for row in db.query(AutomationExecution.id)
        .filter(AutomationExecution.organization_id == DEMO_ORG_ID)
        .all()
    ]
    if execution_ids:
        db.query(StepExecution).filter(StepExecution.execution_id.in_(execution_ids)).delete(
            synchronize_session=False
        )
    db.query(AutomationExecution).filter(
        AutomationExecution.organization_id == DEMO_ORG_ID
    ).delete(synchronize_session=False)
    db.query(Lead).filter(Lead.organization_id == DEMO_ORG_ID).delete(
        synchronize_session=False
    )
    db.query(ProcessDefinition).filter(
        ProcessDefinition.organization_id == DEMO_ORG_ID
    ).delete(synchronize_session=False)
    db.query(Event).filter(Event.organization_id == DEMO_ORG_ID).delete(
        synchronize_session=False
    )
    db.commit()
    now = datetime.now(timezone.utc)
    for i in range(100):
        entity = f"lead-{i}"
        base = now + timedelta(seconds=i * 20)
        for offset, event_type, actor in [
            (0, "lead_created", "system"),
            (2, "human_review", "human"),
            (12, "human_assign", "human"),
        ]:
            db.add(
                Event(
                    organization_id=DEMO_ORG_ID,
                    entity_id=entity,
                    event_type=event_type,
                    timestamp=base + timedelta(seconds=offset),
                    actor=actor,
                    metadata_json={"source": "demo"},
                )
            )
    db.commit()
    return {"organization_id": DEMO_ORG_ID, "events": 300, "journeys": 100}


@app.post("/api/ai/analyze")
def analyze(actor: Actor = Depends(current_actor), db: Session = Depends(get_db)):
    try:
        row = AutomationDesigner.design(db, actor.org_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return serialize_definition(row)


@app.get("/api/opportunities")
def opportunities(actor: Actor = Depends(current_actor), db: Session = Depends(get_db)):
    rows = (
        db.query(ProcessDefinition)
        .filter(ProcessDefinition.organization_id == actor.org_id)
        .order_by(ProcessDefinition.created_at.desc())
        .all()
    )
    return [serialize_definition(r) for r in rows]


@app.post("/api/automations/{automation_id}/deploy")
def deploy(
    automation_id: str,
    actor: Actor = Depends(current_actor),
    db: Session = Depends(get_db),
):
    row = (
        db.query(ProcessDefinition)
        .filter(
            ProcessDefinition.id == automation_id,
            ProcessDefinition.organization_id == actor.org_id,
        )
        .one_or_none()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="automation not found")
    WorkflowInterpreter._validate_spec(row.workflow_spec)
    row.status = "deployed"
    db.commit()
    return serialize_definition(row)


@app.post("/api/automations/{automation_id}/execute")
def execute(
    automation_id: str,
    req: ExecuteRequest,
    actor: Actor = Depends(current_actor),
    db: Session = Depends(get_db),
):
    row = (
        db.query(ProcessDefinition)
        .filter(
            ProcessDefinition.id == automation_id,
            ProcessDefinition.organization_id == actor.org_id,
        )
        .one_or_none()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="automation not found")
    classification = "high_risk" if "spam" in req.email.lower() else "low_risk"
    context = {"entity_id": req.entity_id, "email": req.email, "classification": classification}
    try:
        execution = WorkflowInterpreter.execute(db, row, req.entity_id, context)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"execution failed: {exc}") from exc
    return serialize_execution(execution)


@app.post("/api/executions/{execution_id}/approve")
def approve(
    execution_id: str,
    approved: bool,
    actor: Actor = Depends(current_actor),
    db: Session = Depends(get_db),
):
    execution = (
        db.query(AutomationExecution)
        .filter(
            AutomationExecution.id == execution_id,
            AutomationExecution.organization_id == actor.org_id,
        )
        .one_or_none()
    )
    if execution is None:
        raise HTTPException(status_code=404, detail="execution not found")
    definition = (
        db.query(ProcessDefinition)
        .filter(
            ProcessDefinition.id == execution.automation_id,
            ProcessDefinition.organization_id == actor.org_id,
        )
        .one_or_none()
    )
    if definition is None:
        raise HTTPException(status_code=404, detail="automation not found")
    try:
        out = WorkflowInterpreter.resume(db, definition, execution, approved, actor.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"execution failed: {exc}") from exc
    return serialize_execution(out)


@app.get("/api/executions/{execution_id}")
def execution_history(
    execution_id: str,
    actor: Actor = Depends(current_actor),
    db: Session = Depends(get_db),
):
    execution = (
        db.query(AutomationExecution)
        .filter(
            AutomationExecution.id == execution_id,
            AutomationExecution.organization_id == actor.org_id,
        )
        .one_or_none()
    )
    if execution is None:
        raise HTTPException(status_code=404, detail="execution not found")
    steps = (
        db.query(StepExecution)
        .filter(
            StepExecution.execution_id == execution.id,
            StepExecution.organization_id == actor.org_id,
        )
        .order_by(StepExecution.sequence)
        .all()
    )
    return {
        "execution": serialize_execution(execution),
        "steps": [
            {
                "sequence": s.sequence,
                "step_id": s.step_id,
                "type": s.step_type,
                "status": s.status,
                "input_data": s.input_data,
                "output_data": s.output_data,
                "error_message": s.error_message,
            }
            for s in steps
        ],
    }


@app.post("/api/automations/{automation_id}/roi")
def roi(
    automation_id: str,
    req: ROIRequest,
    actor: Actor = Depends(current_actor),
    db: Session = Depends(get_db),
):
    definition = (
        db.query(ProcessDefinition)
        .filter(
            ProcessDefinition.id == automation_id,
            ProcessDefinition.organization_id == actor.org_id,
        )
        .one_or_none()
    )
    if definition is None:
        raise HTTPException(status_code=404, detail="automation not found")
    mined, _ = ProcessMiner.discover_processes(db, actor.org_id)
    rows = (
        db.query(AutomationExecution)
        .filter(
            AutomationExecution.organization_id == actor.org_id,
            AutomationExecution.automation_id == automation_id,
        )
        .all()
    )
    return ROIEngine.calculate(mined, rows, req)


def serialize_definition(row: ProcessDefinition) -> dict:
    return {
        "id": row.id,
        "organization_id": row.organization_id,
        "name": row.name,
        "description": row.description,
        "automation_score": row.automation_score,
        "confidence": row.confidence,
        "status": row.status,
        "observed_facts": row.observed_facts,
        "ai_interpretation": row.ai_interpretation,
        "workflow_spec": row.workflow_spec,
        "created_at": row.created_at.isoformat(),
    }


def serialize_execution(row: AutomationExecution) -> dict:
    return {
        "id": row.id,
        "automation_id": row.automation_id,
        "entity_id": row.entity_id,
        "status": row.status,
        "current_step_id": row.current_step_id,
        "awaiting_approval": row.awaiting_approval,
        "approval_step_id": row.approval_step_id,
        "approver_id": row.approver_id,
        "approval_decision": row.approval_decision,
        "active_seconds": round(row.active_seconds or 0.0, 6),
        "started_at": row.started_at.isoformat(),
        "end_time": row.end_time.isoformat() if row.end_time else None,
        "error_message": row.error_message,
    }


HTML_CONTENT = """<!doctype html>
<html>
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atlas</title><script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-900">
<div class="max-w-5xl mx-auto p-6 space-y-4">
<h1 class="text-3xl font-bold">Atlas</h1>
<p>Observed process facts, guarded automation, and persisted execution history.</p>
<div id="app" class="space-y-4"></div>
</div>
<script>
let token = null, opp = null, exec = null;
const app = document.getElementById('app');
function escapeHTML(value) {
  return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}
async function request(path, options = {}) {
  const r = await fetch(path, options);
  let payload = {};
  try { payload = await r.json(); } catch (_) {}
  if (!r.ok) {
    app.innerHTML = '<div class="bg-red-50 border border-red-200 p-4 rounded"><b>Request failed (' + r.status + ')</b><pre class="overflow-auto">' + escapeHTML(JSON.stringify(payload, null, 2)) + '</pre></div>';
    return null;
  }
  return payload;
}
function headers() { return {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}; }
async function login() {
  const data = await request('/api/auth/login', {method: 'POST'});
  if (!data) return;
  token = data.access_token;
  app.innerHTML = '<button class="bg-slate-900 text-white px-4 py-2 rounded" onclick="setup()">Setup Demo Data</button><p class="text-sm text-slate-600">Development only. Setup resets previous demo-organization runs.</p>';
}
async function setup() {
  if (!await request('/api/setup', {method: 'POST'})) return;
  opp = await request('/api/ai/analyze', {method: 'POST', headers: headers()});
  if (!opp) return;
  await render();
}
async function deploy() {
  if (!await request('/api/automations/' + opp.id + '/deploy', {method: 'POST', headers: headers()})) return;
  opp.status = 'deployed';
  await render();
}
async function run(high) {
  const data = await request('/api/automations/' + opp.id + '/execute', {
    method: 'POST', headers: headers(),
    body: JSON.stringify({entity_id: 'demo_' + Date.now(), email: high ? 'spam@test.com' : 'good@test.com'})
  });
  if (!data) return;
  exec = data;
  await render();
}
async function decide(approved) {
  const data = await request('/api/executions/' + exec.id + '/approve?approved=' + approved, {method: 'POST', headers: headers()});
  if (!data) return;
  exec = data;
  await render();
}
async function render() {
  if (!opp) return;
  const history = exec ? await request('/api/executions/' + exec.id, {headers: headers()}) : null;
  if (exec && !history) return;
  app.innerHTML = '<div class="bg-white p-6 rounded shadow space-y-4">' +
    '<div><b>Status:</b> ' + escapeHTML(opp.status) + '</div>' +
    '<div><b>Observed:</b><pre>' + escapeHTML(JSON.stringify(opp.observed_facts, null, 2)) + '</pre></div>' +
    '<div><b>Interpretation:</b><p>' + escapeHTML(opp.ai_interpretation) + '</p></div>' +
    '<pre class="overflow-auto">' + escapeHTML(JSON.stringify(opp.workflow_spec, null, 2)) + '</pre>' +
    (opp.status === 'proposed'
      ? '<button class="bg-green-600 text-white px-4 py-2 rounded" onclick="deploy()">Deploy</button>'
      : '<div class="space-x-2"><button class="bg-indigo-600 text-white px-4 py-2 rounded" onclick="run(false)">Low-risk</button><button class="bg-red-600 text-white px-4 py-2 rounded" onclick="run(true)">High-risk</button></div>') +
    (exec ? '<div><b>Execution:</b> ' + escapeHTML(exec.status) + '</div>' +
      (exec.status === 'awaiting_approval'
        ? '<button class="bg-green-600 text-white px-3 py-2 rounded" onclick="decide(true)">Approve</button> <button class="bg-red-600 text-white px-3 py-2 rounded" onclick="decide(false)">Reject</button>'
        : '') + '<pre class="overflow-auto">' + escapeHTML(JSON.stringify(history, null, 2)) + '</pre>' : '') +
    '</div>';
}
login();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_CONTENT


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
