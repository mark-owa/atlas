import os
import tempfile

import pytest
import jwt
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient

# Set these before importing Atlas: import creates its database tables.
_test_directory = tempfile.TemporaryDirectory(prefix="atlas-tests-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(
    _test_directory.name, "atlas.db"
)
os.environ["APP_ENV"] = "development"
os.environ["OPENAI_API_KEY"] = ""
os.environ["JWT_SECRET"] = "atlas_test_secret_not_for_deployment_32_chars"

import atlas

client = TestClient(atlas.app)


@pytest.fixture(autouse=True)
def fresh_db():
    atlas.Base.metadata.drop_all(bind=atlas.engine)
    atlas.Base.metadata.create_all(bind=atlas.engine)
    yield


def token():
    return client.post("/api/auth/login").json()["access_token"]


def headers():
    return {"Authorization": f"Bearer {token()}"}


def setup_demo():
    r = client.post("/api/setup")
    assert r.status_code == 200


def test_auth():
    assert client.get("/api/opportunities").status_code == 401
    assert (
        client.get(
            "/api/opportunities", headers={"Authorization": "Bearer garbage"}
        ).status_code
        == 401
    )
    expired = jwt.encode(
        {
            "org_id": atlas.DEMO_ORG_ID,
            "user_id": atlas.DEMO_USER_ID,
            "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
        },
        atlas.JWT_SECRET,
        algorithm=atlas.JWT_ALGORITHM,
    )
    assert (
        client.get(
            "/api/opportunities", headers={"Authorization": f"Bearer {expired}"}
        ).status_code
        == 401
    )
    no_expiry = jwt.encode(
        {"org_id": atlas.DEMO_ORG_ID, "user_id": atlas.DEMO_USER_ID},
        atlas.JWT_SECRET,
        algorithm=atlas.JWT_ALGORITHM,
    )
    assert (
        client.get(
            "/api/opportunities", headers={"Authorization": f"Bearer {no_expiry}"}
        ).status_code
        == 401
    )
    r = client.post("/api/auth/login", params={"org_id": "rogue"})
    payload = jwt.decode(
        r.json()["access_token"], atlas.JWT_SECRET, algorithms=[atlas.JWT_ALGORITHM]
    )
    assert payload["org_id"] == atlas.DEMO_ORG_ID
    assert payload["user_id"] == atlas.DEMO_USER_ID


def test_graph_validation():
    from atlas import WorkflowInterpreter

    bad = {
        "trigger": "t",
        "steps": [
            {
                "id": "s1",
                "type": "condition",
                "label": "x",
                "field": "a",
                "operator": "==",
                "value": "b",
                "goto_true": "missing",
                "goto_false": "missing",
            }
        ],
    }
    with pytest.raises(ValueError):
        WorkflowInterpreter._validate_spec(bad)
    loop = {
        "trigger": "t",
        "steps": [
            {
                "id": "s1",
                "type": "condition",
                "label": "x",
                "field": "a",
                "operator": "==",
                "value": "b",
                "goto_true": "s1",
                "goto_false": "s1",
            }
        ],
    }
    with pytest.raises(ValueError):
        WorkflowInterpreter._validate_spec(loop)
    dup = {
        "trigger": "t",
        "steps": [
            {
                "id": "s1",
                "type": "api_call",
                "label": "a",
                "action": "send_notification",
                "payload": {},
                "next_step": None,
            },
            {
                "id": "s1",
                "type": "api_call",
                "label": "b",
                "action": "send_notification",
                "payload": {},
                "next_step": None,
            },
        ],
    }
    with pytest.raises(ValueError):
        WorkflowInterpreter._validate_spec(dup)
    unreachable = {
        "trigger": "t",
        "steps": [
            {
                "id": "s1",
                "type": "api_call",
                "label": "a",
                "action": "send_notification",
                "payload": {},
                "next_step": None,
            },
            {
                "id": "s2",
                "type": "api_call",
                "label": "b",
                "action": "send_notification",
                "payload": {},
                "next_step": None,
            },
        ],
    }
    with pytest.raises(ValueError):
        WorkflowInterpreter._validate_spec(unreachable)


def test_malicious_specs():
    from atlas import WorkflowInterpreter

    for typ in ["python", "shell", "exec", "arbitrary"]:
        spec = {"trigger": "t", "steps": [{"id": "1", "type": typ, "code": "bad"}]}
        with pytest.raises(ValueError):
            WorkflowInterpreter._validate_spec(spec)


def test_low_risk_real_side_effect_and_history():
    setup_demo()
    h = headers()
    opp = client.post("/api/ai/analyze", headers=h).json()
    client.post(f"/api/automations/{opp['id']}/deploy", headers=h)
    ex = client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "low1", "email": "good@test.com"},
    ).json()
    assert ex["status"] == "completed"
    db = atlas.SessionLocal()
    lead = (
        db.query(atlas.Lead)
        .filter_by(organization_id=atlas.DEMO_ORG_ID, entity_id="low1")
        .one()
    )
    assert lead.assigned_to == "bot_rep_1"
    assert lead.last_assignment_execution_id == ex["id"]
    db.close()
    hist = client.get(f"/api/executions/{ex['id']}", headers=h).json()
    assert [s["step_id"] for s in hist["steps"]] == ["s1", "s2", "s4"]
    assert hist["execution"]["end_time"] is not None


def test_approval_identity_and_rejection_branch():
    setup_demo()
    h = headers()
    opp = client.post("/api/ai/analyze", headers=h).json()
    client.post(f"/api/automations/{opp['id']}/deploy", headers=h)
    ex = client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "high1", "email": "spam@test.com"},
    ).json()
    assert ex["status"] == "awaiting_approval"
    out = client.post(
        f"/api/executions/{ex['id']}/approve?approved=true&approver_id=attacker",
        headers=h,
    ).json()
    assert out["status"] == "completed"
    assert out["approver_id"] == atlas.DEMO_USER_ID
    db = atlas.SessionLocal()
    assert db.query(atlas.Lead).filter_by(entity_id="high1").count() == 1
    db.close()
    ex2 = client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "rej1", "email": "spam@test.com"},
    ).json()
    out2 = client.post(
        f"/api/executions/{ex2['id']}/approve?approved=false", headers=h
    ).json()
    assert out2["status"] == "completed"
    hist = client.get(f"/api/executions/{ex2['id']}", headers=h).json()
    assert [s["step_id"] for s in hist["steps"]] == ["s1", "s3", "s5"]
    assert hist["steps"][-1]["output_data"]["action_result"]["status"] == "simulated"
    db = atlas.SessionLocal()
    assert db.query(atlas.Lead).filter_by(entity_id="rej1").count() == 0
    db.close()


def test_repeated_assignment_keeps_one_lead():
    setup_demo()
    h = headers()
    opp = client.post("/api/ai/analyze", headers=h).json()
    client.post(f"/api/automations/{opp['id']}/deploy", headers=h)
    client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "same", "email": "good@test.com"},
    ).json()
    e2 = client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "same", "email": "good@test.com"},
    ).json()
    db = atlas.SessionLocal()
    leads = (
        db.query(atlas.Lead)
        .filter_by(organization_id=atlas.DEMO_ORG_ID, entity_id="same")
        .all()
    )
    assert len(leads) == 1
    assert leads[0].last_assignment_execution_id == e2["id"]
    db.close()


def test_cross_tenant():
    setup_demo()
    h = headers()
    opp = client.post("/api/ai/analyze", headers=h).json()
    client.post(f"/api/automations/{opp['id']}/deploy", headers=h)
    ex = client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "x", "email": "spam@test.com"},
    ).json()
    org_b = jwt.encode(
        {
            "org_id": "rogue",
            "user_id": "rogue",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
        },
        atlas.JWT_SECRET,
        algorithm=atlas.JWT_ALGORITHM,
    )
    hb = {"Authorization": f"Bearer {org_b}"}
    assert client.get("/api/opportunities", headers=hb).json() == []
    assert (
        client.post(f"/api/automations/{opp['id']}/deploy", headers=hb).status_code
        == 404
    )
    assert (
        client.post(
            f"/api/automations/{opp['id']}/execute",
            headers=hb,
            json={"entity_id": "rogue-attempt", "email": "x@test.com"},
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/api/executions/{ex['id']}/approve?approved=true", headers=hb
        ).status_code
        == 404
    )
    assert client.get(f"/api/executions/{ex['id']}", headers=hb).status_code == 404
    roi = {
        "hourly_labor_cost": 50,
        "automation_runs_per_month": 100,
        "implementation_cost": 1000,
        "api_cost_per_run": 0.02,
        "avg_human_action_minutes": 5,
    }
    assert (
        client.post(f"/api/automations/{opp['id']}/roi", headers=hb, json=roi).status_code
        == 404
    )


def test_roi_and_failure_rollback(monkeypatch):
    setup_demo()
    h = headers()
    opp = client.post("/api/ai/analyze", headers=h).json()
    client.post(f"/api/automations/{opp['id']}/deploy", headers=h)
    good = client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "roi", "email": "good@test.com"},
    ).json()
    client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "roi2", "email": "spam@test.com"},
    ).json()
    roi = client.post(
        f"/api/automations/{opp['id']}/roi",
        headers=h,
        json={
            "hourly_labor_cost": 50,
            "automation_runs_per_month": 100,
            "implementation_cost": 1000,
            "api_cost_per_run": 0.02,
            "avg_human_action_minutes": 5,
        },
    ).json()
    assert roi["completed_execution_count"] == 1
    assert roi["completed_approval_count"] == 0
    assert roi["automation_minutes_per_month"] > 0
    assert roi["net_monthly_savings"] > 0
    assert roi["roi_percentage"] > 0
    assert roi["payback_period_months"] is not None
    assert roi["projected_approvals_per_month"] == 0
    zero = client.post(
        f"/api/automations/{opp['id']}/roi",
        headers=h,
        json={
            "hourly_labor_cost": 0,
            "automation_runs_per_month": 10,
            "implementation_cost": 0,
            "api_cost_per_run": 10,
            "avg_human_action_minutes": 5,
        },
    ).json()
    assert zero["roi_percentage"] == 0
    assert zero["payback_period_months"] is None
    assert zero["roi_note"]

    def fail_action(self, db, action, payload, execution):
        raise RuntimeError("connector down")

    monkeypatch.setattr(atlas.WorkflowInterpreter, "_action", fail_action)
    failed = client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "failme", "email": "good@test.com"},
    )
    assert failed.status_code == 502
    db = atlas.SessionLocal()
    ex = (
        db.query(atlas.AutomationExecution)
        .filter_by(entity_id="failme")
        .order_by(atlas.AutomationExecution.id.desc())
        .first()
    )
    assert ex.status == "failed"
    assert ex.current_step_id == "s2"
    assert db.query(atlas.Lead).filter_by(entity_id="failme").count() == 0
    steps = (
        db.query(atlas.StepExecution)
        .filter_by(execution_id=ex.id)
        .order_by(atlas.StepExecution.sequence)
        .all()
    )
    assert [s.step_id for s in steps] == ["s1", "s2"]
    assert steps[-1].status == "failed"
    db.close()


def test_ai_failure_and_null_content_fallback(monkeypatch):
    setup_demo()
    h = headers()
    monkeypatch.setattr(atlas, "OPENAI_API_KEY", "fake")

    class BoomCompletions:
        def create(self, **kwargs):
            raise RuntimeError("down")

    class BoomChat:
        completions = BoomCompletions()

    class BoomClient:
        chat = BoomChat()

    monkeypatch.setattr(atlas, "OPENAI_CLIENT_FACTORY", lambda **kwargs: BoomClient())
    r = client.post("/api/ai/analyze", headers=h)
    assert r.status_code == 200
    assert "concentrated" in r.json()["ai_interpretation"]

    class NullMessage:
        content = None

    class NullChoice:
        message = NullMessage()

    class NullResponse:
        choices = [NullChoice()]

    class NullCompletions:
        def create(self, **kwargs):
            return NullResponse()

    class NullChat:
        completions = NullCompletions()

    class NullClient:
        chat = NullChat()

    monkeypatch.setattr(atlas, "OPENAI_CLIENT_FACTORY", lambda **kwargs: NullClient())
    r = client.post("/api/ai/analyze", headers=h)
    assert r.status_code == 200
    assert "concentrated" in r.json()["ai_interpretation"]

    class EmptyChoicesResponse:
        choices = []

    class EmptyCompletions:
        def create(self, **kwargs):
            return EmptyChoicesResponse()

    class EmptyChat:
        completions = EmptyCompletions()

    class EmptyClient:
        chat = EmptyChat()

    monkeypatch.setattr(atlas, "OPENAI_CLIENT_FACTORY", lambda **kwargs: EmptyClient())
    r = client.post("/api/ai/analyze", headers=h)
    assert r.status_code == 200
    assert "concentrated" in r.json()["ai_interpretation"]


def test_assistant_output_does_not_control_graph(monkeypatch):
    setup_demo()
    h = headers()
    monkeypatch.setattr(atlas, "OPENAI_API_KEY", "fake")

    class Message:
        content = '{"steps":[{"type":"python","code":"os.system(\"bad\")"}]}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Completions:
        def create(self, **kwargs):
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    monkeypatch.setattr(atlas, "OPENAI_CLIENT_FACTORY", lambda **kwargs: Client())
    out = client.post("/api/ai/analyze", headers=h).json()
    assert "python" not in {s["type"] for s in out["workflow_spec"]["steps"]}
    assert out["ai_interpretation"].startswith("{")


def test_bad_action_fails_validation():
    from atlas import WorkflowInterpreter

    spec = {
        "trigger": "x",
        "steps": [
            {
                "id": "1",
                "type": "api_call",
                "label": "x",
                "action": "delete_everything",
                "payload": {},
                "next_step": None,
            }
        ],
    }
    with pytest.raises(ValueError):
        WorkflowInterpreter._validate_spec(spec)


def test_path_tampering_is_rejected():
    setup_demo()
    h = headers()
    opp = client.post("/api/ai/analyze", headers=h).json()
    client.post(f"/api/automations/{opp['id']}/deploy", headers=h)
    ex = client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "tamper", "email": "spam@test.com"},
    ).json()
    assert ex["status"] == "awaiting_approval"
    db = atlas.SessionLocal()
    row = db.get(atlas.AutomationExecution, ex["id"])
    row.current_step_id = "s1"
    db.commit()
    db.close()
    r = client.post(f"/api/executions/{ex['id']}/approve?approved=true", headers=h)
    assert r.status_code == 409


def test_repeat_approval_is_rejected():
    setup_demo()
    h = headers()
    opp = client.post("/api/ai/analyze", headers=h).json()
    client.post(f"/api/automations/{opp['id']}/deploy", headers=h)
    ex = client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "repeat", "email": "spam@test.com"},
    ).json()
    r1 = client.post(f"/api/executions/{ex['id']}/approve?approved=true", headers=h)
    assert r1.status_code == 200
    r2 = client.post(f"/api/executions/{ex['id']}/approve?approved=false", headers=h)
    assert r2.status_code == 409


def test_setup_repeated_is_idempotent_at_demo_scale():
    assert client.post("/api/setup").status_code == 200
    assert client.post("/api/setup").status_code == 200
    db = atlas.SessionLocal()
    assert (
        db.query(atlas.Event).filter_by(organization_id=atlas.DEMO_ORG_ID).count()
        == 300
    )
    assert (
        db.query(atlas.Lead).filter_by(organization_id=atlas.DEMO_ORG_ID).count()
        == 0
    )
    db.close()


def test_setup_resets_existing_demo_data():
    setup_demo()
    h = headers()
    opp = client.post("/api/ai/analyze", headers=h).json()
    client.post(f"/api/automations/{opp['id']}/deploy", headers=h)
    client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "to-reset", "email": "good@test.com"},
    )
    db = atlas.SessionLocal()
    assert db.query(atlas.Lead).filter_by(entity_id="to-reset").count() == 1
    assert db.query(atlas.AutomationExecution).count() == 1
    db.close()
    assert client.post("/api/setup").status_code == 200
    db = atlas.SessionLocal()
    assert (
        db.query(atlas.Event).filter_by(organization_id=atlas.DEMO_ORG_ID).count()
        == 300
    )
    assert db.query(atlas.Lead).filter_by(organization_id=atlas.DEMO_ORG_ID).count() == 0
    assert (
        db.query(atlas.AutomationExecution)
        .filter_by(organization_id=atlas.DEMO_ORG_ID)
        .count()
        == 0
    )
    db.close()


def test_roi_validation_rejects_negative_inputs():
    setup_demo()
    h = headers()
    opp = client.post("/api/ai/analyze", headers=h).json()
    client.post(f"/api/automations/{opp['id']}/deploy", headers=h)
    r = client.post(
        f"/api/automations/{opp['id']}/roi",
        headers=h,
        json={
            "hourly_labor_cost": -1,
            "automation_runs_per_month": 10,
            "implementation_cost": 0,
            "api_cost_per_run": 0,
            "avg_human_action_minutes": 1,
        },
    )
    assert r.status_code == 422


def test_roi_with_no_executions_does_not_assume_approvals():
    setup_demo()
    h = headers()
    opp = client.post("/api/ai/analyze", headers=h).json()
    client.post(f"/api/automations/{opp['id']}/deploy", headers=h)
    roi = client.post(
        f"/api/automations/{opp['id']}/roi",
        headers=h,
        json={
            "hourly_labor_cost": 50,
            "automation_runs_per_month": 100,
            "implementation_cost": 1000,
            "api_cost_per_run": 0.02,
            "avg_human_action_minutes": 5,
        },
    ).json()
    assert roi["completed_execution_count"] == 0
    assert roi["completed_approval_count"] == 0
    assert roi["projected_approvals_per_month"] == 0
    assert roi["roi_note"]


def test_miner_uses_stable_id_tiebreaker_for_equal_timestamps():
    setup_demo()
    db = atlas.SessionLocal()
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    db.add_all(
        [
            atlas.Event(
                organization_id=atlas.DEMO_ORG_ID,
                entity_id="tie",
                event_type="b",
                timestamp=ts,
                actor="system",
            ),
            atlas.Event(
                organization_id=atlas.DEMO_ORG_ID,
                entity_id="tie",
                event_type="a",
                timestamp=ts,
                actor="system",
            ),
        ]
    )
    db.commit()
    inserted = (
        db.query(atlas.Event)
        .filter_by(organization_id=atlas.DEMO_ORG_ID, entity_id="tie")
        .order_by(atlas.Event.id)
        .all()
    )
    expected = [r.event_type for r in inserted]
    journeys = atlas.ProcessMiner.discover_processes(db, atlas.DEMO_ORG_ID)[1]
    assert journeys["tie"] == expected
    db.close()


def test_classifier_is_case_insensitive():
    setup_demo()
    h = headers()
    opp = client.post("/api/ai/analyze", headers=h).json()
    client.post(f"/api/automations/{opp['id']}/deploy", headers=h)
    ex = client.post(
        f"/api/automations/{opp['id']}/execute",
        headers=h,
        json={"entity_id": "case", "email": "SPAM@TEST.COM"},
    ).json()
    assert ex["status"] == "awaiting_approval"


def test_non_development_startup_requires_custom_secret():
    import subprocess
    import sys
    import pathlib

    env = os.environ.copy()
    env["APP_ENV"] = "production"
    env["JWT_SECRET"] = "dev_secret_change_in_production_32_chars_min"
    env["DATABASE_URL"] = "sqlite:///:memory:"
    project_dir = str(pathlib.Path(__file__).resolve().parent)
    proc = subprocess.run(
        [sys.executable, "-c", "import atlas"],
        cwd=project_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "JWT_SECRET" in proc.stderr


def test_non_development_startup_requires_minimum_secret_length():
    import pathlib
    import subprocess
    import sys

    env = os.environ.copy()
    env["APP_ENV"] = "production"
    env["JWT_SECRET"] = "short-secret"
    env["DATABASE_URL"] = "sqlite:///:memory:"
    project_dir = str(pathlib.Path(__file__).resolve().parent)
    proc = subprocess.run(
        [sys.executable, "-c", "import atlas"],
        cwd=project_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "32 bytes" in proc.stderr


def test_setup_disabled_outside_development():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(atlas, "APP_ENV", "production")
    try:
        assert client.post("/api/setup").status_code == 403
    finally:
        monkeypatch.undo()


def test_login_disabled_outside_development():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(atlas, "APP_ENV", "production")
    try:
        assert client.post("/api/auth/login").status_code == 403
    finally:
        monkeypatch.undo()


def test_cors_default_is_localhost():
    assert "*" not in atlas.CORS_ORIGINS
    assert "http://localhost:8000" in atlas.CORS_ORIGINS


def test_migrations_not_claimed():
    assert atlas.DATABASE_URL
