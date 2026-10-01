"""Deployment tests cover real state transitions, including rollback."""

import importlib.util
import io
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "deploy_runpod", Path(__file__).parents[1] / "scripts" / "deploy_runpod.py"
)
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)
SHA = "a" * 40
IMAGE = f"ghcr.io/codewithjuber/ruh-serverless:{SHA}"


def test_success_preserves_worker_limit_and_has_zero_minimum():
    mutations = []

    def call(method, path, token, data=None):
        if method == "GET" and path.startswith("/endpoints/"):
            return {"templateId": "old", "workersMax": 3, "workersMin": 0}
        if method == "GET":
            return {
                "containerDiskInGb": 10,
                "imageName": "old:latest",
                "config": {"env": {"SETTING": "preserved"}},
            }
        mutations.append((method, path, data))
        return {"id": "new"}

    result = deployment.deploy(
        "endpoint", IMAGE, SHA, "test-key", call, lambda *_: "verified-job"
    )
    assert result["smoke_job"] == "verified-job"
    assert mutations[0][2]["env"] == {"SETTING": "preserved"}
    assert mutations[1][2] == {"templateId": "new", "workersMin": 0, "workersMax": 3}


def test_failed_smoke_restores_original_template():
    patches = []

    def call(method, path, token, data=None):
        if method == "GET" and path.startswith("/endpoints/"):
            return {"templateId": "old", "workersMax": 3}
        if method == "GET":
            return {"containerDiskInGb": 10}
        if method == "PATCH":
            patches.append(data)
        return {"id": "new"}

    def fail(*_):
        raise RuntimeError("stale worker")

    with pytest.raises(RuntimeError, match="stale worker"):
        deployment.deploy("endpoint", IMAGE, SHA, "test-key", call, fail)
    assert patches[-1] == {"templateId": "old", "workersMin": 0, "workersMax": 3}


def test_mutable_tag_is_rejected_before_api_access():
    with pytest.raises(ValueError, match="immutable"):
        deployment.deploy(
            "endpoint",
            "ghcr.io/codewithjuber/ruh-serverless:latest",
            SHA,
            "test-key",
            lambda *_: pytest.fail("must not call API"),
        )


def test_smoke_waits_for_new_revision_after_stale_worker(monkeypatch):
    responses = iter(
        [
            {"id": "old-job"},
            {"status": "COMPLETED", "output": {"build_sha": "old", "choices": [{}]}},
            {"id": "new-job"},
            {"status": "COMPLETED", "output": {"build_sha": SHA, "choices": [{}]}},
        ]
    )
    monkeypatch.setattr(
        deployment, "urlopen", lambda *a, **kw: io.StringIO(json.dumps(next(responses)))
    )
    monkeypatch.setattr(deployment.time, "sleep", lambda *_: None)
    assert deployment.wait_for_job("endpoint", "test-key", SHA) == "new-job"


def test_smoke_cancels_pending_job_when_polling_fails(monkeypatch):
    paths = []

    def request(req, **kwargs):
        paths.append(req.full_url)
        if "/status/" in req.full_url:
            raise OSError("network unavailable")
        return io.StringIO(json.dumps({"id": "pending"}))

    monkeypatch.setattr(deployment, "urlopen", request)
    with pytest.raises(OSError, match="network unavailable"):
        deployment.wait_for_job("endpoint", "test-key", SHA)
    assert paths[-1].endswith("/cancel/pending")


def test_template_uses_only_writable_rest_fields_and_preserves_launch_settings():
    existing = {
        "startJupyter": True,
        "startSsh": True,
        "dockerEntrypoint": ["python"],
        "dockerStartCmd": ["handler.py"],
        "env": {"KEEP": "value"},
        "containerDiskInGb": 10,
    }
    first = deployment.new_template(existing, IMAGE, SHA)
    second = deployment.new_template(existing, IMAGE, SHA)
    assert "startJupyter" not in first and "startSsh" not in first
    assert first["dockerEntrypoint"] == ["python"]
    assert first["dockerStartCmd"] == ["handler.py"]
    assert first["env"] == existing["env"]
    assert first["name"] != second["name"]
