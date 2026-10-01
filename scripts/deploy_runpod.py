"""Deploy an immutable worker image and verify it, without raising idle costs."""

import json
import os
import re
import sys
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen

API_BASE = "https://rest.runpod.io/v1"
USER_AGENT = "ruh-deployment/1.0 (RunPod API client)"
# All requests use the fixed RunPod HTTPS hosts below; endpoint IDs are validated.
# ruff: noqa: S310


def api(method, path, token, data=None):
    request = Request(
        API_BASE + path,
        data=json.dumps(data).encode() if data is not None else None,
        method=method,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        # Do not log request headers, template configuration, or credential values.
        raise RuntimeError(f"RunPod {method} {path} returned HTTP {exc.code}") from None


def new_template(existing, image, sha):
    # These documented fields preserve existing storage, launch and registry settings.
    fields = (
        "containerDiskInGb",
        "containerRegistryAuthId",
        "volumeInGb",
        "volumeMountPath",
        "ports",
        "dockerEntrypoint",
        "dockerStartCmd",
    )
    template = {field: existing[field] for field in fields if field in existing}
    template.update(
        name=f"ruh-{sha[:12]}-{uuid.uuid4().hex[:8]}",
        imageName=image,
        isServerless=True,
    )
    # startJupyter/startSsh are legacy response-only fields, rejected by REST POST.
    # Preserve existing environment values without putting them in job logs.
    if isinstance(existing.get("env"), dict):
        template["env"] = existing["env"]
    elif isinstance(existing.get("config"), dict):
        env = existing["config"].get("env")
        if env is not None:
            template["env"] = env
    return template


def wait_for_job(endpoint, token, sha, deadline_seconds=600):
    url = f"https://api.runpod.ai/v2/{endpoint}"
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    payload = {
        "input": {
            "messages": [{"role": "user", "content": "knowledge"}],
            "max_tokens": 8,
            "temperature": 0,
        }
    }
    job_id = None
    deadline = time.monotonic() + deadline_seconds
    try:
        while time.monotonic() < deadline:
            if job_id is None:
                with urlopen(
                    Request(
                        url + "/run", data=json.dumps(payload).encode(), headers=headers
                    ),
                    timeout=30,
                ) as response:
                    job_id = json.load(response)["id"]
            with urlopen(
                Request(url + "/status/" + job_id, headers=headers), timeout=30
            ) as response:
                status = json.load(response)
            if status.get("status") == "COMPLETED":
                result = status.get("output") or {}
                completed_id, job_id = job_id, None
                if result.get("build_sha") == sha:
                    if result.get("error") or not result.get("choices"):
                        raise RuntimeError("The deployed worker rejected its smoke job")
                    return completed_id
                # Existing workers can finish during a template transition.
                # Retry within the same deadline; never accept their old image.
            elif status.get("status") in ("FAILED", "CANCELLED", "TIMED_OUT"):
                job_id = None
                raise RuntimeError(
                    "RunPod deployment smoke job failed: " + status["status"]
                )
            time.sleep(5)
        raise TimeoutError(
            "RunPod worker revision was not verified before the deadline"
        )
    finally:
        # An interrupted or failed deployment must not leave queued smoke work.
        if job_id is not None:
            try:
                with urlopen(
                    Request(url + "/cancel/" + job_id, data=b"{}", headers=headers),
                    timeout=30,
                ):
                    pass
            except Exception:
                print(
                    "Warning: the pending smoke job could not be cancelled",
                    file=sys.stderr,
                )


def deploy(endpoint_id, image, sha, token, call=api, smoke=wait_for_job):
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("Expected a full Git commit SHA")
    if not re.fullmatch(r"[a-z0-9]+", endpoint_id):
        raise ValueError("Invalid endpoint ID")
    if image != f"ghcr.io/codewithjuber/ruh-serverless:{sha}":
        raise ValueError("Expected the immutable image tag for this commit")
    endpoint = call("GET", "/endpoints/" + endpoint_id, token)
    previous = endpoint["templateId"]
    original = call("GET", "/templates/" + previous, token)
    template = call("POST", "/templates", token, new_template(original, image, sha))
    update = {
        "templateId": template["id"],
        "workersMin": 0,
        "workersMax": endpoint["workersMax"],
    }
    # Changing the template revision replaces worker image caches without a
    # destructive max-workers-to-zero step. Other endpoint settings are untouched.
    call("PATCH", "/endpoints/" + endpoint_id, token, update)
    try:
        job_id = smoke(endpoint_id, token, sha)
    except Exception:
        call(
            "PATCH",
            "/endpoints/" + endpoint_id,
            token,
            {
                "templateId": previous,
                "workersMin": 0,
                "workersMax": endpoint["workersMax"],
            },
        )
        raise
    return {
        "endpoint": endpoint_id,
        "template": template["id"],
        "image": image,
        "smoke_job": job_id,
        "workersMin": 0,
    }


if __name__ == "__main__":
    key = os.getenv("RUNPOD_API_KEY") or os.getenv("RUNPOD_API_TOKEN")
    if not key:
        raise SystemExit(
            "Set RUNPOD_API_KEY or RUNPOD_API_TOKEN in GitHub Actions secrets"
        )
    revision = os.environ["DEPLOY_SHA"]
    print(
        json.dumps(
            deploy(
                os.getenv("RUNPOD_ENDPOINT_ID", "6o38qhvti4knkk"),
                f"ghcr.io/codewithjuber/ruh-serverless:{revision}",
                revision,
                key,
            )
        )
    )
