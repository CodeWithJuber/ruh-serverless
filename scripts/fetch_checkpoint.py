"""Fetch only the immutable, digest-checked epoch-19 artifact. Never log credentials."""
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler, urlopen
from urllib.error import HTTPError


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def verify(directory, manifest):
    for name, expected in manifest["files"].items():
        path = Path(directory) / name
        if path.stat().st_size != expected["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Checkpoint integrity check failed: {name}")


def fetch(directory="checkpoint"):
    manifest = json.loads(Path("checkpoint-manifest.json").read_text())
    target = Path(directory)
    target.mkdir(exist_ok=True, parents=True)
    headers = {}
    if os.getenv("KAGGLE_API_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["KAGGLE_API_TOKEN"]
    for name, expected in manifest["files"].items():
        filename = quote(manifest["directory"] + "/" + name, safe="")
        url = f'https://www.kaggle.com/api/v1/datasets/download/{manifest["dataset"]}/{filename}?datasetVersionNumber={manifest["version"]}'
        try:
            response = build_opener(NoRedirect).open(Request(url, headers=headers), timeout=60)
        except HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308):
                raise
            location = exc.headers["Location"]
            if urlparse(location).scheme != "https":
                raise ValueError("Checkpoint redirect must use HTTPS")
            # Authentication must never follow the download redirect to another host.
            response = urlopen(Request(location), timeout=60)
        partial = target / (name + ".partial")
        try:
            count = 0
            digest = hashlib.sha256()
            with response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > expected["bytes"]:
                        raise ValueError("Checkpoint exceeds expected size")
                    output.write(chunk)
                    digest.update(chunk)
            if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Checkpoint integrity check failed: {name}")
            partial.replace(target / name)
        finally:
            partial.unlink(missing_ok=True)
    verify(target, manifest)
    print("Checkpoint verified (epoch_19, Bayan v1)")


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        verify("checkpoint", json.loads(Path("checkpoint-manifest.json").read_text()))
    else:
        fetch()
