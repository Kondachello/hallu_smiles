#!/usr/bin/env python3
"""Resolve a public registry commit tag to an immutable OCI digest reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    from datasphere_runtime_image import require_runtime_image
except ImportError:  # pragma: no cover - package import in unit tests
    from scripts.datasphere_runtime_image import require_runtime_image


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_RE = re.compile(
    r"^(?P<registry>[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[1-9][0-9]{0,4})?)"
    r"/(?P<path>[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*)$"
)
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def _anonymous_token(registry: str, path: str, timeout: float) -> str:
    if registry != "ghcr.io":
        raise ValueError(
            "automatic anonymous resolution currently supports ghcr.io only; "
            "pass an explicit --docker-image digest for another registry"
        )
    query = urllib.parse.urlencode(
        {"service": "ghcr.io", "scope": f"repository:{path}:pull"}
    )
    request = urllib.request.Request(
        f"https://ghcr.io/token?{query}",
        headers={"User-Agent": "hallu-smiles-datasphere-runtime-resolver/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    token = payload.get("token")
    if not isinstance(token, str) or not token:
        raise ValueError("registry did not issue an anonymous pull token")
    return token


def resolve(repository: str, commit: str, *, timeout: float = 30.0) -> str:
    match = REPOSITORY_RE.fullmatch(repository)
    if not match:
        raise ValueError("repository must be a lowercase registry/path reference without a tag")
    if not COMMIT_RE.fullmatch(commit):
        raise ValueError("commit must be a full lowercase 40-character Git SHA")
    registry = match.group("registry")
    path = match.group("path")
    token = _anonymous_token(registry, path, timeout)
    request = urllib.request.Request(
        f"https://{registry}/v2/{path}/manifests/{commit}",
        headers={
            "Accept": ACCEPT,
            "Authorization": f"Bearer {token}",
            "User-Agent": "hallu-smiles-datasphere-runtime-resolver/1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        digest = response.headers.get("Docker-Content-Digest", "")
    if not SHA256_RE.fullmatch(digest):
        raise ValueError("registry manifest response has no valid Docker-Content-Digest")
    calculated = "sha256:" + hashlib.sha256(body).hexdigest()
    if calculated != digest:
        raise ValueError(
            f"registry manifest digest mismatch: header={digest}, calculated={calculated}"
        )
    identity = f"{repository}@{digest}"
    return require_runtime_image(identity, registry=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="ghcr.io/kondachello/hallu-smiles-datasphere")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="Wait for a remote build to publish the commit tag before failing.",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    deadline = time.monotonic() + max(0, args.wait_seconds)
    last_error: Exception | None = None
    while True:
        try:
            print(resolve(args.repository, args.commit, timeout=args.timeout))
            return
        except (ValueError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(max(1, args.poll_seconds))
    raise SystemExit(
        f"immutable runtime image for {args.commit} is not available in "
        f"{args.repository}: {last_error}"
    )


if __name__ == "__main__":
    main()
