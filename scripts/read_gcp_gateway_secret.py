#!/usr/bin/env python3
"""Read a named Secret Manager secret to stdout for command substitution only."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--secret", required=True)
    args = parser.parse_args()
    try:
        from google.cloud import secretmanager
    except ImportError as exc:  # pragma: no cover - image-only dependency
        raise SystemExit("google-cloud-secret-manager is required in the GCP runner image") from exc
    name = f"projects/{args.project}/secrets/{args.secret}/versions/latest"
    value = secretmanager.SecretManagerServiceClient().access_secret_version(
        request={"name": name}
    ).payload.data.decode("utf-8")
    if not value:
        raise SystemExit("gateway bearer secret is empty")
    print(value, end="")


if __name__ == "__main__":
    main()
