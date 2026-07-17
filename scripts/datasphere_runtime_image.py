#!/usr/bin/env python3
"""Validate immutable runtime-image identities accepted by DataSphere Jobs."""
from __future__ import annotations

import re


PROJECT_DOCKER_IMAGE_RE = re.compile(r"^b[a-z0-9]{19}$")
OCI_DIGEST_IMAGE_RE = re.compile(
    r"^(?=.{1,512}$)"
    r"[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[1-9][0-9]{0,4})?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
    r"@sha256:[0-9a-f]{64}$"
)


def is_project_docker_image(value: str) -> bool:
    return PROJECT_DOCKER_IMAGE_RE.fullmatch(value) is not None


def is_oci_digest_image(value: str) -> bool:
    return OCI_DIGEST_IMAGE_RE.fullmatch(value) is not None


def require_runtime_image(value: str, *, registry: bool | None = None) -> str:
    valid = is_oci_digest_image(value) if registry else is_project_docker_image(value)
    if registry is None:
        valid = is_project_docker_image(value) or is_oci_digest_image(value)
    if not valid:
        if registry:
            raise ValueError(
                "registry Docker image must be a lowercase OCI reference pinned by "
                "@sha256:<64 lowercase hex characters>"
            )
        if registry is False:
            raise ValueError(
                "DataSphere project Docker resource ID must match b[a-z0-9]{19}"
            )
        raise ValueError("runtime image is neither a project resource nor an OCI digest")
    return value
