"""Stable vocabulary shared by prompts, schemas, graph routing and adapters."""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"


class NliVerdict(StrEnum):
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"


class EvidenceLevel(StrEnum):
    SOURCE_ENTAILED = "source_entailed"
    EXAMPLE_SUPPORTED = "example_supported"
    DEFINITION_ONLY = "definition_only"
    UNKNOWN = "unknown"


class DecisionAction(StrEnum):
    ASSIGN_EXISTING = "assign_existing"
    ADD_CHILD = "add_child"
    ADD_PARENT = "add_parent"
    MULTI_ASSIGN = "multi_assign"
    ALIAS_MERGE = "alias_merge"
    CREATE_BRANCH = "create_branch"
    ROLE_ASSIGN = "role_assign"
    UNKNOWN = "unknown"

