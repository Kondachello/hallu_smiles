"""Immutable file-backed prompt registry with strict rendering and schema validation."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, StrictUndefined
from jsonschema import Draft202012Validator
from langchain_core.messages import HumanMessage, SystemMessage

from .errors import PromptContractError


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    prompt_id: str
    messages: tuple[SystemMessage | HumanMessage, ...]
    output_schema: Mapping[str, Any]
    manifest_sha256: str


class PromptRegistry:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else self._default_root()
        self.root = self.root.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise PromptContractError(f"prompt manifest not found: {manifest_path}")
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PromptContractError("prompt manifest is invalid JSON") from exc
        self.entries = {entry["prompt_id"]: entry for entry in self.manifest.get("entries", [])}
        if not self.entries or len(self.entries) != len(self.manifest.get("entries", [])):
            raise PromptContractError("prompt IDs must be present and unique")
        self.manifest_sha256 = self._hash_tree()
        self.environment = Environment(undefined=StrictUndefined, autoescape=False)
        self._validate_assets()

    @staticmethod
    def _default_root() -> Path:
        source_checkout = Path(__file__).resolve().parents[2] / "prompts" / "v1"
        package_target = Path(__file__).resolve().parents[2] / "share" / "hallugraph-dynamic-typing-agent" / "prompts" / "v1"
        environment_target = Path(sys.prefix) / "share" / "hallugraph-dynamic-typing-agent" / "prompts" / "v1"
        for candidate in (source_checkout, package_target, environment_target):
            if candidate.is_dir():
                return candidate
        return source_checkout

    def _safe_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise PromptContractError(f"invalid prompt asset path: {relative}")
        return path

    def _hash_tree(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in self.root.rglob("*") if item.is_file()):
            digest.update(path.relative_to(self.root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def _validate_assets(self) -> None:
        for prompt_id, entry in self.entries.items():
            for key in ("system", "user", "schema"):
                if key not in entry:
                    raise PromptContractError(f"{prompt_id}: missing {key}")
                self._safe_path(str(entry[key]))
            schema = json.loads(self._safe_path(entry["schema"]).read_text(encoding="utf-8"))
            try:
                Draft202012Validator.check_schema(schema)
            except Exception as exc:
                raise PromptContractError(f"{prompt_id}: invalid JSON Schema") from exc

    def render(self, prompt_id: str, variables: Mapping[str, Any]) -> RenderedPrompt:
        entry = self.entries.get(prompt_id)
        if entry is None:
            raise PromptContractError(f"unknown prompt_id: {prompt_id}")
        expected = set(entry["required_variables"])
        actual = set(variables)
        if actual != expected:
            raise PromptContractError(f"{prompt_id}: variables {sorted(actual)} != {sorted(expected)}")
        system = self._safe_path(entry["system"]).read_text(encoding="utf-8")
        user_template = self._safe_path(entry["user"]).read_text(encoding="utf-8")
        try:
            user = self.environment.from_string(user_template).render(**variables)
        except Exception as exc:
            raise PromptContractError(f"{prompt_id}: rendering failed") from exc
        schema = json.loads(self._safe_path(entry["schema"]).read_text(encoding="utf-8"))
        return RenderedPrompt(
            prompt_id=prompt_id,
            messages=(SystemMessage(content=system), HumanMessage(content=user)),
            output_schema=schema,
            manifest_sha256=self.manifest_sha256,
        )
