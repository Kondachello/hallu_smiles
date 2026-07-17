"""Strict and legacy DSPy adapters for explicitly configured local runtimes.

The research runtime uses native OpenAI ``response_format.type=json_schema``
and deliberately bypasses DSPy's JSON repair and fallback requests.  The old
vLLM 0.6.3 ``guided_json`` adapter remains only for reproducing legacy
artifacts; it is not selected by the new DataSphere profile.  Both adapters
reuse DSPy's typed output schema without changing any KGGen prompt, graph
extraction, or clustering logic.

Imports deliberately stay inside the factory so ordinary offline tests and
remote-provider users do not acquire a DSPy/vLLM dependency.
"""
from __future__ import annotations

import json
import re
import warnings
from functools import wraps
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


STRUCTURED_OUTPUT_PROTOCOL_VERSION = "strict-response-format-v2"
STRUCTURED_OUTPUT_TRANSPORTS = frozenset({"none", "response_format", "guided_json"})
STRUCTURED_OUTPUT_BACKENDS = frozenset({"xgrammar", "guidance"})


class StructuredOutputError(RuntimeError):
    """Base class for deterministic structured-output contract failures."""


class StructuredOutputSchemaError(StructuredOutputError):
    """The requested or returned document does not satisfy its JSON Schema."""


class StructuredOutputParseError(StructuredOutputError):
    """The server returned something other than one strict JSON document."""


def _response_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def validate_completion_envelope(response: Any, *, label: str = "DSPy completion") -> None:
    """Require one complete choice before DSPy discards transport metadata."""
    choices = _response_field(response, "choices")
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        raise StructuredOutputParseError(f"{label} must contain exactly one choice")
    finish_reason = _response_field(choices[0], "finish_reason")
    if finish_reason != "stop":
        raise StructuredOutputParseError(
            f"{label} did not finish cleanly: {finish_reason!r}"
        )


def install_dspy_completion_guard(lm: Any) -> None:
    """Install a version-tolerant finish-reason guard on a DSPy LM instance.

    DSPy 2.6 processes LiteLLM choices into plain strings before an Adapter
    sees them.  Wrapping ``forward`` is the last stable boundary that still
    contains ``choices`` and ``finish_reason``.  The marker makes repeated
    KGExtractor initialisation idempotent.
    """
    if getattr(lm, "_hallu_completion_guard_installed", False):
        return
    original_forward = getattr(lm, "forward", None)
    if not callable(original_forward):
        raise RuntimeError("DSPy LM has no callable forward method to guard")

    @wraps(original_forward)
    def guarded_forward(*args: Any, **kwargs: Any) -> Any:
        response = original_forward(*args, **kwargs)
        validate_completion_envelope(response)
        return response

    lm.forward = guarded_forward
    lm._hallu_completion_guard_installed = True


def strict_json_loads(document: str, *, label: str = "structured output") -> Any:
    """Decode one RFC-8259-style JSON document without permissive extensions.

    Python's standard decoder normally accepts duplicate object keys and the
    non-standard numeric constants ``NaN``/``Infinity``.  Both are ambiguous
    inputs for a scientific structured-output contract: duplicate keys are
    silently resolved by keeping the last value, while non-finite numbers are
    not JSON.  Reject them before independent JSON-Schema validation.
    """
    if not isinstance(document, str):
        raise StructuredOutputParseError(f"{label} is not text")

    def reject_constant(value: str) -> Any:
        raise StructuredOutputParseError(
            f"{label} contains non-JSON numeric constant {value!r}"
        )

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise StructuredOutputParseError(
                    f"{label} contains duplicate object key {key!r}"
                )
            decoded[key] = value
        return decoded

    try:
        return json.loads(
            document,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except StructuredOutputParseError:
        raise
    except json.JSONDecodeError as exc:
        raise StructuredOutputParseError(
            f"{label} is invalid JSON at character {exc.pos}"
        ) from exc


@dataclass(frozen=True)
class StructuredOutputSettings:
    """Validated local structured-output settings derived from ``cfg.llm``."""

    transport: str = "none"
    backend: str = "xgrammar"
    model_revision: str | None = None
    runtime_fingerprint: str | None = None


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def structured_output_settings(llm_config: Any) -> StructuredOutputSettings:
    """Read and validate the explicit structured-output runtime contract.

    ``vllm_guided_json: true`` is retained only so old cached/offline configs
    fail gracefully while Jobs migrate.  It maps to the legacy transport and
    emits a deprecation warning; it never changes ``llm.model``.
    """
    configured_transport = _config_value(llm_config, "structured_output_transport", None)
    legacy_guided = bool(_config_value(llm_config, "vllm_guided_json", False))
    if configured_transport is None:
        transport = "guided_json" if legacy_guided else "none"
        if legacy_guided:
            warnings.warn(
                "llm.vllm_guided_json is deprecated; set "
                "llm.structured_output_transport explicitly",
                DeprecationWarning,
                stacklevel=2,
            )
    else:
        transport = str(configured_transport).strip().lower()
        if legacy_guided and transport != "guided_json":
            raise ValueError(
                "llm.vllm_guided_json conflicts with llm.structured_output_transport"
            )
    if transport not in STRUCTURED_OUTPUT_TRANSPORTS:
        choices = ", ".join(sorted(STRUCTURED_OUTPUT_TRANSPORTS))
        raise ValueError(f"llm.structured_output_transport must be one of: {choices}")

    backend = str(_config_value(llm_config, "structured_output_backend", "xgrammar")).strip().lower()
    if backend not in STRUCTURED_OUTPUT_BACKENDS:
        choices = ", ".join(sorted(STRUCTURED_OUTPUT_BACKENDS))
        raise ValueError(f"llm.structured_output_backend must be one of: {choices}")

    model_revision = _config_value(llm_config, "model_revision", None)
    runtime_fingerprint = _config_value(llm_config, "runtime_fingerprint", None)
    if transport == "response_format" and (not model_revision or not runtime_fingerprint):
        raise ValueError(
            "native response_format requires exact llm.model_revision and "
            "llm.runtime_fingerprint"
        )
    return StructuredOutputSettings(
        transport=transport,
        backend=backend,
        model_revision=str(model_revision) if model_revision else None,
        runtime_fingerprint=str(runtime_fingerprint) if runtime_fingerprint else None,
    )


def json_schema_response_format(
    schema: Mapping[str, Any], *, name: str = "dspy_program_outputs"
) -> dict[str, Any]:
    """Build the native OpenAI/vLLM ``response_format`` wire object."""
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_") or "structured_output"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": safe_name[:64],
            "schema": deepcopy(dict(schema)),
            "strict": True,
        },
    }


def validate_json_document(instance: Any, schema: Mapping[str, Any]) -> None:
    """Independently validate a decoded response against the exact schema."""
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError, ValidationError
    except Exception as exc:  # pragma: no cover - live dependency check catches this
        raise RuntimeError("jsonschema is required for strict structured outputs") from exc
    try:
        Draft202012Validator.check_schema(dict(schema))
        Draft202012Validator(dict(schema)).validate(instance)
    except SchemaError as exc:
        raise StructuredOutputSchemaError(f"invalid structured-output schema: {exc.message}") from exc
    except ValidationError as exc:
        location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        raise StructuredOutputSchemaError(
            f"structured output violates schema at {location}: {exc.message}"
        ) from exc


def dspy_output_schema(signature: Any) -> dict[str, Any]:
    """Return DSPy's exact closed output schema, without legacy rewriting."""
    from dspy.adapters.json_adapter import _get_structured_outputs_response_format

    output_model = _get_structured_outputs_response_format(signature)
    schema = output_model.model_json_schema()
    if not isinstance(schema, Mapping):
        raise StructuredOutputSchemaError("DSPy output model did not produce a JSON Schema object")
    copied = deepcopy(dict(schema))
    # Validate at request construction time so a bad dynamic Literal schema
    # never reaches a paid GPU request.
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(copied)
    except Exception as exc:
        raise StructuredOutputSchemaError("DSPy generated an invalid JSON Schema") from exc
    return copied


def inline_local_json_schema_refs(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Expand local ``#/$defs/...`` references in a Pydantic JSON Schema.

    DSPy serialises ``list[Relation]`` as a perfectly valid Pydantic schema
    whose array item is ``{"$ref": "#/$defs/Relation"}``.  The particular
    vLLM 0.6.3 structured-output backends available in DataSphere accept a
    small schema and start a request for this one, but do not honour that
    local reference: they can emit a bare relation object instead of the
    required ``{"relations": [...]}`` root.  Expanding a local definition at
    its reference site preserves the accepted JSON instances; it merely gives
    the constrained decoder an equivalent schema that does not rely on
    JSON-Pointer resolution.

    Only references into the root ``$defs`` are expanded.  Foreign and cyclic
    references are left intact, together with ``$defs``, rather than risking a
    silent semantic change.  Pydantic's KGGen relation schemas are acyclic and
    use exactly this local-reference form.
    """
    root = deepcopy(dict(schema))
    definitions = root.get("$defs")
    if definitions is None:
        return root
    if not isinstance(definitions, Mapping):
        raise TypeError("JSON Schema $defs must be an object")

    def resolve_local_ref(reference: str) -> Any | None:
        prefix = "#/$defs/"
        if not reference.startswith(prefix):
            return None
        current: Any = definitions
        # JSON Pointer escaping is required for correctness even though
        # Pydantic's current Relation name has no escaped characters.
        for component in reference[len(prefix):].split("/"):
            component = component.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, Mapping) or component not in current:
                raise ValueError(f"unresolvable local JSON Schema reference: {reference}")
            current = current[component]
        return current

    def expand(node: Any, active_refs: frozenset[str] = frozenset()) -> Any:
        if isinstance(node, list):
            return [expand(item, active_refs) for item in node]
        if not isinstance(node, Mapping):
            return deepcopy(node)

        reference = node.get("$ref")
        if isinstance(reference, str):
            target = resolve_local_ref(reference)
            if target is not None:
                # Recursive models are not part of KGGen's output language.
                # Retain their original reference rather than expanding it
                # infinitely or weakening the original schema.
                if reference in active_refs:
                    return {key: expand(value, active_refs) for key, value in node.items()}
                expanded_target = expand(target, active_refs | {reference})
                siblings = {
                    key: expand(value, active_refs)
                    for key, value in node.items()
                    if key != "$ref"
                }
                if not siblings:
                    return expanded_target
                # JSON Schema permits sibling constraints next to ``$ref``.
                # ``allOf`` has the same intersection semantics without
                # overwriting either the target or those constraints.
                return {"allOf": [expanded_target, siblings]}

        return {key: expand(value, active_refs) for key, value in node.items()}

    expanded = expand(root)

    def contains_local_def_ref(node: Any) -> bool:
        if isinstance(node, Mapping):
            if isinstance(node.get("$ref"), str) and node["$ref"].startswith("#/$defs/"):
                return True
            return any(contains_local_def_ref(value) for value in node.values())
        if isinstance(node, list):
            return any(contains_local_def_ref(value) for value in node)
        return False

    # Do not remove definitions if a recursive reference could not be safely
    # inlined.  For KGGen's acyclic Relation schema this removes ``$defs`` and
    # produces the exact flat grammar required by vLLM 0.6.3.
    if not contains_local_def_ref(expanded):
        expanded.pop("$defs", None)
    return expanded


_NON_SEMANTIC_SCHEMA_ANNOTATIONS = frozenset({
    "title",
    "description",
    "examples",
    "default",
    "$comment",
    # ``desc`` is DSPy's legacy field metadata, not a JSON Schema keyword.
    "desc",
})
_SCHEMA_MAP_KEYWORDS = frozenset({"$defs", "patternProperties", "dependentSchemas"})


def canonicalize_vllm_guided_json_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the constraint-bearing JSON Schema accepted by vLLM 0.6.3.

    In addition to local ``$defs`` expansion, remove annotations and DSPy's
    private ``__dspy_*`` metadata.  These keywords do not restrict which JSON
    documents validate; they are prompt/UI metadata.  Keeping them in the
    grammar sent to the old Outlines backend makes it silently ignore the
    enclosing object for KGGen's dynamic fallback Relation schema.  Removing
    them therefore preserves exactly the same JSON language while avoiding
    that backend defect.
    """
    expanded = inline_local_json_schema_refs(schema)

    def clean(node: Any, *, property_map: bool = False) -> Any:
        if isinstance(node, list):
            return [clean(value) for value in node]
        if not isinstance(node, Mapping):
            return deepcopy(node)
        if property_map:
            # A user is allowed to name a JSON property ``title`` or ``desc``;
            # these are property names here, not schema annotations.
            return {str(key): clean(value) for key, value in node.items()}

        result: dict[str, Any] = {}
        for key, value in node.items():
            if key in _NON_SEMANTIC_SCHEMA_ANNOTATIONS or key.startswith("__dspy_"):
                continue
            if key == "properties":
                if not isinstance(value, Mapping):
                    raise TypeError("JSON Schema properties must be an object")
                result[key] = clean(value, property_map=True)
            elif key in _SCHEMA_MAP_KEYWORDS:
                if not isinstance(value, Mapping):
                    raise TypeError(f"JSON Schema {key} must be an object")
                result[key] = clean(value, property_map=True)
            else:
                result[key] = clean(value)
        return result

    return clean(expanded)


def strict_json_schema_adapter() -> Any:
    """Return a DSPy JSON adapter with one strict native-schema request.

    DSPy 2.6's stock :class:`JSONAdapter` catches *any* structured-output
    exception and silently sends a second ``json_object`` request.  Its parser
    also uses ``json_repair``.  Both behaviours are useful for casual chat,
    but invalidate an extraction experiment because a transport failure can
    be hidden or a truncated response can be changed locally.  This adapter:

    * always sends the exact DSPy-generated closed schema through native
      ``response_format``;
    * performs exactly one LM call;
    * parses with ``json.loads`` only; and
    * independently validates the raw document before DSPy type conversion.
    """
    from dspy.adapters.base import Adapter
    from dspy.adapters.json_adapter import JSONAdapter
    from dspy.adapters.utils import parse_value

    class StrictJSONSchemaAdapter(JSONAdapter):
        @staticmethod
        def _request_kwargs(
            lm_kwargs: Mapping[str, Any], signature: Any
        ) -> dict[str, Any]:
            """Return an isolated request containing only the native schema control.

            Calling the DSPy base ``Adapter`` directly below is intentional. In the
            pinned DSPy 2.6.27 implementation, ``JSONAdapter.__call__`` catches
            every structured-output failure and issues a second ``json_object``
            request.  ``ChatAdapter.__call__`` also owns a fallback.  The base
            adapter owns only format -> one LM call -> parse, which is the
            research contract we need and is stable across DSPy 2.6 and 3.x.
            """
            schema = dspy_output_schema(signature)
            request_kwargs = dict(lm_kwargs)
            previous_extra = request_kwargs.get("extra_body")
            if previous_extra is not None:
                if not isinstance(previous_extra, Mapping):
                    raise TypeError("DSPy lm_kwargs.extra_body must be a mapping")
                cleaned_extra = dict(previous_extra)
                cleaned_extra.pop("guided_json", None)
                if cleaned_extra:
                    request_kwargs["extra_body"] = cleaned_extra
                else:
                    request_kwargs.pop("extra_body", None)
            request_kwargs["response_format"] = json_schema_response_format(
                schema,
                name=getattr(signature, "__name__", "dspy_program_outputs"),
            )
            return request_kwargs

        def parse(self, signature: Any, completion: str) -> dict[str, Any]:
            fields = strict_json_loads(completion, label="DSPy completion")
            if not isinstance(fields, dict):
                raise StructuredOutputParseError("structured output root must be an object")

            expected = set(signature.output_fields)
            actual = set(fields)
            if actual != expected:
                raise StructuredOutputSchemaError(
                    "structured output keys differ from the DSPy signature: "
                    f"expected={sorted(expected)!r} actual={sorted(actual)!r}"
                )
            schema = dspy_output_schema(signature)
            validate_json_document(fields, schema)

            converted: dict[str, Any] = {}
            for key, value in fields.items():
                try:
                    converted[key] = parse_value(value, signature.output_fields[key].annotation)
                except Exception as exc:
                    raise StructuredOutputParseError(
                        f"DSPy could not convert structured field {key!r}"
                    ) from exc
            return converted

        def __call__(
            self,
            lm: Any,
            lm_kwargs: dict[str, Any],
            signature: Any,
            demos: list[dict[str, Any]],
            inputs: dict[str, Any],
        ) -> list[dict[str, Any]]:
            return Adapter.__call__(
                self,
                lm,
                self._request_kwargs(lm_kwargs, signature),
                signature,
                demos,
                inputs,
            )

        async def acall(
            self,
            lm: Any,
            lm_kwargs: dict[str, Any],
            signature: Any,
            demos: list[dict[str, Any]],
            inputs: dict[str, Any],
        ) -> list[dict[str, Any]]:
            # Do not let an async DSPy caller silently bypass the schema or
            # re-enter JSONAdapter's fallback path.
            return await Adapter.acall(
                self,
                lm,
                self._request_kwargs(lm_kwargs, signature),
                signature,
                demos,
                inputs,
            )

    return StrictJSONSchemaAdapter()


def is_retryable_llm_exception(exc: BaseException) -> bool:
    """Return true only for transient transport/server failures.

    The check is dependency-light so it works with LiteLLM, httpx, urllib and
    test doubles without importing the live inference stack.  Structured
    schema/parse/Pydantic errors are deterministic and therefore fail fast.
    """
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__

    deterministic_names = {
        "AdapterParseError",
        "JSONDecodeError",
        "SchemaError",
        "ValidationError",
    }
    if any(
        isinstance(item, StructuredOutputError)
        or type(item).__name__ in deterministic_names
        for item in chain
    ):
        return False

    retry_statuses = {408, 409, 425, 429}
    for item in chain:
        response = getattr(item, "response", None)
        status = getattr(item, "status_code", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if status is not None:
            try:
                numeric_status = int(status)
                return numeric_status in retry_statuses or 500 <= numeric_status < 600
            except (TypeError, ValueError):
                return False

    if any(isinstance(item, (TimeoutError, ConnectionError)) for item in chain):
        return True
    transient_names = {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
    }
    return any(type(item).__name__ in transient_names for item in chain)


def vllm_guided_json_adapter() -> Any:
    """Return a DSPy adapter that sends each output schema as ``guided_json``.

    ``JSONAdapter`` already owns DSPy's JSON prompt and parser behaviour.  We
    reuse its private schema builder (pinned together with DSPy in the
    DataSphere runtime) and call ``ChatAdapter`` directly only to avoid
    replacing the vLLM-native constrained parameter with the broken
    ``response_format`` route.  A schema-building failure intentionally falls
    back to DSPy's normal ``JSONAdapter`` instead of silently weakening output
    validation.
    """
    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.adapters.json_adapter import JSONAdapter, _get_structured_outputs_response_format

    class VLLMGuidedJSONAdapter(JSONAdapter):
        def __call__(
            self,
            lm: Any,
            lm_kwargs: dict[str, Any],
            signature: Any,
            demos: list[dict[str, Any]],
            inputs: dict[str, Any],
        ) -> list[dict[str, Any]]:
            try:
                output_model = _get_structured_outputs_response_format(signature)
                schema = canonicalize_vllm_guided_json_schema(output_model.model_json_schema())
            except Exception:
                # Preserve DSPy's own compatibility fallback for signatures
                # which cannot be represented as a closed JSON schema.
                return super().__call__(lm, lm_kwargs, signature, demos, inputs)

            previous_extra = lm_kwargs.get("extra_body", {})
            if previous_extra is None:
                previous_extra = {}
            if not isinstance(previous_extra, dict):
                raise TypeError("DSPy lm_kwargs.extra_body must be a mapping")
            lm_kwargs["extra_body"] = {**previous_extra, "guided_json": schema}
            # Do not send both controls: vLLM 0.6.3.post1 can ignore
            # response_format even when a guided-decoding backend is enabled.
            lm_kwargs.pop("response_format", None)
            # JSONAdapter supplies the JSON-oriented formatting methods via
            # dynamic dispatch.  ChatAdapter supplies the one-call execution
            # path and, because this object is still a JSONAdapter instance,
            # re-raises a parse error rather than retrying unconstrained text.
            return ChatAdapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)

    return VLLMGuidedJSONAdapter()
