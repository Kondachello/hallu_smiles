"""DSPy adapter variants used only by explicitly configured local runtimes.

The project normally leaves adapter selection to DSPy.  vLLM 0.6.3.post1,
however, has a known OpenAI ``response_format`` regression: DSPy's regular
``JSONAdapter`` sends that parameter, while the server can then ignore the
schema.  The same vLLM release exposes its native, constrained-decoding
``guided_json`` request parameter.  This module maps DSPy's *existing* typed
output schema to that parameter without changing any KGGen prompt, graph
extraction, or clustering logic.

Imports deliberately stay inside the factory so ordinary offline tests and
remote-provider users do not acquire a DSPy/vLLM dependency.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any


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
