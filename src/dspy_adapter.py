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

from typing import Any


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
                schema = output_model.model_json_schema()
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
