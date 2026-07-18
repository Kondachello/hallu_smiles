"""Strict OpenAI-compatible JSON-object runtime shared by KGGen and the verifier.

Alibaba Model Studio guarantees a JSON object, not conformance to a caller-provided
JSON Schema.  This module therefore keeps DSPy's complete typed contract in the
prompt, requests JSON mode on the wire, and validates the untouched response against
the exact DSPy-generated closed schema locally.  It deliberately contains no JSON
repair, code-fence stripping, field filtering, or root-object wrapping.
"""
from __future__ import annotations

import json
import math
import platform
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime
from email.utils import parsedate_to_datetime
from importlib import metadata
from typing import Any, Iterator

from tenacity import Retrying, retry_if_exception, stop_after_attempt


RUNTIME_PROTOCOL_VERSION = "dashscope-json-object-strict-local-schema-v1"
_RETRY_INDEX: ContextVar[int] = ContextVar("hallu_provider_retry_index", default=0)
_PENDING_PROVIDER_RESULT: ContextVar[tuple[Any, float, int] | None] = ContextVar(
    "hallu_pending_provider_result", default=None
)
_PROVIDER_SEMAPHORE_LOCK = threading.Lock()
_PROVIDER_SEMAPHORES: dict[tuple[str, str, int], threading.BoundedSemaphore] = {}


class CacheOnlyMissError(RuntimeError):
    """A cache-only replay requested a graph or verdict that was not cached."""


class StructuredOutputError(RuntimeError):
    """Base class for deterministic structured-output contract failures."""


class StructuredOutputParseError(StructuredOutputError):
    """The provider returned something other than one strict JSON document."""


class StructuredOutputSchemaError(StructuredOutputError):
    """A decoded response does not satisfy the exact output schema."""


def config_value(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def runtime_versions() -> dict[str, str]:
    """Return the cache-relevant runtime without importing heavyweight packages."""
    packages = {
        "torch": "torch",
        "kg-gen": "kg-gen",
        "dspy": "dspy",
        "litellm": "litellm",
        "pydantic": "pydantic",
        "jsonschema": "jsonschema",
        "sentence-transformers": "sentence-transformers",
        "transformers": "transformers",
        "tenacity": "tenacity",
    }
    versions: dict[str, str] = {"python": platform.python_version()}
    for label, distribution in packages.items():
        try:
            versions[label] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[label] = "not-installed"
    return versions


def provider_options(llm_config: Any) -> dict[str, Any]:
    """Validate and return the provider options that influence every LLM call."""
    transport = str(
        config_value(llm_config, "structured_output_transport", "json_object")
    ).strip()
    if transport != "json_object":
        raise ValueError("llm.structured_output_transport must be 'json_object'")
    extra_body = _plain(
        config_value(llm_config, "extra_body", {"enable_thinking": False})
    )
    if not isinstance(extra_body, dict):
        raise TypeError("llm.extra_body must be a mapping")
    if extra_body.get("enable_thinking") is not False:
        raise ValueError("llm.extra_body.enable_thinking must be false")
    timeout = float(config_value(llm_config, "request_timeout_s", 180))
    if timeout <= 0:
        raise ValueError("llm.request_timeout_s must be positive")
    return {
        "response_format": {"type": "json_object"},
        "extra_body": extra_body,
        "timeout": timeout,
    }


def provider_semaphore(cfg: Any) -> threading.BoundedSemaphore:
    """Return the shared per-provider concurrency gate for this process."""
    limit = int(config_value(cfg.llm, "concurrency", 1))
    if limit <= 0:
        raise ValueError("llm.concurrency must be positive")
    key = (
        str(cfg.llm.model),
        str(config_value(cfg.llm, "api_base", "")),
        limit,
    )
    with _PROVIDER_SEMAPHORE_LOCK:
        semaphore = _PROVIDER_SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _PROVIDER_SEMAPHORES[key] = semaphore
        return semaphore


def llm_runtime_fingerprint(cfg: Any) -> dict[str, Any]:
    """Stable, secret-free cache identity for provider and package behaviour."""
    options = provider_options(cfg.llm)
    return {
        "protocol": RUNTIME_PROTOCOL_VERSION,
        "model": str(cfg.llm.model),
        "api_base": str(config_value(cfg.llm, "api_base", "")),
        "temperature": float(cfg.llm.temperature),
        "max_tokens": int(config_value(cfg.llm, "max_tokens", 1024)),
        "request_timeout_s": options["timeout"],
        "max_retries": int(config_value(cfg.llm, "max_retries", 1)),
        "retry_backoff_base_s": float(
            config_value(cfg.llm, "retry_backoff_base_s", 0)
        ),
        "concurrency": int(config_value(cfg.llm, "concurrency", 1)),
        "structured_output_transport": "json_object",
        "response_format": options["response_format"],
        "extra_body": options["extra_body"],
        "runtime_versions": runtime_versions(),
    }


def strict_json_loads(document: Any, *, label: str = "structured output") -> Any:
    """Parse exactly one RFC-8259 JSON document, rejecting repairable extensions."""
    if not isinstance(document, str):
        raise StructuredOutputParseError(f"{label} is not text")

    def reject_constant(value: str) -> Any:
        raise StructuredOutputParseError(
            f"{label} contains non-JSON numeric constant {value!r}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except StructuredOutputError:
        raise
    except json.JSONDecodeError as exc:
        raise StructuredOutputParseError(
            f"{label} is invalid JSON at character {exc.pos}"
        ) from exc


def validate_json_document(instance: Any, schema: Mapping[str, Any]) -> None:
    """Validate a decoded document against an unchanged Draft 2020-12 schema."""
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError, ValidationError
    except Exception as exc:  # pragma: no cover - live dependency preflight owns this
        raise RuntimeError("jsonschema is required for strict structured output") from exc
    try:
        Draft202012Validator.check_schema(dict(schema))
        Draft202012Validator(dict(schema)).validate(instance)
    except SchemaError as exc:
        raise StructuredOutputSchemaError(
            f"invalid structured-output schema: {exc.message}"
        ) from exc
    except ValidationError as exc:
        location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        raise StructuredOutputSchemaError(
            f"structured output violates schema at {location}: {exc.message}"
        ) from exc


def dspy_output_schema(signature: Any) -> dict[str, Any]:
    """Build the exact closed Pydantic schema DSPy derives from a signature."""
    try:
        from dspy.adapters.json_adapter import _get_structured_outputs_response_format

        output_model = _get_structured_outputs_response_format(signature)
        schema = output_model.model_json_schema()
    except Exception as exc:
        raise StructuredOutputSchemaError(
            "DSPy could not construct the output schema"
        ) from exc
    if not isinstance(schema, Mapping):
        raise StructuredOutputSchemaError("DSPy output schema is not an object")
    copied = deepcopy(dict(schema))
    validate_schema_only(copied)
    return copied


def validate_schema_only(schema: Mapping[str, Any]) -> None:
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(dict(schema))
    except Exception as exc:
        raise StructuredOutputSchemaError("DSPy generated an invalid JSON Schema") from exc


def strict_json_object_adapter(
    *, extra_body: Mapping[str, Any], usage: Any = None
) -> Any:
    """Create a one-call DSPy JSON adapter with strict, local schema validation."""
    try:
        from dspy.adapters.base import Adapter
        from dspy.adapters.json_adapter import JSONAdapter
        from dspy.adapters.utils import parse_value
    except Exception as exc:  # pragma: no cover - live-only dependency
        raise RuntimeError("DSPy is required for KGGen structured output") from exc

    provider_extra = deepcopy(dict(extra_body))

    class StrictJSONObjectAdapter(JSONAdapter):
        @staticmethod
        def _request_kwargs(lm_kwargs: Mapping[str, Any]) -> dict[str, Any]:
            request_kwargs = dict(lm_kwargs)
            existing_extra = request_kwargs.get("extra_body", {})
            if not isinstance(existing_extra, Mapping):
                raise TypeError("DSPy lm_kwargs.extra_body must be a mapping")
            request_kwargs["extra_body"] = {
                **dict(existing_extra),
                **provider_extra,
            }
            request_kwargs["response_format"] = {"type": "json_object"}
            return request_kwargs

        def parse(self, signature: Any, completion: str) -> dict[str, Any]:
            fields = strict_json_loads(completion, label="DSPy completion")
            if not isinstance(fields, dict):
                raise StructuredOutputSchemaError(
                    "DSPy structured-output root must be an object"
                )
            expected = set(signature.output_fields)
            actual = set(fields)
            if actual != expected:
                raise StructuredOutputSchemaError(
                    "structured output keys differ from the DSPy signature: "
                    f"expected={sorted(expected)!r} actual={sorted(actual)!r}"
                )
            validate_json_document(fields, dspy_output_schema(signature))
            converted: dict[str, Any] = {}
            for key, value in fields.items():
                try:
                    converted[key] = parse_value(
                        value, signature.output_fields[key].annotation
                    )
                except Exception as exc:
                    raise StructuredOutputSchemaError(
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
            # Calling Adapter directly bypasses JSONAdapter's repair/fallback path.
            _PENDING_PROVIDER_RESULT.set(None)
            try:
                result = Adapter.__call__(
                    self,
                    lm,
                    self._request_kwargs(lm_kwargs),
                    signature,
                    demos,
                    inputs,
                )
            except StructuredOutputError as exc:
                _record_pending_provider_result(usage, "contract_error", exc)
                raise
            else:
                _record_pending_provider_result(usage, "success", None)
                return result

        async def acall(
            self,
            lm: Any,
            lm_kwargs: dict[str, Any],
            signature: Any,
            demos: list[dict[str, Any]],
            inputs: dict[str, Any],
        ) -> list[dict[str, Any]]:
            raise RuntimeError(
                "async DSPy calls are disabled by the serial research API runtime"
            )

    return StrictJSONObjectAdapter(use_native_function_calling=False)


def _record_pending_provider_result(
    usage: Any, outcome: str, error: BaseException | None
) -> None:
    pending = _PENDING_PROVIDER_RESULT.get()
    _PENDING_PROVIDER_RESULT.set(None)
    if usage is None or pending is None:
        return
    response, seconds, retry_index = pending
    usage.record_provider_call(
        outcome=outcome,
        seconds=seconds,
        response=response,
        error=error,
        retry_index=retry_index,
    )


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def validate_completion_envelope(response: Any, *, label: str = "completion") -> None:
    """Reject multiple choices and length/content-filter truncation before parsing."""
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        raise StructuredOutputParseError(f"{label} must contain exactly one choice")
    finish_reason = _field(choices[0], "finish_reason")
    if finish_reason != "stop":
        raise StructuredOutputParseError(
            f"{label} did not finish cleanly: {finish_reason!r}"
        )
    message = _field(choices[0], "message")
    content = _field(message, "content")
    if not isinstance(content, str) or not content.strip():
        raise StructuredOutputParseError(f"{label} has empty or non-text content")


def response_content(response: Any) -> str:
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        raise StructuredOutputParseError("completion must contain exactly one choice")
    message = _field(choices[0], "message")
    content = _field(message, "content")
    if not isinstance(content, str) or not content.strip():
        raise StructuredOutputParseError("completion has empty or non-text content")
    return content


@contextmanager
def provider_retry_attempt(index: int) -> Iterator[None]:
    token = _RETRY_INDEX.set(max(0, int(index)))
    try:
        yield
    finally:
        _RETRY_INDEX.reset(token)


def current_retry_index() -> int:
    return _RETRY_INDEX.get()


def configure_dspy_lm(lm: Any, cfg: Any, usage: Any = None) -> Any:
    """Apply one retry/JSON/telemetry policy to KGGen's private DSPy LM."""
    if getattr(lm, "_hallu_api_runtime_configured", False):
        return lm._hallu_api_runtime_adapter
    options = provider_options(cfg.llm)
    semaphore = provider_semaphore(cfg)
    max_attempts = int(config_value(cfg.llm, "max_retries", 1))
    if max_attempts <= 0:
        raise ValueError("llm.max_retries must be positive")
    backoff_base = float(config_value(cfg.llm, "retry_backoff_base_s", 0))
    kwargs = getattr(lm, "kwargs", None)
    if not isinstance(kwargs, dict):
        raise RuntimeError("KGGen DSPy LM has no mutable kwargs mapping")
    kwargs["timeout"] = options["timeout"]
    kwargs["response_format"] = options["response_format"]
    kwargs["extra_body"] = options["extra_body"]
    lm.num_retries = 0
    # The content-addressed graph/verdict caches are the sole research caches.
    if hasattr(lm, "cache"):
        lm.cache = False
    original_forward = getattr(lm, "forward", None)
    if not callable(original_forward):
        raise RuntimeError("KGGen DSPy LM has no callable forward method")

    def guarded_forward(*args: Any, **kwargs: Any) -> Any:
        import time

        response = None
        successful_retry_index = 0
        started = time.perf_counter()
        # Hold the provider slot across Retry-After/backoff so another KGGen
        # chunk cannot violate a provider-wide cooldown while this call waits.
        with semaphore:
            for attempt in Retrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_transient(backoff_base),
                retry=retry_if_exception(is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    successful_retry_index = attempt.retry_state.attempt_number - 1
                    with provider_retry_attempt(successful_retry_index):
                        started = time.perf_counter()
                        try:
                            response = original_forward(*args, **kwargs)
                        except Exception as exc:
                            if usage is not None:
                                usage.record_provider_call(
                                    outcome="failure",
                                    seconds=time.perf_counter() - started,
                                    error=exc,
                                    retry_index=successful_retry_index,
                                )
                            raise
        assert response is not None
        try:
            validate_completion_envelope(response, label="DSPy completion")
        except StructuredOutputError as exc:
            if usage is not None:
                usage.record_provider_call(
                    outcome="contract_error",
                    seconds=time.perf_counter() - started,
                    response=response,
                    error=exc,
                    retry_index=successful_retry_index,
                )
            raise
        _PENDING_PROVIDER_RESULT.set(
            (response, time.perf_counter() - started, successful_retry_index)
        )
        return response

    lm.forward = guarded_forward
    adapter = strict_json_object_adapter(
        extra_body=options["extra_body"], usage=usage
    )
    # KGGen's native chunker creates ordinary ThreadPoolExecutor workers. DSPy
    # contextvars do not cross that boundary, so the strict adapter must also be
    # the process-wide default visible in each worker. The runner constructs one
    # live KGExtractor on its main thread; any conflicting owner fails loudly.
    import dspy

    dspy.configure(lm=lm, adapter=adapter)
    lm._hallu_api_runtime_adapter = adapter
    lm._hallu_api_runtime_configured = True
    return adapter


def strict_kggen_get_relations(
    input_data: str,
    entities: list[str],
    is_conversation: bool = False,
    context: str = "",
) -> list[tuple[str, str, str]]:
    """Run KGGen 0.4's primary relation signature without its repair fallback.

    KGGen's upstream function catches every exception from its primary typed call,
    sends a looser second prompt, and may send a third fixing prompt.  That behaviour
    would hide a provider schema violation, so the API runtime uses the identical
    primary signature and prompt but lets every error reach the explicit retry policy.
    """
    try:
        import dspy
        from kg_gen.steps._2_get_relations import extraction_sig
        from pydantic import BaseModel
    except Exception as exc:  # pragma: no cover - live dependency
        raise RuntimeError("KGGen 0.4 and DSPy are required") from exc

    class Relation(BaseModel):
        """Knowledge graph subject-predicate-object tuple."""

        subject: str = dspy.InputField(desc="Subject entity", examples=["Kevin"])
        predicate: str = dspy.InputField(
            desc="Predicate", examples=["is brother of"]
        )
        object: str = dspy.InputField(desc="Object entity", examples=["Vicky"])

    signature = extraction_sig(Relation, is_conversation, context)
    result = dspy.Predict(signature)(source_text=input_data, entities=entities)
    return [(r.subject, r.predicate, r.object) for r in result.relations]


def install_kggen_relation_contract() -> None:
    """Install the fail-fast relation boundary into KGGen's imported module globals."""
    try:
        import kg_gen.kg_gen as kggen_module
        import kg_gen.steps._2_get_relations as relation_module
        import kg_gen.steps._3_cluster_graph as cluster_module
    except Exception as exc:  # pragma: no cover - live dependency
        raise RuntimeError("KGGen 0.4 is required") from exc
    if getattr(kggen_module, "_hallu_relation_contract_installed", False):
        return
    kggen_module.get_relations = strict_kggen_get_relations
    relation_module.get_relations = strict_kggen_get_relations
    cluster_module._map_batch_items = strict_kggen_map_batch_items
    kggen_module._hallu_relation_contract_installed = True


def strict_kggen_map_batch_items(
    batch: set[str],
    cluster_reps: list[str | None],
    cluster_map: dict[str, Any],
    item_assignments: dict[str, str | None],
    context: str,
    validate: Any,
) -> dict[str, str | None]:
    """KGGen 0.4 `_map_batch_items` with schema failures made fail-fast.

    This is an otherwise line-for-line behavioural copy of the upstream helper.
    Upstream logs and swallows every exception from its validation DSPy call.
    Contract and provider/HTTP/network errors must escape; unrelated local
    validation failures retain upstream's log-and-continue behaviour.
    """
    from kg_gen.steps import _3_cluster_graph as cluster_module

    for index, item in enumerate(batch):
        item_assignments[item] = None
        rep = cluster_reps[index] if index < len(cluster_reps) else None
        target_cluster = cluster_map.get(rep) if rep is not None else None

        if target_cluster:
            if item == target_cluster.representative or item in target_cluster.members:
                item_assignments[item] = target_cluster.representative
                continue

            potential_new_members = target_cluster.members | {item}
            try:
                result = validate(
                    cluster=potential_new_members,
                    context=context,
                )
                validated_items = set(result.validated_items)
                if item in validated_items and len(validated_items) == len(
                    potential_new_members
                ):
                    item_assignments[item] = target_cluster.representative
            except Exception as exc:
                if is_provider_transport_exception(exc):
                    raise
                # Preserve KGGen's handling of non-provider validation bugs.
                cluster_module.logger.error(
                    "Validation failed for item '%s' potentially belonging to "
                    "cluster '%s': %s",
                    item,
                    target_cluster.representative,
                    exc,
                )

    return item_assignments


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def exception_status_code(exc: BaseException) -> int | None:
    for item in _exception_chain(exc):
        status = getattr(item, "status_code", None)
        response = getattr(item, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if status is not None:
            try:
                return int(status)
            except (TypeError, ValueError):
                continue
    return None


def retry_after_seconds(exc: BaseException) -> float | None:
    """Extract Retry-After seconds or HTTP date without retaining other headers."""
    for item in _exception_chain(exc):
        candidates = [getattr(item, "headers", None)]
        response = getattr(item, "response", None)
        candidates.append(getattr(response, "headers", None) if response is not None else None)
        for headers in candidates:
            if not isinstance(headers, Mapping):
                continue
            value = headers.get("retry-after") or headers.get("Retry-After")
            if value is None:
                continue
            try:
                seconds = float(value)
                return max(0.0, seconds) if math.isfinite(seconds) else None
            except (TypeError, ValueError):
                try:
                    retry_at = parsedate_to_datetime(str(value))
                    now = datetime.now(retry_at.tzinfo)
                    return max(0.0, (retry_at - now).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    continue
    return None


def is_retryable_exception(exc: BaseException) -> bool:
    """Retry only network failures, HTTP 429, and HTTP 5xx responses."""
    chain = _exception_chain(exc)
    if any(isinstance(item, StructuredOutputError) for item in chain):
        return False
    status = exception_status_code(exc)
    if status == 429 or (status is not None and 500 <= status < 600):
        return True
    return _is_network_failure(chain)


def _is_network_failure(chain: list[BaseException]) -> bool:
    network_names = {
        "APIConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "NetworkError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "Timeout",
        "TimeoutError",
        "WriteError",
        "WriteTimeout",
    }
    return any(
        isinstance(item, (ConnectionError, TimeoutError))
        or type(item).__name__ in network_names
        for item in chain
    )


def is_provider_transport_exception(exc: BaseException) -> bool:
    """Identify failures that KGGen must never swallow as clustering noise."""
    chain = _exception_chain(exc)
    if any(isinstance(item, StructuredOutputError) for item in chain):
        return True
    if exception_status_code(exc) is not None or _is_network_failure(chain):
        return True
    return any(type(item).__module__.split(".", 1)[0] == "litellm" for item in chain)


def wait_transient(backoff_base: float):
    """Tenacity wait callback that honours Retry-After and exponential backoff."""
    base = float(backoff_base)
    if base < 0:
        raise ValueError("retry_backoff_base_s must be non-negative")

    def wait(retry_state: Any) -> float:
        exponential = base * (2 ** max(0, retry_state.attempt_number - 1))
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        retry_after = retry_after_seconds(exc) if exc is not None else None
        return max(exponential, retry_after or 0.0)

    return wait
