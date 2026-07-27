"""Failure taxonomy. Epistemic abstention is data, never an exception."""


class DynamicTypingError(RuntimeError):
    """Base class for run-failing errors."""


class InputContractError(DynamicTypingError):
    """Input violates source/answer separation or contains forbidden fields."""


class PromptContractError(DynamicTypingError):
    """Prompt manifest, template variables or output schema are invalid."""


class TransportError(DynamicTypingError):
    """Timeout, network, 429 or 5xx failure after the bounded retry policy."""


class ModelProtocolError(DynamicTypingError):
    """Completion is truncated, malformed or violates the strict schema."""


class RegistryInvariantError(DynamicTypingError):
    """Registry cannot be frozen after bounded deterministic/model repair."""


class CacheIntegrityError(DynamicTypingError):
    """Immutable cached content or its provenance hash is inconsistent."""

