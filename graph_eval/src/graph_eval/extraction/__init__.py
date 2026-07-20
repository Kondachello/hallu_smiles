from .base import Extractor, ExtractionError, ExtractionOutput
from .cached import CachedExtractor, extraction_identity
from .fake import FakeExtractor

__all__ = [
    "Extractor",
    "ExtractionOutput",
    "ExtractionError",
    "FakeExtractor",
    "CachedExtractor",
    "extraction_identity",
]

# GatewayExtractor is not eagerly imported (keeps `import graph_eval.extraction`
# free of the optional openai dependency); import it from .gateway when needed.
