"""Outlines compatibility data for JSON-only structured decoding.

No HalluGraph or KGGen request uses Outlines' airport enum.  An empty list is
therefore the exact least-privilege compatibility surface: it makes
``outlines.types.airports`` importable without inventing airport data or
changing any JSON-schema generation behaviour.
"""

AIRPORT_LIST: list[tuple[str, str, str, str]] = []
