"""Offline contract tests for the vLLM 0.6.3 LMFE boolean-schema backport."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "patch_datasphere_lmfe_bool_schema.py"


def _module():
    spec = importlib.util.spec_from_file_location("lmfe_patch", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lmfe_bool_schema_backport_is_narrow_idempotent_and_compilable(tmp_path):
    module = _module()
    source = tmp_path / "jsonschemaobject.py"
    source.write_text(
        "from typing import Any, Dict, Union\n"
        "class JsonSchemaObject:\n"
        "    def validator(\n"
        "        cls, values: Dict[str, Any]\n"
        "    ) -> Any:\n"
        "        exclusive_maximum: Union[float, bool, None] = values.get('exclusiveMaximum')\n"
        "        return values\n",
        encoding="utf-8",
    )

    assert module.patch_source(source) == "patched"
    patched = source.read_text(encoding="utf-8")
    assert module.PATCH_MARKER in patched
    assert "if isinstance(values, bool):" in patched
    assert module.patch_source(source) == "present"


def test_lmfe_backport_refuses_unrecognised_source(tmp_path):
    module = _module()
    source = tmp_path / "jsonschemaobject.py"
    source.write_text("unrecognised\n", encoding="utf-8")
    try:
        module.patch_source(source)
    except RuntimeError as exc:
        assert "refusing an unsafe patch" in str(exc)
    else:
        raise AssertionError("expected an unrecognised third-party file to be rejected")
