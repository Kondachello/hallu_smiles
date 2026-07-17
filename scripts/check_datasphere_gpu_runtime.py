#!/usr/bin/env python3
"""Fail early when the Job's installed PyTorch cannot use the allocated GPU."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-torch-cuda", required=True)
    parser.add_argument("--expected-device-substring", required=True)
    args = parser.parse_args()

    # Import inside main so offline tests do not need torch/CUDA.
    import torch

    if torch.version.cuda != args.expected_torch_cuda:
        raise SystemExit(
            "Unexpected PyTorch CUDA build before GPU initialization: "
            f"expected {args.expected_torch_cuda!r}, got {torch.version.cuda!r}."
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            "PyTorch cannot access CUDA. Check the pinned vLLM/PyTorch version against "
            "the DataSphere g1.1 driver before starting vLLM."
        )
    try:
        probe = torch.empty(1, device="cuda")
        probe.fill_(1)
        torch.cuda.synchronize()
    except Exception as exc:  # torch gives driver-specific exceptions here.
        raise SystemExit(f"PyTorch CUDA smoke-check failed: {exc}") from exc

    index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(index)
    if args.expected_device_substring.casefold() not in device_name.casefold():
        raise SystemExit(
            f"Unexpected GPU: expected name containing {args.expected_device_substring!r}, "
            f"got {device_name!r}."
        )
    payload = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_index": index,
        "device_name": device_name,
        "expected_torch_cuda": args.expected_torch_cuda,
        "expected_device_substring": args.expected_device_substring,
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "status": "ready",
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
