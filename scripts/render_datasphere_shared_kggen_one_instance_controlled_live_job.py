#!/usr/bin/env python3
"""Render the CPU-only real controlled shared-KGGen two-pass Job."""
from __future__ import annotations
import argparse, base64, re
from pathlib import Path
from urllib.parse import urlparse
from datasphere_runtime_image import require_runtime_image

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "datasphere/jobs/shared-kggen-one-instance-controlled-live.template.yaml"
def main() -> None:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--commit",required=True); p.add_argument("--run-id",required=True); p.add_argument("--response-id",required=True); p.add_argument("--gateway-url",required=True); p.add_argument("--docker-image",required=True); p.add_argument("--output",required=True); a=p.parse_args()
 if not re.fullmatch(r"[0-9a-f]{40}",a.commit) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}",a.run_id): raise SystemExit("invalid commit or run id")
 u=urlparse(a.gateway_url)
 if u.scheme!="https" or not u.netloc or u.path.rstrip("/") or u.query or u.fragment: raise SystemExit("--gateway-url must be HTTPS origin")
 image=require_runtime_image(a.docker_image,registry=True); response=str(a.response_id)
 if not response or "\x00" in response: raise SystemExit("invalid response id")
 text=TEMPLATE.read_text(encoding="utf-8")
 text=(text.replace("__GIT_COMMIT__",a.commit).replace("__RUN_ID__",a.run_id).replace("__GATEWAY_URL__",f"https://{u.netloc}").replace("__RESPONSE_ID_B64__",base64.b64encode(response.encode()).decode()).replace("__CACHE_NAMESPACE__",f"v1/{a.commit}/{response}").replace("__DOCKER_IMAGE__",image).replace("__DOCKER_ENV_BLOCK__",f"  docker:\n    image: {image}"))
 if "__" in text: raise RuntimeError("unresolved placeholder")
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(text,encoding="utf-8"); print(out)
if __name__ == "__main__": main()
