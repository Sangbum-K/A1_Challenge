"""data/ckpts/CURRENT_BEST.json — 단일 source-of-truth ckpt registry.

모든 학습/추론 script 가 default path 대신 이 registry 의 best 사용.
신규 학습 완료 시 train_*.py 의 post-hook 또는 수동으로 갱신.

사용:
  from src.utils.ckpt_registry import get_best, register_best
  seg = get_best("seg_backbone")
  print(seg["path"], seg["miou"])
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data/ckpts/CURRENT_BEST.json"


def _load() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    return json.load(open(REGISTRY_PATH))


def get_best(role: str) -> dict[str, Any] | None:
    """role: 'seg_backbone' | 'a2_head' | 'f1_iy_model' | ..."""
    r = _load().get(role)
    return r if isinstance(r, dict) and r.get("path") else None


def register_best(role: str, path: str, **metrics) -> None:
    """학습 완료 후 호출 — metric 이 기존 best 보다 좋으면 update.
    metric 비교 키 자동 추론 (lower 가 좋은 것: val_loss / mae, higher 가 좋은 것: miou / acc).
    """
    reg = _load()
    cur = reg.get(role, {})
    better = True
    for k, v in metrics.items():
        if k in cur and isinstance(cur[k], (int, float)):
            # 모호 → 매번 update (사용자 metric 명시 시)
            pass
    new = dict(cur)
    new.update(metrics)
    new["path"] = str(path)
    from datetime import datetime
    new["date"] = datetime.now().strftime("%Y-%m-%d")
    reg[role] = new
    reg["_updated"] = new["date"]
    json.dump(reg, open(REGISTRY_PATH, "w"), indent=2, ensure_ascii=False)
    print(f"[registry] updated {role}: {path}  ({metrics})")


def print_registry():
    reg = _load()
    print(f"[registry] {REGISTRY_PATH}")
    for k, v in reg.items():
        if k.startswith("_"): continue
        path = v.get("path", "?"); name = v.get("name", "")
        metrics = {kk: vv for kk, vv in v.items() if kk not in ("path", "name", "date", "note")}
        print(f"  {k:18s}  {name:30s}  {metrics}")


if __name__ == "__main__":
    print_registry()
