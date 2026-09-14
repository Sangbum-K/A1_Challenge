"""데이터 파이프라인 자동화 — raw 스캔 → 처리상태 점검 → 신규 데이터 캐시화 오케스트레이션.

아이디어:
  raw 데이터는 {raw_root}/{track}/{policy}/{run}/{CAMERA_*, EGO_INFO, OBJECT_INFO} 구조.
  "데이터셋" = (track, policy). 캐시 파이프라인은 DAG:
     manifest → i1_tok_cache(GPU) → prebake(CPU) → i2_mem_cache(GPU) → inlane_gt → pedal_label
  각 단계는 출력 파일이 있으면 skip (idempotent). 신규/부분처리 데이터만 빠진 단계 실행.

범용성:
  - raw_root: AILabSSD(빠름) 우선, 없으면 AILabDataset, env SCC_RAW_ROOT 로 override.
  - 캐시 출력: SCC_DATA_DIR(env) > AILabDatasetLocal(이 머신) > repo/data/local(타 머신, gitignored).
  - legacy 캐시명(sangam_lead_0520 등)은 manifest 의 (track,policy) 로 역매핑해 인식.

사용:
  python scripts/data_pipeline.py scan                 # 전체 인벤토리 + 처리상태 표
  python scripts/data_pipeline.py scan --track sangam  # 트랙 필터
  python scripts/data_pipeline.py process --new --dry-run   # 신규 데이터 처리계획(실행 안함)
  python scripts/data_pipeline.py process <dataset_id>      # 특정 데이터셋 빠진 단계 실행
"""
from __future__ import annotations
import os, sys, json, glob, argparse, subprocess
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_dir() -> str:
    d = os.environ.get("SCC_DATA_DIR")
    if d:
        return d
    legacy = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"
    if os.path.isdir(legacy):
        return legacy
    return os.path.join(REPO_ROOT, "data", "local")


def raw_roots() -> list[str]:
    env = os.environ.get("SCC_RAW_ROOT")
    if env:
        return [env]
    cands = [
        "/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw",          # SSD (fast)
        "/home/ailab-12/AILabDataset/02_Custom_Dataset/57_A1_E2E/00_raw",
    ]
    return [c for c in cands if os.path.isdir(c)]


DATA = data_dir()
CACHE = os.path.join(DATA, "cache")
PREBAKE = os.path.join(DATA, "prebake")
TRACK_SHORT = {"sangam_track": "sangam", "speedway_track": "speedway", "sangam_qlf": "sangamqlf"}


def dataset_id(track: str, policy: str) -> str:
    """deterministic id (신규 데이터용). legacy 는 manifest 역매핑으로 별도 처리."""
    return f"{TRACK_SHORT.get(track, track)}_{policy}"


def manifest_map() -> dict:
    """기존 manifest 들에서 (track, policy) → logical_name 역매핑 (legacy 인식용)."""
    m = {}
    for f in glob.glob(os.path.join(DATA, "manifest_i1cache_*.json")):
        try:
            j = json.load(open(f))
            name = os.path.basename(f)[len("manifest_i1cache_"):-len(".json")]
            m[(j.get("track"), j.get("policy"))] = name
        except Exception:
            pass
    return m


def discover() -> list[dict]:
    """raw 스캔 → [{track, policy, n_runs, raw_root}]."""
    out = []
    seen = set()
    for root in raw_roots():
        for track in sorted(os.listdir(root)):
            tdir = os.path.join(root, track)
            if not os.path.isdir(tdir) or track not in TRACK_SHORT:
                continue
            for policy in sorted(os.listdir(tdir)):
                pdir = os.path.join(tdir, policy)
                if not os.path.isdir(pdir) or (track, policy) in seen:
                    continue
                runs = [r for r in os.listdir(pdir)
                        if os.path.isdir(os.path.join(pdir, r)) and os.path.isdir(os.path.join(pdir, r, "EGO_INFO"))]
                if not runs:
                    continue
                seen.add((track, policy))
                out.append({"track": track, "policy": policy, "n_runs": len(runs), "raw_root": root})
    return out


def _first(pattern: str):
    g = glob.glob(pattern)
    return g[0] if g else None


def stage_status(name: str) -> dict:
    """logical name 기준 각 단계 출력 존재 여부 (legacy 명명 변형 허용 — glob)."""
    return {
        "manifest": os.path.exists(os.path.join(DATA, f"manifest_i1cache_{name}.json")),
        "i1":       _first(os.path.join(CACHE, f"i1_tok_cache_{name}_*.npz")) is not None,
        "prebake":  _first(os.path.join(PREBAKE, f"f1_prebake_iy_{name}_*.npz")) is not None,
        "i2":       _first(os.path.join(CACHE, f"i2_mem_cache_{name}_*.npz")) is not None,
        "inlane":   os.path.exists(os.path.join(CACHE, f"inlane_gt_{name}.npz")),
        "pedal":    _first(os.path.join(CACHE, f"pedal_label_{name}_*.npz")) is not None,
    }


STAGES = ["manifest", "i1", "prebake", "i2", "inlane", "pedal"]


def overall(st: dict) -> str:
    vals = [st[s] for s in STAGES]
    if all(vals):
        return "DONE"
    if not any(vals):
        return "NEW"
    return "PARTIAL"


def cmd_scan(args):
    mm = manifest_map()
    rows = discover()
    if args.track:
        rows = [r for r in rows if args.track in r["track"]]
    print(f"raw_roots = {raw_roots()}")
    print(f"data_dir  = {DATA}\n")
    hdr = f"{'track':9s} {'policy':28s} {'runs':>4s} {'name':28s} " + " ".join(f"{s:>8s}" for s in STAGES) + "  status"
    print(hdr); print("-" * len(hdr))
    n_new = n_part = n_done = 0
    for r in rows:
        name = mm.get((r["track"], r["policy"]), dataset_id(r["track"], r["policy"]))
        st = stage_status(name)
        ov = overall(st)
        n_new += ov == "NEW"; n_part += ov == "PARTIAL"; n_done += ov == "DONE"
        marks = " ".join(f"{'O' if st[s] else '.':>8s}" for s in STAGES)
        print(f"{TRACK_SHORT.get(r['track'],r['track']):9s} {r['policy']:28s} {r['n_runs']:4d} {name:28s} {marks}  {ov}")
    print(f"\n총 {len(rows)} 데이터셋: DONE {n_done} / PARTIAL {n_part} / NEW {n_new}")
    if n_new or n_part:
        print("처리: python scripts/data_pipeline.py process --new --dry-run")


def cmd_process(args):
    mm = manifest_map()
    rows = discover()
    targets = []
    for r in rows:
        name = mm.get((r["track"], r["policy"]), dataset_id(r["track"], r["policy"]))
        st = stage_status(name)
        ov = overall(st)
        if args.dataset and name != args.dataset:
            continue
        if args.new and ov == "DONE":
            continue
        targets.append((r, name, st))
    if not targets:
        print("처리할 대상 없음 (--new 면 모두 DONE, 또는 dataset_id 불일치)"); return
    print(f"{'DRY-RUN' if args.dry_run else 'EXECUTE'} — {len(targets)} 데이터셋\n")
    for r, name, st in targets:
        missing = [s for s in STAGES if not st[s]]
        print(f"[{name}] track={r['track']} policy={r['policy']} runs={r['n_runs']} → missing: {missing or '없음(DONE)'}")
        if "manifest" in missing:
            print(f"   ! manifest 없음 — control GT 결정 필요(logged vs MPC synth). 자동화 범위 밖, 수동 manifest 생성 후 재실행.")
            continue
        plan = _plan_stages(name, missing)
        for stage, cmd in plan:
            print(f"   - {stage}: {' '.join(cmd)}")
            if not args.dry_run:
                print(f"     >> 실행...", flush=True)
                subprocess.run(cmd, check=True)
    if args.dry_run:
        print("\n실제 실행: --dry-run 제거 (GPU/NFS I/O 큼, 단계별 idempotent).")


def _plan_stages(name: str, missing: list[str]) -> list:
    """빠진 단계 → 기존 스크립트 명령. manifest/prebake 는 별도(상단 참조)."""
    PY = sys.executable
    mani = os.path.join(DATA, f"manifest_i1cache_{name}.json")
    i1 = os.path.join(CACHE, f"i1_tok_cache_{name}_prev20.npz")
    pb = _first(os.path.join(PREBAKE, f"f1_prebake_iy_{name}_*.npz")) or os.path.join(PREBAKE, f"f1_prebake_iy_{name}_prev20.npz")
    i2 = os.path.join(CACHE, f"i2_mem_cache_{name}_v7b_prev20.npz")
    seg = os.path.join(REPO_ROOT, "outputs_multi/seg_pretrain_fx640_v3/best_miou_backbone.pt")
    a2 = os.path.join(REPO_ROOT, "data/ckpts/a2_v7b_d128_alpha075_best_loss.pt")
    plan = []
    if "i1" in missing:
        plan.append(("i1", [PY, "scripts/compute_i1_tok_cache.py", "--manifest", mani, "--out", i1, "--seg-ckpt", seg]))
    if "prebake" in missing:
        plan.append(("prebake", [PY, "scripts/prebake_iy_dataset.py", "--manifest", mani, "--out", pb]))
    if "i2" in missing:
        plan.append(("i2", [PY, "scripts/build_i2_cache.py", "--i1-cache", i1, "--prebake", pb, "--out", i2, "--a2-ckpt", a2]))
    if "inlane" in missing:
        plan.append(("inlane", [PY, "scripts/build_inlane_gt_cache.py", "--ds", name]))
    if "pedal" in missing:
        plan.append(("pedal", [PY, "scripts/gen_ttc_pedal_labels.py", "--ds", name]))
    return plan


def main():
    ap = argparse.ArgumentParser(description="SCC 데이터 파이프라인 자동화")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("scan"); sp.add_argument("--track", default=None)
    pp = sub.add_parser("process")
    pp.add_argument("dataset", nargs="?", default=None)
    pp.add_argument("--new", action="store_true", help="DONE 아닌 모든 데이터셋")
    pp.add_argument("--dry-run", action="store_true", help="계획만 출력(기본 권장)")
    args = ap.parse_args()
    if args.cmd == "scan":
        cmd_scan(args)
    elif args.cmd == "process":
        args.dataset = args.dataset; cmd_process(args)


if __name__ == "__main__":
    main()
