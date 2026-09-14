"""a_2 detector 의 FP(없는 차를 있다고 오검출)가 트랙 지도상 어디서 발생하는지 산점도.

목적: FP 가 특정 구간/배경(그림자·연석·핏 등)에 몰리는지 진단.

모델 2개 비교:
  v7b      (예선, 2-frame)  : A2Model(d_token=256,d_out=128), prev offset 20.
  baseline (현재, 5-frame)  : A2ModelMultiFrame(d_token=256,d_out=128,n_frames=6),
                               prev offsets 10,20,30,40,50.
inference 경로는 scripts/eval_a2_multiframe.py 의 run_v7b / run_mf 를 그대로 재사용.

FP 정의:
  absent(no-lead) frame = aux GT in-lane center([:,16:26,0]) 이 모두 < 0.5.
  그 frame 에서 pred obj prob(sigmoid) center max(obj[:,16:26].max) > 0.5 → FP.

ego (x,y): aux 파일명 {track}__{policy}__{run}__t{tick:07d} →
  RAW/{track}/{policy}/{run}/EGO_INFO/{tick}.txt, parse_ego 재사용.

plot: 좌 sangam(0520+0521), 우 speedway(spd). 전체 frame ego(x,y) 회색,
  FP frame 을 v7b=빨강 / baseline=주황 overlay. 저장 outputs_multi/a2_objsep/fp_on_map.png.
"""
import os, sys, glob, re
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.image_models.a2_model import A2Model, A2ModelMultiFrame
from src.model.image_models.iy_model import yaw_shift_tokens_batch
from scripts.prepare_a2_aux_targets import parse_ego           # ego x,y 파싱 재사용
from scripts.eval_a2_multiframe import run_v7b, run_mf, AUX_PAT, load_cache, build_dyaw_lut

D   = "/home/ailab-12/AILabDatasetLocal/scc_controller_imitator/data"
RAW = "/home/ailab-12/AILabSSD/02_Data/57_A1_E2E/00_raw"
OUT_PNG = ROOT / "outputs_multi" / "a2_objsep" / "fp_on_map.png"

V7B_CKPT = ROOT / "data/ckpts/a2_v7b_d128_alpha075_best_loss.pt"
BASE_CKPT = ROOT / "outputs_multi/a2_b0_abl/mf5f_v2s3set_10_20_30_40_50/best_loss.pt"
BASE_OFFS = [10, 20, 30, 40, 50]      # n_frames=6 (curr + 5 prev)

CENTER = slice(16, 26)

# track 별 dataset 묶음: (label, [(i1_cache, aux_dir), ...])
DATASETS = [
    ("sangam", [
        (f"{D}/cache/i1_tok_cache_sangam_lead_0520_prev20.npz", f"{D}/aux_targets_sangam_lead_0520"),
        (f"{D}/cache/i1_tok_cache_sangam_lead_0521_prev20.npz", f"{D}/aux_targets_sangam_lead_0521"),
    ]),
    ("speedway", [
        (f"{D}/cache/i1_tok_cache_spd_v2_prev20.npz", f"{D}/aux_targets_spd_nolead_0520"),
    ]),
]

# 파일명 → (track, policy, run, tick) (EGO_INFO 경로용; 4-segment split)
def parse_aux_fname(fname: str):
    stem = fname[:-4] if fname.endswith(".npz") else fname
    parts = stem.split("__")
    if len(parts) != 4:
        return None
    track, policy, run, ttick = parts
    if not ttick.startswith("t"):
        return None
    try:
        tick = int(ttick[1:])
    except ValueError:
        return None
    return track, policy, run, tick


_EGO_CACHE = {}
def ego_xy(track, policy, run, tick):
    """RAW EGO_INFO 에서 ego (x,y). 실패 시 None."""
    p = Path(RAW) / track / policy / run / "EGO_INFO" / f"{tick}.txt"
    key = str(p)
    if key in _EGO_CACHE:
        return _EGO_CACHE[key]
    ego = parse_ego(p)
    xy = (float(ego["position"][0]), float(ego["position"][1])) if ego else None
    _EGO_CACHE[key] = xy
    return xy


def collect_samples(pairs, union_offsets):
    """track 의 모든 cache/aux 쌍에서 load_cache(eval 재사용) → 통합 sample list.
    각 sample 에 (track,policy) 부여 (aux dir glob 다시 돌며 fname→full parse 매칭)."""
    all_tokens = []          # per-cache tokens
    all_samples = []         # (cache_idx, ci, gt(40,7), run, tick, pmap, track, policy)
    for ci_cache, (i1p, aux_dir) in enumerate(pairs):
        tokens, k2i, samples = load_cache(i1p, aux_dir, None, union_offsets)
        all_tokens.append(tokens)
        # run__tick → (track, policy) 매핑 (aux dir 파일명에서 full parse)
        rt2tp = {}
        for f in glob.glob(str(Path(aux_dir) / "*.npz")):
            pr = parse_aux_fname(Path(f).name)
            if pr is None:
                continue
            track, policy, run, tick = pr
            rt2tp[(run, tick)] = (track, policy)
        for (ci, gt, run, tick, pmap) in samples:
            tp = rt2tp.get((run, tick))
            track = tp[0] if tp else None
            policy = tp[1] if tp else None
            all_samples.append((ci_cache, ci, gt, run, tick, pmap, track, policy))
    return all_tokens, all_samples


@torch.no_grad()
def infer_v7b(model, all_tokens, all_samples, dev):
    """cache 별로 run_v7b 적용 후 통합 (sample 순서 보존)."""
    objs = np.empty((len(all_samples), 40), dtype=np.float32)
    for ci_cache in range(len(all_tokens)):
        idx_local = [i for i, s in enumerate(all_samples) if s[0] == ci_cache]
        if not idx_local:
            continue
        # run_v7b 가 기대하는 sample 튜플 (ci, gt, run, tick, pmap)
        subs = [(all_samples[i][1], all_samples[i][2], all_samples[i][3],
                 all_samples[i][4], all_samples[i][5]) for i in idx_local]
        o, _ = run_v7b(model, all_tokens[ci_cache], subs, None, dev)
        for k, i in enumerate(idx_local):
            objs[i] = o[k]
    return objs


@torch.no_grad()
def infer_mf(model, all_tokens, all_samples, dev, offs):
    objs = np.empty((len(all_samples), 40), dtype=np.float32)
    for ci_cache in range(len(all_tokens)):
        idx_local = [i for i, s in enumerate(all_samples) if s[0] == ci_cache]
        if not idx_local:
            continue
        subs = [(all_samples[i][1], all_samples[i][2], all_samples[i][3],
                 all_samples[i][4], all_samples[i][5]) for i in idx_local]
        o, _ = run_mf(model, all_tokens[ci_cache], subs, dev, offs, "off", {})
        for k, i in enumerate(idx_local):
            objs[i] = o[k]
    return objs


def set_korean_font():
    try:
        import matplotlib
        from matplotlib import font_manager
        fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print("[font] warn:", e)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dev] {dev}")
    union_offsets = sorted(set([20] + BASE_OFFS))

    # 모델 로드
    v7b = A2Model(d_token=256, d_out=128).to(dev).eval()
    sd = torch.load(str(V7B_CKPT), map_location=dev, weights_only=False)
    v7b.load_state_dict(sd["model"] if "model" in sd else sd)
    print(f"[v7b] loaded {V7B_CKPT.name}")

    base = A2ModelMultiFrame(d_token=256, d_out=128, n_frames=1 + len(BASE_OFFS)).to(dev).eval()
    sb = torch.load(str(BASE_CKPT), map_location=dev, weights_only=False)
    base.load_state_dict(sb["model"] if "model" in sb else sb)
    print(f"[baseline] loaded {BASE_CKPT} (n_frames={1+len(BASE_OFFS)}, offs={BASE_OFFS})")

    per_track = {}    # label → dict(ego_all, fp_v7b_xy, fp_base_xy, n_absent, n_total, ...)
    for label, pairs in DATASETS:
        print(f"\n===== {label} =====")
        all_tokens, all_samples = collect_samples(pairs, union_offsets)
        print(f"  samples N={len(all_samples)}")
        gt = np.stack([s[2] for s in all_samples], axis=0)        # (N,40,7)
        absn = (gt[:, CENTER, 0] < 0.5).all(axis=1)               # in-lane center 전부 absent
        print(f"  absent(no-lead) frames = {int(absn.sum())} / {len(all_samples)} "
              f"({absn.mean()*100:.1f}%)")

        o_v7b = infer_v7b(v7b, all_tokens, all_samples, dev)
        o_base = infer_mf(base, all_tokens, all_samples, dev, BASE_OFFS)
        amax_v7b = o_v7b[:, CENTER].max(axis=1)
        amax_base = o_base[:, CENTER].max(axis=1)
        fp_v7b = absn & (amax_v7b > 0.5)
        fp_base = absn & (amax_base > 0.5)
        print(f"  FP v7b      = {int(fp_v7b.sum())}  ({fp_v7b.sum()/max(int(absn.sum()),1)*100:.1f}% of absent)")
        print(f"  FP baseline = {int(fp_base.sum())}  ({fp_base.sum()/max(int(absn.sum()),1)*100:.1f}% of absent)")

        # ego (x,y) — 전체 frame (회색 궤적) + FP frame
        ego_all, fp_v7b_xy, fp_base_xy = [], [], []
        n_noego = 0
        for i, s in enumerate(all_samples):
            track, policy, run, tick = s[6], s[7], s[3], s[4]
            if track is None or policy is None:
                n_noego += 1; continue
            xy = ego_xy(track, policy, run, tick)
            if xy is None:
                n_noego += 1; continue
            ego_all.append(xy)
            if fp_v7b[i]:
                fp_v7b_xy.append(xy)
            if fp_base[i]:
                fp_base_xy.append(xy)
        print(f"  ego parsed: {len(ego_all)}/{len(all_samples)} (no-ego {n_noego})")
        # absent frame 만의 ego (heatmap 분모 = "차가 없을 때 그 구간을 지난 횟수")
        absent_xy = []
        for i, s in enumerate(all_samples):
            if not absn[i]:
                continue
            track, policy, run, tick = s[6], s[7], s[3], s[4]
            if track is None or policy is None:
                continue
            xy = ego_xy(track, policy, run, tick)
            if xy is not None:
                absent_xy.append(xy)
        per_track[label] = dict(
            ego_all=np.array(ego_all) if ego_all else np.zeros((0, 2)),
            absent_xy=np.array(absent_xy) if absent_xy else np.zeros((0, 2)),
            fp_v7b_xy=np.array(fp_v7b_xy) if fp_v7b_xy else np.zeros((0, 2)),
            fp_base_xy=np.array(fp_base_xy) if fp_base_xy else np.zeros((0, 2)),
            n_absent=int(absn.sum()), n_total=len(all_samples),
            n_fp_v7b=int(fp_v7b.sum()), n_fp_base=int(fp_base.sum()),
        )

    # 빠른 재-plot 용으로 ego/FP 좌표 캐싱
    np.savez_compressed(
        str(OUT_PNG.parent / "fp_on_map_data.npz"),
        **{f"{lab}__{k}": v for lab, T in per_track.items()
           for k, v in T.items() if isinstance(v, np.ndarray)},
        meta=np.array([(lab, T["n_absent"], T["n_total"], T["n_fp_v7b"], T["n_fp_base"])
                       for lab, T in per_track.items()], dtype=object),
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    make_plots(per_track, [d[0] for d in DATASETS])


def make_plots(per_track, labels):
    """① scatter overlay (fp_on_map.png) + ② FP-rate heatmap (fp_rate_heatmap.png).
    per_track[label] = dict(ego_all, absent_xy, fp_v7b_xy, fp_base_xy, n_*)."""
    set_korean_font()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    # ① scatter overlay (FP 만 작게, 트랙은 옅게 — 군집 육안 판단용)
    fig, axes = plt.subplots(1, len(labels), figsize=(7.2 * len(labels), 6.6), squeeze=False)
    for j, label in enumerate(labels):
        ax = axes[0][j]; T = per_track[label]
        if len(T["ego_all"]):
            ax.scatter(T["ego_all"][:, 0], T["ego_all"][:, 1], s=3, c="0.82",
                       alpha=0.6, label=f"track ({len(T['ego_all'])} frames)", zorder=1)
        if len(T["fp_base_xy"]):
            ax.scatter(T["fp_base_xy"][:, 0], T["fp_base_xy"][:, 1], s=14, c="tab:orange",
                       alpha=0.5, marker="s", edgecolors="none",
                       label=f"FP baseline (5f) n={T['n_fp_base']}", zorder=2)
        if len(T["fp_v7b_xy"]):
            ax.scatter(T["fp_v7b_xy"][:, 0], T["fp_v7b_xy"][:, 1], s=10, c="tab:red",
                       alpha=0.45, marker="o", edgecolors="none",
                       label=f"FP v7b (2f) n={T['n_fp_v7b']}", zorder=3)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)
        rv = T["n_fp_v7b"] / max(T["n_absent"], 1) * 100
        rb = T["n_fp_base"] / max(T["n_absent"], 1) * 100
        ax.set_title(f"{label}  (absent {T['n_absent']})\n"
                     f"FP v7b={T['n_fp_v7b']} ({rv:.1f}%)  baseline={T['n_fp_base']} ({rb:.1f}%)",
                     fontsize=11)
        ax.set_xlabel("ego x (m)"); ax.set_ylabel("ego y (m)")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("a_2 detector False-Positive 위치 (없는 차 오검출, in-lane center)  "
                 "회색=트랙궤적 · 빨강=v7b · 주황=baseline", fontsize=12)
    fig.tight_layout(); fig.savefig(str(OUT_PNG), dpi=110)
    print(f"[saved] {OUT_PNG}")

    # ② FP-rate heatmap: 공간 bin 별 (FP frame 수 / absent frame 수) — 군집 정량화
    fig2, axes2 = plt.subplots(2, len(labels), figsize=(7.2 * len(labels), 11), squeeze=False)
    for j, label in enumerate(labels):
        T = per_track[label]
        if not len(T["absent_xy"]):
            continue
        ax_all = np.vstack([T["absent_xy"]] +
                           ([T["fp_v7b_xy"]] if len(T["fp_v7b_xy"]) else []) +
                           ([T["fp_base_xy"]] if len(T["fp_base_xy"]) else []))
        xlo, xhi = ax_all[:, 0].min(), ax_all[:, 0].max()
        ylo, yhi = ax_all[:, 1].min(), ax_all[:, 1].max()
        span = max(xhi - xlo, yhi - ylo); nb = 60
        xe = np.linspace(xlo, xhi, nb + 1); ye = np.linspace(ylo, yhi, nb + 1)
        denom, _, _ = np.histogram2d(T["absent_xy"][:, 0], T["absent_xy"][:, 1], bins=[xe, ye])
        for r, (mdl, xy, nfp) in enumerate([
                ("v7b (2f)", T["fp_v7b_xy"], T["n_fp_v7b"]),
                ("baseline (5f)", T["fp_base_xy"], T["n_fp_base"])]):
            ax = axes2[r][j]
            if len(xy):
                num, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
            else:
                num = np.zeros_like(denom)
            rate = np.full_like(denom, np.nan)
            mvis = denom >= 3        # 통계적으로 의미있는 bin (absent>=3 통과)
            rate[mvis] = num[mvis] / denom[mvis] * 100
            im = ax.imshow(rate.T, origin="lower", extent=[xlo, xhi, ylo, yhi],
                           cmap="hot_r", vmin=0, vmax=100, aspect="equal")
            ax.set_title(f"{label}  {mdl}  FP-rate (absent 대비 %)\n"
                         f"전체 {nfp / max(T['n_absent'],1) * 100:.0f}%", fontsize=10)
            ax.set_xlabel("ego x (m)"); ax.set_ylabel("ego y (m)")
            fig2.colorbar(im, ax=ax, shrink=0.8, label="FP rate %")
    fig2.suptitle("a_2 FP-rate spatial heatmap — bin 별 (FP frame / absent frame). "
                  "밝을수록 그 구간에서 자주 오검출", fontsize=12)
    fig2.tight_layout()
    p2 = OUT_PNG.parent / "fp_rate_heatmap.png"
    fig2.savefig(str(p2), dpi=110)
    print(f"[saved] {p2}")


if __name__ == "__main__":
    main()
