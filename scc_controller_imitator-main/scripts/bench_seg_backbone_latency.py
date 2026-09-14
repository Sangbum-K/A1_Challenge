"""seg backbone 경량화 latency 벤치마크.

두 측정 대상:
  (A) full SegPretrainModel  (encoder + U-Net decoder) — seg pretrain / mIoU 학습 대상
  (B) i_1 token path         (features_only out_indices[-1] → adaptive_avg_pool2d(1,40))
                              — closed_loop_node 실시간성 병목 (매 프레임 i_1 토큰 추출)

입력: (1, 3, 250, 1280). fp32 와 fp16(amp autocast) 둘 다.
warmup 20 + 측정 100 iter, CUDA event 기반 GPU latency.

사용:
  ~/miniforge3/envs/ml_con/bin/python scripts/bench_seg_backbone_latency.py
"""
from __future__ import annotations
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
import torch
import timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model.image_models.seg_pretrain_model import SegPretrainModel

DEVICE = "cuda"
H, W = 250, 1280
WARMUP = 20
ITERS = 100

CANDIDATES = [
    "tf_efficientnetv2_s",   # baseline
    "tf_efficientnetv2_b0",
    "tf_efficientnetv2_b1",
    "tf_efficientnetv2_b2",
    "tf_efficientnetv2_b3",
    "mobilenetv3_large_100",
    "tf_efficientnet_lite0",
]


def _try_flops(model, x):
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9   # GFLOPs
    except Exception:
        return None


@torch.no_grad()
def bench(fn, x, amp: bool) -> float:
    """fn(x) 를 측정. CUDA event ms 평균 반환."""
    # warmup
    for _ in range(WARMUP):
        with torch.amp.autocast("cuda", enabled=amp):
            fn(x)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        with torch.amp.autocast("cuda", enabled=amp):
            fn(x)
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    times.sort()
    # median + mean
    return sum(times) / len(times), times[len(times) // 2]


def i1_token_path_factory(backbone_name: str, seg_ckpt: str | None = None):
    """closed_loop_node 와 동일: features_only out_indices[-1] backbone.
    forward: backbone(x)[-1] → adaptive_avg_pool2d(1,40) → squeeze/transpose."""
    bb = timm.create_model(backbone_name, pretrained=False, features_only=True, out_indices=[-1])
    bb = bb.to(DEVICE).eval()
    for p in bb.parameters():
        p.requires_grad_(False)
    c_b = bb.feature_info.channels()[-1]

    def fwd(x):
        fc = bb(x)[-1]
        fc = torch.nn.functional.adaptive_avg_pool2d(fc, (1, 40)).squeeze(2).transpose(1, 2)
        return fc

    n_params = sum(p.numel() for p in bb.parameters())
    return fwd, bb, n_params, c_b


def main():
    print(f"# device={torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    print(f"# input=(1,3,{H},{W})  warmup={WARMUP}  iters={ITERS}")
    torch.backends.cudnn.benchmark = True

    x = torch.randn(1, 3, H, W, device=DEVICE)

    results = []
    baseline_full_fp32 = None
    baseline_tok_fp32 = None

    for name in CANDIDATES:
        row = {"backbone": name}
        # ---- (B) i_1 token path ----
        tok_fwd, tok_bb, tok_params, c_b = i1_token_path_factory(name)
        row["tok_params_M"] = tok_params / 1e6
        row["last_ch"] = c_b
        m_fp32, _ = bench(tok_fwd, x, amp=False)
        m_fp16, _ = bench(tok_fwd, x, amp=True)
        row["tok_fp32_ms"] = m_fp32
        row["tok_fp16_ms"] = m_fp16
        row["tok_gflops"] = _try_flops(tok_bb, x)
        del tok_bb
        torch.cuda.empty_cache()

        # ---- (A) full SegPretrainModel ----
        model = SegPretrainModel(backbone=name, n_classes=4).to(DEVICE).eval()
        full_params = sum(p.numel() for p in model.parameters())
        row["full_params_M"] = full_params / 1e6
        f_fp32, _ = bench(lambda z: model(z), x, amp=False)
        f_fp16, _ = bench(lambda z: model(z), x, amp=True)
        row["full_fp32_ms"] = f_fp32
        row["full_fp16_ms"] = f_fp16
        row["full_gflops"] = _try_flops(model, x)
        del model
        torch.cuda.empty_cache()

        if name == "tf_efficientnetv2_s":
            baseline_full_fp32 = f_fp32
            baseline_tok_fp32 = m_fp32
        results.append(row)
        print(f"[done] {name}")

    # speedup vs baseline
    for r in results:
        r["full_speedup"] = baseline_full_fp32 / r["full_fp32_ms"]
        r["tok_speedup"]  = baseline_tok_fp32 / r["tok_fp32_ms"]

    print("\n=== i_1 TOKEN PATH (closed_loop bottleneck: features[-1] + pool(1,40)) ===")
    hdr = f"{'backbone':24s} {'params(M)':>9s} {'last_ch':>7s} {'fp32(ms)':>9s} {'fp16(ms)':>9s} {'GFLOPs':>7s} {'vs_v2s':>7s}"
    print(hdr)
    for r in results:
        gf = f"{r['tok_gflops']:.2f}" if r['tok_gflops'] else "  -  "
        print(f"{r['backbone']:24s} {r['tok_params_M']:9.2f} {r['last_ch']:7d} "
              f"{r['tok_fp32_ms']:9.3f} {r['tok_fp16_ms']:9.3f} {gf:>7s} {r['tok_speedup']:6.2f}x")

    print("\n=== FULL SegPretrainModel (encoder + U-Net decoder; mIoU training target) ===")
    hdr = f"{'backbone':24s} {'params(M)':>9s} {'fp32(ms)':>9s} {'fp16(ms)':>9s} {'GFLOPs':>7s} {'vs_v2s':>7s}"
    print(hdr)
    for r in results:
        gf = f"{r['full_gflops']:.2f}" if r['full_gflops'] else "  -  "
        print(f"{r['backbone']:24s} {r['full_params_M']:9.2f} "
              f"{r['full_fp32_ms']:9.3f} {r['full_fp16_ms']:9.3f} {gf:>7s} {r['full_speedup']:6.2f}x")

    with open(os.path.join(ROOT, "outputs_multi", "seg_backbone_latency.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] outputs_multi/seg_backbone_latency.json")


if __name__ == "__main__":
    main()
