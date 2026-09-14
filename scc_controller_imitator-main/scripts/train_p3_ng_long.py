#!/usr/bin/env python
"""nogate p5,p1 chain 완료 후 → p3_ng 를 decaying-max cosine warm restarts lr 로 길게 학습(최종, 끝까지 한번).
- warm: scc_p3_ng.pt(현 best, raw head 유지 — _ng 라 resume:raw 가 head 안 버림).
- cosine: T0=20, Tmult=2, gamma=0.6 → 주기 20/40/80 = 140ep, restart 마다 max lr ×0.6.
- 완료 시 best → scc_p3_ng.pt 최종 갱신 + push."""
import subprocess, os, time
ROOT = "/home/ailab-12/git/scc_controller_imitator"
PY = "/home/ailab-12/miniforge3/envs/ml_con/bin/python"
P1NG = f"{ROOT}/data/ckpts/scc_p1_ng.pt"

print(f"[p3_long] p3_ng cosine 길게 학습 시작 (p1 생략 — 사용자 결정) {time.strftime('%H:%M:%S')}", flush=True)

r = subprocess.run([
    PY, "-u", "scripts/train_prof635_side.py", "--no-gate", "--delta-mode", "raw",
    "--resume", "data/ckpts/scc_p3_ng.pt", "--randomize-base",
    "--cosine-restarts", "--cos-t0", "20", "--cos-tmult", "2", "--cos-gamma", "0.6",
    "--epochs", "140", "--lr", "3e-4", "--pedal-suffix", "p3", "--name", "prof635_side_scc_p3_ng_long",
], cwd=ROOT)
print(f"[p3_long] 완료 rc={r.returncode} {time.strftime('%H:%M:%S')}", flush=True)

if r.returncode == 0:
    bp = f"{ROOT}/outputs_multi/prof635_side_scc_p3_ng_long/best.pt"
    if os.path.exists(bp):
        subprocess.run(["cp", "-f", bp, f"{ROOT}/data/ckpts/scc_p3_ng.pt"])
        subprocess.run(["git", "add", "-f", "data/ckpts/scc_p3_ng.pt"], cwd=ROOT)
        subprocess.run(["git", "commit", "-q", "-m",
                        "ckpt(test): scc_p3_ng 최종 (cosine restarts 140ep 길게) 갱신"], cwd=ROOT)
        subprocess.run(["git", "push", "origin", "acc-test"], cwd=ROOT)
        print("[p3_long] scc_p3_ng.pt 최종 갱신 push", flush=True)
print("[p3_long] DONE", flush=True)
