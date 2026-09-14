#!/usr/bin/env python
"""SCC 5-profile f_2 학습 chain (python wrapper — bash script 가 sandbox 에서 죽어 직접 python).
순서: p3(중간) → p5(보수) → p2 → p4 → p1(공격). gate=2*sigmoid(obj+vrel).
각 profile 독립 (distbrake warm-start). pedal_label_<ds>_p{N} (inlane_gt 물리감속 A + min base)."""
import subprocess, sys, time
ROOT = "/home/ailab-12/git/scc_controller_imitator"
PY = "/home/ailab-12/miniforge3/envs/ml_con/bin/python"
ORDER = ["p3", "p5", "p2", "p4", "p1"]
for p in ORDER:
    print(f"===== profile {p} 시작 {time.strftime('%H:%M:%S')} =====", flush=True)
    r = subprocess.run([
        PY, "-u", "scripts/train_prof635_side.py",
        "--gate-bce", "--gate-bce-keep-vrel", "--vrel-from-a2logits", "--gate-tau", "0.5",
        "--resume", "data/ckpts/prof635_side_distbrake_v1.pt",
        "--delta-mode", "brake_only", "--randomize-base", "--epochs", "25",
        "--pedal-suffix", p, "--name", f"prof635_side_scc_{p}_g2",
    ], cwd=ROOT)
    print(f"===== profile {p} 완료 rc={r.returncode} {time.strftime('%H:%M:%S')} =====", flush=True)
    # hook: 학습 성공 시 테스트용 weight publish + push (multicar prefix=scc_p{N}_f654/f656).
    if r.returncode == 0:
        pr = subprocess.run(["bash", "scripts/publish_test_weights.sh", p], cwd=ROOT)
        print(f"===== profile {p} publish rc={pr.returncode} {time.strftime('%H:%M:%S')} =====", flush=True)
print("===== ALL 5 PROFILE (p3,p5,p2,p4,p1) DONE =====", flush=True)
