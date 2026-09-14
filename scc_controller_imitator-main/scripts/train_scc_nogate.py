#!/usr/bin/env python
"""gate 없는 버전(no_gate + delta raw) p5, p1 학습 chain + 자동 publish.
- p3 는 별도 단일 학습(이미 진행 중, scc_p3_ng.pt 샘플 게시됨). 여기는 p5, p1.
- 각 profile: scc_{p}_side.pt(gate 버전) warm-start(pedal_head 제외 raw 새 init) → final=base+delta.
  완료 시 best.pt → data/ckpts/scc_{p}_ng.pt 복사 + git push (multicar prefix=scc_{p}_ng_f654/f656)."""
import subprocess, os, time
ROOT = "/home/ailab-12/git/scc_controller_imitator"
PY = "/home/ailab-12/miniforge3/envs/ml_con/bin/python"
for p in ["p5", "p1"]:
    print(f"===== nogate {p} 시작 {time.strftime('%H:%M:%S')} =====", flush=True)
    r = subprocess.run([
        PY, "-u", "scripts/train_prof635_side.py", "--no-gate", "--delta-mode", "raw",
        "--resume", f"data/ckpts/scc_{p}_side.pt", "--randomize-base", "--epochs", "15",
        "--pedal-suffix", p, "--name", f"prof635_side_scc_{p}_nogate",
    ], cwd=ROOT)
    print(f"===== nogate {p} 완료 rc={r.returncode} {time.strftime('%H:%M:%S')} =====", flush=True)
    if r.returncode == 0:
        bp = f"{ROOT}/outputs_multi/prof635_side_scc_{p}_nogate/best.pt"
        if os.path.exists(bp):
            subprocess.run(["cp", "-f", bp, f"{ROOT}/data/ckpts/scc_{p}_ng.pt"])
            subprocess.run(["git", "add", "-f", f"data/ckpts/scc_{p}_ng.pt"], cwd=ROOT)
            subprocess.run(["git", "commit", "-q", "-m",
                            f"ckpt(test): scc_{p}_ng (gate 없는 버전, no_gate+raw delta) publish"], cwd=ROOT)
            subprocess.run(["git", "push", "origin", "acc-test"], cwd=ROOT)
            print(f"===== nogate {p} publish+push (scc_{p}_ng.pt) =====", flush=True)
print("===== nogate p5,p1 ALL DONE =====", flush=True)
