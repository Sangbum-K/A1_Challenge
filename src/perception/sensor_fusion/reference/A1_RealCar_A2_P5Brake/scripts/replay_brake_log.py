#!/usr/bin/env python3
"""Re-run 식 0~3 over a recorded frames.jsonl (needs the full A₂ log, a2_bins.raw_40x7).

Works on old A1_RealCamera_A2_21FPS live logs and on this project's live logs,
with numpy only (matplotlib is optional, for the PNG).  The model is not run;
the recorded A₂ numbers are fed to the brake chain exactly as the live runner
would, using the camera timestamps for ḋ.

    python3 scripts/replay_brake_log.py outputs/<run>/frames.jsonl
    python3 scripts/replay_brake_log.py <log> --plan-kph 30 --vrel-mode a2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from a1_brake.brake import BrakeController, config_from_dict, config_to_dict  # noqa: E402
from a1_brake.tracking import (IN_LANE_THETA, TrackerConfig, VehicleTracker,  # noqa: E402
                               measurements_from_a2)


def load_config(path: Path, overrides: dict):
    import yaml  # noqa: PLC0415  (system python and .venv both have it)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return config_from_dict(data)


def iter_frames(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            bins = record.get("a2_bins")
            if not bins or not record.get("finite", True):
                continue
            yield record, np.asarray(bins["raw_40x7"], dtype=np.float64)


def stats(values):
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return None
    return {"n": int(v.size), "p10": float(np.percentile(v, 10)),
            "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "min": float(v.min()), "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, help="frames.jsonl with a2_bins (live --log-full-a2)")
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "brake_p5.yaml")
    ap.add_argument("--ego-speed", type=float, default=0.0, help="actual ego speed m/s")
    ap.add_argument("--plan-kph", type=float, default=None,
                    help="P5 planning speed km/h (default = actual ego speed)")
    ap.add_argument("--vrel-mode", choices=["track", "min", "a2", "ddot"], default=None)
    ap.add_argument("--obj-thr", type=float, default=0.5, help="tracking object threshold")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <log dir>/brake_replay")
    args = ap.parse_args()

    config = load_config(args.config, {"vrel_mode": args.vrel_mode})
    controller = BrakeController(config)
    tracker = VehicleTracker(TrackerConfig(obj_threshold=args.obj_thr))
    meas_xy, lead_xy, all_tracks = [], [], {}
    plan = None if args.plan_kph is None else args.plan_kph / 3.6
    out_dir = args.out or args.frames.parent / "brake_replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    kphs = list(config.what_if_kph)
    header = ["t_s", "seq", "bin", "probability", "score_logit", "distance_m",
              "v_rel_a2_mps", "ddot_mps", "v_rel_used_mps", "v_rel_source", "lead_speed_mps",
              "gate_raw", "gate", "plan_speed_mps", "a_p5_mps2", "a_final_mps2",
              "acc_cmd_mps2", "rel_speed_at_plan_mps", "track_lead_id", "track_lead_x_m",
              "track_lead_y_m", "track_lead_vx_mps", "track_lead_vy_mps",
              "tracks_confirmed"] + [f"acc_cmd_{k:g}kph" for k in kphs]
    rows = []
    t0 = None
    for record, raw in iter_frames(args.frames):
        t = float(record["source_timestamp"])
        t0 = t if t0 is None else t0
        measurements = measurements_from_a2(raw, args.obj_thr)
        track = tracker.update(t, measurements)
        lead_vy = track["lead_track"]["vy_mps"] if track["lead_track"] else None
        out = controller.step(raw, t, args.ego_speed, plan, v_rel_track_mps=lead_vy)
        lead, main = out["lead"], out["main"]
        meas_xy.extend((m.x_m, m.y_m) for m in measurements)
        for tr in track["tracks"]:
            all_tracks.setdefault(tr["id"], []).append((tr["x_m"], tr["y_m"]))
        lt = track["lead_track"]
        if lt:
            lead_xy.append((t - t0, lt["x_m"], lt["y_m"]))
        rows.append([
            t - t0, record.get("seq"), lead["bin"], lead["probability"], lead["score_logit"],
            lead["distance_m"], lead["v_rel_a2_mps"], lead["ddot_mps"], lead["v_rel_used_mps"],
            lead["v_rel_source"], lead["lead_speed_mps"], out["gate"]["raw"],
            out["gate"]["value"], main["plan_speed_mps"], main["a_p5_mps2"],
            main["a_final_mps2"], main["acc_cmd_mps2"], main["lead_rel_speed_at_plan_mps"],
            lt["id"] if lt else None, lt["x_m"] if lt else None, lt["y_m"] if lt else None,
            lt["vx_mps"] if lt else None, lt["vy_mps"] if lt else None, len(track["tracks"]),
        ] + [row["acc_cmd_mps2"] for row in out["what_if"]])
    if not rows:
        raise SystemExit(f"no frame with a2_bins.raw_40x7 in {args.frames} "
                         "(that run was recorded without --log-full-a2)")

    csv_path = out_dir / "brake_replay.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink)
        writer.writerow(header)
        writer.writerows(rows)

    col = {name: [r[i] for r in rows] for i, name in enumerate(header)}
    gate = np.asarray(col["gate"], dtype=np.float64)
    on = gate >= 0.5
    flips = int(np.count_nonzero(on[1:] != on[:-1]))
    duration = col["t_s"][-1] if len(rows) > 1 else 0.0
    gated = [i for i in range(len(rows)) if on[i]]

    def pick(name, idx):
        return [col[name][i] for i in idx]

    summary = {
        "frames_log": str(args.frames),
        "frames": len(rows),
        "duration_s": duration,
        "ego_speed_mps": args.ego_speed,
        "plan_speed_mps": plan if plan is not None else args.ego_speed,
        "config": config_to_dict(config),
        "gate": {
            "on_ratio_ge_0p5": float(on.mean()),
            "flips_per_minute": flips / duration * 60.0 if duration > 0 else None,
            "value": stats(col["gate"]),
        },
        "while_gate_on": {
            "distance_m": stats(pick("distance_m", gated)),
            "v_rel_a2_mps": stats(pick("v_rel_a2_mps", gated)),
            "ddot_mps": stats(pick("ddot_mps", gated)),
            "v_rel_used_mps": stats(pick("v_rel_used_mps", gated)),
            "lead_speed_mps": stats(pick("lead_speed_mps", gated)),
            "v_rel_source_ddot_ratio": (
                sum(1 for s in pick("v_rel_source", gated) if s == "ddot") / len(gated)
                if gated else None),
        },
        "tracking": {
            "lead_track_frame_ratio": sum(1 for v in col["track_lead_id"] if v is not None) / len(rows),
            "distinct_lead_track_ids": len({v for v in col["track_lead_id"] if v is not None}),
            "lead_track_y_m": stats(col["track_lead_y_m"]),
            "lead_track_x_m": stats(col["track_lead_x_m"]),
            "lead_track_vy_mps": stats(col["track_lead_vy_mps"]),
            "confirmed_tracks_per_frame_mean": float(np.mean(col["tracks_confirmed"])),
        },
        "acc_cmd_mps2": {
            "main": stats(col["acc_cmd_mps2"]),
            **{f"what_if_{k:g}kph": stats(col[f"acc_cmd_{k:g}kph"]) for k in kphs},
        },
        "brake_ratio_le_minus_0p5": {
            "main": float(np.mean(np.asarray(col["acc_cmd_mps2"]) <= -0.5)),
            **{f"what_if_{k:g}kph": float(np.mean(np.asarray(col[f"acc_cmd_{k:g}kph"]) <= -0.5))
               for k in kphs},
        },
        "note": ("Replay of recorded A2 numbers. Distance/velocity accuracy is not validated; "
                 "a brake number here says what the rule would request, not that it is right."),
    }
    (out_dir / "brake_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("[plot] matplotlib unavailable; CSV/summary only (run with system python3 for PNG)")
    else:
        t = np.asarray(col["t_s"], dtype=np.float64)

        def arr(name):
            return np.asarray([np.nan if v is None else v for v in col[name]], dtype=np.float64)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(t, arr("distance_m"), lw=0.8, label="d (A2 ch6×40)")
        axes[0].set_ylabel("m"); axes[0].legend(loc="upper right")
        axes[1].plot(t, arr("v_rel_a2_mps"), lw=0.8, label="v_rel A2 (ch3×20)")
        axes[1].plot(t, arr("ddot_mps"), lw=0.8, label="ḋ (1 s LS)")
        axes[1].plot(t, arr("track_lead_vy_mps"), lw=0.8, label="track vy")
        axes[1].plot(t, arr("v_rel_used_mps"), lw=1.2, label="v_rel used")
        axes[1].axhline(0, color="k", lw=0.5); axes[1].set_ylabel("m/s")
        axes[1].legend(loc="upper right")
        axes[2].plot(t, gate, lw=1.0, label="gate g")
        axes[2].plot(t, arr("probability"), lw=0.6, alpha=0.6, label="σ(logit k*)")
        axes[2].set_ylim(-0.05, 1.05); axes[2].legend(loc="upper right")
        for k in kphs:
            axes[3].plot(t, arr(f"acc_cmd_{k:g}kph"), lw=0.8, label=f"what-if {k:g} km/h")
        axes[3].plot(t, arr("acc_cmd_mps2"), "k", lw=1.4, label="main")
        axes[3].set_ylabel("ACC_Cmd m/s²"); axes[3].set_xlabel("t [s]")
        axes[3].legend(loc="lower right", ncol=3, fontsize=8)
        fig.suptitle(f"brake replay: {args.frames.parent.name}")
        fig.tight_layout()
        fig.savefig(out_dir / "brake_replay.png", dpi=110)
        plt.close(fig)
        print(f"[saved] {out_dir / 'brake_replay.png'}")

        fig, ax = plt.subplots(figsize=(7, 9))
        if meas_xy:
            mx, my = zip(*meas_xy)
            ax.scatter(mx, my, s=2, c="0.75", label="A2 object groups")
        for tid, pts in all_tracks.items():
            tx, ty = zip(*pts)
            ax.plot(tx, ty, lw=0.6, alpha=0.6)
        if lead_xy:
            lt_t, lx, ly = zip(*lead_xy)
            sc = ax.scatter(lx, ly, s=6, c=lt_t, cmap="viridis", label="tracked lead")
            fig.colorbar(sc, ax=ax, label="t [s]")
        for theta in IN_LANE_THETA:
            ax.plot([0, 60 * np.tan(theta)], [0, 60], "y--", lw=0.8)
        ax.plot([0], [0], "g^", ms=10)
        ax.set_xlim(-15, 15); ax.set_ylim(0, 60); ax.set_aspect("equal")
        ax.set_xlabel("lateral x [m] (+ right)"); ax.set_ylabel("longitudinal y [m]")
        ax.set_title("vehicle tracks (A2 training geometry fx=640, uncalibrated)")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "tracks_bev.png", dpi=110)
        plt.close(fig)
        print(f"[saved] {out_dir / 'tracks_bev.png'}")
    print(json.dumps({k: summary[k] for k in ("frames", "gate", "while_gate_on", "tracking",
                                              "brake_ratio_le_minus_0p5")},
                     ensure_ascii=False, indent=1))
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
