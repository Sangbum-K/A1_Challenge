"""합성회전로깅데이터셋0303 검증 스크립트.

검증 항목:
  (a) msg 수 == 원본 * 12
  (b) theta=0 변형 trajectory 가 원본과 일치
  (c) theta=180 steer 가 U-turn 방향으로 크게
  (d) trajectory 가 theta 만큼 회전됐는지 수치 (rot_err)
  (e) longitudinal command / vehicle_state 가 원본과 동일
시각화 1~2장 저장.
"""
from __future__ import annotations
import os, sys, glob, math
sys.path.insert(0, "/home/ailab-12/git/scc_controller_imitator/external")
import numpy as np
import rosbag
from autohyu_msgs.msg import ControlInputOutput  # noqa

TOPIC = "/hmi/con/control_io"
OUT_DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/controller_imitator/합성회전로깅데이터셋0303"
SRC_MAP = {
    "ControlIO_2026-04-29-21-59-49_split_000__000": "/home/ailab-12/AILabDataset/02_Custom_Dataset/35_e2e_morai/controller_imitator/Raw_bag_multi/v0301_a1_f4_waypoint_driving",
    "ControlIO_2026-04-30-09-37-15___000": "/home/ailab-12/AILabDataset/02_Custom_Dataset/35_e2e_morai/controller_imitator/Raw_bag_multi/v0301_a1_f4_waypoint_driving",
    "ControlIO_2026-04-30-11-08-22": "/home/ailab-12/AILabDataset/02_Custom_Dataset/35_e2e_morai/controller_imitator/Raw_bag_multi/v0301_a1_f4_waypoint_driving",
    "ControlIO_2026-04-30-12-37-18": "/home/ailab-12/AILabDataset/02_Custom_Dataset/35_e2e_morai/controller_imitator/Raw_bag_multi/v0301_a1_f4_waypoint_driving",
    "tuned_control_dataset_2026-05-15-13-45-56": "/home/ailab-12/AILabDataset/02_Custom_Dataset/35_e2e_morai/controller_imitator/Raw_bag_multi/v0302_a1_f4_waypoint_driving",
}


def rotmat(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, s], [-s, c]])


def count_only(bag_path, topic):
    bag = rosbag.Bag(bag_path)
    info = bag.get_type_and_topic_info()
    c = info.topics[topic].message_count if topic in info.topics else 0
    bag.close()
    return c


def read_first_k(bag_path, k):
    bag = rosbag.Bag(bag_path)
    out = []
    for _, m, t in bag.read_messages(topics=[TOPIC]):
        out.append(m)
        if len(out) >= k:
            break
    bag.close()
    return out


def validate_one(out_bag, src_stem, make_plot=False, limited_src_scenes=None):
    src_dir = SRC_MAP[src_stem]
    src_bag = os.path.join(src_dir, src_stem + ".bag")

    n_out = count_only(out_bag, TOPIC)
    n_src = count_only(src_bag, TOPIC)
    eff_src = limited_src_scenes if limited_src_scenes else n_src
    print(f"\n[{os.path.basename(out_bag)}]")
    print(f"  (a) msg count: src={n_src} out={n_out}  expect={eff_src}*12={eff_src*12}  "
          f"{'OK' if n_out == eff_src * 12 else 'MISMATCH'}")

    orig = read_first_k(src_bag, 1)[0]
    oxy = np.array([[p.x, p.y] for p in orig.trajectory.point])
    msgs = read_first_k(out_bag, 12)

    max_rot_err = 0.0
    for k, m in enumerate(msgs):
        th = math.radians(30 * k)
        xy = np.array([[p.x, p.y] for p in m.trajectory.point])
        expect = oxy @ rotmat(th)
        rot_err = np.abs(xy - expect).max() if xy.size else 0.0
        max_rot_err = max(max_rot_err, rot_err)
        st = np.asarray(m.command_steer)
        speed_ok = np.allclose(np.asarray(m.command_speed), np.asarray(orig.command_speed))
        accp_ok = np.allclose(np.asarray(m.command_accel_pedal), np.asarray(orig.command_accel_pedal))
        brk_ok = np.allclose(np.asarray(m.command_brake_pedal), np.asarray(orig.command_brake_pedal))
        cacc_ok = np.allclose(np.asarray(m.command_acceleration), np.asarray(orig.command_acceleration))
        vx_ok = (m.vehicle_state.vx == orig.vehicle_state.vx)
        yaw_ok = (m.vehicle_state.yaw == orig.vehicle_state.yaw)
        print(f"  th={30*k:3d} N={len(m.trajectory.point):2d} rot_err={rot_err:.2e} "
              f"steer[deg]|max|={np.abs(st).max():6.2f} mean={st.mean():7.2f} | "
              f"spd={speed_ok} accp={accp_ok} brk={brk_ok} cacc={cacc_ok} vx={vx_ok} yaw={yaw_ok}")

    xy0 = np.array([[p.x, p.y] for p in msgs[0].trajectory.point])
    print(f"  (b) theta=0 traj == original: {np.allclose(xy0, oxy, atol=1e-4)}")
    print(f"  (d) max rot_err across 12 variants: {max_rot_err:.2e}")
    s0 = np.abs(np.asarray(msgs[0].command_steer)).max()
    s180 = np.abs(np.asarray(msgs[6].command_steer)).max()
    s180m = np.abs(np.asarray(msgs[6].command_steer)).mean()
    print(f"  (c) steer|max| theta=0:{s0:.2f}deg  theta=180:{s180:.2f}deg(mean|{s180m:.2f}) "
          f"{'OK (180 >> 0)' if s180 > s0 + 10 else 'CHECK'}")

    if make_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax = axes[0]
            cmap = plt.cm.viridis
            for k, m in enumerate(msgs):
                xy = np.array([[p.x, p.y] for p in m.trajectory.point])
                ax.plot(xy[:, 0], xy[:, 1], "-o", ms=2, color=cmap(k / 12.0),
                        label=f"{30*k}deg")
            ax.plot(0, 0, "k*", ms=14, label="ego")
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            ax.set_xlabel("x_fwd [m]"); ax.set_ylabel("y_left [m]")
            ax.set_title("Rotated trajectories (12 thetas, ego-frame)")
            ax.legend(fontsize=6, ncol=2)

            # right: forward-sim recovery path for a few thetas
            sys.path.insert(0, "/home/ailab-12/git/scc_controller_imitator/scripts")
            import gen_synth_rotation_dataset_0303 as G
            ax2 = axes[1]
            for k in [0, 1, 3, 6, 9]:
                m = msgs[k]
                P = np.array([[p.x, p.y] for p in m.trajectory.point])
                # replay forward-sim using stored steer (deg->rad)
                st = np.radians(np.asarray(m.command_steer))
                cx = cy = cyaw = 0.0
                xs, ys = [cx], [cy]
                for steer in st:
                    cx += G.SIM_V * math.cos(cyaw) * G.DT
                    cy += G.SIM_V * math.sin(cyaw) * G.DT
                    cyaw += G.SIM_V / G.WB * math.tan(steer) * G.DT
                    xs.append(cx); ys.append(cy)
                ax2.plot(P[:, 0], P[:, 1], "--", color=cmap(k / 12.0), alpha=0.5)
                ax2.plot(xs, ys, "-", color=cmap(k / 12.0), lw=2, label=f"{30*k}deg recover")
            ax2.plot(0, 0, "k*", ms=14)
            ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
            ax2.set_xlabel("x_fwd [m]"); ax2.set_ylabel("y_left [m]")
            ax2.set_title("Stanley recovery (solid) vs rotated path (dashed)")
            ax2.legend(fontsize=7)
            fig.suptitle(f"synth_rot_0303 scene0  {src_stem}")
            fig.tight_layout()
            outpng = os.path.join(OUT_DIR, "validation_viz.png")
            fig.savefig(outpng, dpi=110)
            print(f"  viz saved: {outpng}")
        except Exception as e:
            print(f"  viz failed: {e}")


def main():
    bags = sorted(glob.glob(os.path.join(OUT_DIR, "synth_rot_0303__*.bag")))
    print(f"found {len(bags)} output bags")
    first = True
    for b in bags:
        stem = os.path.basename(b).replace("synth_rot_0303__", "").replace(".bag", "")
        lim = None
        if len(sys.argv) > 1:
            lim = int(sys.argv[1])  # limited_src_scenes for debug bags
        validate_one(b, stem, make_plot=first, limited_src_scenes=lim)
        first = False


if __name__ == "__main__":
    main()
