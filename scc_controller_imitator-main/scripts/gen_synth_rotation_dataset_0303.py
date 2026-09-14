"""합성회전로깅데이터셋0303 generator.

목적: imitation 제어기(main 모델)가 waypoint 경로에서 heading 이 크게 틀어졌을 때
복귀 steer 를 못 내는 약점 보완. 각 원본 scene 을 yaw 12 방향(0,30,...,330deg)으로
회전시키고, 회전된 경로로 복귀하는 steer 를 Stanley 로 재생성해 합성 scene 12개 생성.

합성으로 바꾸는 건 딱 둘:
  (1) trajectory.point 를 R(theta) 회전 (repo global_to_local 과 동일 form, 각도 = +theta)
  (2) command_steer 를 Stanley(kinematic bicycle forward sim)로 재생성
그 외 모든 필드/헤더/vehicle_state/longitudinal command 는 원본 그대로 복사.

회전 convention (repo ctrl_io_bag_preprocess ... global_to_local 과 일관):
  rotation_matrix(psi) = [[cos psi, sin psi], [-sin psi, cos psi]]  (row-vector 우측곱)
  ego heading 을 +theta 회전 -> ego-frame 경로점은 R(theta) 로 회전:
    p' = p @ rotation_matrix(theta)
  (theta=0 이면 항등 -> 원본 경로 그대로.)

steer 단위: 원본 command_steer 는 tire angle [deg] (lat_traj_unit memory, |.|<=25deg 관측).
  Stanley 내부 계산은 rad(clamp +-0.7rad), 저장 시 deg 로 변환해 원본 필드와 단위 일치.

속도 5 m/s 가정은 Stanley/forward-sim 내부값일 뿐 저장하지 않는다.
"""
from __future__ import annotations
import os, sys, glob, math, time, copy, argparse

sys.path.insert(0, "/home/ailab-12/git/scc_controller_imitator/external")

import numpy as np
import rosbag
from autohyu_msgs.msg import ControlInputOutput  # noqa: F401  (registers type for rosbag)

# ----------------------------------------------------------------------------
TOPIC = "/hmi/con/control_io"
N_ROT = 12                       # 30deg 간격 360deg
THETAS = [math.radians(30 * k) for k in range(N_ROT)]   # 0,30,...,330 deg (rad)

# Stanley / forward-sim constants
STANLEY_K = 1.0
SIM_V = 5.0                      # m/s (내부 가정값, 저장 안 함)
WB = 2.35                        # wheelbase [m]
DT = 0.1                         # forward-sim step [s]
STEER_CLAMP_RAD = 0.7            # rad
RAD2DEG = 180.0 / math.pi

SRC_DIRS = [
    "/home/ailab-12/AILabDataset/02_Custom_Dataset/35_e2e_morai/controller_imitator/Raw_bag_multi/v0301_a1_f4_waypoint_driving",
    "/home/ailab-12/AILabDataset/02_Custom_Dataset/35_e2e_morai/controller_imitator/Raw_bag_multi/v0302_a1_f4_waypoint_driving",
]
OUT_DIR = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/controller_imitator/합성회전로깅데이터셋0303"


# ----------------------------------------------------------------------------
def rot_matrix(psi: float) -> np.ndarray:
    """repo global_to_local 과 동일 form (row-vector 우측곱용)."""
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def rotate_points(xy: np.ndarray, theta: float) -> np.ndarray:
    """ego-frame 경로점 (N,2) 를 R(theta) 회전. p' = p @ rot_matrix(theta)."""
    return xy @ rot_matrix(theta)


def stanley_steer_sequence(traj_xy: np.ndarray, n_steps: int) -> np.ndarray:
    """회전된 경로 traj_xy (N,2, ego-frame x_fwd/y_left)로 복귀하는 steer 시퀀스 생성.

    kinematic bicycle 로 5 m/s 정속 전방 simulation, DT step, n_steps 반복.
    각 step Stanley steer = heading_error + atan2(k*cte, v), clamp +-0.7 rad.
    반환: steer [rad] array, 길이 n_steps.
    """
    # 경로를 polyline 으로. ego 초기 pose = 원점, heading=0 (ego frame 기준).
    P = traj_xy.astype(np.float64)
    if P.shape[0] < 2:
        return np.zeros(n_steps, dtype=np.float64)

    # 차량 상태 (ego frame): x,y,yaw. 시작 = 원점, heading 0.
    cx, cy, cyaw = 0.0, 0.0, 0.0
    steers = np.zeros(n_steps, dtype=np.float64)

    # 경로 segment 의 heading 미리 계산
    seg = P[1:] - P[:-1]
    seg_yaw = np.arctan2(seg[:, 1], seg[:, 0])     # (N-1,)

    for step in range(n_steps):
        # front axle 위치
        fx = cx + WB * math.cos(cyaw)
        fy = cy + WB * math.sin(cyaw)

        # 가장 가까운 경로점 index (front axle 기준)
        d2 = (P[:, 0] - fx) ** 2 + (P[:, 1] - fy) ** 2
        nearest = int(np.argmin(d2))
        # path heading at nearest segment
        seg_idx = min(nearest, seg_yaw.shape[0] - 1)
        path_yaw = seg_yaw[seg_idx]

        # cross-track error: front axle 에서 가장 가까운 점으로의 vector 를
        # path 의 좌측 법선에 투영 (부호 포함)
        dx = P[nearest, 0] - fx
        dy = P[nearest, 1] - fy
        # path 좌측 법선 = (-sin(path_yaw), cos(path_yaw))
        cte = dx * (-math.sin(path_yaw)) + dy * math.cos(path_yaw)

        # heading error (path - vehicle), [-pi,pi] wrap
        he = math.atan2(math.sin(path_yaw - cyaw), math.cos(path_yaw - cyaw))

        steer = he + math.atan2(STANLEY_K * cte, SIM_V)
        steer = max(-STEER_CLAMP_RAD, min(STEER_CLAMP_RAD, steer))
        if not math.isfinite(steer):
            steer = 0.0
        steers[step] = steer

        # kinematic bicycle 전진 (rear-axle ref)
        cx += SIM_V * math.cos(cyaw) * DT
        cy += SIM_V * math.sin(cyaw) * DT
        cyaw += SIM_V / WB * math.tan(steer) * DT
        cyaw = math.atan2(math.sin(cyaw), math.cos(cyaw))

    return steers


def make_rotated_msg(msg, theta: float):
    """원본 msg -> theta 회전 + Stanley steer 재생성한 deep-copy 반환.

    바꾸는 것: trajectory.point[].x/.y (R(theta)), command_steer (Stanley, deg).
    나머지(헤더, vehicle_state, solved_trajectory, longitudinal command, point.yaw 등)는 원본.
    """
    new = copy.deepcopy(msg)

    pts = new.trajectory.point
    N = len(pts)
    if N == 0:
        return new  # 빈 경로는 그대로 (steer 도 빈 채로 둠)

    xy = np.array([[p.x, p.y] for p in pts], dtype=np.float64)
    rot = rotate_points(xy, theta)
    for i, p in enumerate(pts):
        p.x = float(rot[i, 0])
        p.y = float(rot[i, 1])
        # p.yaw 등 나머지 point 필드는 원본 유지 (요구사항: traj x/y 만 회전)

    # Stanley steer 재생성 — 출력 길이 = 원본 command_steer 길이
    raw_len = len(msg.command_steer)
    if raw_len == 0:
        raw_len = N
    steers_rad = stanley_steer_sequence(rot, raw_len)
    steers_deg = (steers_rad * RAD2DEG).astype(np.float32)
    # rosbag serialize 는 list/tuple of float 기대 (float32[])
    new.command_steer = [float(s) for s in steers_deg]

    # longitudinal / vehicle_state / solved_trajectory 등은 deepcopy 로 원본 그대로 유지됨
    return new


def process_bag(src_bag: str, out_bag: str, limit: int | None = None):
    print(f"\n=== {os.path.basename(src_bag)} -> {os.path.basename(out_bag)} ===", flush=True)
    t0 = time.time()
    bag = rosbag.Bag(src_bag)
    info = bag.get_type_and_topic_info()
    total = info.topics[TOPIC].message_count if TOPIC in info.topics else 0
    print(f"  scenes={total} open={time.time()-t0:.1f}s", flush=True)

    n_in = 0
    n_out = 0
    with rosbag.Bag(out_bag, "w", compression="lz4") as ob:
        for topic, msg, t in bag.read_messages(topics=[TOPIC]):
            n_in += 1
            for theta in THETAS:
                rmsg = make_rotated_msg(msg, theta)
                # 원본 timestamp 유지 (12개가 같은 t 를 공유; scene 단위라 무방)
                ob.write(TOPIC, rmsg, t)
                n_out += 1
            if limit and n_in >= limit:
                break
            if n_in % 1000 == 0:
                el = time.time() - t0
                rate = n_in / el if el > 0 else 0
                print(f"  ... {n_in}/{total} scenes ({rate:.0f}/s), {n_out} out msgs, {el:.0f}s",
                      flush=True)
    bag.close()
    sz = os.path.getsize(out_bag) / 1e6
    print(f"  DONE in={n_in} out={n_out} size={sz:.1f}MB time={time.time()-t0:.0f}s", flush=True)
    return n_in, n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="scene limit per bag (debug)")
    ap.add_argument("--only", type=str, default=None,
                    help="substring filter on bag basename")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    src_bags = []
    for d in SRC_DIRS:
        src_bags.extend(sorted(glob.glob(os.path.join(d, "*.bag"))))
    if args.only:
        src_bags = [b for b in src_bags if args.only in os.path.basename(b)]
    print(f"found {len(src_bags)} source bags -> out {OUT_DIR}", flush=True)

    grand_in = grand_out = 0
    for b in src_bags:
        out_name = "synth_rot_0303__" + os.path.splitext(os.path.basename(b))[0] + ".bag"
        out_bag = os.path.join(OUT_DIR, out_name)
        ni, no = process_bag(b, out_bag, limit=args.limit)
        grand_in += ni
        grand_out += no
    print(f"\n==== ALL DONE: in={grand_in} scenes, out={grand_out} msgs (x{N_ROT}) ====", flush=True)


if __name__ == "__main__":
    main()
