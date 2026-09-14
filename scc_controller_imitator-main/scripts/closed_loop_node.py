#!/usr/bin/env python3
"""f_1 closed-loop ROS test node.

Subscribers:
  /app/perc/camera_1            sensor_msgs/Image   (1280x720 RGB)
  /app/loc/vechicle_state       autohyu_msgs/VehicleState

Publishers:
  /app/con/command_steer        std_msgs/Float32 (deg)
  /app/con/command_pedal_accel  std_msgs/Float32 (0..1)
  /app/con/command_pedal_brake  std_msgs/Float32 (0..1)
  /app/con/ml_scc_control_io    autohyu_msgs/ControlInputOutput

Flow:
  image_cb / ego_cb 도착 → 둘 다 있으면 inference
    1) trajectory: raceline csv 에서 ego 최근접점 → vx 기반 등시간격 3s sampling
       (TRAJ_DT × TRAJ_N = 3s, ego frame 변환 후 [x, y, speed])
    2) 이미지 ROI crop (250x1280) + Δyaw shift (prev) + ImageNet norm
    3) ego_5d = [vx, vy, ax, ay, yaw_rate]
    4) SCCControllerImitatorV2 forward → (1, 40, 2) [steer_deg, ax_mps2]
       instant cmd = pred[0, 0, :]
    5) accel/brake pedal: PedalMap.get_pedal_values(vx, ax)
    6) publish 4 topics

DEBUG_VIZ: matplotlib live window (BEV traj + cropped image + steer/accel 시계열)
"""
from __future__ import annotations

# ============================================================================
# MODEL CONFIG (코드 상단 상수 — 모델/가중치 변경 시 여기 수정)
# ============================================================================
import os   # 상단 상수에서 os.environ/REPO_ROOT 사용 → 메인 import 블록(아래)보다 먼저 정의 필요
# 머신 무관 경로: 이 파일(scripts/closed_loop_node.py) 기준 repo root 도출 (하드코딩 /home/ailab-12 제거).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_TYPE: "v2" (legacy) | "iy" (i_y refactor) | "prof635" (#654 frozen main + f_2 residual gated side, v5)
MODEL_TYPE       = os.environ.get("MODEL_TYPE", "prof635")
MODEL_NAME       = "iy_spd_streaming_w20"

# v25h variant toggle (환경변수 MODEL_VARIANT=A 또는 B 로 swap)
# A = image-aware self-attn (main 이 image_mem 도 봄)
# B = Professor main + Side SCC (cross-attn 4L, ax 만 보강) — steer 정통 보존
import os as _os
MODEL_VARIANT = _os.environ.get("MODEL_VARIANT", "A").upper()
print(f"[closed_loop] MODEL_VARIANT = {MODEL_VARIANT}")
# v2 ckpt (legacy)
MODEL_CKPT_V2    = f"{REPO_ROOT}/data/ckpts/f1_cached_v2_k500p_5cyc_safe_best.pt"
# iy ckpt (streaming train — best_loss.pt 갱신될 때마다 자동 반영)
if MODEL_VARIANT == "C":
    # v25c2: f_1-only (main_no_image, side 없음) — Professor #654 → v0301 단독 finetune. image 미사용.
    MODEL_CKPT_IY    = f"{REPO_ROOT}/data/ckpts/iy_v25c2_main_v0301only_best_loss.pt"
elif MODEL_VARIANT == "B":
    MODEL_CKPT_IY    = f"{REPO_ROOT}/data/ckpts/iy_v25k_B4_gttc_accgate_best_loss.pt"  # 2026-05-25 v25k-B4: GT inlane + ACC-style ttc-augment (ttc-NO-OP fix) 위에서 학습. Professor main(main_no_image) + Side SCC 4L cross-attn, traj_zero_after=20. val 0.5808 (steer ~3.9° ax ~2.24). a_2 v7b i_2 cache.
else:
    MODEL_CKPT_IY    = f"{REPO_ROOT}/data/ckpts/iy_v25h_A_best_loss.pt"  # 2026-05-25 v25h-A image-aware self-attn (i_2 cache 입력). ep14 val 0.1649 (steer 0.75° ax 0.90). v25c 0.37 → v25h-A 0.17 (-54%). i_2 cache = a_2 v6 의 fc_proj output (d=128).
# prof635 (v5): #654 frozen main + f_2 residual gated side. v5 best.pt = main+side 전체 포함.
MODEL_CKPT_PROF635 = os.environ.get("PROF635_CKPT",
                     f"{REPO_ROOT}/data/ckpts/prof635_side_distbrake_v1.pt")  # brake_only (DELTA_MODE 기본과 일치). v5=signed.
MAIN_PROF654_CKPT  = f"{REPO_ROOT}/data/ckpts/main_prof654.pth"
# MAIN_CKPT env: 설정 시 prof635 side ckpt 로드 後 이 main 으로 override.
# side ckpt(prof635_side_*.pt)는 main 서브모듈(654)을 포함하므로, 신규 main(655 등) 테스트엔 로드 後 덮어써야 함.
PROF635_MAIN_CKPT  = os.environ.get("MAIN_CKPT", "")
# delta_mode 는 배포 ckpt 학습값과 반드시 일치해야 함. signed=2·tanh(±, 가속가능), brake_only=-2·sigmoid(≤0).
# v5 ckpt 는 signed 학습 → signed. distbrake/randbase ckpt 는 brake_only 학습 → brake_only 로 지정 필수.
PROF635_DELTA_MODE = os.environ.get("DELTA_MODE", "brake_only")
# lead-presence gate = 모델 내부 learned NN MLP (gate_head). rule(threshold/IIR/EMA/distill/beat) 전면 제거.
#   gate = sigmoid(gate_head(a_2 in-lane obj logit)). 학습=추론 동일 forward. 규정-clean (no hardcoded rule).
PROF635_SIDE_OFF    = os.environ.get("SIDE_OFF", "0") == "1"        # 1 → side override 끔(=f_1 단독, 비상용)
PROF635_F2_DELTA_SCALE = float(os.environ.get("F2_DELTA_SCALE", "1.0"))  # f_2 brake delta 증폭(1.0=불변). final=clamp(base+gate·scale·delta). 1.2~1.3=강제동, >1.5 cruise 과제동
# ---- 외부 시계열 gate 옵션 (NO_GATE=1 경로 전용) ----
#   NO_GATE=1: side 가 no_gate(raw delta) → 내부 gate_head 우회. closed_loop final = base + g·delta.
#     · GATE_CKPT 미설정 → g=1 (=base+delta, A1E2EModel-scc 의 g=1 등가, no_gate side 학습값 그대로).
#     · GATE_CKPT=<TemporalMLP ckpt> 설정 → g = TemporalMLP(최근 K프레임 in-lane obj logit window).
#       매 frame a2_logits[0,16:26,0](10dim) 을 deque(maxlen=K) 에 append → np.stack → forward → g.
#       A1E2EModel-scc forward(base + g·delta)와 동일 gate 식 (단일 미분 그래프의 실시간 등가).
#   기존 NO_GATE 미설정(prof635 내부 gate_head) 경로는 불변 (하위호환).
PROF635_GATE_CKPT  = os.environ.get("GATE_CKPT", "")
# ---- ACC 모드 (MODEL_ACC=1): 5-frame a_2 predicted vL_rel 을 f_2 에 주입 (vrel 기반 제동) ----
# 기본 0 → 기존 distbrake 경로(회귀 방지). 1 → predvrel f_2 + 5f a_2 vrel 주입.
# 학습 정합: f_2(predvrel ttc)는 i2_mem/a2_logits=v7b 2-frame + gt_vrel=5f closing(=-vL_rel) 로 학습됨.
#   따라서 ACC ON 에서도 i2_mem/gate 경로는 v7b 유지(MODEL_A2_V6_CKPT), vrel scalar 만 5f 로 산출 후 주입.
MODEL_ACC          = os.environ.get("MODEL_ACC", "0") == "1"
# ACC f_2 ckpt: brake_only(기본) = prof635_side_predvrel_ttc, signed 옵션도 존재(escalation 약함).
ACC_F2_CKPT        = os.environ.get("ACC_F2_CKPT",
                     f"{REPO_ROOT}/data/ckpts/prof635_side_predvrel_ttc.pt")
# 5f a_2 (A2ModelMultiFrame, offsets[10,20,30,40,50], d_out128, 7ch). predicted vL_rel 제공.
ACC_A2_MF5F_CKPT   = os.environ.get("ACC_A2_MF5F_CKPT",
                     f"{REPO_ROOT}/data/ckpts/a2_mf5f_acc.pt")
ACC_PREV_OFFSETS   = [10, 20, 30, 40, 50]   # tick offsets (frame0=curr 제외) — build_pred_vrel_cache 와 동일
# ★ i1 cache 의 1 tick = 10ms (100Hz 녹화; compute_i1_tok_cache "5 20 = 50ms+200ms" 주석 확인).
#   따라서 offsets[10..50] = [100..500]ms. live inference 는 가변 rate(≤20Hz) 이므로
#   inference-tick 단위로 버퍼하면 Δt 가 어긋남 → wall-clock 시간 기반으로 가장 가까운 토큰 fetch.
ACC_TICK_DT        = 0.01                    # 1 tick = 10ms (cache 녹화 주기)
ACC_BIN_LO, ACC_BIN_HI = 16, 25              # in-lane(center) bins (build_pred_vrel_cache 와 동일)
ACC_SCALE_V        = 20.0                    # vL_rel denorm scale
ACC_SCALE_DIST     = 40.0                    # dist denorm scale
ACC_OBJ_THR        = 0.5                     # in-lane lead 판정 obj prob 임계
ACC_VREL_EMA       = float(os.environ.get("ACC_VREL_EMA", "0.6"))   # 주입 closing 시간평활(flicker 억제)
if MODEL_ACC:
    # ACC ON 이면 f_2 ckpt 를 predvrel 로 강제(env PROF635_CKPT 미지정 시). DELTA_MODE 도 brake_only 기본 유지.
    if "PROF635_CKPT" not in os.environ:
        MODEL_CKPT_PROF635 = ACC_F2_CKPT
MODEL_CKPT       = (MODEL_CKPT_PROF635 if MODEL_TYPE == "prof635"
                    else MODEL_CKPT_IY if MODEL_TYPE == "iy" else MODEL_CKPT_V2)
# obj_gate 가 학습된 ckpt 면 True (live inference 에 per-bin gate 적용)
# v25b: main alone NO obj_gate 학습 → False
MODEL_USE_OBJ_GATE = False
MODEL_BACKBONE   = "tf_efficientnetv2_s"
MODEL_D          = 128
MODEL_NHEAD      = 8
MODEL_NLAYERS    = 6
MODEL_TRAJ_SEQ_LEN = 40
MODEL_TRAJ_DIM   = 3       # [x_ego_fwd, y_ego_left, speed]
MODEL_STATE_DIM  = 6       # ego_5d padded to 6
MODEL_NUM_OUTPUT = 2       # [steer_deg, ax_mps2]
MODEL_FX         = 640.0   # i_y 학습 시 fx (fx-aware token shift) — 신규 데이터 fx=640
MODEL_C_B        = 256     # iy 의 i_1 backbone raw channel (tf_efficientnetv2_s.feature[-1])
# v18c-v22 : ControlNet self-attn side branch. v23+ 는 사용 안 함 (cross-attn 으로 대체)
MODEL_USE_SIDE_BRANCH    = False
MODEL_SIDE_NUM_LAYERS    = 3
MODEL_SIDE_DIM_FEEDFORWARD = 512
# v19/v22 : self-attn side 가 ax 채널만 보완. v23+ 는 False (cross-attn 자체로 ax-only)
MODEL_SIDE_AX_ONLY       = False
if MODEL_VARIANT == "C":
    # v25c2: f_1-only (main_no_image, side/image 없음). image 경로 전부 우회.
    MODEL_MAIN_NO_IMAGE      = True
    MODEL_SIDE_CROSS_ATTN    = False
    MODEL_TRAJ_ZERO_AFTER    = 20         # v25c2 학습값과 일치
elif MODEL_VARIANT == "B":
    # v25h-B: Professor main (text-only) + Side SCC 4L cross-attn (ax 만 보강)
    MODEL_MAIN_NO_IMAGE      = True       # main 은 traj+ego 만 (Professor 정통)
    MODEL_SIDE_CROSS_ATTN    = True       # side cross-attn (Q=traj, KV=image_mem)
    MODEL_SIDE_NUM_LAYERS    = 4          # TransformerDecoder × 4 stack
    MODEL_TRAJ_ZERO_AFTER    = 20         # v25k-B4 학습값과 일치 (training --traj-zero-after 20)
else:
    # v25h-A: image-aware self-attn (image_mem 도 transformer 의 token 으로)
    MODEL_MAIN_NO_IMAGE      = False
    MODEL_SIDE_CROSS_ATTN    = False
    MODEL_TRAJ_ZERO_AFTER    = None       # full 40 step
# v25h: i_2 cache 직접 (a2_feat 는 사용 안 함)
MODEL_USE_A2_FEAT        = False
# v25h: i_2 cache (image_mem) 사용 — a_2 v6 실시간 inference 필요. C(main-only)는 image 미사용 → False
MODEL_USE_I2_CACHE       = (MODEL_VARIANT != "C")
# ---- i_2 mem (image_mem) a_2 모델 ----
# 2026-05-27 HOTFIX: f_2/gate/scc 학습 입력 i2_mem_mf20_*.npz cache 는 mf20f_v2s_5to100 으로 생성됨
#   (model=A2ModelMultiFrame, n_frames=21, prev_offsets=[5,10,...,100], d_out=128, ego_motion=0, yaw_shift=off).
#   기존 live 추론은 A2Model 2-frame(curr+prev1) 이라 학습과 불일치 → perception/제어 mismatch.
#   closed_loop 만 21f 로 정합(f_2/scc/gate 재학습 불필요). build_i2_cache_mf20.py 와 동일 규약.
#   prof635(default): mf20 cache 와 정합하는 A2ModelMultiFrame(n_frames=21).
#   iy A/B variant: 옛 2-frame v7b cache 로 학습 → 옛 A2Model(2f) 유지 (mf20 으로 바꾸면 분포 어긋남).
MODEL_A2_V6_CKPT         = os.environ.get("A2_MF20_CKPT", f"{REPO_ROOT}/data/ckpts/a2_mf20_v2s_n20.pt")   # A2_MF20_CKPT env 로 override (db a_2 테스트용). 미설정=배포 mf20. prof635 용 A2ModelMultiFrame(n_frames=21)
MODEL_A2_V6_CKPT_IY2F    = f"{REPO_ROOT}/data/ckpts/a2_v7b_d128_alpha075_best_loss.pt"  # iy A/B 용: 옛 2-frame v7b (이 variant 들은 2f cache 로 학습됨)
MF20_PREV_OFFSETS        = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  # tick offsets (frame0=curr 제외) — build_i2_cache_mf20.py 와 동일
MF20_N_FRAMES            = 1 + len(MF20_PREV_OFFSETS)   # 21
MF20_TICK_DT             = 0.01   # i_1 cache 1 tick = 10ms (100Hz 녹화). live 가변 rate → wall-clock 최근접 fetch
MF20_TOK_BUF_MAXLEN      = 130    # 100 tick(=1.0s) + 여유. 20Hz 기준 ~20개면 1s, 가변 대비 넉넉히.
# i_1/i_2 cache 를 만들 때 쓴 seg backbone 과 반드시 일치해야 함 (token 분포 = 학습 분포).
# i1_tok_cache_*.npz meta.seg_ckpt = seg_pretrain_fx640_v3 (mIoU 0.921). 이전 combined_v1 은 cache 와
# 불일치 → live i2_mem 이 학습과 달라 SCC/제동 거동 붕괴 (2026-05-25 발견, stale 상수 수정).
SEG_BACKBONE_CKPT = f"{REPO_ROOT}/data/ckpts/seg_pretrain_fx640_v3_backbone.pt"
# Optional debug head ckpts (debug viz 켤 때 함께 load).
SEG_FULL_CKPT     = f"{REPO_ROOT}/data/ckpts/seg_pretrain_v2s_full_best.pt"
A2_CKPT           = f"{REPO_ROOT}/data/ckpts/a2_fx640_v4_best_loss.pt"  # 2026-05-23 v4 (Precision 70→98%, dist MAE 5.09→1.03m, vL_rel 5.53→0.94 m/s)

# ============================================================================
# ROS TOPICS
# ============================================================================
# CAR_NS (multicar): 차량별 namespace. set 시 모든 절대 topic 에 /{CAR_NS} prefix
#   (예: CAR_NS=Car1 → /Car1/app/loc/vehicle_state). moraibridge 노드들도 동일 ns 로 묶임
#   (런처가 ROS_NAMESPACE=Car1 로 띄움 → 노드 상대토픽 app/... 이 /Car1/app/... 로 resolve).
#   미설정 시 기존 단일차 동작 그대로 (하위호환).
CAR_NS = os.environ.get("CAR_NS", "").strip().strip("/")
_NS = f"/{CAR_NS}" if CAR_NS else ""
if CAR_NS:
    print(f"[closed_loop] CAR_NS = {CAR_NS}  → topics prefixed with {_NS}")

TOPIC_IMG   = f"{_NS}/raw_img_app/perc/camera_1"
TOPIC_EGO   = f"{_NS}/app/loc/vehicle_state"
TOPIC_STEER = f"{_NS}/app/con/command_steer"
TOPIC_PEDAL_ACCEL = f"{_NS}/app/con/command_pedal_accel"
TOPIC_PEDAL_BRAKE = f"{_NS}/app/con/command_pedal_brake"
TOPIC_ACCEL_CMD   = f"{_NS}/app/con/command_accel"   # m/s² 직접 가속 명령 (vehicle_control 도 받음)
TOPIC_IO    = f"{_NS}/app/con/ml_scc_control_io"
TOPIC_VIZ   = f"{_NS}/debug/scc_imitator_viz"     # debug 시 sensor_msgs/Image publish

# ============================================================================
# DATA / TRAJ CONFIG
# ============================================================================
# raceline 맵: dt0p1s 통합 포맷 {track}_{map}_{P}_{M}.csv (raceline + pit polygon6, cutoff=파일명).
# TRACK(sangam|speedway) × MAP(mild=안정 | high=공격적) env 로 선택. raceline_dt0p1s.load_map 이 glob 으로 찾음.
TRACK        = os.environ.get("TRACK", "sangam").lower()
RACELINE_MAP = os.environ.get("MAP", "mild").lower()
TRAJ_DT      = 0.1     # sample interval (s) — 학습 traj_40 = 40 × 0.1 = 4s
TRAJ_N       = 40      # number of waypoints
PEDAL_MAP_CSV = f"{REPO_ROOT}/data/control/pedal_map_a1_f4.csv"

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
IMG_H, IMG_W  = 720, 1280
ROI_Y0, ROI_Y1 = 235, 485        # 250-pixel strip
CAM_FX        = 320.0             # for Δyaw warp (학습 시 사용한 fx)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ============================================================================
# DEBUG
# ============================================================================
DEBUG_VIZ     = False    # ROS launch arg 로 override 가능 (~debug_viz)
DEBUG_VIZ_SEG = False    # seg head 출력 overlay (~debug_viz_seg). data/ckpts/seg_pretrain_..._best.pt 필요
DEBUG_VIZ_A2  = False    # a_2 head 출력 overlay (~debug_viz_a2). data/ckpts/a2_v1_best_loss.pt 필요
INFER_RATE_HZ = 20       # max inference rate
VIZ_RATE_HZ   = 5        # viz publish rate (별도, 추론보다 낮춤 — matplotlib draw 무거움)
PREV_IMG_DT   = 0.20     # prev image target Δt (s) — 200ms 전 (a_2 v4 PREV_DTICK=20 + prebake_iy prev_dtick=20 와 align)
IMG_QUEUE_LEN = 30       # 이미지 큐 보관 길이 (1.5s)

# ============================================================================
import os, sys, time, math, threading
from collections import deque
from pathlib import Path
import numpy as np
import torch

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Header
# cv_bridge 는 NumPy 1.x 로 compile 됐고 conda env 의 NumPy 2.x 와 호환 안됨 → segfault
# manual decode 로 우회 (rgb8/bgr8 만 지원). 필요시 conda env 에 numpy<2 설치하면 cv_bridge 복귀 가능.
USE_CV_BRIDGE = False
CvBridge = None
if USE_CV_BRIDGE:
    try:
        from cv_bridge import CvBridge as _CvBridge
        CvBridge = _CvBridge
    except Exception:
        CvBridge = None

# scc_controller_imitator (모델 + 내제화된 pedal_map + autohyu_msgs)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, f"{REPO_ROOT}/external")
from src.utils.pedal_map import PedalMap, PedalValues   # noqa: E402
from src.model.image_models.image_scc_models_v2 import SCCControllerImitatorV2   # noqa: E402
from src.model.image_models.iy_model import IyModel   # noqa: E402
from src.model.image_models.prof635_side_model import Prof635SideModel   # noqa: E402
from src.datamodule.prof635_dataset import build_traj6, build_ego6   # noqa: E402
from src.utils import raceline_dt0p1s   # noqa: E402  (dt0p1s waypoint 추출)
from src.utils.temporal_gate import load_gate_filter   # noqa: E402  (NO_GATE 경로 외부 시계열 gate)
import timm   # noqa: E402
# debug heads (optional, 가벼움)
from src.model.image_models.seg_pretrain_model import SegPretrainModel    # noqa: E402
from src.model.image_models.a2_model import A2Model    # noqa: E402
from src.model.image_models.a2_model import A2ModelMultiFrame    # noqa: E402  (ACC 5f vrel)

# autohyu_msgs (내제화: external/autohyu_msgs/ — A1_Rule 빌드 의존성 제거)
# 단 msg MD5 는 A1_Rule build 와 동일해야 sub/pub 호환됨.
try:
    from autohyu_msgs.msg import VehicleState, ControlInputOutput, TrajectoryPoint, Trajectory
except ImportError as e:
    rospy.logwarn(f"autohyu_msgs import 실패: {e}")
    raise

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


# ============================================================================
def _yaw_shift(img: np.ndarray, d_yaw: float, fx: float = CAM_FX) -> np.ndarray:
    """small Δyaw rad → horizontal pixel shift fx * d_yaw."""
    if abs(d_yaw) < 1e-5:
        return img
    shift = int(round(fx * d_yaw))
    if shift == 0:
        return img
    out = np.empty_like(img)
    if shift > 0:
        out[:, shift:] = img[:, :-shift]; out[:, :shift] = img[:, :1]
    else:
        s = -shift
        out[:, :-s] = img[:, s:]; out[:, -s:] = img[:, -1:]
    return out


def _imagenet_normalize(rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, H, W) float32 normalized."""
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def _load_raceline(path: str) -> np.ndarray:
    """semicolon-delimited, columns: s; x; y; yaw_rad; curvature; speed; pit_flag (or 0)."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";") if p.strip()]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) < 6:
                continue
            rows.append(vals[:7] if len(vals) >= 7 else vals + [0.0])
    arr = np.array(rows, dtype=np.float64)
    rospy.loginfo(f"[raceline] loaded {arr.shape[0]} points from {path}")
    return arr   # (N, 7) [s, x, y, yaw, curv, speed, pit]


def _sample_raceline_traj(raceline: np.ndarray, ex: float, ey: float, e_yaw: float, ev: float,
                            n_wp: int = TRAJ_N, dt: float = TRAJ_DT,
                            row_spacing_m: float = 0.5) -> np.ndarray:
    """ego 위치/속도 기반 등시간격 raceline sampling → (n_wp, 3) [x_ego_fwd, y_ego_left, speed].

    1) raceline 중 ego 최근접 idx 찾음 (xy 직접 거리)
    2) row 균등 spacing (0.5 m) 가정 → 각 step v*dt / spacing 만큼 row idx 증가
    3) raceline point → ego frame 변환

    NOTE: 이 raceline csv 는 main+pit 두 segment 가 합쳐져 column[0] (s) 가 non-monotonic.
    그래서 s 컬럼은 무시하고 row index 만 사용. main segment cyclic 은 단순 mod N.
    """
    pts = raceline[:, 1:3]    # (N, 2) (x, y)
    speed_arr = raceline[:, 5]
    N = pts.shape[0]
    # 최근접
    dists2 = (pts[:, 0] - ex) ** 2 + (pts[:, 1] - ey) ** 2
    i0 = int(np.argmin(dists2))

    out = np.zeros((n_wp, 3), dtype=np.float32)
    cur_idx = i0
    for ti in range(n_wp):
        cur_idx = int(cur_idx) % N
        px = pts[cur_idx, 0]; py = pts[cur_idx, 1]
        v  = float(speed_arr[cur_idx])
        # ego frame
        dx = px - ex; dy = py - ey
        c, s = math.cos(-e_yaw), math.sin(-e_yaw)
        x_e = c * dx - s * dy
        y_e = s * dx + c * dy
        out[ti, 0] = float(x_e)
        out[ti, 1] = float(y_e)
        out[ti, 2] = v
        # advance — row 수 ≈ v*dt / spacing
        ds = max(v, 1.0) * dt
        cur_idx = cur_idx + max(1, int(round(ds / row_spacing_m)))
    return out


# ============================================================================
class ClosedLoopNode:
    def __init__(self):
        # multicar: CAR_NS 별 노드 이름 차별화 (충돌 방지). 단일차면 기존 이름 유지.
        _node_name = f"scc_imitator_closed_loop_{CAR_NS}" if CAR_NS else "scc_imitator_closed_loop"
        rospy.init_node(_node_name, anonymous=False)
        # debug viz from param
        self.debug_viz     = bool(rospy.get_param("~debug_viz",     DEBUG_VIZ))
        self.debug_viz_seg = bool(rospy.get_param("~debug_viz_seg", DEBUG_VIZ_SEG))
        self.debug_viz_a2  = bool(rospy.get_param("~debug_viz_a2",  DEBUG_VIZ_A2))
        rospy.loginfo(f"[init] MODEL={MODEL_NAME}  debug_viz={self.debug_viz} "
                       f"seg={self.debug_viz_seg} a2={self.debug_viz_a2}")
        rospy.loginfo(f"[init] CKPT  ={MODEL_CKPT}")

        # build + load model
        # SEED 0 으로 image fusion init — compute_fi_mem_cache.py 와 동일 random 가중치
        # (cached 학습 ckpt 는 fi_mem 만 사용, image fusion 가중치는 cache 시점 random 으로 freeze)
        torch.manual_seed(0)
        np.random.seed(0)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.iy_backbone = None
        self.model = self._build_model().to(self.device).eval()
        if self.iy_backbone is not None:
            self.iy_backbone = self.iy_backbone.to(self.device).eval()
            for p in self.iy_backbone.parameters(): p.requires_grad_(False)
        sd = torch.load(MODEL_CKPT, map_location=self.device, weights_only=False)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        miss, unex = self.model.load_state_dict(sd, strict=False)
        miss_img = sum(1 for k in miss if k.startswith(("img_encoder.", "img_fusion.", "img_proj")))
        miss_other = len(miss) - miss_img
        rospy.loginfo(f"[model] MODEL_TYPE={MODEL_TYPE}  ckpt={MODEL_CKPT}")
        rospy.loginfo(f"[model] ckpt loaded  missing(image stream)={miss_img} "
                       f"missing(other)={miss_other} unexpected={len(unex)}")
        if MODEL_TYPE == "prof635" and PROF635_MAIN_CKPT and os.path.exists(PROF635_MAIN_CKPT):
            mo, uo = self.model.load_main_frozen(PROF635_MAIN_CKPT)
            rospy.logwarn(f"[prof635] main OVERRIDE ← {os.path.basename(PROF635_MAIN_CKPT)} "
                          f"(side ckpt 의 654 main 을 덮어씀, missing={len(mo)} unexpected={len(uo)})")
        if hasattr(self.model, "delta_scale") and PROF635_F2_DELTA_SCALE != 1.0:
            self.model.delta_scale = PROF635_F2_DELTA_SCALE
            rospy.loginfo(f"[prof635] F2_DELTA_SCALE={PROF635_F2_DELTA_SCALE} (brake delta 증폭)")

        # i_2 mem: a_2 mf20 (A2ModelMultiFrame, n_frames=21) — i_2 cache(mf20) 실시간 inference.
        # 학습 시 i2_mem_mf20 cache 가 f_2/gate/scc 의 image_mem 입력. inference 시 매 frame 토큰 deque 에서
        #   prev_offsets[5..100] tick 토큰 fetch → forward_tokens → image_mem(40,128) + a2_logits(40,7).
        # (구 2-frame A2Model → 21f A2ModelMultiFrame 교체. 학습 cache 규약과 정합.)
        self.a2_v6_model = None
        self._i2_mf20 = (MODEL_TYPE == "prof635")   # prof635=21f mf20, iy A/B=옛 2f
        self._i2_tok_buf = None        # deque of (wall_time, tok_raw (40,C_b)) — mf20 prev offset fetch 용
        if self._i2_mf20:
            # prof635: mf20 (A2ModelMultiFrame n_frames=21) — i2_mem_mf20 cache 와 정합.
            if os.path.exists(MODEL_A2_V6_CKPT):
                self.a2_v6_model = A2ModelMultiFrame(
                    img_backbone=MODEL_BACKBONE, d_token=256, n_heads=4, n_ch=7,
                    d_out=128, n_frames=MF20_N_FRAMES).to(self.device).eval()
                sd6 = torch.load(MODEL_A2_V6_CKPT, map_location=self.device, weights_only=False)
                if isinstance(sd6, dict) and "model" in sd6: sd6 = sd6["model"]
                own6 = self.a2_v6_model.state_dict()
                skip = [k for k in list(sd6.keys()) if k in own6 and sd6[k].shape != own6[k].shape]
                for k in skip: del sd6[k]
                m6, u6 = self.a2_v6_model.load_state_dict(sd6, strict=False)
                for p in self.a2_v6_model.parameters(): p.requires_grad_(False)
                self._i2_tok_buf = deque(maxlen=MF20_TOK_BUF_MAXLEN)
                rospy.loginfo(f"[a_2 mf20] loaded {MODEL_A2_V6_CKPT}  n_frames={MF20_N_FRAMES} "
                              f"offsets={MF20_PREV_OFFSETS} missing={len(m6)} unexpected={len(u6)} shape-skip={len(skip)}")
            else:
                rospy.logwarn(f"[a_2 mf20] {MODEL_A2_V6_CKPT} 없음 — MODEL_USE_I2_CACHE OFF (image_mem zero)")
                self.a2_v6_model = None
        elif MODEL_TYPE == "iy" and MODEL_USE_I2_CACHE:
            # iy A/B: 옛 2-frame A2Model (v7b cache 로 학습됨 — mf20 으로 바꾸지 않음).
            if os.path.exists(MODEL_A2_V6_CKPT_IY2F):
                self.a2_v6_model = A2Model(img_backbone=MODEL_BACKBONE, d_token=256, n_heads=4, n_ch=7, d_out=128).to(self.device).eval()
                sd6 = torch.load(MODEL_A2_V6_CKPT_IY2F, map_location=self.device, weights_only=False)
                if isinstance(sd6, dict) and "model" in sd6: sd6 = sd6["model"]
                own6 = self.a2_v6_model.state_dict()
                skip = [k for k in list(sd6.keys()) if k in own6 and sd6[k].shape != own6[k].shape]
                for k in skip: del sd6[k]
                m6, u6 = self.a2_v6_model.load_state_dict(sd6, strict=False)
                for p in self.a2_v6_model.parameters(): p.requires_grad_(False)
                rospy.loginfo(f"[a_2 v7b(2f)] loaded {MODEL_A2_V6_CKPT_IY2F}  missing={len(m6)} unexpected={len(u6)} shape-skip={len(skip)}")
            else:
                rospy.logwarn(f"[a_2 v7b(2f)] {MODEL_A2_V6_CKPT_IY2F} 없음 — MODEL_USE_I2_CACHE OFF (image_mem zero)")
                self.a2_v6_model = None

        # ---- ACC: 5f a_2 (A2ModelMultiFrame) — predicted in-lane vL_rel(closing) 산출 ----
        # f_2(predvrel)는 v7b i2_mem(위 a2_v6_model) + 5f closing(gt_vrel) 로 학습 → 둘 다 필요.
        # i1 토큰(backbone→pool→proj-전 raw (40,C_b))을 5 offset 만큼 버퍼링 → fusion → head[vL_rel].
        self.acc_a2_5f = None
        self._acc_tok_buf = None      # deque of (tick, tok_raw (40,C_b) np) — frame0=curr + prev offsets
        self._acc_vrel_ema = 0.0
        if MODEL_ACC and MODEL_TYPE == "prof635":
            if os.path.exists(ACC_A2_MF5F_CKPT):
                self.acc_a2_5f = A2ModelMultiFrame(
                    img_backbone=MODEL_BACKBONE, d_token=256, n_heads=4, n_ch=7,
                    d_out=128, n_frames=1 + len(ACC_PREV_OFFSETS)).to(self.device).eval()
                sd5 = torch.load(ACC_A2_MF5F_CKPT, map_location=self.device, weights_only=False)
                if isinstance(sd5, dict) and "model" in sd5: sd5 = sd5["model"]
                m5, u5 = self.acc_a2_5f.load_state_dict(sd5, strict=False)
                for p in self.acc_a2_5f.parameters(): p.requires_grad_(False)
                # 토큰 버퍼: (wall_time, token (40,C_b)). 가장 오래된 offset=500ms 커버 + 여유.
                # 20Hz 기준 ~14개면 700ms. 가변 rate 대비 넉넉히 64 보관(오버헤드 무시 가능).
                self._acc_tok_buf = deque(maxlen=64)
                rospy.logwarn(f"[ACC] ON — 5f a_2 {os.path.basename(ACC_A2_MF5F_CKPT)} "
                              f"missing={len(m5)} unexpected={len(u5)} offsets={ACC_PREV_OFFSETS} "
                              f"f_2={os.path.basename(MODEL_CKPT_PROF635)} delta={PROF635_DELTA_MODE}")
            else:
                rospy.logerr(f"[ACC] {ACC_A2_MF5F_CKPT} 없음 — ACC OFF (vrel 주입 안함, distbrake 거동)")
        elif MODEL_ACC:
            rospy.logwarn("[ACC] MODEL_ACC=1 이지만 MODEL_TYPE!=prof635 → 무시")

        # lead-presence gate = 모델 내부 learned NN MLP (Prof635SideModel.gate_head).
        #   추론 시 model.last_gate 가 NN gate output. rule(별도 a_2/temporal filter/IIR) 없음.
        #   (NO_GATE 미설정 = 위 내부 gate_head 경로 그대로 — 하위호환, 아래 외부 gate 영향 없음.)

        # ---- NO_GATE 경로 외부 시계열 gate (GATE_CKPT=<TemporalMLP ckpt> 설정 시) ----
        #   no_gate side 는 final=base+delta 라 self.last_gate(내부 gate_head 값)을 안 씀.
        #   여기 외부 gate net 으로 final = base + g·delta 를 closed_loop 에서 재구성한다.
        #   · GATE_CKPT 미설정 → self._gate_net=None → _inference 에서 g=1 (base+delta).
        #   · GATE_CKPT 설정 → TemporalMLP load + deque(maxlen=K) 버퍼. A1E2EModel-scc forward 와 동일 gate 식.
        self._gate_net = None; self._gate_K = 0; self._obj_win = None
        self._no_gate = (os.environ.get("NO_GATE", "0") == "1")
        if self._no_gate and PROF635_GATE_CKPT:
            if os.path.exists(PROF635_GATE_CKPT):
                self._gate_net, self._gate_K = load_gate_filter(PROF635_GATE_CKPT, map_location=self.device)
                self._gate_net = self._gate_net.to(self.device).eval()
                for p in self._gate_net.parameters(): p.requires_grad_(False)
                self._obj_win = deque(maxlen=self._gate_K)   # 최근 K프레임 in-lane obj logit (10,) window
                rospy.logwarn(f"[gate] 외부 시계열 gate ON — {os.path.basename(PROF635_GATE_CKPT)} "
                              f"K={self._gate_K}. final = base + g(TemporalMLP)·delta "
                              f"(A1E2EModel-scc forward 와 동일).")
            else:
                rospy.logerr(f"[gate] GATE_CKPT={PROF635_GATE_CKPT} 없음 — 외부 gate OFF (g=1, base+delta).")
        elif self._no_gate:
            rospy.loginfo("[gate] NO_GATE=1, GATE_CKPT 미설정 → g=1 (final=base+delta).")

        # debug heads (optional)
        self.seg_model = None; self.a2_model = None
        self._a2_overlay_logits = None   # 제어용 v7b head(i2_mem) 출력 — overlay 가 control 과 동일 a_2 보게
        if self.debug_viz_seg:
            if os.path.exists(SEG_FULL_CKPT):
                self.seg_model = SegPretrainModel(backbone=MODEL_BACKBONE, n_classes=4).to(self.device).eval()
                sd = torch.load(SEG_FULL_CKPT, map_location=self.device, weights_only=False)
                if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
                m, u = self.seg_model.load_state_dict(sd, strict=False)
                rospy.loginfo(f"[seg head] loaded {SEG_FULL_CKPT}  missing={len(m)} unexpected={len(u)}")
            else:
                rospy.logwarn(f"[seg head] {SEG_FULL_CKPT} 없음 — DEBUG_VIZ_SEG off")
                self.debug_viz_seg = False
        # debug_viz_a2 overlay 는 기본적으로 제어 경로의 v7b head(i2_mem) 사용 (control 과 동일 a_2).
        # i2_cache 경로가 없을 때만 standalone v4 a_2 를 fallback 으로 load.
        if self.debug_viz_a2 and not (MODEL_USE_I2_CACHE and self.a2_v6_model is not None):
            if os.path.exists(A2_CKPT):
                self.a2_model = A2Model(img_backbone=MODEL_BACKBONE, d_token=256, n_heads=4, n_ch=7).to(self.device).eval()
                sd = torch.load(A2_CKPT, map_location=self.device, weights_only=False)
                if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
                m, u = self.a2_model.load_state_dict(sd, strict=False)
                rospy.loginfo(f"[a_2 head] loaded {A2_CKPT}  missing={len(m)} unexpected={len(u)}")
            else:
                rospy.logwarn(f"[a_2 head] {A2_CKPT} 없음 — DEBUG_VIZ_A2 off")
                self.debug_viz_a2 = False

        # raceline + pedal map (dt0p1s 통합 맵: raceline + pit polygon + cutoff)
        self.rl_map = raceline_dt0p1s.load_map(TRACK, RACELINE_MAP, f"{REPO_ROOT}/data")
        rospy.loginfo(f"[raceline] {TRACK} {RACELINE_MAP}  N={self.rl_map['N']} "
                      f"P={self.rl_map['P']} M={self.rl_map['M']}  ({os.path.basename(self.rl_map.get('path') or '?')})")
        self.pedal_map = PedalMap()
        ok = self.pedal_map.load_from_file(PEDAL_MAP_CSV)
        if not ok:
            rospy.logerr(f"PedalMap load 실패: {PEDAL_MAP_CSV}")
            sys.exit(1)
        rospy.loginfo(f"[pedal_map] loaded {PEDAL_MAP_CSV}")

        # state
        self.bridge = CvBridge() if CvBridge is not None else None
        self.lock = threading.Lock()
        self.latest_img = None              # (250, 1280, 3) uint8 ROI
        self.latest_img_full = None         # (720, 1280, 3) for viz
        # image queue: list of (recv_time, img_roi, ego_yaw_at_recv).  최신 → 끝
        # prev image 는 (now - PREV_IMG_DT) 에 가장 가까운 entry 에서 가져옴.
        self.img_queue: deque = deque(maxlen=IMG_QUEUE_LEN)
        self.latest_ego = None              # VehicleState msg
        self.last_infer_t = 0.0
        # log for viz
        self.hist = {"t": [], "steer": [], "ax": [], "pedal_a": [], "pedal_b": []}

        # publishers
        self.pub_steer = rospy.Publisher(TOPIC_STEER, Float32, queue_size=1)
        self.pub_accel = rospy.Publisher(TOPIC_PEDAL_ACCEL, Float32, queue_size=1)
        self.pub_brake = rospy.Publisher(TOPIC_PEDAL_BRAKE, Float32, queue_size=1)
        self.pub_accel_cmd = rospy.Publisher(TOPIC_ACCEL_CMD, Float32, queue_size=1)
        self.pub_io    = rospy.Publisher(TOPIC_IO, ControlInputOutput, queue_size=1)
        self.pub_viz   = rospy.Publisher(TOPIC_VIZ, Image, queue_size=1)

        # subscribers
        rospy.Subscriber(TOPIC_IMG, Image, self.image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(TOPIC_EGO, VehicleState, self.ego_cb, queue_size=1)

        # debug viz (matplotlib live)
        if self.debug_viz:
            self._init_viz()

        rospy.loginfo("[init] ready, waiting for image + ego...")

    # --------------------------------------------------------------
    def _build_model(self):
        """MODEL_TYPE 에 따라 V2 / IyModel / Prof635 + 별도 backbone tokenizer 구성."""
        if MODEL_TYPE == "prof635":
            # iy 와 동일 seg backbone tokenizer (fx640_v3) → fc/fp 토큰. i2_mem 은 a2_v6 로 생성.
            # out_indices 생략(=default 전 stage) 후 [-1] 로 마지막 사용. 구버전 timm 은
            # features_only 의 음수 out_indices([-1])를 빈 list 로 처리 → backbone(x)[-1] IndexError.
            backbone = timm.create_model(
                MODEL_BACKBONE, pretrained=False, features_only=True)
            if SEG_BACKBONE_CKPT and os.path.exists(SEG_BACKBONE_CKPT):
                sd = torch.load(SEG_BACKBONE_CKPT, map_location="cpu", weights_only=False)
                own = backbone.state_dict(); n = 0
                for k, v in sd.items():
                    k2 = k.replace("backbone.", "")
                    if k2 in own and own[k2].shape == v.shape:
                        own[k2] = v; n += 1
                backbone.load_state_dict(own)
                rospy.loginfo(f"[model] seg backbone preload (prof635 tokenizer): {n} keys")
            self.iy_backbone = backbone
            # main(#654) frozen weights 는 v5 best.pt(state_dict 에 main+side 전체 포함)로 복원됨.
            m = Prof635SideModel(residual_gate=True, use_traj_in_side=False,
                                 delta_mode=PROF635_DELTA_MODE,
                                 no_gate=(os.environ.get("NO_GATE", "0") == "1"))
            rospy.logwarn(f"[prof635] delta_mode={PROF635_DELTA_MODE} (DELTA_MODE env). "
                          f"ckpt={os.path.basename(MODEL_CKPT_PROF635)} — 둘이 일치해야 함 "
                          f"(v5=signed, distbrake/randbase=brake_only). 불일치 시 delta 오해석.")
            if os.path.exists(MAIN_PROF654_CKPT):
                m.load_main_frozen(MAIN_PROF654_CKPT)   # main 우선 로드(이후 v5 ckpt 가 덮어씀, 동일 가중치)
            return m
        if MODEL_TYPE == "iy":
            # IyModel: live image → backbone (seg preloaded) → (B, 40, C_b) raw token
            # → IyModel(_proj_tok 안에서 C_b → D)
            # out_indices 생략(=default 전 stage) 후 [-1] 로 마지막 사용. 구버전 timm 은
            # features_only 의 음수 out_indices([-1])를 빈 list 로 처리 → backbone(x)[-1] IndexError.
            backbone = timm.create_model(
                MODEL_BACKBONE, pretrained=False, features_only=True)
            if SEG_BACKBONE_CKPT and os.path.exists(SEG_BACKBONE_CKPT):
                sd = torch.load(SEG_BACKBONE_CKPT, map_location="cpu", weights_only=False)
                own = backbone.state_dict()
                n = 0
                for k, v in sd.items():
                    k2 = k.replace("backbone.", "")
                    if k2 in own and own[k2].shape == v.shape:
                        own[k2] = v; n += 1
                backbone.load_state_dict(own)
                rospy.loginfo(f"[model] seg backbone preload (iy tokenizer): {n} keys")
            self.iy_backbone = backbone  # tokenizer (frozen, eval)
            m = IyModel(C_b=MODEL_C_B, d_model=MODEL_D, nhead=MODEL_NHEAD,
                          num_layers=MODEL_NLAYERS, traj_seq_len=MODEL_TRAJ_SEQ_LEN,
                          traj_dim=MODEL_TRAJ_DIM, state_dim=MODEL_STATE_DIM,
                          num_model_output=MODEL_NUM_OUTPUT, fx=MODEL_FX,
                          use_obj_gate=MODEL_USE_OBJ_GATE,
                          use_side_branch=MODEL_USE_SIDE_BRANCH,
                          side_num_layers=MODEL_SIDE_NUM_LAYERS,
                          side_dim_feedforward=MODEL_SIDE_DIM_FEEDFORWARD,
                          side_ax_only=MODEL_SIDE_AX_ONLY,
                          use_a2_feat=MODEL_USE_A2_FEAT,
                          main_no_image=MODEL_MAIN_NO_IMAGE,
                          side_cross_attn=MODEL_SIDE_CROSS_ATTN)
            return m
        # legacy v2
        m = SCCControllerImitatorV2(
            img_backbone=MODEL_BACKBONE,
            d_model=MODEL_D, nhead=MODEL_NHEAD,
            num_layers=MODEL_NLAYERS,
            traj_seq_len=MODEL_TRAJ_SEQ_LEN,
            traj_dim=MODEL_TRAJ_DIM,
            state_dim=MODEL_STATE_DIM,
            num_model_output=MODEL_NUM_OUTPUT,
        )
        if SEG_BACKBONE_CKPT and os.path.exists(SEG_BACKBONE_CKPT):
            sd = torch.load(SEG_BACKBONE_CKPT, map_location="cpu", weights_only=False)
            own = m.img_encoder.backbone.state_dict()
            n = 0
            for k, v in sd.items():
                k2 = k.replace("backbone.", "")
                if k2 in own and own[k2].shape == v.shape:
                    own[k2] = v; n += 1
            m.img_encoder.backbone.load_state_dict(own)
            rospy.loginfo(f"[model] seg backbone preload: {n} keys")
        return m

    # --------------------------------------------------------------
    def image_cb(self, msg: Image):
        rgb = None
        # 1) cv_bridge 시도 (USE_CV_BRIDGE=True 일 때만, NumPy 2.x 환경에서는 segfault 위험)
        if self.bridge is not None:
            try:
                rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            except Exception as e:
                rospy.logwarn_throttle(10, f"cv_bridge fail, fallback to manual: {e}")
        # 2) manual decode (default 경로)
        if rgb is None:
            try:
                if msg.encoding == "rgb8":
                    rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                elif msg.encoding == "bgr8":
                    bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    rgb = bgr[..., ::-1].copy()
                else:
                    rospy.logwarn_throttle(5, f"unsupported encoding: {msg.encoding}")
                    return
            except Exception as e:
                rospy.logerr_throttle(5, f"manual decode fail: {e}")
                return
        if rgb.shape != (IMG_H, IMG_W, 3):
            rospy.logwarn_throttle(2, f"image shape {rgb.shape} != ({IMG_H},{IMG_W},3)")
            return
        roi = rgb[ROI_Y0:ROI_Y1].copy()
        # use msg header stamp if valid, else wall time
        try:
            t_msg = msg.header.stamp.to_sec()
            if t_msg <= 0: t_msg = time.time()
        except Exception:
            t_msg = time.time()
        with self.lock:
            yaw_now = float(self.latest_ego.yaw) if self.latest_ego is not None else 0.0
            self.img_queue.append((t_msg, roi, yaw_now))
            self.latest_img = roi
            self.latest_img_full = rgb
        self._maybe_infer()

    def ego_cb(self, msg: VehicleState):
        with self.lock:
            self.latest_ego = msg
        self._maybe_infer()

    # --------------------------------------------------------------
    def _maybe_infer(self):
        now = time.time()
        with self.lock:
            if self.latest_img is None or self.latest_ego is None or not self.img_queue:
                return
            if now - self.last_infer_t < 1.0 / INFER_RATE_HZ:
                return
            self.last_infer_t = now
            # 가장 최근 entry = curr
            t_curr, img_curr, yaw_curr = self.img_queue[-1]
            img_curr = img_curr.copy()
            # prev: (t_curr - PREV_IMG_DT) 에 가장 가까운 entry. 없으면 curr 자신.
            target_t = t_curr - PREV_IMG_DT
            img_prev = None; prev_yaw = None
            best_dt = float("inf")
            for (t_q, roi_q, yaw_q) in self.img_queue:
                if t_q >= t_curr - 1e-3:  # skip curr 자신
                    continue
                dt = abs(t_q - target_t)
                if dt < best_dt:
                    best_dt = dt
                    img_prev = roi_q; prev_yaw = yaw_q
            if img_prev is not None:
                img_prev = img_prev.copy()
            ego = self.latest_ego

        try:
            self._inference(img_curr, img_prev, ego, prev_yaw)
        except Exception as e:
            rospy.logerr_throttle(2, f"inference fail: {e}")

    def _inference(self, img_curr, img_prev, ego: VehicleState, prev_yaw):
        t0 = time.time()
        # 1) trajectory — dt0p1s 추출 (최근접 pit/track 범위 룰 + 40점 forward + 끝점 외삽)
        traj, _wp_meta = raceline_dt0p1s.sample_traj40(
            self.rl_map, float(ego.x), float(ego.y), float(ego.yaw))
        # 2) image_prev_rot — Δyaw warp 으로 정렬
        if img_prev is None:
            img_prev_rot = img_curr.copy()
        else:
            d_yaw = float(ego.yaw - prev_yaw) if prev_yaw is not None else 0.0
            img_prev_rot = _yaw_shift(img_prev, d_yaw)
        t_prep_img = time.time()

        # 3) tensors
        ic = _imagenet_normalize(img_curr).unsqueeze(0).to(self.device)
        # iy/prof635 는 raw prev (token-level d_yaw shift), v2 는 image plane rotated prev
        ip = _imagenet_normalize(
            img_prev if MODEL_TYPE in ("iy", "prof635") and img_prev is not None else img_prev_rot
        ).unsqueeze(0).to(self.device)
        ego_5d = np.array([ego.vx, ego.vy, ego.ax, ego.ay, ego.yaw_vel], dtype=np.float32)
        e5 = torch.from_numpy(ego_5d).unsqueeze(0).to(self.device)
        t40 = torch.from_numpy(traj).unsqueeze(0).to(self.device)
        t_to_gpu = time.time()

        # ---- prof635 (#654 frozen main + f_2 residual gated side, v5) — self-contained, iy 경로 미접촉 ----
        if MODEL_TYPE == "prof635":
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                fc = self.iy_backbone(ic)[-1]
                fc = torch.nn.functional.adaptive_avg_pool2d(fc, (1, 40)).squeeze(2).transpose(1, 2)
                fp = self.iy_backbone(ip)[-1]
                fp = torch.nn.functional.adaptive_avg_pool2d(fp, (1, 40)).squeeze(2).transpose(1, 2)
                d_yaw_v = float(ego.yaw - prev_yaw) if prev_yaw is not None else 0.0
                d_yaw_t = torch.tensor([d_yaw_v], dtype=torch.float32, device=self.device)
                fx_t = torch.tensor([MODEL_FX], dtype=torch.float32, device=self.device)
                from src.model.image_models.iy_model import yaw_shift_tokens_batch
                fp_rot = yaw_shift_tokens_batch(fp, d_yaw_t, fx=fx_t)
                if self.a2_v6_model is not None:
                    # i_2 mf20 (A2ModelMultiFrame, n_frames=21): curr 토큰을 deque 적재 후
                    #   prev_offsets[5..100] tick 전 시각에 가장 가까운 토큰 fetch → forward_tokens.
                    # yaw_shift off (cache meta 와 동일 → fp_rot 미사용, raw fc 사용), ego_motion=0.
                    #   build_i2_cache_mf20.py 규약: tok_list[0]=curr, [1..]=prev offset raw token.
                    t_now_i2 = time.time()
                    self._i2_tok_buf.append((t_now_i2, fc[0].detach()))   # (wall_time, (40,C_b))
                    frames = [fc[0]]                                      # frame0 = curr
                    last_tok = fc[0]
                    for off in MF20_PREV_OFFSETS:
                        target_t = t_now_i2 - off * MF20_TICK_DT
                        best_tok = last_tok; best_dt = float("inf")
                        for (tq, tokq) in self._i2_tok_buf:
                            dt = abs(tq - target_t)
                            if dt < best_dt:
                                best_dt = dt; best_tok = tokq
                        frames.append(best_tok); last_tok = best_tok
                    tok_list = [t.unsqueeze(0) for t in frames]           # 21 × (1,40,C_b)
                    ego_z = torch.zeros(1, 2, dtype=torch.float32, device=self.device)  # cache ego_motion=0
                    a2_logits, i2_mem = self.a2_v6_model.forward_tokens(
                        tok_list, ego_z, return_mem=True)                 # (1,40,7), (1,40,128)
                else:
                    i2_mem = torch.zeros(1, 40, 128, dtype=torch.float32, device=self.device)
                    a2_logits = torch.full((1, 40, 7), -10.0, dtype=torch.float32, device=self.device)
                traj6 = torch.from_numpy(build_traj6(traj, mode="speed")).unsqueeze(0).to(self.device)
                ego6 = torch.from_numpy(build_ego6(ego_5d)).unsqueeze(0).to(self.device)
                f2_in = {"traj6": traj6, "ego6": ego6, "i2_mem": i2_mem, "a2_logits": a2_logits}
                # ---- ACC: 5f a_2 predicted in-lane closing(=-vL_rel) → f_2 gt_vrel 주입 ----
                # fc(curr raw token, (1,40,C_b))을 tick 버퍼에 적재 → offsets[10..50] 토큰 fetch →
                # A2ModelMultiFrame.forward_tokens → in-lane(16-25) obj>thr 중 최근접 col 의 vL_rel.
                # yaw_shift off, ego_motion=0 (build_pred_vrel_cache.py 와 동일).
                acc_vrel = 0.0
                if self.acc_a2_5f is not None:
                    t_now = time.time()
                    self._acc_tok_buf.append((t_now, fc[0].detach()))   # (wall_time, (40,C_b))
                    frames = [fc[0]]                                    # frame0 = curr
                    last = fc[0]
                    for off in ACC_PREV_OFFSETS:
                        # offset(tick) × 10ms 전 시각에 가장 가까운 버퍼 토큰 (없으면 직전 offset fallback).
                        target_t = t_now - off * ACC_TICK_DT
                        best_tok = last; best_dt = float("inf")
                        for (tq, tokq) in self._acc_tok_buf:
                            dt = abs(tq - target_t)
                            if dt < best_dt:
                                best_dt = dt; best_tok = tokq
                        frames.append(best_tok); last = best_tok
                    tok_list = [t.unsqueeze(0) for t in frames]        # N × (1,40,C_b)
                    ego_z = torch.zeros(1, 2, dtype=torch.float32, device=self.device)
                    logits5 = self.acc_a2_5f.forward_tokens(tok_list, ego_z)   # (1,40,7)
                    obj_p = torch.sigmoid(logits5[0, ACC_BIN_LO:ACC_BIN_HI + 1, 0])
                    vlr   = logits5[0, ACC_BIN_LO:ACC_BIN_HI + 1, 3] * ACC_SCALE_V
                    dst   = logits5[0, ACC_BIN_LO:ACC_BIN_HI + 1, 6] * ACC_SCALE_DIST
                    om = obj_p > ACC_OBJ_THR
                    if bool(om.any()):
                        d = dst.clone(); d[~om] = float("inf")
                        sel = int(torch.argmin(d).item())
                        acc_vrel = float(-vlr[sel].item())            # closing>0 = 접근
                    # EMA 평활 (frame flicker 억제) — no-lead 면 0 으로 감쇠
                    self._acc_vrel_ema = ACC_VREL_EMA * self._acc_vrel_ema + (1.0 - ACC_VREL_EMA) * acc_vrel
                    f2_in["gt_vrel"] = torch.tensor([self._acc_vrel_ema],
                                                    dtype=torch.float32, device=self.device)
                steer_n, _pedal_model = self.model(f2_in)
                if self.device == "cuda":
                    torch.cuda.synchronize()
            steer_deg = float(steer_n.item()) * 40.0
            # lead-presence gate = 모델 내부 learned NN MLP (gate_head) 의 forward output.
            #   model.last_gate = sigmoid(gate_head(a_2 in-lane obj logit)). rule(threshold/IIR/EMA/distill) 없음.
            #   학습과 동일 forward → 학습=추론 일관. no-lead 면 학습된 응답으로 g≈0(가속 보존), lead 면 g→1(제동).
            base_pedal = float(self.model.last_base_pedal.item())
            delta = float(self.model.last_delta.item()) if self.model.last_delta is not None else 0.0
            obj_amax = float(a2_logits[0, 16:26, 0].max().item())   # 로그/진단용 (gate 계산엔 미사용)
            if PROF635_SIDE_OFF:
                gate = 0.0                                          # 비상용: side override OFF (=f_1 단독)
            elif self._no_gate:
                # NO_GATE 경로: side 가 no_gate(raw delta) → final=base+delta. 내부 last_gate 안 씀.
                if self._gate_net is not None:
                    # 외부 시계열 gate(GATE_CKPT): 매 frame in-lane obj logit(16:26, ch0, 10dim) → deque → TemporalMLP.
                    #   A1E2EModel-scc 의 self.gate(obj_window) 과 동일 식 (deque window stack→forward).
                    obj_inlane = a2_logits[0, 16:26, 0].float().detach().cpu().numpy()   # (10,)
                    self._obj_win.append(obj_inlane)
                    win = np.stack(list(self._obj_win))                                  # (<=K,10)
                    if win.shape[0] < self._gate_K:                                      # 좌측 zero-pad (학습 causal 동일)
                        win = np.concatenate(
                            [np.zeros((self._gate_K - win.shape[0], 10), np.float32), win], 0)
                    with torch.no_grad():
                        gw = torch.from_numpy(win[None]).float().to(self.device)         # (1,K,10)
                        gate = float(self._gate_net(gw)[0].item())
                else:
                    gate = 1.0                                      # GATE_CKPT 미설정 → base+delta (no_gate side 학습값)
            else:
                gate = float(self.model.last_gate.item())           # NN gate output (model.forward 에서 계산)
            pedal_s = max(-1.0, min(1.0, base_pedal + gate * delta))
            accel_pedal = max(pedal_s, 0.0); brake_pedal = max(-pedal_s, 0.0)
            self.pub_steer.publish(Float32(data=steer_deg))
            self.pub_accel.publish(Float32(data=accel_pedal))
            self.pub_brake.publish(Float32(data=brake_pedal))
            io = ControlInputOutput(); io.header = Header(stamp=rospy.Time.now()); io.vehicle_state = ego
            self.pub_io.publish(io)
            self._p635_cnt = getattr(self, "_p635_cnt", 0) + 1
            if self._p635_cnt % 20 == 0:
                dt_ms = (time.time() - t0) * 1000.0
                acc_tag = (f" vrel={self._acc_vrel_ema:+.1f}m/s" if self.acc_a2_5f is not None else "")
                rospy.loginfo(f"[prof635{'+ACC' if self.acc_a2_5f is not None else ''}] {dt_ms:.0f}ms "
                              f"i0={_wp_meta['i0']}{'(pit)' if _wp_meta['in_pit'] else ''} "
                              f"steer={steer_deg:+.1f}deg base={base_pedal:+.2f} delta={delta:+.2f} "
                              f"obj_amax={obj_amax:+.2f} gate={gate:.2f}{acc_tag} -> pedal={pedal_s:+.2f} "
                              f"(acc={accel_pedal:.2f} brk={brake_pedal:.2f})")
            # debug viz — iy 경로와 동일 2x2 패널 (early return 전에 호출).
            if self.debug_viz:
                seg_pred = None; a2_pred = None
                if self.debug_viz_seg and self.seg_model is not None:
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                        seg_pred = self.seg_model(ic).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                if self.debug_viz_a2:
                    a2_arr = a2_logits[0].float().cpu().numpy()
                    a2_arr[:, 0] = 1.0 / (1.0 + np.exp(-a2_arr[:, 0]))   # sigmoid obj
                    a2_pred = a2_arr
                # ax 슬롯에 pedal_s 전달(시계열에 pedal 추이 표시 — prof635 는 ax 대신 단일 pedal)
                self._update_viz(traj, img_curr, steer_deg, pedal_s, accel_pedal, brake_pedal,
                                 seg_pred=seg_pred, a2_pred=a2_pred, ego_vx=float(ego.vx))
            return
        # 4) forward — amp 비활성 (학습이 fp32 → fp16 cast 시 transformer softmax overflow → NaN)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            if MODEL_TYPE == "iy":
                # backbone → (B, C_b, H', 40) → pool → (B, 40, C_b)
                fc = self.iy_backbone(ic)[-1]
                fc = torch.nn.functional.adaptive_avg_pool2d(fc, (1, 40)).squeeze(2).transpose(1, 2)
                fp = self.iy_backbone(ip)[-1]
                fp = torch.nn.functional.adaptive_avg_pool2d(fp, (1, 40)).squeeze(2).transpose(1, 2)
                d_yaw_v = float(ego.yaw - prev_yaw) if prev_yaw is not None else 0.0
                d_yaw_t = torch.tensor([d_yaw_v], dtype=torch.float32, device=self.device)
                fx_t    = torch.tensor([MODEL_FX], dtype=torch.float32, device=self.device)
                # v23+ : traj 일부 zero (학습 환경 align)
                if MODEL_TRAJ_ZERO_AFTER is not None and MODEL_TRAJ_ZERO_AFTER < MODEL_TRAJ_SEQ_LEN:
                    t40 = t40.clone()
                    t40[:, MODEL_TRAJ_ZERO_AFTER:, :] = 0.0
                model_in = {
                    "tok_curr": fc, "tok_prev_raw": fp, "d_yaw": d_yaw_t,
                    "ego_5d": e5, "traj_40": t40, "fx": fx_t,
                }
                # a_2 features: live a_2 inference 미구현 → zero (학습 시 a2_proj 가 a2_feat 입력 처리)
                if MODEL_USE_A2_FEAT:
                    model_in["a2_feat"] = torch.zeros(fc.shape[0], 40, 7,
                                                       dtype=torch.float32, device=self.device)
                # v25h: i_2 cache 실시간 생성 (a_2 v6 forward)
                if MODEL_USE_I2_CACHE and self.a2_v6_model is not None:
                    # yaw shift prev (학습 시 yaw_shift_tokens_batch 와 동일)
                    from src.model.image_models.iy_model import yaw_shift_tokens_batch
                    fp_rot = yaw_shift_tokens_batch(fp, d_yaw_t, fx=fx_t)
                    # a_2 v6: encoder.proj → fusion → fc_proj
                    tc_proj = self.a2_v6_model.encoder.proj(fc.transpose(1, 2)).transpose(1, 2)
                    tp_proj = self.a2_v6_model.encoder.proj(fp_rot.transpose(1, 2)).transpose(1, 2)
                    # build_i2_cache.py 가 ego_mot=zeros 로 i2_mem 캐시를 생성 → 학습 분포와 일치시키려면
                    # live 도 zeros 여야 함 (real ego_mot 입력 시 i2_mem 이 학습과 달라져 side/SCC 거동 붕괴).
                    ego_mot = torch.zeros_like(e5[:, :2])
                    fi_mem_256 = self.a2_v6_model.fusion(tc_proj, tp_proj, ego_mot)
                    i2_mem = self.a2_v6_model.fc_proj(fi_mem_256)         # (1, 40, 128)
                    model_in["i2_mem"] = i2_mem
                    # v7b head logits — obj_gate inject + debug overlay 둘 다 동일 a_2(제어 경로) 사용
                    if MODEL_USE_OBJ_GATE or self.debug_viz_a2:
                        a2_logits_live = self.a2_v6_model.head(i2_mem)    # (1, 40, 7)
                        if MODEL_USE_OBJ_GATE:
                            model_in["a2_logits"] = a2_logits_live
                        if self.debug_viz_a2:
                            self._a2_overlay_logits = a2_logits_live.detach()
                out = self.model(model_in)
                pred = out[0] if isinstance(out, tuple) else out
            else:
                pred = self.model({"image_curr": ic, "image_prev_rot": ip,
                                     "ego_5d": e5, "traj_30": t40})
            if self.device == "cuda":
                torch.cuda.synchronize()
        t_fwd = time.time()
        # pred (1, 40, 2) [steer_deg, ax_mps2]
        pred_np = pred[0].float().cpu().numpy()
        steer_deg = float(pred_np[0, 0])
        ax_target = float(pred_np[0, 1])

        # 5) pedal conversion
        pv: PedalValues = self.pedal_map.get_pedal_values(float(ego.vx), ax_target)
        accel_pedal = float(pv.accel_pedal)
        brake_pedal = float(pv.brake_pedal)

        # 6) publish
        self.pub_steer.publish(Float32(data=steer_deg))
        self.pub_accel.publish(Float32(data=accel_pedal))
        self.pub_brake.publish(Float32(data=brake_pedal))
        self.pub_accel_cmd.publish(Float32(data=float(ax_target)))   # m/s² 직접 가속도 (vehicle_control subscribe)

        io = ControlInputOutput()
        io.header = Header(stamp=rospy.Time.now())
        io.vehicle_state = ego
        # input trajectory (ego frame → just fill x, y, speed)
        tr_in = Trajectory()
        tr_in.header = io.header
        for i in range(traj.shape[0]):
            tp = TrajectoryPoint()
            tp.time = float(i * TRAJ_DT)
            tp.x = float(traj[i, 0]); tp.y = float(traj[i, 1])
            tp.speed = float(traj[i, 2])
            tr_in.point.append(tp)
        io.trajectory = tr_in
        # solved trajectory (predicted steer/ax 40-step)
        tr_out = Trajectory(); tr_out.header = io.header
        for i in range(pred_np.shape[0]):
            tp = TrajectoryPoint()
            tp.time = float(i * 0.025)   # long_traj dt (training convention)
            tr_out.point.append(tp)
        io.solved_trajectory = tr_out
        io.command_steer        = pred_np[:, 0].tolist()
        io.command_acceleration = pred_np[:, 1].tolist()
        # pedal arrays — first only (모델은 ax 만 출력, pedal 은 매핑 결과)
        io.command_accel_pedal  = [accel_pedal]
        io.command_brake_pedal  = [brake_pedal]
        io.command_speed        = traj[:, 2].tolist()
        self.pub_io.publish(io)

        # debug head inference (optional)
        seg_pred = None; a2_pred = None
        if self.debug_viz_seg and self.seg_model is not None:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                seg_logits = self.seg_model(ic)   # (1, 4, H, W)
                seg_pred = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        if self.debug_viz_a2 and self._a2_overlay_logits is not None:
            # 제어 경로의 v7b head(i2_mem) 출력 그대로 — overlay = 컨트롤러가 실제 보는 perception
            a2_arr = self._a2_overlay_logits[0].float().cpu().numpy()
            a2_arr[:, 0] = 1.0 / (1.0 + np.exp(-a2_arr[:, 0]))   # sigmoid obj
            a2_pred = a2_arr
        elif self.debug_viz_a2 and self.a2_model is not None:
            # fallback: i2_cache 경로 미사용 시에만 standalone a_2
            ego_mot = torch.tensor([[float(ego.vx), float(ego.yaw_vel)]], dtype=torch.float32).to(self.device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                a2_logits = self.a2_model(ic, ip, ego_mot)   # (1, 40, 7)
            a2_arr = a2_logits[0].float().cpu().numpy()
            a2_arr[:, 0] = 1.0 / (1.0 + np.exp(-a2_arr[:, 0]))   # sigmoid obj
            a2_pred = a2_arr

        t_done = time.time()
        rospy.loginfo_throttle(
            1.0,
            f"[infer] traj+img={1000*(t_prep_img-t0):5.1f}ms  "
            f"to_gpu={1000*(t_to_gpu-t_prep_img):5.1f}ms  "
            f"fwd={1000*(t_fwd-t_to_gpu):5.1f}ms  "
            f"pub={1000*(t_done-t_fwd):5.1f}ms  "
            f"TOTAL={1000*(t_done-t0):5.1f}ms  |  "
            f"steer={steer_deg:+.2f}° ax={ax_target:+.2f}m/s² "
            f"pedal_a={accel_pedal:.2f} pedal_b={brake_pedal:.2f}"
        )

        # debug viz
        if self.debug_viz:
            self._update_viz(traj, img_curr, steer_deg, ax_target, accel_pedal, brake_pedal,
                              seg_pred=seg_pred, a2_pred=a2_pred, ego_vx=float(ego.vx))

    # --------------------------------------------------------------
    def _init_viz(self):
        """matplotlib backend Agg + 고정 layout (tight_layout 매번 X)."""
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self._fig = Figure(figsize=(12, 7), dpi=70)
        self._canvas = FigureCanvasAgg(self._fig)
        self._axes = np.array([
            [self._fig.add_subplot(2, 2, 1), self._fig.add_subplot(2, 2, 2)],
            [self._fig.add_subplot(2, 2, 3), self._fig.add_subplot(2, 2, 4)],
        ])
        self._fig.suptitle(f"closed-loop: {MODEL_NAME}")
        self._fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.07,
                                    wspace=0.20, hspace=0.30)   # tight_layout 대체 (1회만)
        self._last_viz_t = 0.0

    def _update_viz(self, traj, img_curr, steer_deg, ax_target, accel_pedal, brake_pedal,
                     seg_pred=None, a2_pred=None, ego_vx=0.0):
        """viz composite → sensor_msgs/Image publish (rqt_image_view 로 확인).
        seg_pred: (H, W) int 4-class (or None)
        a2_pred:  (40, 7) [obj_prob, vL_abs, vY_abs, vL_rel, vY_rel, prox, dist] (or None)
        ego_vx: 자차 종방향 속도 (m/s) — viz 표시용
        """
        # rate limit
        now = time.time()
        if now - self._last_viz_t < 1.0 / VIZ_RATE_HZ:
            # history 만 누적, render skip
            self.hist["t"].append(now); self.hist["steer"].append(steer_deg)
            self.hist["ax"].append(ax_target); self.hist["pedal_a"].append(accel_pedal)
            self.hist["pedal_b"].append(brake_pedal)
            for k in self.hist: self.hist[k] = self.hist[k][-200:]
            return
        self._last_viz_t = now
        # history
        self.hist["t"].append(time.time())
        self.hist["steer"].append(steer_deg)
        self.hist["ax"].append(ax_target)
        self.hist["pedal_a"].append(accel_pedal)
        self.hist["pedal_b"].append(brake_pedal)
        for k in self.hist: self.hist[k] = self.hist[k][-200:]

        # ax 0,0: BEV traj
        from matplotlib.patches import Rectangle
        ax = self._axes[0, 0]; ax.cla()
        ax.add_patch(Rectangle((-2.5, -1), 5, 2, ec="k", fc="gray", alpha=0.6))
        ax.plot(traj[:, 0], traj[:, 1], "C0.-", ms=4)
        ax.set_xlabel("x_fwd (m)"); ax.set_ylabel("y_left (m)")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 60); ax.set_ylim(-10, 10)
        ax.set_title("input traj (BEV, ego frame)")

        # ax 0,1: image_curr (ROI) — seg overlay (있으면) + a_2 bin bars (있으면)
        self._axes[0, 1].cla()
        disp = img_curr.copy()
        if seg_pred is not None and seg_pred.shape == img_curr.shape[:2]:
            # 4-class palette (bg=black, road=gray, vehicle=red, wall=skyblue)
            PAL = np.array([[0,0,0], [80,80,80], [255,80,80], [80,200,255]], dtype=np.uint8)
            rgb_seg = PAL[seg_pred]
            disp = (disp.astype(np.float32) * 0.6 + rgb_seg.astype(np.float32) * 0.4).astype(np.uint8)
        self._axes[0, 1].imshow(disp); self._axes[0, 1].axis("off")
        # SCC 상태 + 자차 속도 overlay: in-lane(16-25) lead 인지 → ACC 동작 여부
        lead_p = float(a2_pred[16:26, 0].max()) if a2_pred is not None else 0.0
        lead_det = lead_p > 0.5
        scc_active = lead_det and (ax_target < -0.5)   # lead 인지 + 감속 중 = ACC 동작
        scc_txt = "SCC: ACTIVE" if scc_active else ("SCC: lead?" if lead_det else "SCC: idle")
        scc_col = "#00ff66" if scc_active else ("#ffcc00" if lead_det else "#888888")
        self._axes[0, 1].text(
            0.01, 0.98, f"EGO {ego_vx:4.1f} m/s   LEAD {'Y' if lead_det else 'N'} (p={lead_p:.2f})\n{scc_txt}   ax={ax_target:+.1f} m/s²",
            transform=self._axes[0, 1].transAxes, va="top", ha="left", fontsize=10, color=scc_col,
            family="monospace", bbox=dict(boxstyle="round", fc="black", alpha=0.55, ec="none"))
        ttl = f"image_curr  steer={steer_deg:+.2f}° ax={ax_target:+.2f}m/s²"
        if seg_pred is not None: ttl += "  +seg"
        if a2_pred is not None:
            from matplotlib.patches import Rectangle as _R
            bin_w = img_curr.shape[1] / 40
            H = img_curr.shape[0]
            for bi in range(40):
                p = float(a2_pred[bi, 0])
                if p < 0.3: continue
                d   = float(a2_pred[bi, 6])
                vrel = float(a2_pred[bi, 3])
                c = (1.0, 0.2, 0.2) if vrel < 0 else (0.2, 0.8, 0.2)
                self._axes[0, 1].add_patch(_R((bi*bin_w, H-22), bin_w, 22,
                                                 fc=c, alpha=0.55, ec="white", lw=0.5))
                self._axes[0, 1].text(bi*bin_w + bin_w/2, H-11, f"{d:.0f}",
                                        ha="center", va="center", fontsize=6,
                                        color="white", fontweight="bold")
            ttl += "  +a_2"
        self._axes[0, 1].set_title(ttl, fontsize=9)

        # ax 1,0: steer time series
        ts = np.array(self.hist["t"]); ts = ts - ts[0] if len(ts) else ts
        ax = self._axes[1, 0]; ax.cla()
        ax.plot(ts, self.hist["steer"], "C3-")
        ax.set_xlabel("t (s)"); ax.set_ylabel("steer (deg)")
        ax.grid(True, alpha=0.3); ax.set_title("steer time series")

        # ax 1,1: ax / pedals
        ax = self._axes[1, 1]; ax.cla()
        ax.plot(ts, self.hist["ax"], "C0-", label="ax_target")
        ax.plot(ts, self.hist["pedal_a"], "C2-", label="accel_pedal")
        ax.plot(ts, self.hist["pedal_b"], "C1-", label="brake_pedal")
        ax.set_xlabel("t (s)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_title("ax / pedals time series")

        # tight_layout 제거 (init 때 subplots_adjust 로 고정)
        self._canvas.draw()
        buf, (w, h) = self._canvas.print_to_buffer()
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        rgb  = np.ascontiguousarray(rgba[..., :3])

        # publish
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "scc_viz"
        msg.height = h; msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = rgb.tobytes()
        self.pub_viz.publish(msg)


# ============================================================================
def main():
    node = ClosedLoopNode()
    rospy.spin()


if __name__ == "__main__":
    main()
