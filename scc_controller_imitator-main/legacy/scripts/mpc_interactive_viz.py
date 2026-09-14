#!/usr/bin/env python3
"""Interactive MPC visualization (10 Hz).

요소:
- BEV 평면: ego(원점) + reference trajectory (preset) + draggable lead object + MPC 예측 경로
- 우상단: control output 시계열 (steer, accel)
- 하단 sliders: ego_vx, ego_ax, obj_vx, obj_vy, traj_speed, n_iter
- Radio: trajectory preset
- Buttons: reset, pause, warm-reset

사용:
  /usr/bin/python3 scripts/mpc_interactive_viz.py             # interactive (DISPLAY 필요)
  /usr/bin/python3 scripts/mpc_interactive_viz.py --headless  # N tick 돌리고 PNG 저장

(`/usr/bin/python3` 가 casadi/acados 설치된 환경)

DISPLAY 가 없으면 (SSH no -X), X-forwarding 켜거나 (`ssh -X`), VNC 사용 또는 --headless.
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, "/home/ailab-12/git/A1_Rule/devel/lib/python3/dist-packages")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "external/acados_scc/scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
BUILD_CACHE = os.path.join(ROOT, "external/acados_scc/build_cache")
os.makedirs(BUILD_CACHE, exist_ok=True)
os.chdir(BUILD_CACHE)

import matplotlib
# backend 선택은 main() 에서 (--headless 면 Agg)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation

from MPC.NMPC_class_delayed_kinematic_cat import (
    Nonlinear_Delayed_Kinematic_Cat_Model_Predictive_Controller as MPC_Class,
)
from synthesize_scc_k1 import make_vs_msg, lead_to_objs_msg, run_cycle, LANE_HALF_WIDTH

# weights.ini 경로 (MPC class 와 동일 dir)
WEIGHTS_INI_PATH = os.path.join(
    os.path.dirname(os.path.abspath(
        __import__("MPC.NMPC_class_delayed_kinematic_cat", fromlist=["x"]).__file__
    )), "weights.ini"
)

# UI 에 표시할 주요 weight key 들
WEIGHT_DISPLAY_KEYS = ("ddelta_w", "jerk_w", "dddelta_w", "d_dot_w", "d_w", "mu_w", "s_w", "speed_w")


def read_weights_summary(path: str = WEIGHTS_INI_PATH) -> dict:
    """현 weights.ini 의 [weights] section 에서 표시 대상 값만 dict 로 반환."""
    import configparser
    cfg = configparser.ConfigParser()
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            cfg.read_file(f)
        return {k: float(cfg["weights"][k]) for k in WEIGHT_DISPLAY_KEYS if k in cfg["weights"]}
    except Exception as e:
        return {"_error": str(e)}
from synthesize_scc_kn_random import build_random_traj_msg
from autohyu_msgs.msg import Trajectory, TrajectoryPoint


def compute_lane_boundary(traj_xy: np.ndarray, half_width: float):
    """traj_xy (N, 2) → (left_xy (N,2), right_xy (N,2)) lane 경계."""
    N = len(traj_xy)
    left = np.zeros_like(traj_xy); right = np.zeros_like(traj_xy)
    for i in range(N):
        if i == 0:
            tangent = traj_xy[1] - traj_xy[0]
        elif i == N - 1:
            tangent = traj_xy[-1] - traj_xy[-2]
        else:
            tangent = traj_xy[i+1] - traj_xy[i-1]
        n_t = np.linalg.norm(tangent)
        if n_t > 1e-6:
            t = tangent / n_t
            normal = np.array([-t[1], t[0]])
            left[i]  = traj_xy[i] + normal * half_width
            right[i] = traj_xy[i] - normal * half_width
        else:
            left[i] = right[i] = traj_xy[i]
    return left, right


def compute_obj_n(obj_xy: np.ndarray, ref_xy: np.ndarray) -> tuple[float, float]:
    """obj 의 lateral offset n (반시계 좌측 = +) 및 longitudinal s 반환. synth_k1 의 방식과 동일."""
    dxs = ref_xy[:, 0] - obj_xy[0]; dys = ref_xy[:, 1] - obj_xy[1]
    k = int(np.argmin(np.hypot(dxs, dys)))
    seg = ref_xy[k + 1] - ref_xy[k] if k + 1 < len(ref_xy) else ref_xy[k] - ref_xy[k - 1]
    seg_n = np.hypot(seg[0], seg[1])
    if seg_n < 1e-6:
        return 0.0, 0.0
    seg = seg / seg_n
    rel = obj_xy - ref_xy[k]
    n_val = seg[0] * rel[1] - seg[1] * rel[0]
    # arc length to k
    diffs = np.diff(ref_xy[:k+1], axis=0) if k > 0 else np.zeros((0, 2))
    s_val = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))) if len(diffs) > 0 else 0.0
    return n_val, s_val


def build_traj_msg_with_origin(pts: np.ndarray) -> Trajectory:
    """MPC 가 요구하는 origin (0,0,0,speed[0]) prepended trajectory msg."""
    traj = Trajectory()
    p0 = TrajectoryPoint()
    p0.x = 0.0; p0.y = 0.0; p0.yaw = 0.0; p0.speed = float(pts[0, 3])
    traj.point.append(p0)
    for i in range(len(pts)):
        p = TrajectoryPoint()
        p.x = float(pts[i, 0]); p.y = float(pts[i, 1])
        p.yaw = float(pts[i, 2]); p.speed = float(pts[i, 3])
        p.time = (i + 1) * 0.1
        traj.point.append(p)
    return traj


# -------------------- trajectory presets --------------------
TRAJ_POOL_NPZ = os.path.join(ROOT, "data/v0237_traj_pool.npz")
_POOL_CACHE = None


def _load_pool():
    global _POOL_CACHE
    if _POOL_CACHE is None:
        z = np.load(TRAJ_POOL_NPZ)
        _POOL_CACHE = (z["trajs"], z["v0s"])     # (P, 40, 4), (P,)
    return _POOL_CACHE


def _rescale_traj(pts: np.ndarray, target_speed: float) -> np.ndarray:
    """pool trajectory 의 speed channel 만 target_speed 로 균일 변경 (x,y,yaw 그대로)."""
    out = pts.copy()
    out[:, 3] = target_speed
    return out


def _pick_pool_idx(speed: float, kind: str) -> int:
    """speed 근처 pool idx 선택. kind 에 따라 curvature 별."""
    trajs, v0s = _load_pool()
    speed_match = np.abs(v0s - speed) < 5.0
    if not speed_match.any():
        return 0
    # final lateral displacement 으로 curvature 판단
    final_y = trajs[:, -1, 1]
    if kind == "straight":
        cand = np.where(speed_match & (np.abs(final_y) < 1.5))[0]
    elif kind == "left_turn":
        cand = np.where(speed_match & (final_y > 5.0))[0]
    elif kind == "right_turn":
        cand = np.where(speed_match & (final_y < -5.0))[0]
    else:
        cand = np.where(speed_match)[0]
    if len(cand) == 0:
        return 0
    return int(cand[len(cand) // 2])  # 중간값


def _build_preset(kind: str, speed: float) -> np.ndarray:
    """real pool 에서 kind 에 맞는 traj 뽑고 speed channel 만 target 으로 교체."""
    idx = _pick_pool_idx(speed, kind)
    trajs, _ = _load_pool()
    return _rescale_traj(trajs[idx], speed)


PRESETS = {
    "straight":   lambda speed: _build_preset("straight",   speed),
    "left_turn":  lambda speed: _build_preset("left_turn",  speed),
    "right_turn": lambda speed: _build_preset("right_turn", speed),
}
LOADED_SAMPLE_KEY = "loaded sample (real)"


# -------------------- real synth sample loader (well-conditioned input) --------------------
SYNTH_DIR = os.path.join(ROOT, "data/scc_synth_k100_v1")
_SAMPLE_CACHE = None    # all (fp, idx) pairs across fwd50 pkls


def _index_synth_samples():
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE is not None: return _SAMPLE_CACHE
    import glob, pickle
    pairs = []
    for fp in sorted(glob.glob(os.path.join(SYNTH_DIR, "fwd50_*.pkl"))):
        if os.path.getsize(fp) < 1000: continue
        try:
            with open(fp, "rb") as f:
                d = pickle.load(f)
            for i in range(len(d["samples"])):
                pairs.append((fp, i))
        except Exception:
            continue
    _SAMPLE_CACHE = pairs
    return pairs


def load_random_synth_sample(rng: np.random.RandomState | None = None) -> dict:
    """fwd50 pkl 에서 random sample → dict with ego_vx, obj_*, traj_40 (40, 4)."""
    import pickle
    pairs = _index_synth_samples()
    if not pairs:
        raise FileNotFoundError(f"no synth pkls in {SYNTH_DIR}")
    rng = rng or np.random.RandomState()
    fp, i = pairs[int(rng.randint(len(pairs)))]
    with open(fp, "rb") as f:
        d = pickle.load(f)
    s = d["samples"][i]
    ego = np.asarray(s["ego_5d"], dtype=np.float32)
    traj = np.asarray(s["traj_30"], dtype=np.float32)   # (30 or 40, 4)
    # logged 는 (30,4) — 마지막 row 반복 padding 으로 (40,4) 로 늘림
    if traj.shape[0] < 40:
        pad = np.tile(traj[-1:], (40 - traj.shape[0], 1))
        traj = np.concatenate([traj, pad], axis=0)
    li = np.asarray(s["lead_info"], dtype=np.float32)
    out = dict(
        ego_vx=float(ego[0]),
        ego_ax=float(ego[2]),
        traj_40=traj,
        traj_speed=float(traj[0, 3]) if traj[0, 3] > 0 else float(ego[0]),
        lead_valid=bool(s["lead_valid"]),
        obj_x=float(li[0]) if s["lead_valid"] else 40.0,
        obj_y=float(li[1]) if s["lead_valid"] else 0.0,
        obj_vx=float(li[2]) if s["lead_valid"] else float(ego[0]),
        obj_vy=float(li[3]) if s["lead_valid"] else 0.0,
        obj_L=float(li[4]) if s["lead_valid"] else 4.5,
        obj_W=float(li[5]) if s["lead_valid"] else 2.0,
        frame_id=s.get("frame_id", "?"),
        traj_tag=s.get("traj_tag", "?"),
    )
    return out


# NOTE: forward_integrate 제거됨. MPC pred_X[:, 0:2] 가 이미 Cartesian (ego-local) 이라 직접 사용.
# Frenet (s, n, mu) 는 MPC 내부의 SCC 종방향 추적용으로만 사용.
# 참조: ~/.claude/projects/-home-ailab-12-git-scc-controller-imitator/memory/mpc_internals.md


# -------------------- main app --------------------
class MPCInteractive:
    def __init__(self):
        # ---- state ----
        # 초기 baseline: real synth sample 로드 (MPC 가 well-conditioned input 보장).
        # slider 로 변형해서 MPC behavior 관찰.
        self._rng = np.random.RandomState(0)
        self.state = dict(
            ego_vx=20.0, ego_ax=0.0,
            obj_x=40.0, obj_y=0.0, obj_L=4.5, obj_W=2.0,
            obj_vx=20.0, obj_vy=0.0,
            traj_speed=20.0,
            traj_preset=LOADED_SAMPLE_KEY,
            loaded_traj=None,
            loaded_frame_id="?",
            n_iter=10,
            paused=False,
        )
        self.dragging = False
        self.last_solve_ms = 0.0
        self.last_status = "?"
        self.tick_counter = 0
        # weights.ini 감시
        self._weights_mtime = 0.0
        self._weights_summary = {}
        self._weights_reloaded_at_tick = -999
        self._refresh_weights_if_changed()

        # ---- MPC (먼저 생성 후 sample load) ----
        self.mpc = MPC_Class()
        self._reset_mpc_warm()
        try:
            self._load_synth_sample()
        except FileNotFoundError as e:
            print(f"warn: {e} — fallback to preset 'straight'")
            self.state["traj_preset"] = "straight"

        # ---- figure ----
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("MPC interactive viz — drag object, adjust sliders, watch MPC follow", fontsize=12)
        gs = self.fig.add_gridspec(3, 2, width_ratios=[2.5, 1], height_ratios=[3, 1, 0.4],
                                    hspace=0.35, wspace=0.25)
        self.ax_bev = self.fig.add_subplot(gs[0, 0])
        self.ax_ctrl = self.fig.add_subplot(gs[0, 1])
        self.ax_info = self.fig.add_subplot(gs[1, :])
        self.ax_info.axis("off")
        self.ax_info_text = self.ax_info.text(0.01, 0.5, "", fontsize=10, family="monospace",
                                               va="center", transform=self.ax_info.transAxes)

        # widgets area
        self.fig.subplots_adjust(bottom=0.30)
        self._build_widgets()
        self._build_bev_artists()
        self._build_ctrl_artists()

        # mouse events
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

        # timer (10 Hz)
        self.anim = FuncAnimation(self.fig, self.tick, interval=100, blit=False, cache_frame_data=False)

    def _reset_mpc_warm(self):
        self.mpc.is_MPC_init = True
        self.mpc.pred_X = None
        self.mpc.pred_U = None
        self.mpc.X_init = None
        self.mpc.fallback_count = 0
        self.mpc.last_mpc_status = None
        self._warmup_needed = True

    def _refresh_weights_if_changed(self):
        """weights.ini mtime 변경 시 summary 다시 읽기 + 'reloaded' flash 표시 트리거."""
        try:
            m = os.path.getmtime(WEIGHTS_INI_PATH)
        except OSError:
            return
        if abs(m - self._weights_mtime) > 1e-6:
            self._weights_mtime = m
            self._weights_summary = read_weights_summary(WEIGHTS_INI_PATH)
            self._weights_reloaded_at_tick = self.tick_counter
            print(f"weights.ini reloaded at tick={self.tick_counter}: {self._weights_summary}")

    def _load_synth_sample(self):
        """real synth fwd50 sample 1 개 random 로드 → state 갱신."""
        s = load_random_synth_sample(self._rng)
        self.state["loaded_traj"] = s["traj_40"]
        self.state["loaded_frame_id"] = f"{s['frame_id']}  [{s['traj_tag']}]"
        # ego/obj 값도 sample 에 맞춰 셋업 (MPC well-conditioned start 보장)
        self.state["ego_vx"] = s["ego_vx"]
        self.state["ego_ax"] = s["ego_ax"]
        self.state["traj_speed"] = s["traj_speed"]
        if s["lead_valid"]:
            self.state["obj_x"] = s["obj_x"]
            self.state["obj_y"] = s["obj_y"]
            self.state["obj_vx"] = s["obj_vx"]
            self.state["obj_vy"] = s["obj_vy"]
            self.state["obj_L"] = s["obj_L"]
            self.state["obj_W"] = s["obj_W"]
        # 슬라이더가 있으면 동기화 (init 후엔 _sync_sliders 가 처리)
        self._reset_mpc_warm()

    # ---------------- widgets ----------------
    def _build_widgets(self):
        # sliders
        sl_specs = [
            ("ego_vx (m/s)",   "ego_vx",      0.0,  50.0),
            ("ego_ax (m/s²)",  "ego_ax",     -5.0,   5.0),
            ("obj_vx (m/s)",   "obj_vx",      0.0,  50.0),
            ("obj_vy (m/s)",   "obj_vy",    -10.0,  10.0),
            ("traj_speed",     "traj_speed",  5.0,  50.0),
            ("MPC iters/tick", "n_iter",      1,    10),
        ]
        self.sliders = {}
        for i, (label, key, lo, hi) in enumerate(sl_specs):
            ax = self.fig.add_axes([0.10, 0.21 - i * 0.025, 0.45, 0.020])
            s = Slider(ax, label, lo, hi, valinit=self.state[key],
                        valstep=(1 if isinstance(self.state[key], int) else 0.5))
            s.on_changed(self._make_slider_cb(key))
            self.sliders[key] = s

        # radio: trajectory preset (+ loaded sample option)
        rad_labels = [LOADED_SAMPLE_KEY] + list(PRESETS.keys())
        rad_ax = self.fig.add_axes([0.62, 0.04, 0.18, 0.20])
        active_idx = rad_labels.index(self.state["traj_preset"]) if self.state["traj_preset"] in rad_labels else 0
        self.radio = RadioButtons(rad_ax, rad_labels, active=active_idx)
        self.radio.on_clicked(self._on_preset_change)

        # buttons
        btn_ax_reset = self.fig.add_axes([0.84, 0.18, 0.12, 0.04])
        self.btn_reset = Button(btn_ax_reset, "Reset state")
        self.btn_reset.on_clicked(self._on_reset)

        btn_ax_pause = self.fig.add_axes([0.84, 0.12, 0.12, 0.04])
        self.btn_pause = Button(btn_ax_pause, "Pause MPC")
        self.btn_pause.on_clicked(self._on_pause)

        btn_ax_warm = self.fig.add_axes([0.84, 0.06, 0.12, 0.04])
        self.btn_warm = Button(btn_ax_warm, "Reset warm chain")
        self.btn_warm.on_clicked(lambda e: self._reset_mpc_warm())

        btn_ax_load = self.fig.add_axes([0.84, 0.00, 0.12, 0.04])
        self.btn_load = Button(btn_ax_load, "Load random sample")
        self.btn_load.on_clicked(self._on_load_sample)

    def _sync_sliders(self):
        """state 의 ego/obj/traj_speed 값을 slider 에 반영."""
        for k in ["ego_vx", "ego_ax", "obj_vx", "obj_vy", "traj_speed"]:
            if k in self.sliders:
                self.sliders[k].set_val(self.state[k])

    def _on_load_sample(self, event):
        _ = event
        try:
            self._load_synth_sample()
            self.state["traj_preset"] = LOADED_SAMPLE_KEY
            self._sync_sliders()
            # radio 도 동기화
            self.radio.set_active(0)
        except FileNotFoundError as e:
            print(f"warn: {e}")

    def _make_slider_cb(self, key):
        def cb(val):
            if isinstance(self.state[key], int):
                self.state[key] = int(val)
            else:
                self.state[key] = float(val)
        return cb

    def _on_preset_change(self, label):
        self.state["traj_preset"] = label

    def _on_reset(self, event):
        self.state.update(
            ego_vx=20.0, ego_ax=0.0,
            obj_x=30.0, obj_y=0.0,
            obj_vx=18.0, obj_vy=0.0,
            traj_speed=25.0,
        )
        for k, s in self.sliders.items():
            s.set_val(self.state[k])
        self._reset_mpc_warm()

    def _on_pause(self, event):
        self.state["paused"] = not self.state["paused"]
        self.btn_pause.label.set_text("Resume MPC" if self.state["paused"] else "Pause MPC")

    # ---------------- artists ----------------
    def _build_bev_artists(self):
        ax = self.ax_bev
        ax.set_xlim(-10, 80)
        ax.set_ylim(-30, 30)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x [m] (ego forward)")
        ax.set_ylabel("y [m] (ego left)")
        ax.set_title("BEV  (drag the red object)")
        # ego
        ax.add_patch(Rectangle((-2.5, -1.0), 5, 2, ec="k", fc="lightgray", alpha=0.8))
        ax.text(0, 0, "ego", ha="center", va="center", fontsize=9)
        # reference trajectory line + points
        self.line_ref, = ax.plot([], [], "k--", lw=1.5, alpha=0.7, label="reference traj")
        self.sc_ref = ax.scatter([], [], c="k", s=8, alpha=0.5)
        # lane boundary (filter: |n| ≤ LANE_HALF_WIDTH)
        self.line_lane_l, = ax.plot([], [], color="gray", ls=":", lw=1.0, alpha=0.6,
                                       label=f"lane ±{LANE_HALF_WIDTH:.1f}m")
        self.line_lane_r, = ax.plot([], [], color="gray", ls=":", lw=1.0, alpha=0.6)
        # MPC predicted trajectory
        self.line_mpc, = ax.plot([], [], "orange", lw=2.5, label="MPC predicted")
        self.sc_mpc = ax.scatter([], [], c="orange", s=14)
        # object box + velocity arrow
        self.obj_patch = Rectangle((self.state["obj_x"] - self.state["obj_L"] / 2,
                                     self.state["obj_y"] - self.state["obj_W"] / 2),
                                    self.state["obj_L"], self.state["obj_W"],
                                    ec="red", fc="salmon", alpha=0.7, picker=True)
        ax.add_patch(self.obj_patch)
        self.obj_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                       arrowprops=dict(arrowstyle="->", color="red", lw=2))
        self.obj_label = ax.text(0, 0, "", color="red", fontsize=9, ha="center", fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)

    def _build_ctrl_artists(self):
        ax = self.ax_ctrl
        ax.set_xlim(0, 4.0)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.set_title("MPC control output (steer / accel)")
        ax.set_xlabel("t (s)")
        self.line_steer, = ax.plot([], [], "C0-", lw=1.8, label="steer (deg)")
        self.ax_ctrl_2 = ax.twinx()
        self.ax_ctrl_2.set_ylim(-5, 5)
        self.line_accel, = self.ax_ctrl_2.plot([], [], "C3-", lw=1.8, label="accel (m/s²)")
        ax.set_ylabel("steer deg", color="C0")
        self.ax_ctrl_2.set_ylabel("accel m/s²", color="C3")
        # combined legend
        ax.legend(handles=[self.line_steer, self.line_accel], loc="upper right", fontsize=9)

    # ---------------- mouse drag ----------------
    def on_press(self, event):
        _ = event   # silence linter
        if event.inaxes != self.ax_bev or event.xdata is None: return
        # within object box?
        x0 = self.state["obj_x"] - self.state["obj_L"] / 2
        y0 = self.state["obj_y"] - self.state["obj_W"] / 2
        if (x0 <= event.xdata <= x0 + self.state["obj_L"]
            and y0 <= event.ydata <= y0 + self.state["obj_W"]):
            self.dragging = True

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax_bev or event.xdata is None: return
        self.state["obj_x"] = float(event.xdata)
        self.state["obj_y"] = float(event.ydata)

    def on_release(self, event):
        self.dragging = False

    # ---------------- tick ----------------
    def tick(self, _frame):
        if self.state["paused"]:
            self._render_info()
            return
        self.tick_counter += 1
        # 매 tick weights.ini mtime 체크 (저비용)
        self._refresh_weights_if_changed()

        # build messages
        ego = np.array([0, 0, 0, self.state["ego_vx"], 0.0, 0.0])
        vs = make_vs_msg(ego, ax=self.state["ego_ax"])
        # trajectory: loaded sample (real) 이면 캐시된 raw, 아니면 preset
        if self.state["traj_preset"] == LOADED_SAMPLE_KEY and self.state["loaded_traj"] is not None:
            traj_local = self.state["loaded_traj"].copy()
            # traj_speed slider 로 속도만 rescale (구조는 real sample)
            traj_local[:, 3] = self.state["traj_speed"]
        else:
            traj_local = PRESETS[self.state["traj_preset"]](self.state["traj_speed"])
        traj_msg = build_random_traj_msg(traj_local)   # 40 points, pool 컨벤션
        lead = (self.state["obj_x"], self.state["obj_y"],
                self.state["obj_L"], self.state["obj_W"],
                np.array([self.state["obj_vx"], self.state["obj_vy"]]))
        objs_msg = lead_to_objs_msg(lead)

        # MPC: warm-chain. 첫 tick 만 20 iter warmup, 이후는 n_iter (보통 3-5).
        # "Reset warm chain" 버튼 으로 cold restart 가능.
        n_iters = 20 if self._warmup_needed else self.state["n_iter"]
        self._warmup_needed = False
        t0 = time.perf_counter()
        cio = None
        for _ in range(n_iters):
            out = run_cycle(self.mpc, vs, traj_msg, objs_msg)
            _, _, _, _, cio_now = out
            if cio_now is not None:
                cio = cio_now
        self.last_solve_ms = (time.perf_counter() - t0) * 1000
        self.last_mpc_status = getattr(self.mpc, "last_mpc_status", None)
        # last_mpc_status 만 사용 — fallback_count 는 cumulative 라 신호 약함
        self.fallback_active = (self.last_mpc_status != "success")

        # update artists — ref traj + lane boundary
        ref_xy = traj_local[:, :2]
        self.line_ref.set_data(ref_xy[:, 0], ref_xy[:, 1])
        self.sc_ref.set_offsets(ref_xy)
        left, right = compute_lane_boundary(ref_xy, LANE_HALF_WIDTH)
        self.line_lane_l.set_data(left[:, 0], left[:, 1])
        self.line_lane_r.set_data(right[:, 0], right[:, 1])
        # lead 의 n 계산 → lane 안/밖 표시
        obj_xy_arr = np.array([self.state["obj_x"], self.state["obj_y"]])
        self.last_obj_n, self.last_obj_s = compute_obj_n(obj_xy_arr, ref_xy)
        in_lane = abs(self.last_obj_n) <= LANE_HALF_WIDTH
        self.obj_patch.set_edgecolor("red" if in_lane else "gray")
        self.obj_patch.set_facecolor("salmon" if in_lane else "lightgray")
        self.obj_patch.set_alpha(0.7 if in_lane else 0.4)
        self.last_obj_in_lane = in_lane
        # object box + arrow
        self.obj_patch.set_xy((self.state["obj_x"] - self.state["obj_L"] / 2,
                                self.state["obj_y"] - self.state["obj_W"] / 2))
        end_x = self.state["obj_x"] + self.state["obj_vx"] * 0.5
        end_y = self.state["obj_y"] + self.state["obj_vy"] * 0.5
        self.obj_arrow.set_position((self.state["obj_x"], self.state["obj_y"]))
        self.obj_arrow.xy = (end_x, end_y)
        self.obj_label.set_position((self.state["obj_x"], self.state["obj_y"] + self.state["obj_W"] / 2 + 1.5))
        self.obj_label.set_text(f"obj ({self.state['obj_x']:.1f}, {self.state['obj_y']:+.1f})")

        # control output (always plot if cio returned, even fallback)
        if cio is not None and "steer" in cio and "acceleration" in cio:
            steer = np.asarray(cio["steer"], dtype=float)
            accel = np.asarray(cio["acceleration"], dtype=float)
            t_arr = np.arange(len(steer)) * 0.05
            self.line_steer.set_data(t_arr, steer)
            self.line_accel.set_data(t_arr, accel)
            sm = max(0.2, float(np.abs(steer).max()) * 1.3)
            am = max(1.0, float(np.abs(accel).max()) * 1.3)
            self.ax_ctrl.set_ylim(-sm, sm)
            self.ax_ctrl_2.set_ylim(-am, am)
            self.last_status = f"ok (mpc_status={getattr(self.mpc, 'last_mpc_status', '?')})"
        else:
            self.last_status = "fail"

        # BEV 예측 trajectory: MPC 가 success 면 pred_X[:, 0:2] 직접 사용 (Cartesian).
        # fail 이면 ego_vx 상수 직진 으로 fallback 시각화.
        px = getattr(self.mpc, "pred_X", None)
        succeeded = (getattr(self.mpc, "last_mpc_status", None) == "success") and (px is not None) and (px.shape[0] > 1)
        if succeeded:
            xs, ys = px[:, 0], px[:, 1]
            self.line_mpc.set_data(xs, ys)
            self.sc_mpc.set_offsets(np.column_stack([xs[::8], ys[::8]]))
        else:
            # 직진 ghost trace
            tt = np.arange(81) * 0.05
            xs = self.state["ego_vx"] * tt
            ys = np.zeros_like(xs)
            self.line_mpc.set_data(xs, ys)
            self.sc_mpc.set_offsets(np.column_stack([xs[::8], ys[::8]]))

        self._render_info()

    def _render_info(self):
        s = self.state
        approach = s["ego_vx"] - s["obj_vx"]
        dist = float(np.hypot(s["obj_x"], s["obj_y"]))
        t_close = (dist - 5) / approach if approach > 0.5 else float("inf")
        mpc_status = getattr(self, "last_mpc_status", "?")
        fallback = getattr(self, "fallback_active", False)
        flag = "⚠ FALLBACK" if fallback else "✓"
        color = "red" if fallback else "black"
        loaded_fid = s.get("loaded_frame_id", "?")
        obj_n = getattr(self, "last_obj_n", 0.0)
        obj_s_val = getattr(self, "last_obj_s", 0.0)
        in_lane = getattr(self, "last_obj_in_lane", True)
        lane_flag = f"✓ IN LANE  (filter passes)" if in_lane else f"× OUT (|n|>{LANE_HALF_WIDTH}, FILTERED → MPC ignores)"
        # weights summary (recently changed → flash)
        w = self._weights_summary
        recently_reloaded = (self.tick_counter - self._weights_reloaded_at_tick) <= 5
        w_flash = "  ← RELOADED" if recently_reloaded else ""
        w_line = "  ".join(f"{k}={int(w[k]) if w.get(k, 0) >= 1 else w.get(k, 0):>5}"
                            for k in WEIGHT_DISPLAY_KEYS if k in w)
        info = (
            f"tick={self.tick_counter:>4d}   solve={self.last_solve_ms:>5.1f}ms ({s['n_iter']} iters)   "
            f"MPC: {flag} (status={mpc_status})   paused={s['paused']}\n"
            f"ego: vx={s['ego_vx']:.1f}  ax={s['ego_ax']:+.2f}        "
            f"obj: pos=({s['obj_x']:.1f}, {s['obj_y']:+.1f})  vel=({s['obj_vx']:.1f}, {s['obj_vy']:+.1f})   "
            f"approach={approach:+.1f} m/s  TTC={t_close:.1f}s\n"
            f"obj frenet: n={obj_n:+.2f}m  s={obj_s_val:.1f}m   {lane_flag}\n"
            f"traj: {s['traj_preset']} @ {s['traj_speed']:.1f} m/s    sample: {loaded_fid}\n"
            f"weights.ini: {w_line}{w_flash}"
        )
        self.ax_info_text.set_text(info)
        self.ax_info_text.set_color(color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true",
                    help="DISPLAY 없이 N tick 돌리고 snapshot PNG 저장 (스모크 테스트)")
    ap.add_argument("--ticks", type=int, default=30,
                    help="--headless 일 때 실행할 tick 수")
    ap.add_argument("--save", default="outputs_multi/mpc_viz_headless.png",
                    help="--headless snapshot 저장 경로")
    args = ap.parse_args()

    if args.headless:
        matplotlib.use("Agg")
    else:
        matplotlib.use("Qt5Agg")

    app = MPCInteractive()
    if args.headless:
        print(f"headless mode: {args.ticks} ticks, then save → {args.save}")
        for k in range(args.ticks):
            app.tick(k)
        out = args.save
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        app.fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"saved {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    else:
        print("MPC interactive viz running — close window to quit.")
        plt.show()


if __name__ == "__main__":
    main()
