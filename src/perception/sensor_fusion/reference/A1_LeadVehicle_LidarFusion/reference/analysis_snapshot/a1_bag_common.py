"""Shared helpers for analyze_mapping_bag.py / analyze_localization_bag.py.

Run with ROS 2 Humble and the A1 workspace sourced (rosbag2_py, gsof_msgs,
mola_msgs message definitions are needed to deserialize the bag).
"""
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import yaml

GPS_EPOCH_UNIX = 315964800.0   # 1980-01-06T00:00:00Z
GPS_UTC_LEAP = 18.0            # GPS - UTC since 2017
KST = dt.timezone(dt.timedelta(hours=9))
MOLA_PLUGIN = "libmola_metric_maps.so"


# ----------------------------------------------------------------- reporting
class Report:
    def __init__(self):
        self.lines = []
        self.summary = {}

    def h(self, title):
        self.add("")
        self.add(f"== {title}")

    def add(self, line=""):
        print(line)
        self.lines.append(line)

    def warn(self, line):
        self.add(f"  [주의] {line}")

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.summary, f, ensure_ascii=False, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def kst(t):
    return dt.datetime.fromtimestamp(float(t), KST).strftime("%Y-%m-%d %H:%M:%S KST")


# ----------------------------------------------------------------- rosbag2
def bag_storage_id(bag_dir):
    with open(os.path.join(bag_dir, "metadata.yaml"), encoding="utf-8") as f:
        meta = yaml.safe_load(f)["rosbag2_bagfile_information"]
    return meta.get("storage_identifier", "mcap"), meta


def read_bag(bag_dir, topics, progress=True):
    """Return {topic: [(receive_time_s, msg), ...]} for the topics present in the bag."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_id, meta = bag_storage_id(bag_dir)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    present = [t for t in topics if t in types]
    missing = [t for t in topics if t not in types]
    if not present:
        return {}, missing
    reader.set_filter(rosbag2_py.StorageFilter(topics=present))
    cls = {t: get_message(types[t]) for t in present}
    out = {t: [] for t in present}
    n = 0
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        out[topic].append((t_ns * 1e-9, deserialize_message(raw, cls[topic])))
        n += 1
        if progress and n % 20000 == 0:
            print(f"  ... {n} messages")
    return out, missing


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def fix_time_base(recv, hdr):
    """Convert GPS-epoch header stamps to Unix time if the driver ran with time_source=gps.

    Returns (hdr_unix, mode, host_minus_stamp_median).
    mode: 'gps' (converted), 'host' (stamps already in the host clock) or 'unknown'.
    """
    recv, hdr = np.asarray(recv, float), np.asarray(hdr, float)
    if len(hdr) == 0:
        return hdr, "unknown", float("nan")
    d = float(np.median(recv - hdr))
    if abs(d - (GPS_EPOCH_UNIX - GPS_UTC_LEAP)) < 3600:
        conv = hdr + GPS_EPOCH_UNIX - GPS_UTC_LEAP
        return conv, "gps", float(np.median(recv - conv))
    if abs(d) < 3600:
        return hdr, "host", d
    return hdr, "unknown", d


# ----------------------------------------------------------------- geometry
def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_yaw_deg(q):
    q = np.atleast_2d(q)
    return np.degrees(np.arctan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                                 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)))


def rotz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_to_R(roll_deg, pitch_deg, yaw_deg):
    r, p, y = (math.radians(v) for v in (roll_deg, pitch_deg, yaw_deg))
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(np.asarray(a))))


def umeyama_rigid(X, Y):
    """Rigid R, t (no scale) with Y ~ R X + t, X/Y shape (n, d)."""
    mx, my = X.mean(0), Y.mean(0)
    H = (X - mx).T @ (Y - my)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(X.shape[1])
    D[-1, -1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, my - R @ mx


def interp_rows(t_src, Y, t_q):
    Y = np.atleast_2d(Y.T).T
    return np.stack([np.interp(t_q, t_src, Y[:, j]) for j in range(Y.shape[1])], -1)


def path_length(xy):
    return float(np.linalg.norm(np.diff(np.asarray(xy)[:, :2], axis=0), axis=1).sum())


def total_turning_deg(xy):
    d = np.diff(np.asarray(xy)[:, :2], axis=0)
    ok = np.linalg.norm(d, axis=1) > 0.05
    h = np.unwrap(np.arctan2(d[ok, 1], d[ok, 0]))
    return float(np.degrees(h[-1] - h[0])) if len(h) > 1 else 0.0


# ----------------------------------------------------------------- WGS84
_A, _F = 6378137.0, 1 / 298.257223563
_E2 = _F * (2 - _F)


def geodetic_to_ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    N = _A / np.sqrt(1 - _E2 * np.sin(la) ** 2)
    return np.stack([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo),
                     (N * (1 - _E2) + h) * np.sin(la)], -1)


class Enu:
    def __init__(self, lat0, lon0, h0):
        self.lat0, self.lon0, self.h0 = float(lat0), float(lon0), float(h0)
        la, lo = math.radians(self.lat0), math.radians(self.lon0)
        self.R = np.array([[-math.sin(lo), math.cos(lo), 0],
                           [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                           [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
        self.origin = geodetic_to_ecef(self.lat0, self.lon0, self.h0)

    def from_geodetic(self, lat, lon, h):
        return (geodetic_to_ecef(np.asarray(lat), np.asarray(lon), np.asarray(h)) - self.origin) @ self.R.T

    def to_geodetic(self, enu):
        X = np.atleast_2d(enu) @ self.R + self.origin
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        lon = np.arctan2(y, x)
        p = np.hypot(x, y)
        lat = np.arctan2(z, p * (1 - _E2))
        for _ in range(6):
            N = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
            h = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1 - _E2 * N / (N + h)))
        return np.degrees(lat), np.degrees(lon), h


# ----------------------------------------------------------------- MOLA map files
def map_prefix(path):
    for ext in (".mm", ".simplemap", ".georef.yaml", ".georef"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


class Georef:
    """T_enu_to_map (map frame expressed in ENU) + ENU origin, from <prefix>.georef.yaml or extracted from <prefix>.mm."""

    def __init__(self, prefix, work_dir):
        yml = prefix + ".georef.yaml"
        if not os.path.isfile(yml):
            mm = prefix + ".mm"
            if not os.path.isfile(mm):
                raise FileNotFoundError(f"{yml} / {mm} 둘 다 없음")
            yml = os.path.join(work_dir, "georef_extracted.yaml")
            cmd = ["mm-georef", "-m", mm, "-g", yml, "--extract-from-map", "-l", MOLA_PLUGIN]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.isfile(yml):
                raise RuntimeError("mm-georef --extract-from-map 실패 (맵에 georef가 없거나 플러그인 문제):\n" + r.stdout[-800:] + r.stderr[-800:])
        with open(yml, encoding="utf-8") as f:
            g = yaml.safe_load(f)
        if not g.get("defined", True):
            raise RuntimeError(f"{yml}: georeferencing이 정의되지 않음 (defined: false)")
        m = g["T_enu_to_map"]["mean"]
        self.source = yml
        self.R_em = rpy_to_R(m["roll_deg"], m["pitch_deg"], m["yaw_deg"])
        self.t_em = np.array([m["x"], m["y"], m["z"]], float)
        self.yaw_deg = float(m["yaw_deg"])
        self.pitch_deg, self.roll_deg = float(m["pitch_deg"]), float(m["roll_deg"])
        cov = np.array(g["T_enu_to_map"]["cov"]["data"], float).reshape(6, 6)
        self.std = np.sqrt(np.diag(cov))   # x y z yaw pitch roll (rad)
        gc = g["geo_coord"]
        self.enu = Enu(gc["lat"], gc["lon"], gc["altitude"])

    def enu_to_map(self, p):
        return (np.atleast_2d(p) - self.t_em) @ self.R_em

    def map_to_enu(self, p):
        return np.atleast_2d(p) @ self.R_em.T + self.t_em

    def map_yaw_to_enu(self, yaw_deg):
        return np.asarray(yaw_deg) + self.yaw_deg

    def enu_yaw_to_map(self, yaw_deg):
        return np.asarray(yaw_deg) - self.yaw_deg


def load_keyframes(prefix=None, tum_path=None, work_dir=None):
    """Keyframe poses as array [t, x, y, z, qx, qy, qz, qw] (TUM). From a .simplemap via sm-cli, or a TUM file."""
    if tum_path is None:
        sm = prefix + ".simplemap"
        if not os.path.isfile(sm):
            raise FileNotFoundError(sm)
        if shutil.which("sm-cli") is None:
            raise RuntimeError("sm-cli 없음 (ROS/mola 환경 source 필요)")
        tum_path = os.path.join(work_dir, "keyframes.tum")
        r = subprocess.run(["sm-cli", "export-keyframes", sm, "--output", tum_path], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("sm-cli export-keyframes 실패:\n" + r.stdout[-500:] + r.stderr[-500:])
    kf = np.loadtxt(tum_path)
    if kf.ndim == 1:
        kf = kf[None]
    return kf


# ----------------------------------------------------------------- GNSS / TF extraction
def extract_tf_static(tf_msgs):
    """{(parent, child): (xyz, quat)} from /tf_static messages."""
    out = {}
    for _, m in tf_msgs:
        for tr in m.transforms:
            t, q = tr.transform.translation, tr.transform.rotation
            out[(tr.header.frame_id, tr.child_frame_id)] = (np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w]))
    return out


def extract_tf_series(tf_msgs, parent, child):
    """Array [[header_t, recv_t, x, y, z, qx, qy, qz, qw], ...] for one dynamic transform."""
    rows = []
    for recv, m in tf_msgs:
        for tr in m.transforms:
            if tr.header.frame_id == parent and tr.child_frame_id == child:
                t, q = tr.transform.translation, tr.transform.rotation
                rows.append([stamp_s(tr), recv, t.x, t.y, t.z, q.x, q.y, q.z, q.w])
    return np.array(rows) if rows else np.zeros((0, 9))


class InsData:
    """INS/GNSS series from the raw driver topics (preferred) or the relay topics."""

    def __init__(self, bag, rep):
        self.ok = False
        nav = bag.get("/gsof_client/navsat") or bag.get("/a1_gnss/navsat")
        odom = bag.get("/gsof_client/odom") or bag.get("/a1_gnss/odom")
        ins = bag.get("/gsof_client/gsof/ins_solution_49")
        rms = bag.get("/gsof_client/gsof/ins_solution_rms_50")
        self.odom_child = None
        if not nav:
            rep.warn("NavSatFix 토픽 없음 (/gsof_client/navsat 또는 /a1_gnss/navsat) → INS 비교 생략")
            return
        recv = np.array([r for r, _ in nav])
        hdr = np.array([stamp_s(m) for _, m in nav])
        self.t, self.time_mode, self.host_minus_stamp = fix_time_base(recv, hdr)
        self.recv = recv
        self.lla = np.array([[m.latitude, m.longitude, m.altitude] for _, m in nav])
        self.navsat_status = np.array([m.status.status for _, m in nav])
        self.cov0 = np.array([m.position_covariance[0] for _, m in nav])
        if odom:
            self.odom_child = odom[0][1].child_frame_id
            orc = np.array([r for r, _ in odom])
            oh = np.array([stamp_s(m) for _, m in odom])
            self.odom_t, _, _ = fix_time_base(orc, oh)
            self.odom_xyz = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z] for _, m in odom])
            self.odom_q = np.array([[m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w] for _, m in odom])
            self.odom_cov0 = np.array([m.pose.covariance[0] for _, m in odom])
        else:
            self.odom_t = None
        if ins:
            irc = np.array([r for r, _ in ins])
            ih = np.array([stamp_s(m) for _, m in ins])
            self.ins_t, _, _ = fix_time_base(irc, ih)
            self.speed = np.array([m.total_speed for _, m in ins])
            self.heading = np.array([m.heading for _, m in ins])          # compass deg
            self.gnss_fix = np.array([m.status.gnss for _, m in ins])
            self.imu_align = np.array([m.status.imu_alignment for _, m in ins])
        else:
            self.ins_t = None
        if rms:
            rrc = np.array([r for r, _ in rms])
            rh = np.array([stamp_s(m) for _, m in rms])
            self.rms_t, _, _ = fix_time_base(rrc, rh)
            self.rms_h = np.array([math.hypot(m.pos_rms_error.north, m.pos_rms_error.east) for _, m in rms])
            self.rms_d = np.array([m.pos_rms_error.down for _, m in rms])
            self.rms_hdg = np.array([m.attitude_rms_error_heading for _, m in rms])
        else:
            self.rms_t = None
        self.ok = True

    def heading_at(self, t):
        if self.ins_t is None:
            return None
        return np.interp(t, self.ins_t, unwrap_deg(self.heading))

    def summarize(self, rep):
        FIX = {0: "no fix", 1: "SPS", 2: "DGPS", 3: "PPS", 4: "RTK fixed", 5: "RTK float"}
        ALIGN = {0: "GPS only", 1: "coarse", 2: "degraded", 3: "aligned", 4: "full nav", 5: "unknown"}
        rep.h("INS/GNSS 상태")
        rep.add(f"  구간: {kst(self.t[0])} → {kst(self.t[-1])} ({self.t[-1] - self.t[0]:.0f} s), NavSatFix {len(self.t)}개")
        rep.add(f"  타임스탬프: {self.time_mode} ({'GPS epoch → UTC 변환함' if self.time_mode == 'gps' else 'host 시각'}); 수신시각 − 스탬프 중앙값 {self.host_minus_stamp:+.3f} s"
                + (" ← host 클럭 − GPS 클럭 + 지연" if self.time_mode == "gps" else " ← 전송/처리 지연"))
        s = {"time_mode": self.time_mode, "host_minus_stamp_s": self.host_minus_stamp, "n_navsat": int(len(self.t))}
        if self.ins_t is not None:
            fh = {FIX.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(self.gnss_fix, return_counts=True))}
            ah = {ALIGN.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(self.imu_align, return_counts=True))}
            n = len(self.gnss_fix)
            rep.add("  GNSS fix: " + ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in fh.items()))
            rep.add("  INS 정렬: " + ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in ah.items()))
            mv = self.speed > 0.3
            rep.add(f"  속도: 최대 {self.speed.max():.2f} m/s, 주행 평균 {self.speed[mv].mean() if mv.any() else 0:.2f} m/s, 정차(<0.3 m/s) {100 * (~mv).mean():.0f}%")
            s.update(fix_hist=fh, align_hist=ah, speed_max=float(self.speed.max()), rtk_fixed_frac=float((self.gnss_fix == 4).mean()))
            if (self.gnss_fix == 4).mean() < 0.95:
                rep.warn("RTK fixed 비율이 95% 미만 → georef/비교 정확도 저하")
            # RTK fixed 구간 목록 (연속 구간, 2 s 미만 끊김은 무시)
            segs, i = [], 0
            fixed = (self.gnss_fix == 4)
            while i < len(fixed):
                if not fixed[i]:
                    i += 1; continue
                j = i
                while j < len(fixed) - 1 and (fixed[j + 1] or (self.ins_t[min(j + 200, len(fixed) - 1)] - self.ins_t[j] < 2.0 and fixed[min(j + 200, len(fixed) - 1)])):
                    j += 1
                segs.append((i, j)); i = j + 1
            t0 = self.ins_t[0]
            dist = np.r_[0, np.cumsum(np.diff(self.ins_t) * self.speed[:-1])]
            lines = [f"t+{self.ins_t[a] - t0:.0f}–{self.ins_t[b] - t0:.0f} s ({kst(self.ins_t[a])[11:19]}–{kst(self.ins_t[b])[11:19]}, {dist[b] - dist[a]:.0f} m)" for a, b in segs if self.ins_t[b] - self.ins_t[a] >= 5]
            rep.add("  RTK fixed 구간: " + ("; ".join(lines) if lines else "없음") + f"  | 전체 주행거리 {dist[-1]:.0f} m 중 fixed {sum(dist[b] - dist[a] for a, b in segs):.0f} m")
            s["rtk_segments_s"] = [[float(self.ins_t[a] - t0), float(self.ins_t[b] - t0)] for a, b in segs]
            # 헤딩 점프/데이터 끊김: map→odom 점프의 직접 원인
            dh = np.abs(wrap_deg(np.diff(self.heading)))
            dtm = np.diff(self.ins_t)
            jumps = np.flatnonzero((dh > 5) | (dh / np.maximum(dtm, 1e-3) > 90))
            gaps = np.flatnonzero(dtm > 0.5)
            s.update(heading_jumps=int(len(jumps)), data_gaps=int(len(gaps)))
            if len(jumps):
                rep.warn(f"INS 헤딩 점프 {len(jumps)}회 (연속 샘플 간 >5° 또는 >90°/s): t+"
                         + ", ".join(f"{self.ins_t[i] - self.ins_t[0]:.0f}" for i in jumps[:10]) + ("..." if len(jumps) > 10 else "")
                         + f" s; 그 시점 INS 정렬 {sorted(set(self.imu_align[jumps].astype(int).tolist()))} → odom→base_link와 map→odom이 같이 튐")
            if len(gaps):
                rep.add(f"  INS 데이터 끊김(>0.5 s) {len(gaps)}회, 최대 {dtm.max():.2f} s")
        if self.rms_t is not None:
            rep.add(f"  INS 보고 RMS: 수평 중앙값 {100 * np.median(self.rms_h):.1f} cm (최대 {self.rms_h.max():.2f} m), 수직 중앙값 {100 * np.median(self.rms_d):.1f} cm, 헤딩 중앙값 {np.median(self.rms_hdg):.2f}°")
            s.update(rms_h_median=float(np.median(self.rms_h)), rms_h_max=float(self.rms_h.max()))
        if self.odom_t is not None:
            rep.add(f"  odom: child_frame '{self.odom_child}', {len(self.odom_t)}개, pose.covariance[0] {'-1 포함 → GSOF #50 미수신 구간 있음' if (self.odom_cov0 < 0).any() else '정상'}")
        rep.add(f"  고도 범위 {np.ptp(self.lla[:, 2]):.2f} m (GNSS 고도 노이즈 참고)")
        s["alt_span_m"] = float(np.ptp(self.lla[:, 2]))
        rep.summary["ins"] = s


def extrinsics_from_tf_static(tfs, base="base_link", rep=None):
    """base_link->ins and base_link->lidar from /tf_static; returns dict name -> (xyz, R)."""
    out = {}
    for (p, c), (xyz, q) in tfs.items():
        if p == base and c in ("ins", "hesai_lidar"):
            out[c] = (xyz, quat_to_R(q))
            if rep:
                yaw = quat_yaw_deg(q)[0]
                rep.add(f"  /tf_static {p}→{c}: xyz ({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) m, yaw {yaw:+.1f}°")
    return out


def lag_scan(fn, lags):
    """fn(lag) -> cost; returns (best_lag, best_cost, cost_at_0)."""
    costs = np.array([fn(l) for l in lags])
    i = int(np.argmin(costs))
    j = int(np.argmin(np.abs(lags)))
    return float(lags[i]), float(costs[i]), float(costs[j])


def setup_matplotlib():
    """Headless matplotlib with a Korean-capable font; returns pyplot."""
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    import matplotlib.font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"):
        if cand in names:
            matplotlib.rcParams["font.family"] = cand
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    return plt


def arc_indices(xy, step):
    """Indices of poses spaced at least `step` m apart along the path (always starts with 0)."""
    xy = np.asarray(xy)[:, :2]
    idx, acc = [0], 0.0
    for i in range(1, len(xy)):
        acc += float(np.linalg.norm(xy[i] - xy[i - 1]))
        if acc >= step:
            idx.append(i)
            acc = 0.0
    return np.array(idx)


def motion_direction_deg(p, R, step=0.5):
    """Median travel direction in the body frame (deg, 0 = +x), from poses `step` m apart. Returns (deg, n)."""
    idx = arc_indices(p, step)
    if len(idx) < 4:
        return float("nan"), 0
    ps, Rs = np.asarray(p)[idx], np.asarray(R)[idx]
    mv = ps[2:] - ps[:-2]
    loc = np.einsum("nji,nj->ni", Rs[1:-1], mv)
    ang = np.degrees(np.arctan2(loc[:, 1], loc[:, 0]))
    return float(np.median(ang)), int(len(ang))


def lag_from_heading(t, yaw_enu_deg, ins, lags, speed_min=0.3):
    """Time lag to add to `t` so INS heading(t+lag) best follows yaw_enu(t), judged by the spread of their
    difference while moving (a constant mounting offset is removed). Independent of the lever arm.
    Returns (lag, spread_at_lag_deg, spread_at_0_deg, median_offset_deg) or None without INS heading."""
    if ins.ins_t is None:
        return None
    yaw_enu_deg = np.asarray(yaw_enu_deg)

    def spread(lag):
        hd = ins.heading_at(t + lag)
        sp = np.interp(t + lag, ins.ins_t, ins.speed)
        d = wrap_deg(yaw_enu_deg - (90.0 - hd))
        m = sp > speed_min
        if m.sum() < 10:
            return float("inf"), float("nan")
        med = float(np.median(d[m]))
        return float(np.percentile(np.abs(wrap_deg(d[m] - med)), 68)), med

    res = [spread(l) for l in lags]
    i = int(np.argmin([r[0] for r in res]))
    j = int(np.argmin(np.abs(lags)))
    return float(lags[i]), res[i][0], res[j][0], res[i][1]


def default_out_dir(bag_dir, kind):
    base = os.path.basename(os.path.normpath(bag_dir))
    return os.path.join(os.path.dirname(os.path.normpath(bag_dir)), f"analysis_{kind}_{base}")


def make_work_dir():
    return tempfile.mkdtemp(prefix="a1_analysis_")
