#!/usr/bin/env python3
"""RTK-INS로 키프레임 포즈를 다시 잡아(=GNSS-anchored) 신뢰도 높은 맵을 재구성.

  python3 ~/a1_analysis/gnss_anchor_map.py --bag <매핑 bag> --simplemap <map.simplemap> --out <디렉터리>

원리
  위치   : RTK fixed 구간은 GNSS 위치(cm), 그 외 구간은 라이다 오도메트리의 상대 이동량(차체 프레임)
           + DGPS/SPS 위치(공분산 가중)를 최소제곱으로 융합  → z 드리프트 제거
  자세   : INS roll/pitch/heading (중력 기준) + 장착 yaw 보정(데이터로 추정)
  시각   : 라이다–GNSS 스탬프 차이를 헤딩으로 추정 (host 클럭 드리프트는 선형 모델)
  결과 맵: ENU 프레임 (x 동, y 북, z 위; 원점 = 첫 RTK fix). georef는 항등변환 + 원점 좌표.
출력: <out>/map.simplemap (포즈 교체), map.mm (sm2mm), map.georef.yaml, keyframes.tum, anchor_report/, *.png
필요: ~/a1_analysis/smpose (C++ 유틸, simplemap 키프레임 포즈 교체), sm2mm, mm-georef
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a1_bag_common import (MOLA_PLUGIN, Enu, InsData, Report, extract_tf_static, extrinsics_from_tf_static, interp_rows,  # noqa: E402
                           kst, make_work_dir, quat_to_R, quat_yaw_deg, read_bag, rotz, setup_matplotlib, stamp_s, unwrap_deg,
                           wrap_deg)

HERE = os.path.dirname(os.path.abspath(__file__))
TOPICS = ["/gsof_client/navsat", "/gsof_client/odom", "/gsof_client/gsof/ins_solution_49", "/tf_static"]


def R_to_quat(R):
    """Rotation matrix -> (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s])


def R_to_ypr_deg(R):
    return (math.degrees(math.atan2(R[1, 0], R[0, 0])), math.degrees(math.asin(max(-1, min(1, -R[2, 0])))),
            math.degrees(math.atan2(R[2, 1], R[2, 2])))


def ypr_to_R(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = (f(math.radians(a)) for a in (yaw, yaw, pitch, pitch, roll, roll) for f in (math.cos, math.sin)) if False else \
        (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), math.cos(math.radians(pitch)), math.sin(math.radians(pitch)),
         math.cos(math.radians(roll)), math.sin(math.radians(roll)))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def heading_lag(t_kf, yaw_kf, ins, lag_range, rep):
    """Lag (array per keyframe) so that INS heading(t+lag) follows LO yaw(t); windowed linear model for clock drift."""
    def spread(l, m=None):
        tt = t_kf if m is None else t_kf[m]
        yk = yaw_kf if m is None else yaw_kf[m]
        hd = ins.heading_at(tt + l)
        sp = np.interp(tt + l, ins.ins_t, ins.speed)
        mm = sp > 0.5
        if mm.sum() < 15:
            return np.inf
        d = wrap_deg(yk - (90.0 - hd))[mm]
        return float(np.percentile(np.abs(wrap_deg(d - np.median(d))), 68))
    lags = np.round(np.arange(-lag_range, lag_range + 1e-9, 0.05), 2)
    c = [spread(l) for l in lags]
    lag = float(lags[int(np.argmin(c))])
    fine = np.round(np.arange(lag - 0.05, lag + 0.051, 0.01), 3)
    lag = float(fine[int(np.argmin([spread(l) for l in fine]))])
    rep.add(f"  전역 lag {lag:+.2f} s (헤딩 차 산포 {min(c):.2f}°, lag 0: {spread(0.0):.2f}°)")
    rows = []
    lw = np.round(np.arange(lag - 1.0, lag + 1.001, 0.02), 2)
    for ws in np.arange(t_kf[0], t_kf[-1], 150.0):
        m = (t_kf >= ws) & (t_kf < ws + 150.0)
        if m.sum() < 30:
            continue
        cw = [spread(l, m) for l in lw]
        if np.isfinite(min(cw)):
            rows.append((ws + 75.0 - t_kf[0], float(lw[int(np.argmin(cw))])))
    lag_arr = np.full(len(t_kf), lag)
    if len(rows) >= 3:
        rows = np.array(rows)
        b, a = np.polyfit(rows[:, 0], rows[:, 1], 1)
        rep.add("  구간별 lag: " + ", ".join(f"t+{r[0]:.0f}s→{r[1]:+.2f}" for r in rows) + f" → 선형 {a:+.2f} {b * 1e3:+.3f} ms/s (드리프트 {b * 1e6:+.0f} ppm)")
        if abs(b) > 5e-5:
            lag_arr = a + b * (t_kf - t_kf[0])
    return lag_arr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--simplemap", required=True, help="키프레임+스캔 원본 (.simplemap)")
    ap.add_argument("--out", required=True, help="출력 디렉터리")
    ap.add_argument("--lever", type=float, nargs=3, help="base_link→ins xyz (기본: bag /tf_static)")
    ap.add_argument("--mount-yaw", type=float, help="INS 장착 yaw [deg] (기본: 데이터로 추정)")
    ap.add_argument("--lag-range", type=float, default=5.0)
    ap.add_argument("--odom-frac", type=float, default=0.01, help="LO 상대 이동량 σ = frac × 이동거리")
    ap.add_argument("--pipeline", default=os.path.join(HERE, "sm2mm_full_keyframe_map.yaml"))
    ap.add_argument("--no-build", action="store_true", help="포즈만 계산 (simplemap/mm 생성 생략)")
    ap.add_argument("--fit-lever", action="store_true", help="GNSS/LO 증분으로 추정한 레버암을 사용 (기본: tf_static 값)")
    ap.add_argument("--sensor-pose", type=float, nargs=6, metavar=("X", "Y", "Z", "YAW", "PITCH", "ROLL"),
                    help="base_link→lidar 를 이 값(m, deg)으로 덮어써서 스캔을 다시 놓음 (라이다 장착 기울기 보정)")
    args = ap.parse_args()

    rep = Report()
    work = make_work_dir()
    os.makedirs(args.out, exist_ok=True)
    # --- keyframes (LO poses)
    tum = os.path.join(work, "kf_lo.tum")
    subprocess.run(["sm-cli", "export-keyframes", args.simplemap, "--output", tum], check=True, capture_output=True)
    kf = np.loadtxt(tum)
    t_kf, p_lo, q_lo = kf[:, 0], kf[:, 1:4], kf[:, 4:8]
    R_lo = np.array([quat_to_R(q) for q in q_lo])
    yaw_lo = quat_yaw_deg(q_lo)
    n = len(kf)
    rep.h("입력")
    rep.add(f"  simplemap {args.simplemap}: 키프레임 {n}개 {kst(t_kf[0])} → {kst(t_kf[-1])}")

    # --- bag
    bag, _ = read_bag(args.bag, TOPICS)
    tfs = extract_tf_static(bag.get("/tf_static", []))
    ext = extrinsics_from_tf_static(tfs, "base_link", rep)
    lever = np.array(args.lever) if args.lever else (ext["ins"][0] if "ins" in ext else np.zeros(3))
    ins = InsData(bag, rep)
    if not ins.ok or ins.ins_t is None or ins.odom_t is None:
        rep.add("INS 데이터 부족 (navsat/odom/ins_solution_49 필요)"); sys.exit(1)
    nav = bag["/gsof_client/navsat"] if "/gsof_client/navsat" in bag else bag["/a1_gnss/navsat"]
    cov_h = np.array([m.position_covariance[0] for _, m in nav])
    cov_v = np.array([m.position_covariance[8] for _, m in nav])

    # --- time lag
    rep.h("시각 정렬")
    lag_arr = heading_lag(t_kf, yaw_lo, ins, args.lag_range, rep)
    tq = t_kf + lag_arr
    inb = (tq > ins.t[0]) & (tq < ins.t[-1])
    rep.add(f"  bag 시간 안 키프레임 {inb.sum()}/{n} (밖: {(~inb).sum()}개 → GNSS 없이 오도메트리로만 연결)")

    # --- ENU, GNSS at keyframes
    fix_kf = np.round(np.interp(tq, ins.ins_t, ins.gnss_fix)).astype(int)
    first_fix = np.flatnonzero((ins.gnss_fix == 4))
    i0 = int(np.searchsorted(ins.t, ins.ins_t[first_fix[0]])) if len(first_fix) else 0
    enu = Enu(*ins.lla[i0])
    g_ins = interp_rows(ins.t, enu.from_geodetic(ins.lla[:, 0], ins.lla[:, 1], ins.lla[:, 2]), tq)
    sh = np.sqrt(np.maximum(np.interp(tq, ins.t, cov_h), 0.0))
    sv = np.sqrt(np.maximum(np.interp(tq, ins.t, cov_v), 0.0))
    sh = np.where(fix_kf == 4, np.maximum(sh, 0.02), np.maximum(sh, 0.5))
    sv = np.where(fix_kf == 4, np.maximum(sv, 0.03), np.maximum(sv, 1.0))
    sv = np.where(sv < sh, sh, sv)

    # --- INS attitude at keyframes (odom quaternion = INS body in ENU/FLU)
    yo = unwrap_deg(quat_yaw_deg(ins.odom_q))
    ypr = np.array([R_to_ypr_deg(quat_to_R(q)) for q in ins.odom_q])
    yaw_i = np.interp(tq, ins.odom_t, yo)
    pitch_i = np.interp(tq, ins.odom_t, ypr[:, 1])
    roll_i = np.interp(tq, ins.odom_t, ypr[:, 2])
    # mount yaw: INS yaw vs LO yaw (LO frame -> ENU yaw from a 2D rigid fit on RTK samples)
    rep.h("자세")
    m_fit = inb & (fix_kf == 4)
    A, B = p_lo[m_fit, :2], g_ins[m_fit, :2]
    H = (A - A.mean(0)).T @ (B - B.mean(0))
    yaw_fit = math.degrees(math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
    sp_kf = np.interp(tq, ins.ins_t, ins.speed)
    mv = inb & (sp_kf > 1.0)
    mount = float(np.median(wrap_deg(yaw_i[mv] - (yaw_lo[mv] + yaw_fit)))) if args.mount_yaw is None else args.mount_yaw
    rep.add(f"  LO 맵 프레임 → ENU yaw (RTK {m_fit.sum()}점 정합) {yaw_fit:.2f}°; INS 장착 yaw (INS yaw − 차체 yaw) = {mount:+.2f}°"
            + ("" if args.mount_yaw is None else " (지정값)"))
    rep.add(f"  INS roll/pitch 범위: roll [{roll_i.min():.1f}, {roll_i.max():.1f}]°, pitch [{pitch_i.min():.1f}, {pitch_i.max():.1f}]° (INS 본체 기준)")
    R_base = np.array([ypr_to_R(yaw_i[k], pitch_i[k], roll_i[k]) @ rotz(-mount) for k in range(n)])
    # yaw consistency check vs LO relative yaw increments
    dyaw_lo = wrap_deg(np.diff(yaw_lo))
    dyaw_ins = wrap_deg(np.diff(np.array([R_to_ypr_deg(R)[0] for R in R_base])))
    rep.add(f"  키프레임 간 yaw 증분 LO vs INS: 차이 중앙값 {np.median(dyaw_ins - dyaw_lo):+.3f}°, 68% {np.percentile(np.abs(dyaw_ins - dyaw_lo), 68):.3f}°")

    # --- lever-arm check from increments: g(i+1)-g(i) = R_i d_i + (R_{i+1}-R_i) lever  (lag-insensitive)
    d_body = np.einsum("nji,nj->ni", R_lo[:-1], p_lo[1:] - p_lo[:-1])  # LO increments in body frame
    rep.h("레버암 검증 (GNSS 증분 − LO 증분, RTK 연속 쌍)")
    pair = inb[:-1] & inb[1:] & (fix_kf[:-1] == 4) & (fix_kf[1:] == 4)
    Ai = (R_base[1:] - R_base[:-1])[pair]
    bi = (g_ins[1:] - g_ins[:-1])[pair] - np.einsum("nij,nj->ni", R_base[:-1][pair], d_body[pair])
    keep = np.ones(pair.sum(), bool)
    lever_fit, sig = np.zeros(3), np.zeros(3)
    for _ in range(3):
        A2, b2 = Ai[keep].reshape(-1, 3), bi[keep].reshape(-1)
        lever_fit, *_ = np.linalg.lstsq(A2, b2, rcond=None)
        r = bi - np.einsum("nij,j->ni", Ai, lever_fit)
        rn = np.linalg.norm(r[:, :2], axis=1)
        keep = rn < 3 * np.sqrt((rn[keep] ** 2).mean())
    try:
        cov = np.linalg.inv(A2.T @ A2) * ((b2 - A2 @ lever_fit) ** 2).mean()
        sig = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        pass
    dyaw_step = np.abs(wrap_deg(np.diff(yaw_i)))[pair]
    rep.add(f"  쌍 {pair.sum()}개 (회전 구간이 정보 제공: 스텝당 |Δyaw| 중앙값 {np.median(dyaw_step):.2f}°, 최대 {dyaw_step.max():.1f}°)")
    rep.add(f"  추정 base_link→ins = ({lever_fit[0]:+.3f}, {lever_fit[1]:+.3f}, {lever_fit[2]:+.3f}) m  ±({sig[0]:.3f}, {sig[1]:.3f}, {sig[2]:.3f})")
    rep.add(f"  설정값          = ({lever[0]:+.3f}, {lever[1]:+.3f}, {lever[2]:+.3f}) m → 차이 ({lever_fit[0] - lever[0]:+.3f}, {lever_fit[1] - lever[1]:+.3f}, {lever_fit[2] - lever[2]:+.3f})")
    rep.add("  (z는 회전이 yaw 위주라 관측성 낮음 → ± 큰 값이면 무시)")
    rep.summary["lever_fit"] = dict(value=lever_fit.tolist(), sigma=sig.tolist(), n_pairs=int(pair.sum()))
    if args.fit_lever:
        lever = lever_fit.copy()
        rep.add("  → --fit-lever: 추정값을 사용")

    # --- least squares fusion of positions (per axis)
    rep.h("위치 융합 (RTK + 오도메트리 증분 + 약한 DGPS/SPS)")
    g_base = g_ins - np.einsum("nij,j->ni", R_base, lever)        # GNSS-implied base_link position
    d_enu = np.einsum("nij,nj->ni", R_base[:-1], d_body)
    step = np.linalg.norm(d_body, axis=1)
    so = np.maximum(0.02, args.odom_frac * step)
    P = np.zeros((n, 3))
    for ax in range(3):
        rows, cols, vals, b, w = [], [], [], [], []
        r = 0
        for i in range(n - 1):                                     # odometry
            rows += [r, r]; cols += [i + 1, i]; vals += [1.0, -1.0]; b.append(d_enu[i, ax]); w.append(1.0 / so[i]); r += 1
        for i in np.flatnonzero(inb):                               # GNSS
            s = sh[i] if ax < 2 else sv[i]
            rows.append(r); cols.append(i); vals.append(1.0); b.append(g_base[i, ax]); w.append(1.0 / s); r += 1
        Am = np.zeros((r, n)); Am[rows, cols] = vals
        w = np.array(w); Aw = Am * w[:, None]; bw = np.array(b) * w
        P[:, ax] = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    res_g = (P - g_base)[inb]
    fx = fix_kf[inb] == 4
    res_o = (P[1:] - P[:-1]) - d_enu
    rep.add(f"  GNSS 잔차 (RTK fixed {fx.sum()}개): 수평 RMS {np.sqrt((res_g[fx, :2] ** 2).sum(1).mean()):.3f} m, z RMS {np.sqrt((res_g[fx, 2] ** 2).mean()):.3f} m")
    if (~fx).sum():
        rep.add(f"  GNSS 잔차 (비RTK {(~fx).sum()}개): 수평 중앙값 {np.median(np.hypot(res_g[~fx, 0], res_g[~fx, 1])):.2f} m")
    rep.add(f"  오도메트리 증분 잔차: 수평 RMS {np.sqrt((res_o[:, :2] ** 2).sum(1).mean()):.3f} m, z RMS {np.sqrt((res_o[:, 2] ** 2).mean()):.3f} m (스텝 평균 {step.mean():.2f} m)")
    big = np.flatnonzero(np.linalg.norm(res_o[:, :2], axis=1) > 0.5)
    if len(big):
        rep.warn(f"오도메트리 증분과 0.5 m 넘게 어긋난 스텝 {len(big)}개 (t+" + ", ".join(f"{t_kf[i] - t_kf[0]:.0f}" for i in big[:8]) + "... s) → GNSS 점프 또는 LO 점프 구간")
    rep.add(f"  시작–끝 높이차: LO {p_lo[-1, 2] - p_lo[0, 2]:+.2f} m → 융합 {P[-1, 2] - P[0, 2]:+.2f} m; z 범위 LO {np.ptp(p_lo[:, 2]):.1f} m → 융합 {np.ptp(P[:, 2]):.1f} m (GNSS 고도 범위 {np.ptp(g_ins[inb, 2]):.1f} m)")
    rep.summary.update(n_kf=int(n), rtk_kf=int(fx.sum()), mount_yaw_deg=mount, yaw_fit_deg=yaw_fit, lever=lever.tolist(),
                       gnss_res_h_rms=float(np.sqrt((res_g[fx, :2] ** 2).sum(1).mean())), gnss_res_z_rms=float(np.sqrt((res_g[fx, 2] ** 2).mean())),
                       odom_res_h_rms=float(np.sqrt((res_o[:, :2] ** 2).sum(1).mean())), dz_start_end=float(P[-1, 2] - P[0, 2]), z_span=float(np.ptp(P[:, 2])),
                       lat0=enu.lat0, lon0=enu.lon0, h0=enu.h0)

    # --- outputs
    tum_out = os.path.join(args.out, "keyframes.tum")
    with open(tum_out, "w") as f:
        for k in range(n):
            q = R_to_quat(R_base[k])
            f.write(f"{t_kf[k]:.6f} {P[k, 0]:.4f} {P[k, 1]:.4f} {P[k, 2]:.4f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    geo = {"type": "mp2p_icp::Georeferencing", "defined": True, "geo_coord": {"lat": enu.lat0, "lon": enu.lon0, "altitude": enu.h0},
           "T_enu_to_map": {"mean": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
                            "cov": {"rows": 6, "cols": 6, "data": [float(v) for v in np.diag([0.02 ** 2] * 3 + [math.radians(0.05) ** 2] * 3).ravel()]}}}
    with open(os.path.join(args.out, "map.georef.yaml"), "w") as f:
        yaml.safe_dump(geo, f, sort_keys=False)
    rep.h("출력")
    rep.add(f"  {tum_out} (ENU 프레임 base_link 포즈), map.georef.yaml (항등 + 원점 lat {enu.lat0:.8f} lon {enu.lon0:.8f} h {enu.h0:.2f})")
    if not args.no_build:
        smpose = os.path.join(HERE, "smpose", "build", "smpose")
        sm_out = os.path.join(args.out, "map.simplemap")
        cmd = [smpose, args.simplemap, tum_out, sm_out] + ([f"{v}" for v in args.sensor_pose] if args.sensor_pose else [])
        if args.sensor_pose:
            rep.add(f"  라이다 sensorPose 덮어쓰기: xyz ({args.sensor_pose[0]}, {args.sensor_pose[1]}, {args.sensor_pose[2]}) yaw/pitch/roll ({args.sensor_pose[3]}, {args.sensor_pose[4]}, {args.sensor_pose[5]})°")
        r1 = subprocess.run(cmd, capture_output=True, text=True)
        rep.add(f"  smpose (포즈 교체 simplemap): exit {r1.returncode} → {sm_out}" + ("" if r1.returncode == 0 else "\n" + r1.stdout[-500:] + r1.stderr[-500:]))
        if r1.returncode == 0:
            mm_out = os.path.join(args.out, "map.mm")
            r2 = subprocess.run(["sm2mm", "-i", sm_out, "-o", mm_out, "-p", args.pipeline, "-l", MOLA_PLUGIN, "--no-progress-bar"], capture_output=True, text=True)
            rep.add(f"  sm2mm: exit {r2.returncode} → {mm_out}" + ("" if r2.returncode == 0 else "\n" + r2.stdout[-500:] + r2.stderr[-500:]))
            if r2.returncode == 0:
                r3 = subprocess.run(["mm-georef", "-m", mm_out, "-g", os.path.join(args.out, "map.georef.yaml"), "--inject-to-map", "-l", MOLA_PLUGIN], capture_output=True, text=True)
                rep.add(f"  mm-georef inject: exit {r3.returncode}")
    # --- plots
    plt = setup_matplotlib()
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax[0].plot(g_ins[inb & (fix_kf == 4), 0], g_ins[inb & (fix_kf == 4), 1], ".", color="#2a78d6", ms=3, label="RTK fixed GNSS")
    ax[0].plot(g_ins[inb & (fix_kf != 4), 0], g_ins[inb & (fix_kf != 4), 1], "x", color="#fab219", ms=4, label="비RTK GNSS")
    ax[0].plot(P[:, 0], P[:, 1], "-", color="#d03b3b", lw=1.2, label="융합 키프레임 포즈")
    ax[0].set_aspect("equal"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8); ax[0].set_title("ENU 평면 (x 동, y 북)"); ax[0].set_xlabel("E [m]"); ax[0].set_ylabel("N [m]")
    ax[1].plot(dist, p_lo[:, 2] - p_lo[0, 2], color="#9aa", lw=1, label="LO z (원 맵, 시작 기준)")
    ax[1].plot(dist[inb], g_ins[inb, 2], ".", color="#2a78d6", ms=2, label="GNSS 고도 (ENU)")
    ax[1].plot(dist, P[:, 2], color="#d03b3b", lw=1.2, label="융합 z")
    ax[1].set_xlabel("주행거리 [m]"); ax[1].set_ylabel("z [m]"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8); ax[1].set_title("높이 프로파일")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "anchor_profile.png"), dpi=120); plt.close(fig)
    rep.save(os.path.join(args.out, "anchor_report"))


if __name__ == "__main__":
    main()
