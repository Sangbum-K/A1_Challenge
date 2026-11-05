"""Offline, manually seeded LiDAR lead tracking audit. Never sends control.

Times named video_s use the first LiDAR *receive* time, as render_lidar.py did.
Filtering and derivatives use scan midpoint timestamps. All positions use
base x forward, y left (recorded LiDAR mounting yaw +90 degrees).
"""
import argparse
import csv
import json
import pathlib
import pickle
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

BAG = '/home/ailab/.ros/a1_localization/bags/a1_map_20261008_111019'
MAP = '/home/ailab/git/A1_localization/maps/track_run2/anchored_final'


def reader(topics):
    import rosbag2_py
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id='mcap'),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return r


def points(m):
    a = np.frombuffer(m.data, dtype=np.dtype({
        'names': ['x', 'y', 'z', 't'],
        'formats': ['<f4', '<f4', '<f4', '<f8'],
        'offsets': [0, 4, 8, 18], 'itemsize': m.point_step}))
    return np.column_stack([-a['y'], a['x'], a['z']]), a['t']


def ground_clusters(p, dense=False):
    seeds = p[(abs(p[:, 0]) < 25) & (abs(p[:, 1]) < 7) & (p[:, 2] < -.5)]
    rng = np.random.default_rng(42)
    if len(seeds) > 1500:
        seeds = seeds[rng.choice(len(seeds), 1500, replace=False)]
    best = -1
    plane = np.array([0., 0., 1., 1.05])
    for _ in range(60):
        if len(seeds) < 3:
            break
        s = seeds[rng.choice(len(seeds), 3, replace=False)]
        n = np.cross(s[1]-s[0], s[2]-s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        if abs(n[2]) < .96:
            continue
        if n[2] < 0:
            n = -n
        d = -s[0] @ n
        score = (abs(seeds @ n+d) < .12).sum()
        if score > best:
            best, plane = score, np.r_[n, d]
    mask = abs(seeds @ plane[:3]+plane[3]) < .12
    if mask.sum() > 20:
        ab = np.linalg.lstsq(np.c_[seeds[mask, :2], np.ones(mask.sum())],
                             seeds[mask, 2], rcond=None)[0]
        plane = np.r_[-ab[:2], 1., -ab[2]]
        plane /= np.linalg.norm(plane[:3])
    heights = p @ plane[:3]+plane[3]
    q = p[(heights > .22) & (heights < 2.8)]
    if not len(q):
        return plane, []
    _, idx = np.unique(np.floor(q/.16).astype(np.int32), axis=0, return_index=True)
    v = q[idx]
    tree = cKDTree(v)
    graph = tree.sparse_distance_matrix(tree, .65, output_type='coo_matrix')
    _, lab = connected_components(graph, directed=False)
    cs = []
    for k in np.unique(lab):
        c = v[lab == k]
        if len(c) < (5 if dense else 3):
            continue
        lo, hi = np.percentile(c, [2, 98], axis=0)
        dim = hi-lo
        # A distant car can hit a single laser ring: requiring vertical extent
        # discarded the user's seed after 285.8 s despite visible returns.
        # These are unclassified candidates; temporal association rejects clutter.
        if max(dim[:2]) > 8 or max(dim[:2]) < (.35 if dense else .30) or (dense and dim[2] < .20):
            continue
        cs.append(dict(c=np.median(c, axis=0), lo=lo, hi=hi,
                       n=len(c), p=c.astype(np.float32)))
    return plane, cs


def load_ins(out):
    from rclpy.serialization import deserialize_message
    from gsof_msgs.msg import NavigationSolution49
    cache = out/'ins.npz'
    if cache.exists():
        return dict(np.load(cache))
    r = reader(['/gsof_client/gsof/ins_solution_49'])
    rows = []
    while r.has_next():
        _, raw, ns = r.read_next()
        m = deserialize_message(raw, NavigationSolution49)
        stamp = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
        if ns*1e-9-stamp > 3e8:
            stamp += 315964800-18
        rows.append([stamp, ns*1e-9, m.velocity.east, m.velocity.north,
                     m.heading, m.total_speed, m.status.gnss])
    a = np.array(rows)
    # Reuse the existing offline heading alignment implementation and raw LO
    # keyframes; not the original, incorrectly oriented relay twist.x.
    sys.path.insert(0, '/home/ailab/a1_analysis')
    from a1_bag_common import Report, quat_yaw_deg, unwrap_deg
    from gnss_anchor_map import heading_lag
    kf = np.loadtxt(MAP+'/lo_keyframes.tum')
    ins = SimpleNamespace(ins_t=a[:, 0], speed=a[:, 5],
        heading_at=lambda t: np.interp(t, a[:, 0], unwrap_deg(a[:, 4])))
    rep = Report()
    lag = heading_lag(kf[:, 0], quat_yaw_deg(kf[:, 4:8]), ins, 4., rep)
    slope, offset = np.polyfit(kf[:, 0]-kf[0, 0], lag, 1)
    # INS UTC = lidar_time + offset + slope*(lidar_time-kf_start).
    common_t = kf[0, 0]+(a[:, 0]-kf[0, 0]-offset)/(1+slope)
    # Reuse this run's fitted INS mounting yaw, explicitly a provisional value.
    yaw = np.unwrap(np.deg2rad(90-a[:, 4]-89.87))
    omega = np.gradient(yaw, common_t)
    # Current bag's later offline mounting estimate, NOT the stale TF lever.
    lever = np.array([0., -.47])
    c, s = np.cos(yaw), np.sin(yaw)
    rotlever = np.column_stack([c*lever[0]-s*lever[1], s*lever[0]+c*lever[1]])
    vel = a[:, 2:4]-omega[:, None]*np.column_stack([-rotlever[:, 1], rotlever[:, 0]])
    pos = cumulative_trapezoid(vel, common_t, axis=0, initial=0.)
    result = dict(t=common_t, yaw=yaw, omega=omega, vel=vel, pos=pos,
                  gnss=a[:, 6], raw=a, lag_slope=slope, lag_offset=offset,
                  lag_epoch=kf[0, 0])
    np.savez(cache, **result)
    (out/'timing.json').write_text(json.dumps(dict(
        lag_at_keyframe_start_s=offset, lag_slope_s_per_s=slope,
        keyframe_start_unix=kf[0, 0], mounting_yaw_deg=89.87,
        lever_xy_m=lever.tolist(), alignment_log=rep.lines,
        limitations=['Offline fitted timing and mounting; not ground truth.',
                    'Planar yaw deskew only; road tilt handled by local ground plane.',
                    'Absolute velocities inherit INS/mounting/timing errors.']), indent=2))
    return result


def pose_at(ins, t):
    t = np.atleast_1d(t)
    p = np.column_stack([np.interp(t, ins['t'], ins['pos'][:, i]) for i in range(2)])
    v = np.column_stack([np.interp(t, ins['t'], ins['vel'][:, i]) for i in range(2)])
    yaw = np.interp(t, ins['t'], ins['yaw'])
    return p, v, yaw


def rotation(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])


def extract(args, out):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    ins = load_ins(out)
    r = reader(['/lidar_points'])
    _, _, first = r.read_next()
    r.seek(first+int(args.start*1e9))
    n = 0
    started = time.monotonic()
    with (out/'frames.pkl').open('wb') as f:
        while r.has_next():
            _, raw, ns = r.read_next()
            elapsed = (ns-first)*1e-9
            if elapsed > args.end:
                break
            tick = time.perf_counter()
            m = deserialize_message(raw, PointCloud2)
            p, ts = points(m)
            ref = (ts.min()+ts.max())*.5
            valid = (np.isfinite(p).all(1) & (p[:, 0] > -45) & (p[:, 0] < 85)
                     & (abs(p[:, 1]) < 35) & (p[:, 2] > -4) & (p[:, 2] < 4))
            p, ts = p[valid], ts[valid]
            ep, ev, yaw = pose_at(ins, ref)
            R = rotation(yaw[0])
            # Each packet has its own time; transform all to the scan midpoint.
            unique, inv = np.unique(ts, return_inverse=True)
            pp, _, yy = pose_at(ins, unique)
            c, s = np.cos(yy)[inv], np.sin(yy)[inv]
            world = np.column_stack([c*p[:, 0]-s*p[:, 1], s*p[:, 0]+c*p[:, 1]])+pp[inv]
            p[:, :2] = (world-ep[0]) @ R
            plane, cs = ground_clusters(p, args.dense)
            compute_ms = (time.perf_counter()-tick)*1000
            # Every candidate retains its downsampled observed surface.
            frame = dict(video_s=elapsed, t=ref, recv_ns=ns, pos=ep[0], vel=ev[0],
                         yaw=yaw[0], omega=float(np.interp(ref, ins['t'], ins['omega'])),
                         plane=plane, cs=cs, p=p[::max(1, len(p)//4000)].astype(np.float32),
                         compute_ms=compute_ms)
            pickle.dump(frame, f, protocol=5)
            n += 1
            if n % 200 == 0:
                print(f'extract {n} scans | video {elapsed:.1f}s | wall {time.monotonic()-started:.1f}s', flush=True)
    print(f'extracted {n} scans', flush=True)


def frames(out):
    with (out/'frames.pkl').open('rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def track(args, out):
    # A manually initialized, causal 2D constant-velocity Kalman tracker.
    # No automatic reinitialization after loss: nearby cars are not the seed ID.
    state = None
    P = None
    last = None
    hit = None
    dead = False
    rows = []
    associations = []
    for frame in frames(out):
        t, R = frame['t'], rotation(frame['yaw'])
        dt = 0 if last is None else t-last
        last = t
        cs = frame['cs']
        chosen = -1
        innovation = np.nan
        if state is None and not dead:
            if cs:
                ds = [np.linalg.norm(c['c'][:2]-[args.seed_x, args.seed_y]) for c in cs]
                if min(ds) < 3:
                    chosen = int(np.argmin(ds))
                    xy = R @ cs[chosen]['c'][:2]+frame['pos']
                    state = np.r_[xy, frame['vel']]
                    P = np.diag([.3**2, .3**2, 3.**2, 3.**2])
                    hit = t
        elif state is not None and not dead:
            F = np.eye(4)
            F[:2, 2:] = np.eye(2)*dt
            G = np.vstack([np.eye(2)*dt*dt/2, np.eye(2)*dt])
            state = F @ state
            P = F @ P @ F.T+G @ G.T*3.**2
            costs = []
            for j, c in enumerate(cs):
                z = R @ c['c'][:2]+frame['pos']
                err = z-state[:2]
                dist = np.linalg.norm(err)
                S = P[:2, :2]+np.eye(2)*.3**2
                cost = err @ np.linalg.solve(S, err)
                # Fixed Euclidean limit prevents increasingly uncertain state
                # from silently snapping to roadside clutter.
                if cost < 9.21 and dist < min(3., 1.3+1.2*(t-hit)):
                    costs.append((cost, j, z, dist))
            if costs:
                _, chosen, z, innovation = min(costs, key=lambda x:x[0])
                K = np.linalg.solve(P[:2, :2]+np.eye(2)*.3**2, P[:2, :]).T
                state = state+K @ (z-state[:2])
                P = (np.eye(4)-K @ np.eye(4)[:2]) @ P
                hit = t
            if t-hit > args.coast:
                dead = True
        status = 'lost' if dead or state is None else ('measured' if chosen >= 0 else 'predicted')
        row = dict(video_s=frame['video_s'], scan_time_unix=t, status=status,
                   candidate_index=chosen, points=0, range_m=np.nan,
                   x_surface_m=np.nan, y_surface_m=np.nan,
                   rear_surface_x_m=np.nan, vx_world=np.nan, vy_world=np.nan,
                   relative_long_mps=np.nan, lead_long_mps=np.nan,
                   lead_speed_mps=np.nan, ego_long_mps=float(frame['vel'] @ R[:, 0]),
                   range_rate_mps=np.nan, innovation_m=innovation,
                   compute_ms=frame['compute_ms'], yaw=frame['yaw'],
                   measurement_age_s=np.nan if hit is None else t-hit,
                   raw_surface_x_m=np.nan, raw_surface_y_m=np.nan)
        if status != 'lost':
            rel = (state[:2]-frame['pos']) @ R
            vrel = (state[2:]-frame['vel']) @ R
            vv = state[2:] @ R
            row.update(x_surface_m=rel[0], y_surface_m=rel[1], range_m=np.linalg.norm(rel),
                       vx_world=state[2], vy_world=state[3], relative_long_mps=vrel[0],
                       lead_long_mps=vv[0], lead_speed_mps=np.linalg.norm(state[2:]),
                       range_rate_mps=float(vrel @ rel/max(np.linalg.norm(rel), 1e-6)))
            if chosen >= 0:
                row['points'] = cs[chosen]['n']
                row['rear_surface_x_m'] = cs[chosen]['lo'][0]
                row['raw_surface_x_m'], row['raw_surface_y_m'] = map(float, cs[chosen]['c'][:2])
        rows.append(row)
        associations.append(chosen)
    suffix = '_'+args.tag if args.tag else ''
    with (out/f'track{suffix}.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.save(out/f'associations{suffix}.npy', associations)
    measured = [r for r in rows if r['status']=='measured']
    print(json.dumps(dict(scans=len(rows), measured=len(measured),
                         last_measurement_s=measured[-1]['video_s'] if measured else None,
                         first=rows[0]), indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['extract', 'track'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=float, default=266.)
    ap.add_argument('--end', type=float, default=921.)
    ap.add_argument('--seed-x', type=float, default=29.)
    ap.add_argument('--seed-y', type=float, default=-3.)
    ap.add_argument('--coast', type=float, default=2.,
                    help='Maximum prediction-only identity retention; not a valid measurement.')
    ap.add_argument('--dense', action='store_true', help='Reproduce initial vertical-extent baseline.')
    ap.add_argument('--tag', default='', help='Suffix for a sensitivity run; leaves nominal results intact.')
    args = ap.parse_args()
    out = pathlib.Path(args.out);out.mkdir(parents=True, exist_ok=True)
    suffix = '_'+args.tag if args.tag else ''
    (out/(args.mode+suffix+'_config.json')).write_text(json.dumps(vars(args), indent=2))
    (extract if args.mode=='extract' else track)(args, out)


if __name__=='__main__':
    main()
