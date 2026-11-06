"""Bounded C++ A2 benchmark: real camera or timestamped synthetic 21 Hz images.
Python only publishes/observes; all image preprocessing and inference run in C++.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from a1_fusion_cpp.msg import A2Features

ROOT = Path(__file__).resolve().parents[2]
WS_ROOT = ROOT.parents[2] if ROOT.parts[-3:]==('src','perception','sensor_fusion') else ROOT.parent  # A1_Total: <ws>/src/perception/sensor_fusion; A1_Challenge: <repo>/sensor_fusion

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hz', type=float, default=21)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    n = rclpy.create_node('a1_gpu_benchmark')
    images, outputs, children, logs = [], [], [], []
    topic = '/test/gpu_image' if args.synthetic else '/camera_driver_gv/vis/image_raw'
    def on_image(m):
        images.append((time.monotonic(), m.header.stamp.sec + m.header.stamp.nanosec/1e9))
    def on_a2(m):
        stamp = m.header.stamp.sec + m.header.stamp.nanosec/1e9
        outputs.append((time.monotonic(), stamp, time.time()-stamp, m.history_ready, bool(np.isfinite(m.raw).all())))
    subs = [n.create_subscription(Image, topic, on_image, qos_profile_sensor_data),
            n.create_subscription(A2Features, '/lead_fusion/a2', on_a2, qos_profile_sensor_data)]
    pub = n.create_publisher(Image, topic, qos_profile_sensor_data) if args.synthetic else None
    def launch(name, cmd):
        log = (args.out/(name+'.log')).open('w'); logs.append(log)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(proc)
        return proc
    result = {}
    try:
        for _ in range(10): rclpy.spin_once(n, timeout_sec=.05)
        if n.count_publishers('/lead_fusion/a2'):
            raise RuntimeError('A2 publisher already active; use an isolated ROS_DOMAIN_ID')
        proc = launch('a2', [str(WS_ROOT/'install/lib/a1_fusion_cpp/a2_camera_node'), '--ros-args',
            '-p', 'device:=cuda', '-p', 'model_dir:='+str(ROOT/'cpp/models_gpu'), '-p', 'image_topic:='+topic])
        deadline = time.monotonic()+120
        while n.count_subscribers(topic) < 2 and time.monotonic() < deadline:
            if proc.poll() is not None: raise RuntimeError('C++ node exited; see a2.log')
            rclpy.spin_once(n, timeout_sec=.05)
        if n.count_subscribers(topic) < 2: raise RuntimeError('C++ warmup/subscription timed out')
        if not args.synthetic and not n.count_publishers(topic):
            launch('camera', ['ros2', 'launch', str(ROOT/'cpp/launch/hikrobot_gpu.launch.py'),
                'guid:=169.254.222.118', 'exposure_us:=3000', 'fps:='+str(args.hz)])
        msg = Image(); msg.width=1280; msg.height=720; msg.encoding='bgr8'; msg.step=3840
        msg.header.frame_id='hikrobot_camera_vis'
        msg.data=np.random.default_rng(42).integers(0,256,(720,1280,3),dtype=np.uint8).tobytes()
        start=time.monotonic(); end=start+args.seconds+5; next_frame=start
        while time.monotonic()<end:
            if proc.poll() is not None: raise RuntimeError('C++ node exited during measurement')
            if pub and time.monotonic()>=next_frame:
                msg.header.stamp=n.get_clock().now().to_msg(); pub.publish(msg)
                next_frame+=1/args.hz
                if next_frame<time.monotonic()-1/args.hz: next_frame=time.monotonic()+1/args.hz
            rclpy.spin_once(n,timeout_sec=.001 if pub else .02)
        # Discard startup and history filling. Persist raw timings for independent audit.
        im=[r for r in images if r[0]>=start+5]
        out=[r for r in outputs if r[0]>=start+5]
        def hz(rows): return (len(rows)-1)/(rows[-1][0]-rows[0][0]) if len(rows)>1 else 0
        result=dict(synthetic=args.synthetic, requested_hz=args.hz, measurement_s=args.seconds,
            image_count=len(im), a2_count=len(out), image_hz=hz(im), a2_hz=hz(out),
            history_ready_fraction=float(np.mean([r[3] for r in out])) if out else 0,
            finite=all(r[4] for r in out) if out else False,
            latency_ms_percentiles=dict(zip(['p50','p95','p99'],np.percentile([r[2]*1000 for r in out],[50,95,99]).tolist())) if out else None,
            rows=out, image_rows=im)
        result['pass_21hz']=(result['a2_hz']>=20.9 and result['history_ready_fraction']>=.99 and result['finite'])
        (args.out/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','image_rows')},indent=2))
    finally:
        for proc in children:
            if proc.poll() is None: os.killpg(proc.pid,signal.SIGINT)
        for proc in children:
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        for f in logs: f.close()
        n.destroy_node(); rclpy.shutdown()
    if not result.get('pass_21hz'): raise SystemExit(2)

if __name__=='__main__': main()
