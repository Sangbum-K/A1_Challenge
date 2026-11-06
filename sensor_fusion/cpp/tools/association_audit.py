"""Start the local C++ inspection pipeline, collect evidence, stop only owned processes."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from a1_fusion_cpp.msg import A2Features, Association
ROOT=Path(__file__).resolve().parents[2]
WS_ROOT=ROOT.parents[2] if ROOT.parts[-3:]==('src','perception','sensor_fusion') else ROOT.parent  # A1_Total: <ws>/src/perception/sensor_fusion; A1_Challenge: <repo>/sensor_fusion
def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=45);p.add_argument('--with-shadow',action='store_true')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/association_check');p.add_argument('--show-window',action='store_true')
    args=p.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    rclpy.init();n=rclpy.create_node('a1_association_audit');times=defaultdict(list);rows=[];children=[];logs=[]
    def received(kind,m):
        times[kind].append(time.monotonic())
        if kind=='association':rows.append(dict(stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9,status=m.status,
            primary_id=m.primary_track_id,standby_id=m.standby_track_id,primary_status=m.primary_status,primary_camera_age_s=m.primary_camera_age_s,
            candidate_tracks=list(m.candidate_track_ids),lidar_candidates=list(m.lidar_candidate_track_ids),ambiguous=m.association_ambiguous,
            unique=m.candidate_unique,verified=m.verified_match,track=m.candidate_track_id,
            bin=[m.bin_left,m.bin_right],ground=m.ground_points,nonground=m.nonground_points,
            image_delta_s=m.image_delta_s,a2_delta_s=m.a2_delta_s,processing_ms=m.processing_ms,
            tracks=[dict(raw_center=list(t.raw_center_m),raw_size=list(t.raw_size_m),id=t.id,confirmed=t.confirmed,center=list(t.center_m),size=list(t.size_m),
                geometry_status=t.geometry_status,selection_reason=t.selection_reason,
                measured=t.measured,prediction_valid=t.prediction_valid,observation_age_s=t.observation_age_s,position_covariance_xy=list(t.position_covariance_xy),
                lidar_candidate=t.lidar_candidate,camera_matched=t.camera_matched,matched_bins=[t.matched_bin_left,t.matched_bin_right],range_m=t.range_m,
                relative_velocity=list(t.relative_velocity_mps),near_surface_x=t.near_surface_x_m,box=list(t.image_box_xyxy),camera_evidence=t.camera_evidence,camera_state=t.camera_state,far_vehicle=t.far_vehicle) for t in m.tracks]))
    topics={'camera':('/camera_driver_gv/vis/image_raw',Image),'lidar':('/lidar_points',PointCloud2),
        'a2':('/lead_fusion/a2',A2Features),'association':('/lead_fusion/association',Association)}
    subs=[n.create_subscription(cls,topic,lambda m,k=k:received(k,m),qos_profile_sensor_data) for k,(topic,cls) in topics.items()]
    commands={
      'camera':['bash',str(ROOT/'run.sh'),'camera-driver-gpu','exposure_auto:=Continuous','exposure_min_us:=1000','exposure_max_us:=20000','gain_auto:=Continuous','gain_max_db:=12'],
      'lidar':[str(ROOT/'install/lib/hesai_ros_driver/hesai_ros_driver_node'),'--ros-args','-p','config_path:='+str(ROOT/'cpp/config/lidar_host_time.yaml')],
      'a2':['bash',str(ROOT/'run.sh'),'camera-gpu'],
      'association':[str(WS_ROOT/'install/lib/a1_fusion_cpp/association_node'),'--ros-args','--params-file',str(ROOT/'cpp/config/association.yaml'),
        '-p','preview_path:='+str(args.out/'latest.jpg'),'-p','csv_path:='+str(args.out/'association.csv'),'-p','show_window:='+str(args.show_window).lower()]
    }
    if args.with_shadow:
        commands['shadow']=[str(WS_ROOT/'install/lib/a1_fusion_cpp/fusion_node'),'--ros-args','--params-file',str(ROOT/'cpp/config/shadow.yaml')]
    try:
        for _ in range(10):rclpy.spin_once(n,timeout_sec=.05)
        # Avoid silently mixing old binaries/configuration into a verification run.
        occupied=[k for k,(topic,_) in topics.items() if n.count_publishers(topic)]
        if args.with_shadow:
            occupied += [t for t in ['/lead_fusion/estimate','/lead_fusion/ego_motion'] if n.count_publishers(t)]
        if occupied:raise RuntimeError('Publishers already active: '+str(occupied)+'. Use a separate ROS_DOMAIN_ID / stop the intended old run.')
        for k,cmd in commands.items():
            f=(args.out/(k+'.log')).open('w');logs.append(f)
            children.append((k,subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=args.out)))
        end=time.monotonic()+args.seconds
        while time.monotonic()<end:
            rclpy.spin_once(n,timeout_sec=.01)
            failed=[(k,p.returncode) for k,p in children if p.poll() is not None]
            if failed:raise RuntimeError('Pipeline process exited: '+str(failed))
    finally:
        for _,proc in children:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
        for _,proc in children:
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for f in logs:f.close()
        summary={k:dict(count=len(t),hz=(len(t)-1)/(t[-1]-t[0]) if len(t)>1 else 0) for k,t in times.items()}
        summary.update(association_frames=len(rows),unique_frames=sum(r['unique'] for r in rows),verified_frames=sum(r['verified'] for r in rows),
            statuses={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
            latest=rows[-1] if rows else None,children={k:p.returncode for k,p in children})
        # Nonfinite missing-input deltas are encoded as null, not nonstandard JSON NaN.
        def clean(v):
            import math
            if isinstance(v,float) and not math.isfinite(v):return None
            if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
            if isinstance(v,list):return [clean(x) for x in v]
            return v
        (args.out/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
        (args.out/'frames.json').write_text(json.dumps(clean(rows),indent=2,allow_nan=False)+'\n')
        print(json.dumps(clean(summary),indent=2));n.destroy_node();rclpy.shutdown()
    if not rows:raise SystemExit(2)
if __name__=='__main__':main()
