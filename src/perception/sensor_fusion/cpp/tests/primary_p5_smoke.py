"""Synthetic primary -> P5 contract, no sensor drivers and no vehicle commands."""
import argparse,subprocess,time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from a1_fusion_cpp.msg import Association,ClusterTrack,BrakeEstimate
from gsof_msgs.msg import NavigationSolution49

def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',required=True);args=p.parse_args()
 rclpy.init();n=rclpy.create_node('primary_p5_contract_test');received=[]
 navpub=n.create_publisher(NavigationSolution49,'/test/ins',qos_profile_sensor_data)
 targetpub=n.create_publisher(Association,'/lead_fusion/association',qos_profile_sensor_data)
 sub=n.create_subscription(BrakeEstimate,'/lead_fusion/estimate',received.append,10)
 proc=subprocess.Popen([args.binary,'--ros-args','-p','mode:=association','-p','ego_source:=ins','-p','ins_topic:=/test/ins','-p','ins_mount_verified:=true','-p','clock_verified:=true','-p','extrinsics_verified:=true','-p','primary_motion_verified:=true'],stdout=subprocess.DEVNULL)
 def spin(seconds):
  end=time.monotonic()+seconds
  while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.005)
 def frame(primary=42,age=0.,clock=True,matched=True,ambiguous=False):
  ts=n.get_clock().now().to_msg()
  nav=NavigationSolution49();nav.header.stamp=ts;nav.heading=90.;nav.status.imu_alignment=4;nav.status.gnss=4;nav.velocity.east=10.;navpub.publish(nav);spin(.015)
  m=Association();m.header.stamp=ts;m.header.frame_id='lidar_vehicle';m.primary_track_id=primary;m.standby_track_id=43 if primary==42 else 42;m.clock_verified=clock;m.calibration_verified=True;m.primary_status="measured" if age==0 else "predicted";m.primary_camera_age_s=age;m.association_ambiguous=ambiguous
  for ident,x in [(42,20.),(43,35.)]:
   tr=ClusterTrack();tr.id=ident;tr.confirmed=True;tr.prediction_valid=True;tr.measured=age==0;tr.camera_matched=matched;tr.observation_age_s=age
   tr.center_m=[x,0.,0.];tr.size_m=[2.,1.8,1.5];tr.apparent_velocity_mps=[-2.,0.];tr.position_covariance_xy=[.1,0.,0.,.1];m.tracks.append(tr)
  begin=len(received);targetpub.publish(m);spin(.08)
  result=[v for v in received[begin:] if abs((v.header.stamp.sec-ts.sec)+(v.header.stamp.nanosec-ts.nanosec)/1e9)<1e-5]
  assert result,('no P5 response',[(v.reason,v.header.stamp) for v in received[-3:]]);return result[-1],m
 try:
  deadline=time.monotonic()+8
  while time.monotonic()<deadline and not (navpub.get_subscription_count() and targetpub.get_subscription_count()):spin(.05)
  assert targetpub.get_subscription_count(),'no association subscriber'
  spin(.3) # Allow the independent result publisher discovery to finish.
  first,_=frame();assert first.valid and first.source=='lidar_primary' and first.track_id==42 and first.gap_m==19. and first.relative_speed_mps==-2.,first
  predicted,m=frame(age=.2);assert not predicted.valid and predicted.reason=='primary_camera_unconfirmed'
  unmatched,_=frame(matched=False);assert not unmatched.valid and unmatched.reason=='primary_camera_unconfirmed'
  ambiguous,_=frame(ambiguous=True);assert not ambiguous.valid and ambiguous.reason=='primary_camera_unconfirmed'
  back,_=frame();assert back.valid and back.track_id==42 and back.raw_p5_mps2==first.raw_p5_mps2
  expired,_=frame(age=.4);assert not expired.valid
  swapped,_=frame(primary=43);assert swapped.valid and swapped.track_id==43 and swapped.gap_m==34.
  missing,_=frame(primary=-1);assert not missing.valid and missing.reason=='no_valid_primary'
  unverified,_=frame(clock=False);assert not unverified.valid
  spin(.4);assert not received[-1].valid,'input loss did not invalidate output'
  print('Primary only (standby ignored), camera-required, held/ambiguous rejection, reacquisition, ID switch, missing/clock/stale invalidation: PASS')
 finally:
  proc.terminate()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
  n.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
