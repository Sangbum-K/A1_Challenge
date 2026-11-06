"""Synthetic geometry test; never presented as a real-sensor accuracy result."""
import argparse,subprocess,time,csv,tempfile
from array import array
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,PointCloud2,PointField
from a1_fusion_cpp.msg import A2Features,Association

def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--map-filter',action='store_true');p.add_argument('--roi-center-deg',type=float,default=0.);args=p.parse_args()
 rclpy.init();n=rclpy.create_node('association_synthetic_test');rows=[];fronts=[];screens=[];blank_pixels=array('B',bytes(720*3840))
 image_pub=n.create_publisher(Image,'/test/image',qos_profile_sensor_data)
 cloud_pub=n.create_publisher(PointCloud2,'/test/cloud',qos_profile_sensor_data)
 a2_pub=n.create_publisher(A2Features,'/lead_fusion/a2',qos_profile_sensor_data)
 front_sub=n.create_subscription(PointCloud2,'/lead_fusion/front_points',fronts.append,qos_profile_sensor_data)
 sub=n.create_subscription(Association,'/lead_fusion/association',rows.append,qos_profile_sensor_data)
 tmp=tempfile.TemporaryDirectory();csv_path=tmp.name+'/association.csv'
 proc=subprocess.Popen([args.binary,'--ros-args','-p','csv_path:='+csv_path,'-p','image_topic:=/test/image','-p','lidar_topic:=/test/cloud','-p','sensor_height_m:=1.8','-p','camera_position_m:=[0.8, 0.0, -0.3]','-p','cluster_tolerance_m:=0.45','-p','corridor_half_width_m:=12.0','-p',f'lidar_roi_center_deg:={args.roi_center_deg}'],stdout=subprocess.DEVNULL)
 try:
  deadline=time.monotonic()+10
  while not cloud_pub.get_subscription_count() and time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=.05)
  assert cloud_pub.get_subscription_count(),'no inspection subscriber'
  xx,yy=np.meshgrid(np.arange(2,40,.25),np.arange(-10,10,.25));ground=np.stack([xx.ravel(),yy.ravel(),-1.8+np.random.default_rng(42).normal(0,.01,xx.size)],axis=1)
  def box(x,y=0,length=2.):
   # LiDAR samples visible surfaces, not a dense solid-volume cube.
   yy,zz=np.meshgrid(np.arange(-.9,.9,.15),np.arange(-1.3,.4,.15))
   front=np.stack([np.full(yy.size,x-length/2),yy.ravel(),zz.ravel()],axis=1)
   xx,yy=np.meshgrid(np.arange(x-length/2,x+length/2,.15),np.arange(-.9,.9,.15))
   roof=np.stack([xx.ravel(),yy.ravel(),np.full(xx.size,.35)],axis=1)
   return np.concatenate([front,roof])+np.array([0,y,0])
  map_active=False
  def phase(two=False,send_camera=True,multi=False,three=False,drop=False,background=False,rear_only=False,rotated=False):
   begin=len(rows);end=time.monotonic()+2.5;next_frame=time.monotonic();frame_index=0
   xyz=np.concatenate([ground,box(15),np.array([[-5.,-2.,0.],[-5.,2.,0.],[-5.,-8.,0.]])]+([box(25)] if two else []))
   if rotated:
    car=box(15,length=4.5)-np.array([15,0,0]);angle=np.deg2rad(45.)
    car=car@np.array([[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]])+np.array([15,0,0])
    xyz=np.concatenate([ground,car])
   if background:xyz=np.concatenate([xyz,box(11,5)])
   if rear_only:
    yy,zz=np.meshgrid(np.arange(-.85,.86,.15),np.arange(-.6,.01,.15))
    rear=np.stack([np.full(yy.size,15.),yy.ravel(),zz.ravel()],axis=1)
    xyz=np.concatenate([ground,rear])
   if multi:xyz=np.concatenate([ground,box(20,3),box(20,-3)]+([box(30,0)] if three else []))
   while time.monotonic()<end:
    if time.monotonic()>=next_frame:
     ts=n.get_clock().now().to_msg();t=ts.sec+ts.nanosec/1e9
     if send_camera:
      im=Image();im.header.stamp=ts;im.header.frame_id='hikrobot_camera_vis';im.width=1280;im.height=720;im.encoding='bgr8';im.step=3840;im.data=blank_pixels;image_pub.publish(im)
      a=A2Features();a.header=im.header;a.history_ready=True;a.raw=[float('nan')]*280
      for k in range(40):a.raw[k*7]=8. if ((15<=k<=17 or 22<=k<=24 or (three and 19<=k<=20)) if multi else 18<=k<=21) else -8.
      a2_pub.publish(a)
     points=xyz
     if drop and frame_index in (10,11):points=np.concatenate([ground,box(25)])
     arr=np.zeros(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('padding','<f4'),('timestamp','<f8')])
     arr['x'],arr['y'],arr['z']=points.T;arr['timestamp']=t
     cloud=PointCloud2();cloud.header.stamp=ts;cloud.header.frame_id='hesai_lidar';cloud.width=len(arr);cloud.height=1;cloud.point_step=24;cloud.row_step=24*len(arr);cloud.data=array('B',arr.tobytes())
     # Input LiDAR axes: -Y forward, X left. Convert body points into LiDAR before publishing.
     arr['x']=points[:,1];arr['y']=-points[:,0];cloud.data=array('B',arr.tobytes())
     cloud.fields=[PointField(name=name,offset=offset,datatype=kind,count=1) for name,offset,kind in [('x',0,7),('y',4,7),('z',8,7),('timestamp',16,8)]]
     if map_active:
      from std_msgs.msg import Float32
      from geometry_msgs.msg import TransformStamped
      quality_pub.publish(Float32(data=.9));epoch_pub.publish(map_header)
      transform=TransformStamped();transform.header.stamp=ts;transform.header.frame_id='map';transform.child_frame_id='hesai_lidar'
      transform.transform.rotation.z=np.sqrt(.5);transform.transform.rotation.w=np.sqrt(.5);tf_pub.sendTransform(transform)
     cloud_pub.publish(cloud);next_frame+=.1;frame_index+=1
    rclpy.spin_once(n,timeout_sec=.003)
   return rows[begin:]
  turning=phase(rotated=True)
  assert any(r.primary_status=='measured' and any(tr.id==r.primary_track_id and tr.raw_size_m[1]>2.6 for tr in r.tracks) for r in turning), 'rotated car rejected by axis-aligned width'
  print('Rotated 4.5 m car with >2.6 m axis-aligned width acquired: PASS')
  thin=phase(rear_only=True)
  assert any(r.primary_track_id>=0 and r.primary_status=='measured' for r in thin), 'camera-supported thin rear surface never acquired'
  background=phase(background=True)
  chosen=[(r,tr) for r in background for tr in r.tracks if tr.id==r.primary_track_id and r.primary_status=='measured']
  assert chosen and all(abs(tr.center_m[1])<1 for r,tr in chosen), 'near off-camera background selected as lead'
  assert any(tr.selection_reason=='camera_evidence_low' for r in background for tr in r.tracks), 'background rejection was not audited'
  print('Thin rear acquisition + closer off-camera background excluded: PASS')
  one=phase();valid=[r for r in one if r.candidate_unique]
  assert valid,[(r.status,[(t.id,t.confirmed,list(t.center_m),list(t.image_box_xyxy),t.camera_evidence) for t in r.tracks],r.ground_points) for r in one[-3:]]
  assert all(not r.verified_match for r in valid),'unverified calibration incorrectly accepted'
  assert all(abs(r.image_delta_s-r.a2_delta_s)<.005 for r in valid),'features paired with different image'
  two=phase(two=True);assert any(sum(t.camera_state=='occluded' and t.confirmed for t in r.tracks)==1 and r.primary_track_id>=0 for r in two),[(r.status,[(t.range_m,t.camera_state) for t in r.tracks]) for r in two[-5:]]
  assert all(tr.range_m<20 for r in two[-5:] for tr in r.tracks if tr.id==r.primary_track_id),'hidden car replaced the nearer target'
  phase() # Restore a single camera-supported primary before the dropout.
  dropout=phase(drop=True)
  predicted=[r for r in dropout if r.primary_status=='predicted']
  assert predicted,('one/two scan dropout did not retain primary',[(r.primary_track_id,r.primary_status,[(tr.id,round(tr.center_m[0],2),tr.measured,round(tr.observation_age_s,2),list(tr.size_m)) for tr in r.tracks]) for r in dropout])
  primary_id=predicted[0].primary_track_id
  assert all(r.primary_track_id==primary_id for r in predicted)
  assert dropout[-1].primary_track_id==primary_id and dropout[-1].primary_status=='measured',('reconnection changed primary ID',[(r.primary_track_id,r.primary_status,[(tr.id,round(tr.center_m[0],2),tr.measured,round(tr.observation_age_s,2)) for tr in r.tracks]) for r in dropout])
  assert all(any(tr.id==primary_id and not tr.measured and tr.prediction_valid and tr.observation_age_s<=.500001 for tr in r.tracks) for r in predicted)
  print('ROS dropout prediction + primary hold + same-ID reacquisition: PASS')
  multi=phase(multi=True);both=[r for r in multi if len(r.candidate_track_ids)==2]
  assert both,[(r.status,list(r.candidate_track_ids),[(t.range_m,t.camera_evidence) for t in r.tracks]) for r in multi[-3:]]
  assert all(not r.candidate_unique and not r.association_ambiguous and not r.verified_match for r in both)
  assert all(len(set(r.candidate_track_ids))==2 and sum(t.camera_matched for t in r.tracks)>=2 for r in both)
  capped=phase(multi=True,three=True)
  assert all(len(r.candidate_track_ids)<=2 and len(r.lidar_candidate_track_ids)<=2 for r in capped)
  assert any(len(r.candidate_track_ids)==2 for r in capped)
  print('Two independent camera/LiDAR matches + three-object cap: PASS')
  if args.map_filter:
   from rclpy.qos import QoSProfile, DurabilityPolicy
   from std_msgs.msg import Header, Float32
   from tf2_ros import TransformBroadcaster
   map_qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
   map_pub=n.create_publisher(PointCloud2,'/localization/map_surfaces',map_qos)
   epoch_pub=n.create_publisher(Header,'/localization/map_surfaces_epoch',map_qos)
   quality_pub=n.create_publisher(Float32,'/lidar_odometry/pose_quality',qos_profile_sensor_data)
   tf_pub=TransformBroadcaster(n)
   yy,zz=np.meshgrid(np.arange(4,6,.1),np.arange(-1.4,.5,.1))
   surface=np.zeros((yy.size,7),dtype='<f4');surface[:,0]=10.;surface[:,1]=yy.ravel();surface[:,2]=zz.ravel();surface[:,3]=1.;surface[:,6]=2.
   map_msg=PointCloud2();map_msg.header.frame_id='map';map_msg.header.stamp=n.get_clock().now().to_msg();map_header=map_msg.header
   map_msg.width=len(surface);map_msg.height=1;map_msg.point_step=28;map_msg.row_step=28*len(surface);map_msg.data=array('B',surface.tobytes())
   map_msg.fields=[PointField(name=name,offset=4*i,datatype=7,count=1) for i,name in enumerate(['x','y','z','normal_x','normal_y','normal_z','curvature'])]
   map_pub.publish(map_msg);map_active=True
   filtered=phase(background=True)
   assert any(r.primary_status=='measured' and any(tr.id==r.primary_track_id and abs(tr.center_m[1])<1 for tr in r.tracks) for r in filtered),'map filtering lost foreground lead'
   audit=list(csv.DictReader(open(csv_path)))
   assert any(row['map_filter_status']=='active' and int(row['map_wall_removed'])>0 for row in audit),'map wall was not removed'
   print('Association pipeline: mapped wall removed, foreground lead retained: PASS')
  stale=phase(send_camera=False);assert stale[-1].status=='no_target' and not stale[-1].candidate_unique
  assert stale[-1].primary_track_id==-1,'camera loss left an unrelated LiDAR primary'
  diagnostics=list(csv.DictReader(open(csv_path.replace('.csv','.tracks.csv'))))
  assert diagnostics and all(None not in row for row in diagnostics),'track diagnostics missing/misaligned'
  assert any(row['reason']=='selected' and row['footprint_valid']=='1' and float(row['raw_dy'])>2.6 and float(row['footprint_width_m'])<2.6 for row in diagnostics), 'rotated footprint diagnostics missing'
  assert any(row['reason']=='selected' for row in diagnostics)
  assert any(row['reason']=='camera_evidence_low' for row in diagnostics)
  assert all(3<=tr.range_m<=50 for row in rows for tr in row.tracks if tr.lidar_candidate and tr.measured)
  csv_rows=list(csv.DictReader(open(csv_path)))
  assert any(len(row['candidate_ids'].split(';'))==2 for row in csv_rows),'two target IDs not logged'
  assert all(None not in row for row in csv_rows),'CSV column mismatch'
  assert fronts,'front hemisphere output missing'
  front=fronts[-1];assert front.header.frame_id=='lidar_vehicle'
  off=next(f.offset for f in front.fields if f.name=='x')
  xs=np.ndarray((front.height,front.width),dtype='<f4',buffer=front.data,offset=off,strides=(front.row_step,front.point_step))
  yo=next(f.offset for f in front.fields if f.name=='y')
  ys=np.ndarray((front.height,front.width),dtype='<f4',buffer=front.data,offset=yo,strides=(front.row_step,front.point_step))
  angle=np.deg2rad(args.roi_center_deg)
  assert np.all(np.cos(angle)*xs+np.sin(angle)*ys>=-1e-5) and np.any(xs>8),'wrong sector half-plane'
  if args.roi_center_deg==-45:assert np.any(xs<0),'right rotated sector did not include right-rear test point'
  screen_sub=n.create_subscription(Image,'/lead_fusion/association/image',screens.append,qos_profile_sensor_data)
  deadline=time.monotonic()+2.
  while time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=.01)
  assert screens,'dashboard output missing'
  screen=screens[-1];assert (screen.width,screen.height)==(1600,860)
  pixels=np.ndarray((screen.height,screen.width,3),dtype=np.uint8,buffer=screen.data,strides=(screen.step,3,1))
  assert tuple(pixels[5,5])==(20,20,110),'dashboard did not mark stopped LiDAR stale'
  print('1600x860 dashboard + no-cloud stale banner: PASS')
  print('Configured hemisphere membership and lidar_vehicle frame: PASS')
  print('Patchwork++ ground / single projected candidate / NaN camera metrics ignored / hidden car keeps its track / camera loss releases target: PASS')
 finally:
  proc.terminate()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
  tmp.cleanup();n.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
