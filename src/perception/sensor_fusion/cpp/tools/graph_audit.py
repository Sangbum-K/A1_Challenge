"""Read-only ROS graph/rate audit. Does not publish application data or stop nodes."""
import argparse,json,os,time
from collections import defaultdict
from pathlib import Path
import rclpy
from rclpy.qos import qos_profile_sensor_data
from a1_fusion_cpp.msg import A2Features,Association,BrakeEstimate,EgoMotion
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=8);args=p.parse_args()
rclpy.init();n=rclpy.create_node('a1_readonly_graph_audit_'+str(os.getpid()))
times=defaultdict(list)
def got(key,m):times[key].append(time.monotonic())
subs=[n.create_subscription(cls,topic,lambda m,k=topic:got(k,m),qos_profile_sensor_data)
      for cls,topic in [(A2Features,'/lead_fusion/a2'),(Association,'/lead_fusion/association'),
                        (BrakeEstimate,'/lead_fusion/estimate'),(EgoMotion,'/lead_fusion/ego_motion')]]
end=time.monotonic()+args.seconds
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.02)
report=dict(domain=os.environ.get('ROS_DOMAIN_ID','0'),nodes=n.get_node_names_and_namespaces(),
 rates={k:dict(count=len(v),hz=(len(v)-1)/(v[-1]-v[0]) if len(v)>1 else 0) for k,v in times.items()},topics={})
for topic,types in n.get_topic_names_and_types():
 report['topics'][topic]=dict(types=types,publishers=[dict(node=e.node_name,namespace=e.node_namespace) for e in n.get_publishers_info_by_topic(topic)])
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(domain=report['domain'],nodes=report['nodes'],rates=report['rates']),indent=2))
n.destroy_node();rclpy.shutdown()
