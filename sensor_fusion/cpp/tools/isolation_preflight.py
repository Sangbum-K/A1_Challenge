"""Reject duplicate perception nodes; shared mode permits localization/BSW."""
import os
import time
import rclpy
rclpy.init()
node = rclpy.create_node('a1_private_domain_preflight')
deadline = time.monotonic() + 2
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=.05)
private = os.environ.get('A1_FUSION_TRANSPORT', 'direct') == 'bridge'
owned = {'a1_fusion', 'a1_association', 'a2_camera', 'a1_a2_camera',
         'a1_lidar_bridge_sink', 'a1_lidar_readonly_bridge_source'}
foreign = [ns.rstrip('/') + '/' + name for name, ns in node.get_node_names_and_namespaces()
           if name != node.get_name() and not name.startswith('_ros2cli_daemon')
           and (private or name in owned)]
node.destroy_node()
rclpy.shutdown()
if foreign:
    raise SystemExit('Conflicting application nodes: ' + ', '.join(sorted(foreign)))
print('Perception ownership check passed; domain=' + os.environ.get('ROS_DOMAIN_ID', '0'))
