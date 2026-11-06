#include "a1_fusion/map_filter_ros.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <thread>
#include <iostream>
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
int main(int argc,char** argv){rclcpp::init(argc,argv);try{
 auto node=std::make_shared<rclcpp::Node>("map_filter_synthetic_test");a1::inspection::MapFilterRos filter(*node);
 auto maps=node->create_publisher<sensor_msgs::msg::PointCloud2>("/localization/map_surfaces",rclcpp::QoS(1).reliable().transient_local());
 auto epochs=node->create_publisher<std_msgs::msg::Header>("/localization/map_surfaces_epoch",rclcpp::QoS(1).reliable().transient_local());
 auto quality=node->create_publisher<std_msgs::msg::Float32>("/lidar_odometry/pose_quality",rclcpp::SensorDataQoS());
 tf2_ros::TransformBroadcaster tf(*node);Eigen::Matrix3d rotation=Eigen::AngleAxisd(M_PI/2,Eigen::Vector3d::UnitZ()).toRotationMatrix();
 CHECK(filter.begin(node->now().seconds(),"hesai_lidar",rotation)=="no_map");
 a1::inspection::Surfaces points;for(int y=-4;y<=4;++y)for(int z=-4;z<=4;++z){pcl::PointNormal p;p.x=13;p.y=20+y*.1;p.z=3+z*.1;p.normal_x=1;p.normal_y=p.normal_z=0;p.curvature=2;points.push_back(p);}
 sensor_msgs::msg::PointCloud2 map;pcl::toROSMsg(points,map);map.header.frame_id="map";map.header.stamp=node->now();maps->publish(map);
 std_msgs::msg::Header epoch=map.header;double stamp=0;
 auto pump=[&](double seconds,float q,bool send_quality=true,bool send_tf=true,bool send_correction=true){auto until=std::chrono::steady_clock::now()+std::chrono::duration<double>(seconds);
  while(std::chrono::steady_clock::now()<until){stamp=node->now().seconds();epochs->publish(epoch);
   if(send_quality){std_msgs::msg::Float32 msg;msg.data=q;quality->publish(msg);}
   // MOLA publishes map->odom about 0.25 s behind the scan; odom->lidar follows the INS in real time.
   if(send_correction){geometry_msgs::msg::TransformStamped msg;msg.header.stamp=rclcpp::Time(int64_t((stamp-.25)*1e9));msg.header.frame_id="map";msg.child_frame_id="odom";msg.transform.translation.x=10;msg.transform.translation.y=20;msg.transform.rotation.w=1;tf.sendTransform(msg);}
   if(send_tf){geometry_msgs::msg::TransformStamped msg;msg.header.stamp=rclcpp::Time(int64_t(stamp*1e9));msg.header.frame_id="odom";msg.child_frame_id="hesai_lidar";msg.transform.translation.z=1;msg.transform.rotation.z=std::sqrt(.5);msg.transform.rotation.w=std::sqrt(.5);tf.sendTransform(msg);}
   rclcpp::spin_some(node);std::this_thread::sleep_for(std::chrono::milliseconds(10));rclcpp::spin_some(node);
  }
 };
 pump(1.,.9);CHECK(filter.begin(stamp,"hesai_lidar",rotation)=="active");
 auto in_map=[&](double x,double y,double z){return Eigen::Vector3d(filter.map_from_body()*Eigen::Vector3d(x,y,z));};
 CHECK(filter.match(in_map(3,0,2),0.)==2);CHECK(filter.match(in_map(2.5,0,2),0.)==0); // Correct rotation + lever arm, nearby foreground preserved.
 CHECK(filter.active());CHECK(!filter.has_structure());CHECK(filter.covered(in_map(3,0,2)));
 pump(.1,.1);CHECK(filter.begin(stamp,"hesai_lidar",rotation)=="localization_untrusted");CHECK(filter.match(in_map(3,0,2),0.)==0);CHECK(!filter.active());
 pump(.1,.9);CHECK(filter.begin(stamp,"missing_lidar",rotation)=="tf_unavailable");
 pump(.4,.9,false);CHECK(filter.begin(stamp,"hesai_lidar",rotation)=="localization_untrusted");
 pump(.1,.9);epoch.stamp.sec+=1;pump(.1,.9);CHECK(filter.begin(stamp,"hesai_lidar",rotation)=="map_epoch_unavailable");
 epoch=map.header;pump(.1,.9);CHECK(filter.begin(stamp,"hesai_lidar",rotation)=="active");
 pump(.8,.9,true,true,false);CHECK(filter.begin(stamp,"hesai_lidar",rotation)=="tf_stale"); // map->odom stopped: newest correction is over 0.5 s old.
 pump(.1,.9);CHECK(filter.begin(stamp,"hesai_lidar",rotation)=="active");
 pump(.3,.9,true,false);CHECK(filter.begin(stamp,"hesai_lidar",rotation)=="tf_unavailable"); // odom->lidar stopped: the scan stamp is newer than the INS.
 pump(.1,.9);CHECK(filter.begin(stamp,"hesai_lidar",rotation)=="active");
 maps.reset();pump(.5,.9);CHECK(filter.begin(stamp,"hesai_lidar",rotation)=="map_source_unavailable");
 std::cout<<"ROS map filter: transform through a lagging map->odom, foreground, low/stale quality, missing/stale TF, epoch mismatch, dead map server: PASS\n";
 rclcpp::shutdown();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';rclcpp::shutdown();return 1;}}
