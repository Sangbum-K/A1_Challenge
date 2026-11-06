#pragma once
#include "a1_fusion/map_surfaces.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pcl_conversions/pcl_conversions.h>
#include <chrono>
namespace a1::inspection {
// A map from the localization launch, a live matching epoch, recent good ICP
// quality and a timestamped TF are all required before any point is removed.
class MapFilterRos {
 public:
  explicit MapFilterRos(rclcpp::Node& node):node_(node){
   enabled_=node_.declare_parameter("map_filter_enabled",true);
   frame_=node_.declare_parameter<std::string>("map_frame","map");
   if(!enabled_)return;
   buffer_=std::make_unique<tf2_ros::Buffer>(node_.get_clock());listener_=std::make_unique<tf2_ros::TransformListener>(*buffer_);
   map_sub_=node_.create_subscription<sensor_msgs::msg::PointCloud2>("/localization/map_surfaces",rclcpp::QoS(1).reliable().transient_local(),[this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m){
    valid_=false;if(m->header.frame_id!=frame_)return;
    // Refuse missing normal/label fields instead of silently defaulting them.
    for(const char* field:{"x","y","z","normal_x","normal_y","normal_z","curvature"}){
     bool found=false;for(const auto& f:m->fields)if(f.name==field&&f.datatype==7&&f.count==1&&f.offset+4<=m->point_step)found=true;if(!found)return;
    }
    if(m->is_bigendian||m->row_step<uint64_t(m->width)*m->point_step||m->data.size()<uint64_t(m->height)*m->row_step)return;
    auto points=pcl::make_shared<Surfaces>();pcl::fromROSMsg(*m,*points);surfaces_.set(points);epoch_=rclcpp::Time(m->header.stamp).nanoseconds();valid_=!surfaces_.empty();
   });
   epoch_sub_=node_.create_subscription<std_msgs::msg::Header>("/localization/map_surfaces_epoch",rclcpp::QoS(1).reliable().transient_local(),[this](std_msgs::msg::Header::ConstSharedPtr m){
    beat_epoch_=m->frame_id==frame_?rclcpp::Time(m->stamp).nanoseconds():-1;beat_time_=steady();
   });
   quality_sub_=node_.create_subscription<std_msgs::msg::Float32>("/lidar_odometry/pose_quality",rclcpp::SensorDataQoS(),[this](std_msgs::msg::Float32::ConstSharedPtr m){quality_=m->data;quality_time_=steady();quality_ros_time_=node_.now().seconds();});
  }
  std::string begin(double stamp,const std::string& lidar_frame,const Eigen::Matrix3d& body_from_lidar){
   active_=false;
   if(!enabled_)return "disabled";
   if(!valid_)return "no_map";
   if(node_.count_publishers("/localization/map_surfaces")!=1||node_.count_publishers("/localization/map_surfaces_epoch")!=1)return "map_source_unavailable";
   if(beat_epoch_!=epoch_||steady()-beat_time_>.5)return "map_epoch_unavailable";
   if(node_.count_publishers("/lidar_odometry/pose_quality")!=1||!std::isfinite(quality_)||quality_<.5||quality_>1||steady()-quality_time_>.3||std::abs(stamp-quality_ros_time_)>.3)return "localization_untrusted";
   if(lidar_frame.empty())return "no_lidar_frame";
   try{
    const auto tf=buffer_->lookupTransform(frame_,lidar_frame,rclcpp::Time(int64_t(stamp*1e9)),rclcpp::Duration::from_seconds(0));
    const auto& p=tf.transform.translation;const auto& r=tf.transform.rotation;
    Eigen::Quaterniond q(r.w,r.x,r.y,r.z);Eigen::Vector3d t(p.x,p.y,p.z);
    if(!t.allFinite()||!q.coeffs().allFinite()||std::abs(q.norm()-1)>.01)return "invalid_tf";
    map_from_body_.linear()=q.normalized().toRotationMatrix()*body_from_lidar.transpose();map_from_body_.translation()=t;
   }catch(const tf2::TransformException&){return "tf_unavailable";}
   active_=true;return "active";
  }
  bool active()const{return active_;}
  bool has_structure()const{return surfaces_.has_structure();}
  const Eigen::Isometry3d& map_from_body()const{return map_from_body_;}
  // Callers pass map points so they can correct for motion within the scan first.
  int match(const Eigen::Vector3d& map_point,double range)const{return active_?surfaces_.match(map_point,range):0;}
  bool is_structure(const Eigen::Vector3d& map_point,double range)const{return active_&&surfaces_.is_structure(map_point,range);}
  bool covered(const Eigen::Vector3d& map_point)const{return active_&&surfaces_.covered(map_point);}
 private:
  static double steady(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
  rclcpp::Node& node_;bool enabled_{},valid_{},active_{};std::string frame_;MapSurfaces surfaces_;
  int64_t epoch_{-1},beat_epoch_{-2};double beat_time_{-1},quality_time_{-1},quality_ros_time_{-1},quality_{NAN};
  Eigen::Isometry3d map_from_body_=Eigen::Isometry3d::Identity();
  std::unique_ptr<tf2_ros::Buffer> buffer_;std::unique_ptr<tf2_ros::TransformListener> listener_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
  rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr epoch_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr quality_sub_;
};
}
