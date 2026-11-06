#pragma once
#include "a1_fusion/core.hpp"
#include <gsof_msgs/msg/navigation_solution49.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/time.hpp>
#include <string>
namespace a1 {
// ALIGNED(3) or FULL_NAV(4). A1_localization's 2026-10-08 bags stayed at ALIGNED for the whole
// drive, also with RTK fixed, and never reported FULL_NAV (docs/setup/gnss_integration.md there).
inline bool navigation_aligned(const gsof_msgs::msg::NavigationSolution49& m){
  const auto alignment=m.status.imu_alignment;
  return (alignment==3||alignment==4)&&m.status.gnss>=1&&m.status.gnss<=6;
}
// Empty result means e holds a usable sample. Position stays for EgoHistory to integrate:
// map-corrected positions and stationary holds must never enter the continuous frame.
inline std::string parse_gnss_odom(const nav_msgs::msg::Odometry& m,Ego& e){
  const auto& q=m.pose.pose.orientation;Eigen::Quaterniond rotation(q.w,q.x,q.y,q.z);
  const auto& v=m.twist.twist.linear;const auto& w=m.twist.twist.angular;
  const Vec body_velocity(v.x,v.y,v.z);
  if(m.header.frame_id!="odom"||m.child_frame_id!="base_link"||!rotation.coeffs().allFinite()||
     std::abs(rotation.norm()-1)>0.01||!body_velocity.allFinite()||!std::isfinite(w.z)||
     !std::isfinite(m.twist.covariance[0]))return "gnss_odom_frame_or_fields_invalid";
  // trimble_driver_ros writes -1 until GSOF #50 (RMS) arrives; the relay passes it through.
  if(m.twist.covariance[0]<0)return "gnss_odom_rms_missing";
  rotation.normalize();const auto matrix=rotation.toRotationMatrix();
  // Same planar contract as primary target motion; banking needs a 6DoF model.
  if(matrix(2,2)<std::cos(10.*M_PI/180.))return "gnss_odom_tilt_unsupported";
  e.stamp=rclcpp::Time(m.header.stamp).seconds();e.valid=true;e.source="gnss_odometry";
  e.yaw=std::atan2(matrix(1,0),matrix(0,0));e.yaw_rate=w.z;
  e.velocity=Eigen::AngleAxisd(e.yaw,Vec::UnitZ())*body_velocity;
  return {};
}
}
