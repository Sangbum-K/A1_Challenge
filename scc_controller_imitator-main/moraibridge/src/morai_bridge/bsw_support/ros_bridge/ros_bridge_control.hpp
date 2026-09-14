/**
 * @file        ros_bridge_control.hpp
 * @brief       ROS bridge for control
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-01 updated by Yuseung Na: Refactoring 
 */

#ifndef __ROS_BRIDGE_CONTROL__
#define __ROS_BRIDGE_CONTROL__
#pragma once

// ROS header
#include <ros/ros.h>
#include <tf/tf.h>

// ROS Message Header
#include <autohyu_msgs/ControlInputOutput.h>
#include <autohyu_msgs/ControlInfo.h>

// Interface Header
#include "interface_vehicle_state.hpp"
#include "interface_trajectories.hpp"
#include "interface_control.hpp"
#include "interface_planning_control.hpp"

// Bridge Header
#include "ros_bridge_std_msgs.hpp"
#include "ros_bridge_vehicle_state.hpp"
#include "ros_bridge_trajectories.hpp"

namespace ros_bridge {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    interface::ControlInputOutput GetControlInputOutput(const autohyu_msgs::ControlInputOutput& msg) {
        interface::ControlInputOutput control_io;

        control_io.header = ros_bridge::GetHeader(msg.header);
        control_io.header.frame_id = "world";
        
        control_io.vehicle_state = ros_bridge::GetVehicleState(msg.vehicle_state);
        control_io.trajectory = ros_bridge::GetTrajectory(msg.trajectory);
        control_io.solved_trajectory = ros_bridge::GetTrajectory(msg.solved_trajectory);
        
        for (auto steer : msg.command_steer) {
            control_io.command_steer.push_back(steer);
        }
        for (auto accel_pedal : msg.command_accel_pedal) {
            control_io.command_accel_pedal.push_back(accel_pedal);
        }
        for (auto brake_pedal : msg.command_brake_pedal) {
            control_io.command_brake_pedal.push_back(brake_pedal);
        }
        for (auto speed : msg.command_speed) {
            control_io.command_speed.push_back(speed);
        }
        for (auto accel : msg.command_acceleration) {
            control_io.command_acceleration.push_back(accel);
        }

        return control_io;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    autohyu_msgs::ControlInputOutput UpdateControlInputOutput(const interface::VehicleState& vehicle_state,
                                                              const interface::Trajectory& trajectory,
                                                              const interface::ControlTrajectory& solved_trajectory,
                                                              const std::vector<double>& command_steer,
                                                              const std::vector<double>& command_accel_pedal,
                                                              const std::vector<double>& command_brake_pedal,
                                                              const std::vector<double>& command_speed,
                                                              const std::vector<double>& command_acceleration) {
        autohyu_msgs::ControlInputOutput msg;
        msg.header = UpdateHeader(vehicle_state.header);

        msg.vehicle_state   = ros_bridge::UpdateVehicleState(vehicle_state);
        msg.trajectory      = ros_bridge::UpdateTrajectory(trajectory);

        for (auto point : solved_trajectory.control_point) {
            autohyu_msgs::TrajectoryPoint tpoint;
            tpoint.time         = point.time;
            tpoint.distance     = point.distance;
            tpoint.direction    = 0; // Forward
            tpoint.x            = point.x;
            tpoint.y            = point.y;
            tpoint.z            = 0.0;
            tpoint.yaw          = point.yaw;
            tpoint.curvature    = point.curvature;
            tpoint.speed        = point.vx;
            tpoint.acceleration = point.ax;
            msg.solved_trajectory.point.push_back(tpoint);
        }

        for (int i = 0; i < command_steer.size(); i++) {
            msg.command_steer.push_back(static_cast<float>(command_steer[i]));
        }
        for (int i = 0; i < command_accel_pedal.size(); i++) {
            msg.command_accel_pedal.push_back(static_cast<float>(command_accel_pedal[i]));
        }
        for (int i = 0; i < command_brake_pedal.size(); i++) {
            msg.command_brake_pedal.push_back(static_cast<float>(command_brake_pedal[i]));
        }
        for (int i = 0; i < command_speed.size(); i++) {
            msg.command_speed.push_back(static_cast<float>(command_speed[i]));
        }
        for (int i = 0; i < command_acceleration.size(); i++) {
            msg.command_acceleration.push_back(static_cast<float>(command_acceleration[i]));
        }

        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizPredictedPath(const interface::ControlTrajectory& control_trajectory) {
        visualization_msgs::MarkerArray msg;
        visualization_msgs::Marker      line_marker, point_marker;

        line_marker.header.frame_id = control_trajectory.header.frame_id;
        // line_marker.header.frame_id = "base_link";
        line_marker.header.stamp    = ros::Time(control_trajectory.header.stamp);
        line_marker.ns              = "control_trajectory";
        line_marker.action          = visualization_msgs::Marker::ADD;
        line_marker.type            = visualization_msgs::Marker::LINE_STRIP;
        line_marker.lifetime        = ros::Duration(0.1);
        line_marker.scale.x = 0.15;
        line_marker.color.r = 1.0f;
        line_marker.color.g = 0.078f;
        line_marker.color.b = 0.57;
        line_marker.color.a = 0.3f;

        point_marker.header.frame_id = control_trajectory.header.frame_id;
        // point_marker.header.frame_id = "base_link";
        point_marker.header.stamp    = ros::Time(control_trajectory.header.stamp);
        point_marker.ns              = "control_points";
        point_marker.action          = visualization_msgs::Marker::ADD;
        point_marker.type            = visualization_msgs::Marker::CYLINDER;
        point_marker.lifetime        = ros::Duration(0.1);
        point_marker.scale.x = 0.15;
        point_marker.scale.y = 0.15;
        point_marker.scale.z = 0.01;
        point_marker.color.r = 1.0f;
        point_marker.color.g = 0.078f;
        point_marker.color.b = 0.57;
        point_marker.color.a = 0.8f;

        uint16_t marker_id = 0;
        for (auto control_point : control_trajectory.control_point) {
            geometry_msgs::Point point;
            point.x = control_point.x;
            point.y = control_point.y;
            point.z = 0.2;
            line_marker.points.push_back(point);

            point_marker.id              = marker_id++;
            point_marker.pose.position.x = control_point.x;
            point_marker.pose.position.y = control_point.y;
            point_marker.pose.position.z = 0.25;
            point_marker.scale.z         = 0.5;

            point_marker.pose.orientation.x = 0.0;
            point_marker.pose.orientation.y = 0.0;
            point_marker.pose.orientation.z = 0.0;
            point_marker.pose.orientation.w = 1.0;

            msg.markers.push_back(point_marker);
        }
        msg.markers.push_back(line_marker);

        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizPredictedSpeedProfile(const interface::ControlTrajectory& control_trajectory) {
        visualization_msgs::MarkerArray msg;
        visualization_msgs::Marker      point_marker;

        point_marker.header.frame_id = "ego_frame";
        point_marker.header.stamp    = ros::Time(control_trajectory.header.stamp);
        point_marker.ns              = "control_trajectory";
        point_marker.action          = visualization_msgs::Marker::ADD;
        point_marker.type            = visualization_msgs::Marker::CYLINDER;
        point_marker.lifetime        = ros::Duration(0.1);

        point_marker.scale.x = 0.15;
        point_marker.scale.y = 0.15;
        point_marker.scale.z = 0.01;

        point_marker.color.r = 1.0f;
        point_marker.color.g = 0.078f;
        point_marker.color.b = 0.57;
        point_marker.color.a = 0.3f;

        uint16_t marker_id = 0;
        for ( auto control_point : control_trajectory.control_point ) {
            point_marker.id              = marker_id++;
            point_marker.pose.position.x = control_point.x;
            point_marker.pose.position.y = control_point.y;
            point_marker.pose.position.z = control_point.vx * interface::KPH2MPS/2;
            point_marker.scale.z         = control_point.vx * interface::KPH2MPS;

            point_marker.pose.orientation.x = 0.0;
            point_marker.pose.orientation.y = 0.0;
            point_marker.pose.orientation.z = 0.0;
            point_marker.pose.orientation.w = 1.0;

            msg.markers.push_back(point_marker);
        }

        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizControlReference(const interface::ControlTrajectory& trajectory) {
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker      marker;

        marker.header.frame_id = trajectory.header.frame_id;
        marker.header.stamp    = ros::Time(trajectory.header.stamp);
        marker.action          = visualization_msgs::Marker::ADD;
        marker.type            = visualization_msgs::Marker::CYLINDER;
        marker.lifetime        = ros::Duration(0.1);

        marker.scale.x = 0.15;
        marker.scale.y = 0.15;

        marker.color.r = 0.0f;
        marker.color.g = 0.78f;
        marker.color.b = 0.58f;
        marker.color.a = 0.5f;

        uint16_t marker_id = 0;
        for ( auto dp : trajectory.control_point ) {
            marker.id              = marker_id++;
            marker.ns              = "curvature";
            marker.color.r = 0.0f;
            marker.color.g = 0.78f;
            marker.color.b = 0.58f;
            marker.pose.position.x = dp.x;
            marker.pose.position.y = dp.y;
            marker.pose.position.z = dp.curvature*50; // scale up for viz
            marker.scale.x         = 0.15;
            marker.scale.y         = 0.15;
            marker.scale.z         = dp.curvature*100;
            marker.pose.orientation.x = 0.;
            marker.pose.orientation.y = 0.;
            marker.pose.orientation.z = 0.;
            marker.pose.orientation.w = 1.;
            marker_array.markers.push_back(marker);

            marker.id              = marker_id++;
            marker.ns              = "yaw";
            marker.color.r = 1.0f;
            marker.pose.position.z = dp.yaw*interface::RAD2DEG/4; // scale up for viz
            marker.scale.z         = dp.yaw*interface::RAD2DEG/2;
            marker_array.markers.push_back(marker);

            marker.id              = marker_id++;
            marker.ns              = "speed";
            marker.color.r = 0.6f;
            marker.color.g = 0.6f;
            marker.pose.position.z = dp.vx*interface::KPH2MPS/2; // scale up for viz
            marker.scale.z         = dp.vx*interface::KPH2MPS;
            marker_array.markers.push_back(marker);
        }
        return marker_array;
    }

    autohyu_msgs::ControlInfo UpdateControlInfos(const interface::PathTrackingInfos& tracking_errors) {
        autohyu_msgs::ControlInfo msg;
        msg.cross_track_error = tracking_errors.cross_track_error;
        msg.yaw_error = tracking_errors.yaw_error;
        msg.speed_error = tracking_errors.speed_error;
        msg.target_speed = tracking_errors.target_speed;
        msg.current_speed = tracking_errors.current_speed;
        msg.opt_cost = tracking_errors.opt_cost;
        msg.opt_time = tracking_errors.opt_time;
        msg.solve_time = tracking_errors.solve_time;
        msg.total_time = tracking_errors.total_time;
        msg.sqp_iter = tracking_errors.sqp_iter;
        msg.qp_iter = tracking_errors.qp_iter;
        return msg;
    }
} // namespace ros_bridge

#endif  // __ROS_BRIDGE_CONTROL__