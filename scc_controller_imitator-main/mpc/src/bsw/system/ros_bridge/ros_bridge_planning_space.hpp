/**
 * @file        ros_bridge_planning_space.hpp
 * @brief       ROS bridge for planning_space
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2024-12-14 created by Yuseung Na
 */

#ifndef __ROS_BRIDGE_PLANNING_SPACE__
#define __ROS_BRIDGE_PLANNING_SPACE__
#pragma once

// ROS header
#include <ros/ros.h>
#include <tf/tf.h>

// ROS Message Header
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <autohyu_msgs/PlanningSpace.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>

// Interface Header
#include "interface_constants.hpp"
#include "interface_planning_space.hpp"
#include "interface_trajectories.hpp"

// Bridge Header
#include "ros_bridge_std_msgs.hpp"
#include "ros_bridge_trajectories.hpp"
#include "ros_bridge_objects.hpp"

namespace ros_bridge {    
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //   
    interface::PlanningSpaceState GetPlanningSpaceState(const autohyu_msgs::PlanningSpaceState& msg) {
        interface::PlanningSpaceState planning_space_state;
        planning_space_state.stamp      = msg.stamp;
        planning_space_state.x          = msg.x;
        planning_space_state.y          = msg.y;
        planning_space_state.heading    = msg.heading;
        planning_space_state.vx         = msg.vx;
        planning_space_state.vy         = msg.vy;
        planning_space_state.ax         = msg.ax;
        planning_space_state.ay         = msg.ay;
        planning_space_state.tire_angle = msg.tire_angle;

        return planning_space_state;
    }

    interface::PlanningSpacePoint GetPlanningSpacePoint(const autohyu_msgs::PlanningSpacePoint& msg) {
        interface::PlanningSpacePoint planning_space_point;
        planning_space_point.x            = msg.x;
        planning_space_point.y            = msg.y;
        planning_space_point.heading      = msg.heading;
        planning_space_point.speed_limit  = msg.speed_limit;
        planning_space_point.traffic_light_id = msg.traffic_light_id;
        planning_space_point.traffic_light_color = msg.traffic_light_color;

        return planning_space_point;
    }

    interface::PlanningSpaceVectorMap GetPlanningSpaceVectorMap(const autohyu_msgs::PlanningSpaceVectorMap& msg) {
        interface::PlanningSpaceVectorMap planning_space_vector_map;
        planning_space_vector_map.id = msg.id;

        for (const auto& point : msg.point) {
            planning_space_vector_map.point.push_back(GetPlanningSpacePoint(point));
        }

        return planning_space_vector_map;
    }

    interface::PlanningSpaceTrafficLight GetPlanningSpaceTrafficLight(const autohyu_msgs::PlanningSpaceTrafficLight& msg) {
        interface::PlanningSpaceTrafficLight planning_space_traffic_light;
        planning_space_traffic_light.header = GetHeader(msg.header);
        planning_space_traffic_light.id = msg.id;
        planning_space_traffic_light.type = static_cast<interface::TrafficLightType>(msg.type);
        planning_space_traffic_light.color = static_cast<interface::TrafficLightColor>(msg.color);
        planning_space_traffic_light.check_state = msg.check_state;

        for (const auto& point : msg.stop_line_points) {
            interface::Point stop_line_point;
            stop_line_point.x = point.x;
            stop_line_point.y = point.y;
            stop_line_point.z = point.z;
            planning_space_traffic_light.stop_line_points.push_back(stop_line_point);
        }

        return planning_space_traffic_light;
    }

    interface::PlanningSpace GetPlanningSpace(const autohyu_msgs::PlanningSpace& msg) {
        interface::PlanningSpace planning_space;
        planning_space.header = GetHeader(msg.header);

        // Ego
        for (const auto& state : msg.ego.state) {
            planning_space.ego.state.push_back(GetPlanningSpaceState(state));
        }

        // Objects
        for (const auto& object : msg.objects) {
            interface::PlanningSpaceObject planning_space_object;
            planning_space_object.id             = object.id;
            planning_space_object.length         = object.length;
            planning_space_object.width          = object.width;
            planning_space_object.height         = object.height;
            planning_space_object.classification = object.classification;

            for (const auto& state : object.state) {
                planning_space_object.state.push_back(GetPlanningSpaceState(state));
            }

            planning_space.objects.push_back(planning_space_object);
        }

        // Vector Map
        for (const auto& lane : msg.lanes) {
            planning_space.lanes.push_back(GetPlanningSpaceVectorMap(lane));
        }
        for (const auto& route : msg.routes) {
            planning_space.routes.push_back(GetPlanningSpaceVectorMap(route));
        }
        for (const auto& crosswalk : msg.crosswalks) {
            planning_space.crosswalks.push_back(GetPlanningSpaceVectorMap(crosswalk));
        }
        for (const auto& goal_route : msg.goal_routes) {
            planning_space.goal_routes.push_back(GetPlanningSpaceVectorMap(goal_route));
        }
        
        // Drivable Area
        planning_space.drivable_area.header = GetHeader(msg.drivable_area.header);
        planning_space.drivable_area.id = msg.drivable_area.id;
        
        // Left Boundary
        for (const auto& point : msg.drivable_area.left_boundary.point) {
            interface::TrajectoryBoundaryPoint boundary_point;
            boundary_point.x = point.x;
            boundary_point.y = point.y;
            boundary_point.z = point.z;
            planning_space.drivable_area.left_boundary.point.push_back(boundary_point);
        }
        
        // Right Boundary
        for (const auto& point : msg.drivable_area.right_boundary.point) {
            interface::TrajectoryBoundaryPoint boundary_point;
            boundary_point.x = point.x;
            boundary_point.y = point.y;
            boundary_point.z = point.z;
            planning_space.drivable_area.right_boundary.point.push_back(boundary_point);
        }
        
        // Center Points
        for (const auto& point : msg.drivable_area.point) {
            interface::TrajectoryPoint trajectory_point;
            trajectory_point.time = point.time;
            trajectory_point.direction = static_cast<interface::TrajectoryDirection>(point.direction);
            trajectory_point.x = point.x;
            trajectory_point.y = point.y;
            trajectory_point.z = point.z;
            trajectory_point.yaw = point.yaw;
            trajectory_point.curvature = point.curvature;
            trajectory_point.distance = point.distance;
            trajectory_point.speed = point.speed;
            trajectory_point.acceleration = point.acceleration;
            planning_space.drivable_area.point.push_back(trajectory_point);
        }
        
        // Traffic Light
        planning_space.traffic_light = GetPlanningSpaceTrafficLight(msg.traffic_light);
        
        // Predicted Objects - use existing GetPredictObjects function
        planning_space.predict_objects = ros_bridge::GetPredictObjects(msg.predict_objects);

        // Imitation Trajectory
        interface::Trajectory trajectory;
        trajectory.header = GetHeader(msg.imitation_trajectory.header);
        trajectory.id = msg.imitation_trajectory.id;

        for (const auto& point : msg.imitation_trajectory.point) {            
            interface::TrajectoryPoint tp;
            tp.time = point.time;
            tp.direction = static_cast<interface::TrajectoryDirection>(point.direction);
            tp.x = point.x;
            tp.y = point.y;
            tp.z = point.z;
            tp.yaw = point.yaw;
            tp.curvature = point.curvature;
            tp.distance = point.distance;
            tp.speed = point.speed;
            tp.acceleration = point.acceleration;

            trajectory.point.push_back(tp);
        }
        planning_space.imitation_trajectory = trajectory;
        
        return planning_space;
    }
    
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    autohyu_msgs::PlanningSpace UpdatePlanningSpace(const interface::PlanningSpace& planning_space) {
        autohyu_msgs::PlanningSpace msg;

        msg.header = ros_bridge::UpdateHeader(planning_space.header);

        // Ego
        for (const auto& state : planning_space.ego.state) {
            autohyu_msgs::PlanningSpaceState planning_space_state;
            planning_space_state.stamp      = state.stamp;
            planning_space_state.x          = state.x;
            planning_space_state.y          = state.y;
            planning_space_state.heading    = state.heading;
            planning_space_state.vx         = state.vx;
            planning_space_state.vy         = state.vy;
            planning_space_state.ax         = state.ax;
            planning_space_state.ay         = state.ay;
            planning_space_state.tire_angle = state.tire_angle;

            msg.ego.state.push_back(planning_space_state);
        }

        // Objects
        for (const auto& object : planning_space.objects) {
            autohyu_msgs::PlanningSpaceObject planning_space_object;
            planning_space_object.id             = object.id;
            planning_space_object.length         = object.length;
            planning_space_object.width          = object.width;
            planning_space_object.height         = object.height;
            planning_space_object.classification = object.classification;

            for (const auto& state : object.state) {
                autohyu_msgs::PlanningSpaceState planning_space_state;
                planning_space_state.stamp      = state.stamp;
                planning_space_state.x          = state.x;
                planning_space_state.y          = state.y;
                planning_space_state.heading    = state.heading;
                planning_space_state.vx         = state.vx;
                planning_space_state.vy         = state.vy;
                planning_space_state.ax         = state.ax;
                planning_space_state.ay         = state.ay;
                planning_space_state.tire_angle = 0.0;

                planning_space_object.state.push_back(planning_space_state);
            }

            msg.objects.push_back(planning_space_object);
        }

        // Vector Map
        for (const auto& lane : planning_space.lanes) {
            autohyu_msgs::PlanningSpaceVectorMap planning_space_vector_map;
            planning_space_vector_map.id = lane.id;
            planning_space_vector_map.is_dividing_lane = lane.is_dividing_lane;

            for (const auto& point : lane.point) {
                autohyu_msgs::PlanningSpacePoint planning_space_point;
                planning_space_point.x                   = point.x;
                planning_space_point.y                   = point.y;
                planning_space_point.heading             = point.heading;
                planning_space_point.traffic_light_id    = point.traffic_light_id;
                planning_space_point.traffic_light_color = point.traffic_light_color;
                planning_space_point.speed_limit         = point.speed_limit;
                planning_space_vector_map.point.push_back(planning_space_point);
            }

            for (const auto& left_boundary : lane.left_boundary) {
                autohyu_msgs::PlanningSpacePoint left_boundary_point;
                left_boundary_point.x                   = left_boundary.x;
                left_boundary_point.y                   = left_boundary.y;
                left_boundary_point.heading             = left_boundary.heading;
                left_boundary_point.traffic_light_id    = left_boundary.traffic_light_id;
                left_boundary_point.traffic_light_color = left_boundary.traffic_light_color;
                left_boundary_point.speed_limit         = left_boundary.speed_limit;
                planning_space_vector_map.left_boundary.push_back(left_boundary_point);
            }

            for (const auto& right_boundary : lane.right_boundary) {
                autohyu_msgs::PlanningSpacePoint right_boundary_point;
                right_boundary_point.x                   = right_boundary.x;
                right_boundary_point.y                   = right_boundary.y;
                right_boundary_point.heading             = right_boundary.heading;
                right_boundary_point.traffic_light_id    = right_boundary.traffic_light_id;
                right_boundary_point.traffic_light_color = right_boundary.traffic_light_color;
                right_boundary_point.speed_limit         = right_boundary.speed_limit;
                planning_space_vector_map.right_boundary.push_back(right_boundary_point);
            }

            msg.lanes.push_back(planning_space_vector_map);
        }

        for (const auto& route : planning_space.routes) {
            autohyu_msgs::PlanningSpaceVectorMap planning_space_vector_map;
            planning_space_vector_map.id = route.id;

            for (const auto& point : route.point) {
                autohyu_msgs::PlanningSpacePoint planning_space_point;
                planning_space_point.x                   = point.x;
                planning_space_point.y                   = point.y;
                planning_space_point.heading             = point.heading;
                planning_space_point.traffic_light_id    = point.traffic_light_id;
                planning_space_point.traffic_light_color = point.traffic_light_color;
                planning_space_point.speed_limit          = point.speed_limit;

                planning_space_vector_map.point.push_back(planning_space_point);
            }

            msg.routes.push_back(planning_space_vector_map);
        }

        for (const auto& crosswalk : planning_space.crosswalks) {
            autohyu_msgs::PlanningSpaceVectorMap planning_space_vector_map;
            planning_space_vector_map.id = crosswalk.id;

            for (const auto& point : crosswalk.point) {
                autohyu_msgs::PlanningSpacePoint planning_space_point;
                planning_space_point.x                   = point.x;
                planning_space_point.y                   = point.y;
                planning_space_point.heading             = point.heading;
                planning_space_point.traffic_light_id    = point.traffic_light_id;
                planning_space_point.traffic_light_color = point.traffic_light_color;
                planning_space_point.speed_limit         = point.speed_limit;
                planning_space_vector_map.point.push_back(planning_space_point);
            }

            msg.crosswalks.push_back(planning_space_vector_map);
        }

        for (const auto& goal_route : planning_space.goal_routes) {
            autohyu_msgs::PlanningSpaceVectorMap planning_space_vector_map;
            planning_space_vector_map.id = goal_route.id;
            for (const auto& point : goal_route.point) {
                autohyu_msgs::PlanningSpacePoint planning_space_point;
                planning_space_point.x                    = point.x;
                planning_space_point.y                    = point.y;
                planning_space_point.heading              = point.heading;
                planning_space_point.traffic_light_id     = point.traffic_light_id;
                planning_space_point.traffic_light_color  = point.traffic_light_color;
                planning_space_point.speed_limit          = point.speed_limit;
                planning_space_vector_map.point.push_back(planning_space_point);
            }

            msg.goal_routes.push_back(planning_space_vector_map);
        }
        
        // Drivable Area
        msg.drivable_area.header = UpdateHeader(planning_space.drivable_area.header);
        msg.drivable_area.header.frame_id = "world";
        msg.drivable_area.id = planning_space.drivable_area.id;
        
        // Left Boundary
        for (const auto& point : planning_space.drivable_area.left_boundary.point) {
            autohyu_msgs::TrajectoryBoundaryPoint boundary_point;
            boundary_point.x = point.x;
            boundary_point.y = point.y;
            boundary_point.z = point.z;
            msg.drivable_area.left_boundary.point.push_back(boundary_point);
        }
        
        // Right Boundary
        for (const auto& point : planning_space.drivable_area.right_boundary.point) {
            autohyu_msgs::TrajectoryBoundaryPoint boundary_point;
            boundary_point.x = point.x;
            boundary_point.y = point.y;
            boundary_point.z = point.z;
            msg.drivable_area.right_boundary.point.push_back(boundary_point);
        }
        
        // Center Points
        for (const auto& point : planning_space.drivable_area.point) {
            autohyu_msgs::TrajectoryPoint trajectory_point;
            trajectory_point.time = point.time;
            trajectory_point.direction = point.direction;
            trajectory_point.x = point.x;
            trajectory_point.y = point.y;
            trajectory_point.z = point.z;
            trajectory_point.yaw = point.yaw;
            trajectory_point.curvature = point.curvature;
            trajectory_point.distance = point.distance;
            trajectory_point.speed = point.speed;
            trajectory_point.acceleration = point.acceleration;
            msg.drivable_area.point.push_back(trajectory_point);
        }

        // Stop Line
        autohyu_msgs::PlanningSpaceTrafficLight traffic_light;
        traffic_light.header = UpdateHeader(planning_space.header);
        traffic_light.header.frame_id = "world";
        traffic_light.id = planning_space.traffic_light.id;
        traffic_light.type = planning_space.traffic_light.type;
        traffic_light.color = planning_space.traffic_light.color;
        traffic_light.check_state = planning_space.traffic_light.check_state;

        for (const auto& point : planning_space.traffic_light.stop_line_points) {
            geometry_msgs::Point pose;
            pose.x = point.x;
            pose.y = point.y;
            pose.z = point.z;
            traffic_light.stop_line_points.push_back(pose);
        }
        msg.traffic_light = traffic_light;

        // Predicted Objects - use existing UpdatePredictObjects function
        msg.predict_objects = ros_bridge::UpdatePredictObjects(planning_space.predict_objects);   

        // Imitation Trajectory
        msg.imitation_trajectory.header = UpdateHeader(planning_space.imitation_trajectory.header);
        msg.imitation_trajectory.header.frame_id = "world";
        msg.imitation_trajectory.id = planning_space.imitation_trajectory.id;

        for (auto point : planning_space.imitation_trajectory.point) {
            autohyu_msgs::TrajectoryPoint tpoint;
            tpoint.time         = point.time;
            tpoint.direction    = point.direction;
            tpoint.x            = point.x;
            tpoint.y            = point.y;
            tpoint.z            = point.z;
            tpoint.yaw          = point.yaw;
            tpoint.curvature    = point.curvature;
            tpoint.distance     = point.distance;
            tpoint.speed        = point.speed;
            tpoint.acceleration = point.acceleration;

            msg.imitation_trajectory.point.push_back(tpoint);
        }
        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizPlanningSpace(const interface::PlanningSpace& planning_space) {
        visualization_msgs::MarkerArray msg;

        // Ego
        visualization_msgs::Marker ego_marker;
        ego_marker.header = ros_bridge::UpdateHeader(planning_space.header);
        ego_marker.ns = "ego";
        ego_marker.id = 0;
        ego_marker.type = visualization_msgs::Marker::LINE_STRIP;
        ego_marker.action = visualization_msgs::Marker::ADD;
        ego_marker.scale.x         = interface::VEHICLE_WIDTH;
        ego_marker.color.r         = 0.0;
        ego_marker.color.g         = 1.0;
        ego_marker.color.b         = 0.0;
        ego_marker.color.a         = 0.3;
        ego_marker.lifetime        = ros::Duration(0.1);
        for (const auto& state : planning_space.ego.state) {
            geometry_msgs::Point point;
            point.x = state.x;
            point.y = state.y;
            point.z = 0.0;
            ego_marker.points.push_back(point);
        }
        msg.markers.push_back(ego_marker);

        // Objects
        for (const auto& object : planning_space.objects) {
            visualization_msgs::Marker object_marker;
            object_marker.header = ros_bridge::UpdateHeader(planning_space.header);
            object_marker.ns = "object";
            object_marker.id = object.id;
            object_marker.type = visualization_msgs::Marker::LINE_STRIP;
            object_marker.action = visualization_msgs::Marker::ADD;
            object_marker.scale.x         = object.width;
            object_marker.color.r         = 1.0;
            object_marker.color.g         = 1.0;
            object_marker.color.b         = 1.0;
            object_marker.color.a         = 0.3;
            object_marker.lifetime        = ros::Duration(0.1);
            for (const auto& state : object.state) {
                geometry_msgs::Point point;
                point.x = state.x;
                point.y = state.y;
                point.z = 0.0;
                object_marker.points.push_back(point);
            }
            msg.markers.push_back(object_marker);

            visualization_msgs::Marker position_marker;
            visualization_msgs::Marker track_info_marker;

            position_marker.header = ros_bridge::UpdateHeader(planning_space.header);
            position_marker.ns = "position";
            position_marker.id = object.id;
            position_marker.action = visualization_msgs::Marker::ADD;
            position_marker.type = visualization_msgs::Marker::CUBE;
            position_marker.lifetime = ros::Duration(0.1);

            track_info_marker.header = ros_bridge::UpdateHeader(planning_space.header);
            track_info_marker.ns = "track_info";
            track_info_marker.id = object.id;
            track_info_marker.action = visualization_msgs::Marker::ADD;
            track_info_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            track_info_marker.lifetime = ros::Duration(0.1);

            auto object_state = object.state.back();
            double track_velocity = sqrt(object_state.vx*object_state.vx + object_state.vy*object_state.vy);
            double longitudinal_accel = object_state.ax * cos(object_state.heading) + object_state.ay * sin(object_state.heading);

            std::ostringstream display_text;
            display_text << "ID (" << object.id << ")";
            display_text << "\nCLASS (" << object.classification << ")";
            display_text << "\nVel:   " << std::fixed << std::setprecision(3) << std::round(track_velocity * 1000.0) / 1000.0 << "m/s";

            track_info_marker.text = display_text.str();

            // Line width
            position_marker.scale.x = object.length;
            position_marker.scale.y = object.width;
            position_marker.scale.z = object.height;
            
            track_info_marker.scale.z = 1.0;

            // Color space
            position_marker.color.r = 0.0f;
            position_marker.color.g = 1.0f;
            position_marker.color.b = 1.0f;
            position_marker.color.a = 0.7f;

            track_info_marker.color.r = 1.0f;
            track_info_marker.color.g = 1.0f;
            track_info_marker.color.b = 1.0f;
            track_info_marker.color.a = 1.0f;

            position_marker.pose.position.x = object_state.x;
            position_marker.pose.position.y = object_state.y;
            position_marker.pose.position.z = object.height/2;
            position_marker.pose.orientation = tf::createQuaternionMsgFromYaw(object_state.heading);

            track_info_marker.pose.position.x = object_state.x;
            track_info_marker.pose.position.y = object_state.y;
            track_info_marker.pose.position.z = 3.0;
            track_info_marker.pose.orientation = tf::createQuaternionMsgFromYaw(object_state.heading);

            msg.markers.push_back(position_marker);
            msg.markers.push_back(track_info_marker);
        }
        
        for (const auto& object : planning_space.predict_objects.object) {
            visualization_msgs::Marker object_marker;
            object_marker.header = ros_bridge::UpdateHeader(planning_space.header);
            object_marker.ns = "predicted_object";
            object_marker.id = object.id;
            object_marker.type = visualization_msgs::Marker::LINE_STRIP;
            object_marker.action = visualization_msgs::Marker::ADD;
            object_marker.scale.x         = object.dimension.width;
            object_marker.color.r         = 1.0;
            object_marker.color.g         = 0.5;
            object_marker.color.b         = 0.0;
            object_marker.color.a         = 0.5;
            object_marker.lifetime        = ros::Duration(0.1);
            for (const auto& state : object.state) {
                geometry_msgs::Point point;
                point.x = state.x;
                point.y = state.y;
                point.z = 0.0;
                object_marker.points.push_back(point);
            }
            msg.markers.push_back(object_marker);
        }

        // Vector Map
        for (const auto& lane : planning_space.lanes) {
            visualization_msgs::Marker marker;
            marker.header = ros_bridge::UpdateHeader(planning_space.header);
            marker.ns = "lane";
            marker.id = lane.id;
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.action = visualization_msgs::Marker::ADD;
            marker.scale.x         = 0.5;
            marker.scale.y         = 0.5;
            marker.scale.z         = 0.5;
            marker.color.r         = 0.0;
            marker.color.g         = 0.5;
            marker.color.b         = 1.0;
            marker.color.a         = 1.0;
            marker.lifetime        = ros::Duration(0.1);
            for (const auto& point : lane.point) {
                geometry_msgs::Point point_msg;
                point_msg.x = point.x;
                point_msg.y = point.y;
                point_msg.z = 0.0;
                marker.points.push_back(point_msg);
            }
            msg.markers.push_back(marker);            
            marker.points.clear();

            marker.ns = "left_boundary";
            marker.id = lane.id;
            marker.color.r         = 1.0;
            marker.color.g         = 0.0;
            marker.color.b         = 0.0;
            marker.color.a         = 0.5;
            for (const auto& lbp : lane.left_boundary) {
                geometry_msgs::Point point_msg;
                point_msg.x = lbp.x;
                point_msg.y = lbp.y;
                point_msg.z = 0.0;
                marker.points.push_back(point_msg);
            }
            msg.markers.push_back(marker);            
            marker.points.clear();

            marker.ns = "right_boundary";
            marker.id = lane.id;
            marker.color.r         = 0.0;
            marker.color.g         = 1.0;
            marker.color.b         = 0.0;
            marker.color.a         = 0.5;
            for (const auto& rbp : lane.right_boundary) {
                geometry_msgs::Point point_msg;
                point_msg.x = rbp.x;
                point_msg.y = rbp.y;
                point_msg.z = 0.0;
                marker.points.push_back(point_msg);
            }
            msg.markers.push_back(marker);
            marker.points.clear();
        }

        for (const auto& route : planning_space.routes) {
            visualization_msgs::Marker marker;
            marker.header = ros_bridge::UpdateHeader(planning_space.header);
            marker.ns = "route";
            marker.id = route.id;
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.action = visualization_msgs::Marker::ADD;
            marker.scale.x         = 0.7;
            marker.scale.y         = 0.7;
            marker.scale.z         = 0.7;
            marker.color.r         = 1.0;
            marker.color.g         = 0.0;
            marker.color.b         = 1.0;
            marker.color.a         = 0.5;
            marker.lifetime        = ros::Duration(0.1);
            for (const auto& point : route.point) {
                geometry_msgs::Point point_msg;
                point_msg.x = point.x;
                point_msg.y = point.y;
                point_msg.z = 0.0;
                marker.points.push_back(point_msg);
            }
            msg.markers.push_back(marker);
        }

        for (const auto& crosswalk : planning_space.crosswalks) {
            visualization_msgs::Marker marker;
            marker.header = ros_bridge::UpdateHeader(planning_space.header);
            marker.ns = "crosswalk";
            marker.id = crosswalk.id;
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.action = visualization_msgs::Marker::ADD;
            marker.scale.x         = 0.5;
            marker.color.r         = 0.0;
            marker.color.g         = 0.0;
            marker.color.b         = 1.0;
            marker.color.a         = 0.8;
            marker.lifetime        = ros::Duration(0.1);
            for (const auto& point : crosswalk.point) {
                geometry_msgs::Point point_msg;
                point_msg.x = point.x;
                point_msg.y = point.y;
                point_msg.z = 0.0;
                marker.points.push_back(point_msg);
            }
            marker.points.push_back(marker.points.front()); // Close the polygon
            msg.markers.push_back(marker);
        }

        for (const auto& goal_routes : planning_space.goal_routes) {
            visualization_msgs::Marker marker;
            marker.header = ros_bridge::UpdateHeader(planning_space.header);
            marker.ns = "goal_routes";
            marker.id = goal_routes.id;
            marker.type = visualization_msgs::Marker::SPHERE_LIST;
            marker.action = visualization_msgs::Marker::ADD;
            marker.scale.x         = 0.7;
            marker.scale.y         = 0.7;
            marker.scale.z         = 0.7;
            marker.color.r         = 1.0;
            marker.color.g         = 1.0;
            marker.color.b         = 0.0;
            marker.color.a         = 1.0;
            marker.lifetime        = ros::Duration(0.1);
            for (const auto& point : goal_routes.point) {
                geometry_msgs::Point point_msg;
                point_msg.x = point.x;
                point_msg.y = point.y;
                point_msg.z = 0.0;
                marker.points.push_back(point_msg);
            }
            msg.markers.push_back(marker);
        }
        for (const auto& goal_routes : planning_space.goal_routes) {
            // 속도 벡터를 z축으로 시각화 (LINE_LIST)
            visualization_msgs::Marker speed_vectors_marker;
            speed_vectors_marker.header = ros_bridge::UpdateHeader(planning_space.header);
            speed_vectors_marker.ns = "goal_routes_speed_limit";
            speed_vectors_marker.id = goal_routes.id;
            speed_vectors_marker.type = visualization_msgs::Marker::LINE_LIST;
            speed_vectors_marker.action = visualization_msgs::Marker::ADD;
            speed_vectors_marker.lifetime = ros::Duration(0.1);
            
            // 속도 벡터 선 두께
            speed_vectors_marker.scale.x = 0.2f;  // 적당한 굵기
            
            // 속도 벡터 색깔 (빨간색)
            speed_vectors_marker.color.r = 0.0f;
            speed_vectors_marker.color.g = 0.0f;
            speed_vectors_marker.color.b = 1.0f;
            speed_vectors_marker.color.a = 0.9f;
            
            // 각 점에서 z축으로 속도 벡터 추가
            for (const auto& point : goal_routes.point) {
                // 시작점 (지면)
                geometry_msgs::Point start_point;
                start_point.x = point.x;
                start_point.y = point.y;
                start_point.z = 0.0;
                
                // 끝점 (속도제한에 비례한 높이)
                geometry_msgs::Point end_point;
                end_point.x = point.x;
                end_point.y = point.y;
                
                end_point.z = point.speed_limit / 4.0; //  * 2.0;
                
                speed_vectors_marker.points.push_back(start_point);
                speed_vectors_marker.points.push_back(end_point);
            }
            msg.markers.push_back(speed_vectors_marker);
        }

        for (const auto& dividing_lane : planning_space.lanes) {
            visualization_msgs::Marker marker;
            marker.header = ros_bridge::UpdateHeader(planning_space.header);
            marker.ns = "dividing_lane";
            marker.id = dividing_lane.id;
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.action = visualization_msgs::Marker::ADD;
            marker.scale.x         = 0.5;

            if (dividing_lane.is_dividing_lane) {
                marker.color.r         = 1.0;
                marker.color.g         = 0.0;
                marker.color.b         = 0.0;
            } else {
                marker.color.r         = 0.0;
                marker.color.g         = 1.0;
                marker.color.b         = 0.0;
            }

            marker.color.a         = 0.5;   
            marker.lifetime        = ros::Duration(0.1);
            for (const auto& point : dividing_lane.point) {
                geometry_msgs::Point point_msg;
                point_msg.x = point.x;
                point_msg.y = point.y;
                point_msg.z = 0.0;
                marker.points.push_back(point_msg);
            }
            msg.markers.push_back(marker);
        }

        // Stop Line Visualization
        if (!planning_space.traffic_light.stop_line_points.empty()) {
            visualization_msgs::Marker stop_line_marker;
            stop_line_marker.header = ros_bridge::UpdateHeader(planning_space.traffic_light.header);
            stop_line_marker.header.frame_id = "world";
            stop_line_marker.ns = "traffic_light_stop_line";
            stop_line_marker.id = 0;
            stop_line_marker.type = visualization_msgs::Marker::SPHERE_LIST;
            stop_line_marker.action = visualization_msgs::Marker::ADD;
            stop_line_marker.scale.x = 0.5;
            stop_line_marker.scale.y = 0.5;
            stop_line_marker.scale.z = 0.5;
            stop_line_marker.color.r = 1.0;
            stop_line_marker.color.g = 1.0;
            stop_line_marker.color.b = 0.0;  // Yellow
            stop_line_marker.color.a = 1.0;
            stop_line_marker.lifetime = ros::Duration(0.1);
            
            for (const auto& point : planning_space.traffic_light.stop_line_points) {
                geometry_msgs::Point point_msg;
                point_msg.x = point.x;
                point_msg.y = point.y;
                point_msg.z = point.z;
                stop_line_marker.points.push_back(point_msg);
            }
            msg.markers.push_back(stop_line_marker);
        }

        // Imitation Trajectory Visualization
        visualization_msgs::Marker trajectory_marker;
        trajectory_marker.header = UpdateHeader(planning_space.imitation_trajectory.header);
        trajectory_marker.ns = "trajectory";
        trajectory_marker.id = 0;
        trajectory_marker.action = visualization_msgs::Marker::ADD;
        trajectory_marker.type = visualization_msgs::Marker::LINE_STRIP;
        trajectory_marker.lifetime = ros::Duration(0.1);

        // Line width
        trajectory_marker.scale.x = interface::VEHICLE_WIDTH;

        // Color space
        trajectory_marker.color.r = 0.7f;
        trajectory_marker.color.g = 0.0f;
        trajectory_marker.color.b = 0.7f;
        trajectory_marker.color.a = 0.7f;  

        for (uint16_t i = 0; i < (uint16_t)planning_space.imitation_trajectory.point.size(); i++) {
            geometry_msgs::Point point;
            point.x = planning_space.imitation_trajectory.point[i].x;
            point.y = planning_space.imitation_trajectory.point[i].y;
            point.z = planning_space.imitation_trajectory.point[i].z;

            if (planning_space.imitation_trajectory.point[i].speed > 0.0) {
                trajectory_marker.points.push_back(point);
            }
        }
        msg.markers.push_back(trajectory_marker);

        // Drivable Area Visualization
        visualization_msgs::Marker left_boundary_marker;
        left_boundary_marker.header = ros_bridge::UpdateHeader(planning_space.drivable_area.header);
        left_boundary_marker.header.frame_id = "world";
        left_boundary_marker.ns = "drivable_area_left_boundary";
        left_boundary_marker.id = 0;
        left_boundary_marker.type = visualization_msgs::Marker::LINE_STRIP;
        left_boundary_marker.action = visualization_msgs::Marker::ADD;
        left_boundary_marker.scale.x         = 0.2;
        left_boundary_marker.color.r         = 1.0;
        left_boundary_marker.color.g         = 0.0;
        left_boundary_marker.color.b         = 1.0;
        left_boundary_marker.color.a         = 0.8;
        left_boundary_marker.lifetime        = ros::Duration(0.1);
        for (const auto& point : planning_space.drivable_area.left_boundary.point) {
            geometry_msgs::Point point_msg;
            point_msg.x = point.x;
            point_msg.y = point.y;
            point_msg.z = 0.0;
            left_boundary_marker.points.push_back(point_msg);
        }
        msg.markers.push_back(left_boundary_marker);

        visualization_msgs::Marker right_boundary_marker;
        right_boundary_marker.header = ros_bridge::UpdateHeader(planning_space.drivable_area.header);
        right_boundary_marker.header.frame_id = "world";
        right_boundary_marker.ns = "drivable_area_right_boundary";
        right_boundary_marker.id = 0;
        right_boundary_marker.type = visualization_msgs::Marker::LINE_STRIP;
        right_boundary_marker.action = visualization_msgs::Marker::ADD;
        right_boundary_marker.scale.x         = 0.2;
        right_boundary_marker.color.r         = 1.0;
        right_boundary_marker.color.g         = 1.0;
        right_boundary_marker.color.b         = 0.0;
        right_boundary_marker.color.a         = 0.8;
        right_boundary_marker.lifetime        = ros::Duration(0.1);
        for (const auto& point : planning_space.drivable_area.right_boundary.point) {
            geometry_msgs::Point point_msg;
            point_msg.x = point.x;
            point_msg.y = point.y;
            point_msg.z = 0.0;
            right_boundary_marker.points.push_back(point_msg);
        }
        msg.markers.push_back(right_boundary_marker);

        return msg;
    }    
} // namespace ros_bridge

#endif  // __ROS_BRIDGE_PLANNING_SPACE__