/**
 * @file        ros_bridge_trajectories.hpp
 * @brief       ROS bridge for trajectories
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-01 updated by Yuseung Na: Refactoring 
 */

#ifndef __ROS_BRIDGE_TRAJECTORIES__
#define __ROS_BRIDGE_TRAJECTORIES__
#pragma once

// ROS header
#include <ros/ros.h>
#include <tf/tf.h>

// ROS Message Header
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_rviz_plugins/OverlayText.h>
#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/PolygonStamped.h>
#include <autohyu_msgs/Trajectories.h>
#include <autohyu_msgs/PlanningSpace.h>
#include <autohyu_msgs/PlanningInputOutput.h>
#include <autohyu_msgs/BehaviorTrajectories.h>
#include <autohyu_msgs/DAgger.h>
#include <autohyu_msgs/EventPoints.h>

// Interface Header
#include "interface_vehicle_state.hpp"
#include "interface_constants.hpp"
#include "interface_objects.hpp"
#include "interface_trajectories.hpp"
#include "function_print.hpp"

// Bridge Header
#include "ros_bridge_std_msgs.hpp"
#include "ros_bridge_route.hpp"
#include "ros_bridge_planning_space.hpp"
#include "ros_bridge_geometry_msgs.hpp"

namespace ros_bridge {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //    
    interface::BehaviorTrajectory GetBehaviorTrajectory(const autohyu_msgs::BehaviorTrajectory& msg) {
        interface::BehaviorTrajectory behavior_trajectory;

        behavior_trajectory.header = GetHeader(msg.header);
        behavior_trajectory.id = msg.id;
        behavior_trajectory.on_ego = msg.on_ego;

        for (const auto& point : msg.left_boundary.point) {
            interface::TrajectoryBoundaryPoint bpoint;
            bpoint.x = point.x;
            bpoint.y = point.y;
            bpoint.z = point.z;

            behavior_trajectory.left_boundary.point.push_back(bpoint);
        }
        for (const auto& point : msg.right_boundary.point) {
            interface::TrajectoryBoundaryPoint bpoint;
            bpoint.x = point.x;
            bpoint.y = point.y;
            bpoint.z = point.z;

            behavior_trajectory.right_boundary.point.push_back(bpoint);
        }
        for (const auto& traffic_sign : msg.traffic_sign) {
            behavior_trajectory.traffic_sign.push_back(ros_bridge::GetLaneletTrafficSign(traffic_sign));
        }
        for (const auto& turn_signal : msg.turn_signal) {
            behavior_trajectory.turn_signal.push_back(ros_bridge::GetLaneletTrafficSign(turn_signal));
        }
        for (const auto& traffic_light : msg.traffic_light) {
            behavior_trajectory.traffic_light.push_back(ros_bridge::GetLaneletTrafficLight(traffic_light));
        }
        for (const auto& point : msg.point) {
            interface::BehaviorTrajectoryPoint btp;
            btp.time = point.time;
            btp.distance = point.distance;
            btp.x = point.x;
            btp.y = point.y;
            btp.z = point.z;
            btp.curvature = point.curvature;
            btp.speed = point.speed;
            btp.max_speed = point.max_speed;
            btp.on_shortest_path = point.on_shortest_path;
            btp.lane_num = point.lane_num;

            behavior_trajectory.point.push_back(btp);
        }

        return behavior_trajectory;
    }

    interface::BehaviorTrajectories GetBehaviorTrajectories(const autohyu_msgs::BehaviorTrajectories& msg) {
        interface::BehaviorTrajectories behavior_trajectories;
        behavior_trajectories.header         = GetHeader(msg.header);
        behavior_trajectories.lk_trajectory  = GetBehaviorTrajectory(msg.lk_trajectory);
        // TODO
        behavior_trajectories.llc_trajectories.push_back(GetBehaviorTrajectory(msg.llc_trajectory));
        behavior_trajectories.rlc_trajectories.push_back(GetBehaviorTrajectory(msg.rlc_trajectory));

        return behavior_trajectories;
    }

    interface::Trajectory GetTrajectory(const autohyu_msgs::Trajectory& msg) {
        interface::Trajectory trajectory;
        trajectory.header = GetHeader(msg.header);
        trajectory.id = msg.id;
        trajectory.behavior_zone      = static_cast<interface::BehaviorType>(msg.behavior_zone);
        trajectory.next_behavior_zone = static_cast<interface::BehaviorType>(msg.next_behavior_zone);
        trajectory.decision_behavior  = static_cast<interface::BehaviorType>(msg.decision_behavior);
        trajectory.behavior_distance  = msg.behavior_distance;
        trajectory.is_target          = msg.is_target;

        auto copy_boundary = [](const autohyu_msgs::TrajectoryBoundary& src,
                                interface::TrajectoryBoundary& dst) {
            for (const auto& p : src.point) {
                interface::TrajectoryBoundaryPoint bp;
                bp.x = p.x; bp.y = p.y; bp.z = p.z;
                bp.s = p.s; bp.n = p.n;
                dst.point.push_back(bp);
            }
        };
        copy_boundary(msg.left_boundary,   trajectory.left_boundary);
        copy_boundary(msg.right_boundary,  trajectory.right_boundary);
        copy_boundary(msg.center_boundary, trajectory.center_boundary);

        for (const auto& point : msg.point) {
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
            tp.s = point.s;
            tp.n = point.n;
            tp.ds = point.ds;
            tp.dn = point.dn;
            tp.dds = point.dds;
            tp.ddn = point.ddn;
            tp.yaw_sn = point.yaw_sn;

            trajectory.point.push_back(tp);
        }

        return trajectory;
    }

    std::vector<interface::Point> GetEventPoints(const visualization_msgs::MarkerArray& msg) {
        std::vector<interface::Point> event_points;
        for (const auto& marker : msg.markers) {
            if (marker.points.size() > 0) {
                if (marker.ns == "red_points") {
                    for (const auto& point : marker.points) {
                        interface::Point p;
                        p.x = point.x;
                        p.y = point.y;
                        p.z = point.z;

                        event_points.push_back(p);
                    }
                }
            }
        }

        return event_points;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    autohyu_msgs::BehaviorTrajectory UpdateBehaviorTrajectory(const interface::BehaviorTrajectory& behavior_trajectory) {
        autohyu_msgs::BehaviorTrajectory msg;

        msg.header      = UpdateHeader(behavior_trajectory.header);
        msg.id          = behavior_trajectory.id;
        msg.on_ego      = behavior_trajectory.on_ego;

        for (const auto& point : behavior_trajectory.left_boundary.point) {
            autohyu_msgs::TrajectoryBoundaryPoint bpoint;
            bpoint.x = point.x;
            bpoint.y = point.y;
            bpoint.z = point.z;

            msg.left_boundary.point.push_back(bpoint);
        }
        for (const auto& point : behavior_trajectory.right_boundary.point) {
            autohyu_msgs::TrajectoryBoundaryPoint bpoint;
            bpoint.x = point.x;
            bpoint.y = point.y;
            bpoint.z = point.z;

            msg.right_boundary.point.push_back(bpoint);
        }
        for (const auto& traffic_sign : behavior_trajectory.traffic_sign) {
            msg.traffic_sign.push_back(ros_bridge::UpdateLaneletTrafficSign(traffic_sign));
        }
        for (const auto& turn_signal : behavior_trajectory.turn_signal) {
            msg.turn_signal.push_back(ros_bridge::UpdateLaneletTrafficSign(turn_signal));
        }
        for (const auto& traffic_light : behavior_trajectory.traffic_light) {
            msg.traffic_light.push_back(ros_bridge::UpdateLaneletTrafficLight(traffic_light));
        }        
        for (const auto& point : behavior_trajectory.point) {
            autohyu_msgs::BehaviorTrajectoryPoint bpoint;
            bpoint.time = point.time;
            bpoint.distance = point.distance;
            bpoint.x = point.x;
            bpoint.y = point.y;
            bpoint.z = point.z;
            bpoint.curvature = point.curvature;
            bpoint.speed = point.speed;
            bpoint.max_speed = point.max_speed;
            bpoint.on_shortest_path = point.on_shortest_path;
            bpoint.lane_num = point.lane_num;

            msg.point.push_back(bpoint);
        }

        return msg;
    }

    autohyu_msgs::BehaviorTrajectories UpdateRouteCandidates(const interface::BehaviorTrajectories& behavior_trajectories) {
        autohyu_msgs::BehaviorTrajectories msg;
        msg.header = UpdateHeader(behavior_trajectories.header);
        msg.header.frame_id = "world";
        
        if (behavior_trajectories.lk_trajectory.point.size() > 0) {
            msg.lk_trajectory = UpdateBehaviorTrajectory(behavior_trajectories.lk_trajectory);
        }
        // TODO
        if (behavior_trajectories.llc_trajectories.size() > 0) {
            auto llc_trajectory = behavior_trajectories.llc_trajectories.at(0);
            if (llc_trajectory.point.size() > 0) {
                msg.llc_trajectory = UpdateBehaviorTrajectory(llc_trajectory);
            }
        }
        if (behavior_trajectories.rlc_trajectories.size() > 0) {
            auto rlc_trajectory = behavior_trajectories.rlc_trajectories.at(0);
            if (rlc_trajectory.point.size() > 0) {
                msg.rlc_trajectory = UpdateBehaviorTrajectory(rlc_trajectory);
            }
        }

        return msg;
    }

    autohyu_msgs::BehaviorTrajectories UpdateBehaviorTrajectories(const interface::BehaviorTrajectories& behavior_trajectories) {
        autohyu_msgs::BehaviorTrajectories msg;
        msg.header = UpdateHeader(behavior_trajectories.header);
        msg.header.frame_id = "world";
        
        if (behavior_trajectories.lk_trajectory.point.size() > 0 && behavior_trajectories.lk_trajectory.id == 1) {
            msg.lk_trajectory = UpdateBehaviorTrajectory(behavior_trajectories.lk_trajectory);
        }
        // TODO
        if (behavior_trajectories.llc_trajectories.size() > 0) {
            auto llc_trajectory = behavior_trajectories.llc_trajectories.at(0);
            if (llc_trajectory.point.size() > 0) {
                msg.llc_trajectory = UpdateBehaviorTrajectory(llc_trajectory);
            }
        }
        if (behavior_trajectories.rlc_trajectories.size() > 0) {
            auto rlc_trajectory = behavior_trajectories.rlc_trajectories.at(0);
            if (rlc_trajectory.point.size() > 0) {
                msg.rlc_trajectory = UpdateBehaviorTrajectory(rlc_trajectory);
            }
        }

        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizBehaviorTrajectories(const interface::BehaviorTrajectories& behavior_trajectories,
                                                                   const int& lk_shortest_num,
                                                                   const std::vector<int>& llc_shortest_num,
                                                                   const std::vector<int>& rlc_shortest_num) {
        visualization_msgs::MarkerArray msg;
        
        visualization_msgs::Marker speed_marker;
        speed_marker.header = UpdateHeader(behavior_trajectories.header);
        speed_marker.header.frame_id = "world";
        speed_marker.id = 0;
        speed_marker.type = visualization_msgs::Marker::LINE_LIST;
        speed_marker.action = visualization_msgs::Marker::ADD;
        speed_marker.lifetime = ros::Duration(0.1);
        speed_marker.scale.x = 0.1;
        speed_marker.color.r = 1.0;
        speed_marker.color.g = 1.0;
        speed_marker.color.b = 1.0;
        speed_marker.color.a = 0.4;
        speed_marker.pose.orientation.x = 0.0;
        speed_marker.pose.orientation.y = 0.0;
        speed_marker.pose.orientation.z = 0.0;
        speed_marker.pose.orientation.w = 0.0;

        visualization_msgs::Marker curvature_marker = speed_marker;
        curvature_marker.color.r = 0.7;
        curvature_marker.color.g = 0.0;
        curvature_marker.color.b = 0.7;
        curvature_marker.color.a = 0.6;

        visualization_msgs::Marker path_marker;
        path_marker.header = UpdateHeader(behavior_trajectories.header);
        path_marker.header.frame_id = "world";
        path_marker.id = 0;
        path_marker.type = visualization_msgs::Marker::LINE_STRIP;
        path_marker.action = visualization_msgs::Marker::ADD;
        path_marker.lifetime = ros::Duration(0.1);
        path_marker.color.a = 0.7;
        path_marker.scale.x = 0.3;
        path_marker.pose.orientation.x = 0.0;
        path_marker.pose.orientation.y = 0.0;
        path_marker.pose.orientation.z = 0.0;
        path_marker.pose.orientation.w = 0.0;

        visualization_msgs::Marker lb_marker;
        lb_marker.header = UpdateHeader(behavior_trajectories.header);
        lb_marker.header.frame_id = "world";
        lb_marker.id = 0;
        lb_marker.type = visualization_msgs::Marker::LINE_STRIP;
        lb_marker.action = visualization_msgs::Marker::ADD;
        lb_marker.lifetime = ros::Duration(0.1);
        lb_marker.color.a = 0.7;
        lb_marker.scale.x = 0.5;
        lb_marker.pose.orientation.x = 0.0;
        lb_marker.pose.orientation.y = 0.0;
        lb_marker.pose.orientation.z = 0.0;
        lb_marker.pose.orientation.w = 0.0;

        visualization_msgs::Marker rb_marker = lb_marker;

        visualization_msgs::Marker shortest_marker;
        shortest_marker.header = UpdateHeader(behavior_trajectories.header);
        shortest_marker.header.frame_id = "world";
        shortest_marker.id = 0;
        shortest_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        shortest_marker.action = visualization_msgs::Marker::ADD;
        shortest_marker.lifetime = ros::Duration(0.1);
        shortest_marker.color.r = 1.0;
        shortest_marker.color.g = 1.0;
        shortest_marker.color.b = 1.0;
        shortest_marker.color.a = 0.7;
        shortest_marker.scale.x = 1.0;
        shortest_marker.scale.y = 1.0;
        shortest_marker.scale.z = 1.0;
        shortest_marker.pose.orientation.x = 0.0;
        shortest_marker.pose.orientation.y = 0.0;
        shortest_marker.pose.orientation.z = 0.0;
        shortest_marker.pose.orientation.w = 1.0;
        
        // Viz Start
        path_marker.ns = "lk_path";
        speed_marker.ns = "lk_speed";
        curvature_marker.ns = "lk_curvature";
        lb_marker.ns = "lk_left_bound";
        rb_marker.ns = "lk_right_bound";
        shortest_marker.ns = "lk_shortest_num";
        path_marker.color.r = 0.0f;
        path_marker.color.g = 0.38f;
        path_marker.color.b = 0.68f;
        lb_marker.color = path_marker.color;
        rb_marker.color = path_marker.color;
        for (const auto& point : behavior_trajectories.lk_trajectory.point) {
            geometry_msgs::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            path_marker.points.push_back(p);
            speed_marker.points.push_back(p);
            curvature_marker.points.push_back(p);
            p.z = point.speed*interface::KPH2MPS;
            speed_marker.points.push_back(p);
            p.z = point.curvature * 1000.0;
            curvature_marker.points.push_back(p);
        }
        for (const auto& point : behavior_trajectories.lk_trajectory.left_boundary.point) {
            geometry_msgs::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            lb_marker.points.push_back(p);
        }
        for (const auto& point : behavior_trajectories.lk_trajectory.right_boundary.point) {
            geometry_msgs::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            rb_marker.points.push_back(p);
        }
        shortest_marker.pose.position.x = behavior_trajectories.lk_trajectory.point[0].x;
        shortest_marker.pose.position.y = behavior_trajectories.lk_trajectory.point[0].y;
        shortest_marker.pose.position.z = behavior_trajectories.lk_trajectory.point[0].z + 2.0;
        shortest_marker.text = std::to_string(lk_shortest_num);

        if (path_marker.points.size() > 0) {
            msg.markers.push_back(path_marker);
            msg.markers.push_back(speed_marker);
            msg.markers.push_back(curvature_marker);
            msg.markers.push_back(lb_marker);
            msg.markers.push_back(rb_marker);
            msg.markers.push_back(shortest_marker);
        }
        path_marker.points.clear();
        speed_marker.points.clear();
        curvature_marker.points.clear();
        lb_marker.points.clear();
        rb_marker.points.clear();        

        path_marker.ns = "llc_path";
        speed_marker.ns = "llc_speed";
        curvature_marker.ns = "llc_curvature";
        lb_marker.ns = "llc_left_bound";
        rb_marker.ns = "llc_right_bound";
        shortest_marker.ns = "llc_shortest_num";

        int idx = 0;
        for (const auto& llc_trajectory : behavior_trajectories.llc_trajectories) {
            if (llc_trajectory.point.size() > 0) {
                path_marker.id = idx++;
                speed_marker.id = path_marker.id;
                curvature_marker.id = path_marker.id;
                lb_marker.id = path_marker.id;
                rb_marker.id = path_marker.id;
                shortest_marker.id = path_marker.id;
                
                if (llc_trajectory.id == 0) {
                    path_marker.color.r = 1.0;
                    path_marker.color.g = 0.7;
                    path_marker.color.b = 0.0;
                    path_marker.color.a = 0.2;
                    path_marker.scale.x = 0.3;
                }
                else if (llc_trajectory.id == 1) {
                    path_marker.color.r = 0.0;
                    path_marker.color.g = 1.0;
                    path_marker.color.b = 0.0;
                    path_marker.color.a = 0.5;
                    path_marker.scale.x = interface::VEHICLE_WIDTH;
                }
                else if (llc_trajectory.id == 2) {
                    path_marker.color.r = 1.0;
                    path_marker.color.g = 0.0;
                    path_marker.color.b = 0.0;
                    path_marker.color.a = 0.3;
                    path_marker.scale.x = 0.3;
                }

                lb_marker.color = path_marker.color;
                rb_marker.color = path_marker.color;

                for (const auto& point : llc_trajectory.point) {
                    geometry_msgs::Point p;
                    p.x = point.x;
                    p.y = point.y;
                    p.z = point.z;
                    path_marker.points.push_back(p);
                    speed_marker.points.push_back(p);
                    curvature_marker.points.push_back(p);
                    p.z = point.speed*interface::KPH2MPS;
                    speed_marker.points.push_back(p);
                    p.z = point.curvature * 1000.0;
                    curvature_marker.points.push_back(p);
                }
                for (const auto& point : llc_trajectory.left_boundary.point) {
                    geometry_msgs::Point p;
                    p.x = point.x;
                    p.y = point.y;
                    p.z = point.z;
                    lb_marker.points.push_back(p);
                }
                for (const auto& point : llc_trajectory.right_boundary.point) {
                    geometry_msgs::Point p;
                    p.x = point.x;
                    p.y = point.y;
                    p.z = point.z;
                    rb_marker.points.push_back(p);
                }
                shortest_marker.pose.position.x = llc_trajectory.point[0].x;
                shortest_marker.pose.position.y = llc_trajectory.point[0].y;
                shortest_marker.pose.position.z = llc_trajectory.point[0].z + 2.0;
                shortest_marker.text = std::to_string(llc_shortest_num.at(path_marker.id));

                msg.markers.push_back(path_marker);
                msg.markers.push_back(speed_marker);
                msg.markers.push_back(curvature_marker);
                msg.markers.push_back(lb_marker);
                msg.markers.push_back(rb_marker);
                msg.markers.push_back(shortest_marker);
            }
            path_marker.points.clear();
            speed_marker.points.clear();
            curvature_marker.points.clear();
            lb_marker.points.clear();
            rb_marker.points.clear();
        }    

        path_marker.ns = "rlc_path";
        speed_marker.ns = "rlc_speed";
        curvature_marker.ns = "rlc_curvature";
        lb_marker.ns = "rlc_left_bound";
        rb_marker.ns = "rlc_right_bound";
        shortest_marker.ns = "rlc_shortest_num";   

        idx = 0;
        for (const auto& rlc_trajectory : behavior_trajectories.rlc_trajectories) {
            if (rlc_trajectory.point.size() > 0) {
                path_marker.id = idx++;
                speed_marker.id = path_marker.id;
                curvature_marker.id = path_marker.id;
                lb_marker.id = path_marker.id;
                rb_marker.id = path_marker.id;
                shortest_marker.id = path_marker.id;

                if (rlc_trajectory.id == 0) {
                    path_marker.color.r = 1.0;
                    path_marker.color.g = 0.7;
                    path_marker.color.b = 0.0;
                    path_marker.color.a = 0.2;
                    path_marker.scale.x = 0.3;
                }
                else if (rlc_trajectory.id == 1) {
                    path_marker.color.r = 0.0;
                    path_marker.color.g = 1.0;
                    path_marker.color.b = 0.0;
                    path_marker.color.a = 0.5;
                    path_marker.scale.x = interface::VEHICLE_WIDTH;
                }
                else if (rlc_trajectory.id == 2) {
                    path_marker.color.r = 1.0;
                    path_marker.color.g = 0.0;
                    path_marker.color.b = 0.0;
                    path_marker.color.a = 0.3;
                    path_marker.scale.x = 0.3;
                }

                lb_marker.color = path_marker.color;
                rb_marker.color = path_marker.color;

                for (const auto& point : rlc_trajectory.point) {
                    geometry_msgs::Point p;
                    p.x = point.x;
                    p.y = point.y;
                    p.z = point.z;
                    path_marker.points.push_back(p);
                    speed_marker.points.push_back(p);
                    curvature_marker.points.push_back(p);
                    p.z = point.speed*interface::KPH2MPS;
                    speed_marker.points.push_back(p);
                    p.z = point.curvature * 1000.0;
                    curvature_marker.points.push_back(p);
                }
                for (const auto& point : rlc_trajectory.left_boundary.point) {
                    geometry_msgs::Point p;
                    p.x = point.x;
                    p.y = point.y;
                    p.z = point.z;
                    lb_marker.points.push_back(p);
                }
                for (const auto& point : rlc_trajectory.right_boundary.point) {
                    geometry_msgs::Point p;
                    p.x = point.x;
                    p.y = point.y;
                    p.z = point.z;
                    rb_marker.points.push_back(p);
                }
                shortest_marker.pose.position.x = rlc_trajectory.point[0].x;
                shortest_marker.pose.position.y = rlc_trajectory.point[0].y;
                shortest_marker.pose.position.z = rlc_trajectory.point[0].z + 2.0;
                shortest_marker.text = std::to_string(rlc_shortest_num.at(path_marker.id));

                msg.markers.push_back(path_marker);
                msg.markers.push_back(speed_marker);
                msg.markers.push_back(curvature_marker);
                msg.markers.push_back(lb_marker);
                msg.markers.push_back(rb_marker);
                msg.markers.push_back(shortest_marker);
            }
            path_marker.points.clear();
            speed_marker.points.clear();
            curvature_marker.points.clear();
            lb_marker.points.clear();
            rb_marker.points.clear();
        }

        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizEventPoints(const std::vector<std::pair<std::string, interface::Point>>& event_points) {
        visualization_msgs::MarkerArray msg;
        visualization_msgs::Marker red_marker;
        visualization_msgs::Marker yellow_marker;
        visualization_msgs::Marker green_marker;
        visualization_msgs::Marker gray_marker;

        red_marker.header.frame_id = "world";
        red_marker.header.stamp = ros::Time::now();
        red_marker.ns = "red_points";
        red_marker.id = 0;
        red_marker.type = visualization_msgs::Marker::SPHERE_LIST;
        red_marker.action = visualization_msgs::Marker::ADD;
        red_marker.scale.x = 0.5;
        red_marker.scale.y = 0.5;
        red_marker.scale.z = 0.5;
        red_marker.color.r = 1.0;
        red_marker.color.g = 0.0;
        red_marker.color.b = 0.0;
        red_marker.color.a = 1.0;
        red_marker.lifetime = ros::Duration(0.1);

        yellow_marker = red_marker;
        yellow_marker.ns = "yellow_points";
        yellow_marker.color.r = 1.0;
        yellow_marker.color.g = 1.0;
        yellow_marker.color.b = 0.0;        

        green_marker = red_marker;
        green_marker.ns = "green_points";
        green_marker.color.r = 0.0;
        green_marker.color.g = 1.0;
        green_marker.color.b = 0.0;

        gray_marker = red_marker;
        gray_marker.ns = "detected_points";
        gray_marker.color.r = 0.5;
        gray_marker.color.g = 0.5;
        gray_marker.color.b = 0.5;   
        gray_marker.color.a = 0.5;

        for (const auto& point : event_points) {
            geometry_msgs::Point p;
            p.x = point.second.x;
            p.y = point.second.y;
            p.z = 1.0;

            if (point.first == "red") {
                red_marker.points.push_back(p);
            } 
            else if (point.first == "yellow") {
                yellow_marker.points.push_back(p);
            }
            else if (point.first == "green") {
                green_marker.points.push_back(p);
            }
            else {
                gray_marker.points.push_back(p);
            }
        }

        msg.markers.push_back(red_marker);
        msg.markers.push_back(yellow_marker);
        msg.markers.push_back(green_marker);
        msg.markers.push_back(gray_marker);

        return msg;
    }

    autohyu_msgs::Trajectories UpdateTrajectoriesWithCost(const interface::Trajectories& trajectories) {
        autohyu_msgs::Trajectories msg;
        msg.header.frame_id = "world";
        msg.header = UpdateHeader(trajectories.header);

        for (auto trajectory : trajectories.trajectory) {
            autohyu_msgs::Trajectory o_trajectory;
            for (auto point : trajectory.point) {
                autohyu_msgs::TrajectoryPoint tpoint;
                tpoint.time = point.time;
                tpoint.x = point.x;
                tpoint.y = point.y;
                tpoint.z = point.z;
                tpoint.yaw = point.yaw;
                tpoint.curvature = point.curvature;
                tpoint.distance = point.distance - trajectory.point.at(0).distance;
                tpoint.speed = point.speed;
                tpoint.acceleration = point.acceleration;

                o_trajectory.point.push_back(tpoint);
            }

            msg.trajectory.push_back(o_trajectory);
        }
        return msg;
    }

    autohyu_msgs::Trajectory UpdateTrajectory(const interface::Trajectory& trajectory) {
        autohyu_msgs::Trajectory msg;
        msg.header              = UpdateHeader(trajectory.header);
        msg.header.frame_id     = "world";
        msg.id                  = trajectory.id;
        msg.behavior_zone       = static_cast<uint8_t>(trajectory.behavior_zone);
        msg.next_behavior_zone  = static_cast<uint8_t>(trajectory.next_behavior_zone);
        msg.decision_behavior   = static_cast<uint8_t>(trajectory.decision_behavior);
        msg.behavior_distance   = trajectory.behavior_distance;
        msg.is_target           = trajectory.is_target ? 1 : 0;

        auto copy_boundary = [](const interface::TrajectoryBoundary& src,
                                autohyu_msgs::TrajectoryBoundary& dst) {
            for (const auto& p : src.point) {
                autohyu_msgs::TrajectoryBoundaryPoint bp;
                bp.x = p.x; bp.y = p.y; bp.z = p.z;
                bp.s = p.s; bp.n = p.n;
                dst.point.push_back(bp);
            }
        };
        copy_boundary(trajectory.left_boundary,   msg.left_boundary);
        copy_boundary(trajectory.right_boundary,  msg.right_boundary);
        copy_boundary(trajectory.center_boundary, msg.center_boundary);

        for (auto point : trajectory.point) {
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
            tpoint.s            = point.s;
            tpoint.n            = point.n;
            tpoint.ds           = point.ds;
            tpoint.dn           = point.dn;
            tpoint.dds          = point.dds;
            tpoint.ddn          = point.ddn;
            tpoint.yaw_sn       = point.yaw_sn;

            msg.point.push_back(tpoint);
        }

        return msg;
    }

    autohyu_msgs::Trajectories UpdateTrajectories(const interface::Trajectories& trajectories) {

        autohyu_msgs::Trajectories msg;
        msg.header.frame_id = "world";
        msg.header = UpdateHeader(trajectories.header);
        for (auto trajectory : trajectories.trajectory) {
            autohyu_msgs::Trajectory o_trajectory;
            for (auto point : trajectory.point) {
                autohyu_msgs::TrajectoryPoint tpoint;
                tpoint.time = point.time;
                tpoint.x = point.x;
                tpoint.y = point.y;
                tpoint.z = point.z;
                tpoint.yaw = point.yaw;
                tpoint.curvature = point.curvature;
                tpoint.distance = point.distance - trajectory.point.at(0).distance;
                tpoint.speed = point.speed;
                tpoint.acceleration = point.acceleration;

                o_trajectory.point.push_back(tpoint);
            }
            msg.trajectory.push_back(o_trajectory);
        }
        
        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizTrajectory(const interface::Trajectory& trajectory) {
        visualization_msgs::MarkerArray msg;

        visualization_msgs::Marker path_marker;
        visualization_msgs::Marker speed_marker;
        visualization_msgs::Marker curvature_marker;
        visualization_msgs::Marker trajectory_marker;

        path_marker.header = UpdateHeader(trajectory.header);
        path_marker.ns = "spatial";
        path_marker.id = 0;
        path_marker.action = visualization_msgs::Marker::ADD;
        path_marker.type = visualization_msgs::Marker::LINE_STRIP;
        path_marker.lifetime = ros::Duration(0.1);

        speed_marker.header = UpdateHeader(trajectory.header);
        speed_marker.ns = "temporal";
        speed_marker.id = 1;
        speed_marker.action = visualization_msgs::Marker::ADD;
        speed_marker.type = visualization_msgs::Marker::LINE_LIST;
        speed_marker.lifetime = ros::Duration(0.1);

        curvature_marker.header = UpdateHeader(trajectory.header);
        curvature_marker.ns = "curvature";
        curvature_marker.id = 2;
        curvature_marker.action = visualization_msgs::Marker::ADD;
        curvature_marker.type = visualization_msgs::Marker::LINE_LIST;
        curvature_marker.lifetime = ros::Duration(0.1);

        trajectory_marker.header = UpdateHeader(trajectory.header);
        trajectory_marker.ns = "trajectory";
        trajectory_marker.id = 3;
        trajectory_marker.action = visualization_msgs::Marker::ADD;
        trajectory_marker.type = visualization_msgs::Marker::LINE_STRIP;
        trajectory_marker.lifetime = ros::Duration(0.1);

        // Line width
        path_marker.scale.x = 0.25f;
        speed_marker.scale = path_marker.scale;
        curvature_marker.scale = path_marker.scale;
        trajectory_marker.scale.x = interface::VEHICLE_WIDTH;

        // Color space
        path_marker.color.r = 0.0f;
        path_marker.color.g = 0.78f;
        path_marker.color.b = 0.58f;
        path_marker.color.a = 0.7f;
        speed_marker.color = path_marker.color; 
        curvature_marker.color.r = 0.7f;
        curvature_marker.color.g = 0.0f;
        curvature_marker.color.b = 0.7f;
        curvature_marker.color.a = 0.7f;  
        trajectory_marker.color.a = 1.0f;

        std::vector<std_msgs::ColorRGBA> colors;
        std::vector<double> speeds;
        for (uint16_t i = 0; i < (uint16_t)trajectory.point.size(); i++) {
            geometry_msgs::Point point;
            point.x = trajectory.point[i].x;
            point.y = trajectory.point[i].y;
            point.z = trajectory.point[i].z;
            path_marker.points.push_back(point);
            speed_marker.points.push_back(point);
            curvature_marker.points.push_back(point);            
            
            if (trajectory.point[i].speed > 0.0) {
                trajectory_marker.points.push_back(point);
                speeds.push_back(trajectory.point[i].speed);
            }
            
            point.z = trajectory.point[i].speed * interface::KPH2MPS;
            speed_marker.points.push_back(point);
            point.z = trajectory.point[i].curvature * 1000.0;
            curvature_marker.points.push_back(point);
        }

        double max_speed = *std::max_element(speeds.begin(), speeds.end());
        for (const auto& speed : speeds) {
            float ratio = 1.0;//speed / max_speed;

            std_msgs::ColorRGBA color;
            color.r = std::min(1.0f - ratio, 1.0f);
            color.g = std::min(0.78f * (ratio), 1.0f);
            color.b = std::min(0.58f * (ratio), 1.0f);
            color.a = 0.7f;

            colors.push_back(color);
        }
        trajectory_marker.colors = colors;

        msg.markers.push_back(path_marker);
        msg.markers.push_back(speed_marker);
        msg.markers.push_back(curvature_marker);
        msg.markers.push_back(trajectory_marker);

        return msg;
    }


    visualization_msgs::MarkerArray UpdateRvizOptimalTrajectory(const interface::Trajectory& trajectory) {
        
        visualization_msgs::MarkerArray msg;
        uint16_t id = 0;
        
        visualization_msgs::Marker path_marker;
        visualization_msgs::Marker speed_marker;

        path_marker.header = UpdateHeader(trajectory.header);
        path_marker.ns = "optimal_spatial";
        path_marker.id = id++;
        path_marker.action = visualization_msgs::Marker::ADD;
        path_marker.type = visualization_msgs::Marker::LINE_STRIP;
        path_marker.lifetime = ros::Duration(0.1);

        speed_marker.header = UpdateHeader(trajectory.header);
        speed_marker.ns = "optimal_temporal";
        speed_marker.id = id;
        speed_marker.action = visualization_msgs::Marker::ADD;
        speed_marker.type = visualization_msgs::Marker::LINE_LIST;
        speed_marker.lifetime = ros::Duration(0.1);

        // Line width
        path_marker.scale.x = 0.15f;
        speed_marker.scale = path_marker.scale;

        // Deep blue color for optimal trajectory
        path_marker.color.r = 0.0f;
        path_marker.color.g = 0.0f;
        path_marker.color.b = 0.8f;  // Deep blue
        path_marker.color.a = 0.9f;  // High alpha for better visibility
        
        speed_marker.color = path_marker.color;

        for (uint16_t i = 0; i < (uint16_t)trajectory.point.size(); i++) {
            geometry_msgs::Point point;
            point.x = trajectory.point[i].x;
            point.y = trajectory.point[i].y;
            point.z = trajectory.point[i].z;
            path_marker.points.push_back(point);
            speed_marker.points.push_back(point);
            point.z = trajectory.point[i].speed * interface::KPH2MPS;
            speed_marker.points.push_back(point);
        }
        msg.markers.push_back(path_marker);
        msg.markers.push_back(speed_marker);

        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizNonoptTrajectory(const interface::Trajectory& trajectory) {
        visualization_msgs::MarkerArray msg;
        visualization_msgs::Marker trajectory_marker;

        trajectory_marker.header = UpdateHeader(trajectory.header);
        trajectory_marker.ns = "trajectory";
        trajectory_marker.id = 0;
        trajectory_marker.action = visualization_msgs::Marker::ADD;
        trajectory_marker.type = visualization_msgs::Marker::LINE_STRIP;
        trajectory_marker.lifetime = ros::Duration(0.1);

        // Line width
        trajectory_marker.scale.x = interface::VEHICLE_WIDTH;

        // Color space
        trajectory_marker.color.r = 0.78f;
        trajectory_marker.color.g = 0.0f;
        trajectory_marker.color.b = 0.58f;
        trajectory_marker.color.a = 0.2f;

        for (uint16_t i = 0; i < (uint16_t)trajectory.point.size(); i++) {
            geometry_msgs::Point point;
            point.x = trajectory.point[i].x;
            point.y = trajectory.point[i].y;
            point.z = trajectory.point[i].z; 
            
            if (trajectory.point[i].speed > 0.0) {
                trajectory_marker.points.push_back(point);
            }
        }

        msg.markers.push_back(trajectory_marker);

        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizTrajectories(const interface::Trajectories& trajectories) {

        visualization_msgs::MarkerArray msg;
        uint16_t id = 0;
        
        for (auto trajectory : trajectories.trajectory) {
            visualization_msgs::Marker path_marker;
            visualization_msgs::Marker speed_marker;

            path_marker.header = UpdateHeader(trajectory.header);
            path_marker.ns = "spatial";
            path_marker.id = id++;
            path_marker.action = visualization_msgs::Marker::ADD;
            path_marker.type = visualization_msgs::Marker::LINE_STRIP;
            path_marker.lifetime = ros::Duration(0.1);

            speed_marker.header = UpdateHeader(trajectory.header);
            speed_marker.ns = "temporal";
            speed_marker.id = id;
            speed_marker.action = visualization_msgs::Marker::ADD;
            speed_marker.type = visualization_msgs::Marker::LINE_LIST;
            speed_marker.lifetime = ros::Duration(0.1);

            // Line width
            path_marker.scale.x = 0.1f;
            speed_marker.scale = path_marker.scale;

            path_marker.color.r = 0.0f;
            path_marker.color.g = 0.38f;
            path_marker.color.b = 0.68f;
            path_marker.color.a = 0.3f;
            speed_marker.color = path_marker.color;

            for (uint16_t i = 0; i < (uint16_t)trajectory.point.size(); i++) {
                geometry_msgs::Point point;
                point.x = trajectory.point[i].x;
                point.y = trajectory.point[i].y;
                point.z = trajectory.point[i].z;
                path_marker.points.push_back(point);
                speed_marker.points.push_back(point);
                point.z = trajectory.point[i].speed * interface::KPH2MPS;
                speed_marker.points.push_back(point);
            }
            msg.markers.push_back(path_marker);
            msg.markers.push_back(speed_marker);
        }

        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizPaths(const interface::Paths& paths) {
        
        visualization_msgs::MarkerArray msg;
        uint16_t id = 0;
        
        for (auto path : paths.path) {
            visualization_msgs::Marker path_marker;

            path_marker.header = UpdateHeader(path.header);
            path_marker.ns = "spatial";
            path_marker.id = id++;
            path_marker.action = visualization_msgs::Marker::ADD;
            path_marker.type = visualization_msgs::Marker::LINE_STRIP;
            path_marker.lifetime = ros::Duration(0.1);

            // Line width
            path_marker.color.r = 0.0f;
            path_marker.color.g = 0.78f;
            path_marker.color.b = 0.58f;
            path_marker.color.a = 0.2f;
            path_marker.scale.x = interface::VEHICLE_WIDTH;

            for (const auto& point : path.point) {
                geometry_msgs::Point p;
                p.x = point.x;
                p.y = point.y;
                p.z = point.z;
                path_marker.points.push_back(p);
            }
            
            msg.markers.push_back(path_marker);
        }

        return msg;
    }

    visualization_msgs::Marker UpdateRvizRouteCandidate(const interface::BehaviorTrajectory& route_candidate, 
                                                        const std::string& ns,
                                                        const int& id = 0) {
        visualization_msgs::Marker marker;
        marker.header = UpdateHeader(route_candidate.header);
        marker.header.frame_id = "world";
        marker.ns = ns;
        marker.id = id;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.color.a = 1.0;
        marker.lifetime = ros::Duration(0.1);

        if (route_candidate.spatial_trajectory.point.size() == 0) {
            return marker;
        }

        for (const auto& point : route_candidate.spatial_trajectory.point) {
            geometry_msgs::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;

            marker.points.push_back(p);
        }

        if (route_candidate.on_ego == 0) {
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
        } 
        else if (route_candidate.on_ego == 1) {
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        }
        else if (route_candidate.on_ego == 2) {
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
        }

        return marker;
    }

    visualization_msgs::Marker UpdateRvizRouteBoundary(const interface::TrajectoryBoundary& route_boundary, 
                                                       const std::string& ns) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.ns = ns;
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.color.a = 1.0;

        if (route_boundary.point.size() == 0) {
            return marker;
        }

        for (const auto& point : route_boundary.point) {
            geometry_msgs::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;

            marker.points.push_back(p);
        }

        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;

        return marker;
    }

    

    visualization_msgs::Marker UpdateRvizRouteBoundary(const interface::TrajectoryBoundary& route_boundary, 
                                                       const std::string& ns,
                                                       const int& id = 0) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.lifetime = ros::Duration(0.1);
        marker.ns = ns;
        marker.id = id;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.color.a = 1.0;

        if (route_boundary.point.size() == 0) {
            return marker;
        }

        for (const auto& point : route_boundary.point) {
            geometry_msgs::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;

            marker.points.push_back(p);
        }

        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;

        return marker;
    }

    visualization_msgs::MarkerArray UpdateRvizRouteCandidates(const interface::BehaviorTrajectories& route_candidates) {
        visualization_msgs::MarkerArray msg;
        visualization_msgs::Marker marker;

        // Ego Route
        marker = UpdateRvizRouteCandidate(route_candidates.lk_trajectory, "ego_route", 0);
        msg.markers.push_back(marker);
        marker = UpdateRvizRouteBoundary(route_candidates.lk_trajectory.left_boundary, "ego_route_left", 0);
        msg.markers.push_back(marker);
        marker = UpdateRvizRouteBoundary(route_candidates.lk_trajectory.right_boundary, "ego_route_right", 0);
        msg.markers.push_back(marker);

        // Left Route
        int idx = 0;
        for (const auto& llc : route_candidates.llc_trajectories) {
            marker = UpdateRvizRouteCandidate(llc, "left_route", idx++);
            msg.markers.push_back(marker);
            marker = UpdateRvizRouteBoundary(llc.left_boundary, "left_route_left", idx);
            msg.markers.push_back(marker);
            marker = UpdateRvizRouteBoundary(llc.right_boundary, "left_route_right", idx);
            msg.markers.push_back(marker);
        }

        // Right Route
        idx = 0;
        for (const auto& rlc : route_candidates.rlc_trajectories) {
            marker = UpdateRvizRouteCandidate(rlc, "right_route", idx++);
            msg.markers.push_back(marker);
            marker = UpdateRvizRouteBoundary(rlc.left_boundary, "right_route_left", idx);
            msg.markers.push_back(marker);
            marker = UpdateRvizRouteBoundary(rlc.right_boundary, "right_route_right", idx);
            msg.markers.push_back(marker);
        }

        return msg;
    }

    jsk_recognition_msgs::BoundingBoxArray UpdateRvizWarningObjects(const interface::DetectObjects3D& llc_objects, 
                                                                    const interface::DetectObjects3D& rlc_objects) {        
        jsk_recognition_msgs::BoundingBoxArray msg;
        msg.header = UpdateHeader(llc_objects.header);  
        msg.header.frame_id = "world";      

        for (auto& object : llc_objects.object) {
            jsk_recognition_msgs::BoundingBox marker;
            marker.header = msg.header;
            marker.label = 1;

            marker.dimensions.x = object.dimension.length;
            marker.dimensions.y = object.dimension.width;
            marker.dimensions.z = object.dimension.height;

            marker.pose.position.x = object.state.x;
            marker.pose.position.y = object.state.y;
            marker.pose.position.z = object.dimension.height/2;

            tf2::Quaternion myQuaternion;
            myQuaternion.setRPY(0, 0, object.state.yaw);
            myQuaternion = myQuaternion.normalize();
            marker.pose.orientation.x = myQuaternion.getX();
            marker.pose.orientation.y = myQuaternion.getY();
            marker.pose.orientation.z = myQuaternion.getZ();
            marker.pose.orientation.w = myQuaternion.getW();

            msg.boxes.push_back(marker);
        }

        for (auto& object : rlc_objects.object) {
            jsk_recognition_msgs::BoundingBox marker;
            marker.header = msg.header;
            marker.label = 2;

            marker.dimensions.x = object.dimension.length;
            marker.dimensions.y = object.dimension.width;
            marker.dimensions.z = object.dimension.height;

            marker.pose.position.x = object.state.x;
            marker.pose.position.y = object.state.y;
            marker.pose.position.z = object.dimension.height/2;

            tf2::Quaternion myQuaternion;
            myQuaternion.setRPY(0, 0, object.state.yaw);
            myQuaternion = myQuaternion.normalize();
            marker.pose.orientation.x = myQuaternion.getX();
            marker.pose.orientation.y = myQuaternion.getY();
            marker.pose.orientation.z = myQuaternion.getZ();
            marker.pose.orientation.w = myQuaternion.getW();

            msg.boxes.push_back(marker);
        }

        return msg;
    }

    jsk_recognition_msgs::BoundingBoxArray UpdateRvizTargetObjects(const interface::DetectObjects3D& objects) {        
        jsk_recognition_msgs::BoundingBoxArray msg;
        msg.header = UpdateHeader(objects.header);        

        for (auto& object : objects.object) {
            jsk_recognition_msgs::BoundingBox marker;
            marker.header = msg.header;
            marker.label = object.classification;

            marker.dimensions.x = object.dimension.length;
            marker.dimensions.y = object.dimension.width;
            marker.dimensions.z = object.dimension.height;

            marker.pose.position.x = object.state.x;
            marker.pose.position.y = object.state.y;
            marker.pose.position.z = object.dimension.height/2;

            tf2::Quaternion myQuaternion;
            myQuaternion.setRPY(0, 0, object.state.yaw);
            myQuaternion = myQuaternion.normalize();
            marker.pose.orientation.x = myQuaternion.getX();
            marker.pose.orientation.y = myQuaternion.getY();
            marker.pose.orientation.z = myQuaternion.getZ();
            marker.pose.orientation.w = myQuaternion.getW();

            msg.boxes.push_back(marker);
        }

        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizTargetLines(const interface::TargetLines& lines) {
        visualization_msgs::MarkerArray msg;
        visualization_msgs::Marker line_marker;
        visualization_msgs::Marker info_marker;

        for (const auto& target_line : lines.line) {            
            line_marker.header = UpdateHeader(lines.header);
            line_marker.header.frame_id = "world";
            line_marker.ns = "line";
            line_marker.id = 0;
            line_marker.action = visualization_msgs::Marker::ADD;
            line_marker.type = visualization_msgs::Marker::LINE_STRIP;
            line_marker.lifetime = ros::Duration(0.1);

            info_marker.header = UpdateHeader(lines.header);
            info_marker.header.frame_id = "world";
            info_marker.ns = "info";
            info_marker.id = 1;
            info_marker.action = visualization_msgs::Marker::ADD;
            info_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            info_marker.lifetime = ros::Duration(0.1);

            // Line width
            line_marker.scale.x = 0.1f;
            info_marker.scale.z = 0.5f;

            // Color space
            line_marker.color.r = 0.0f;
            line_marker.color.g = 1.0f;
            line_marker.color.b = 0.0f;
            line_marker.color.a = 1.0f;
            info_marker.color.r = 1.0f;
            info_marker.color.g = 1.0f;
            info_marker.color.b = 1.0f;
            info_marker.color.a = 1.0f;

            geometry_msgs::Point point;
            point.x = target_line.lx;
            point.y = target_line.ly;
            point.z = 0.0;
            line_marker.points.push_back(point);
            
            point.x = target_line.rx;
            point.y = target_line.ry;
            point.z = 0.0;
            line_marker.points.push_back(point);

            info_marker.pose.position.x = point.x;
            info_marker.pose.position.y = point.y;
            info_marker.pose.position.z = 0.0;
            info_marker.text = "Target Line Dist: " + std::to_string(target_line.dist) + "m";

            msg.markers.push_back(line_marker);
            msg.markers.push_back(info_marker);
        }

        return msg;        
    }

    visualization_msgs::MarkerArray UpdateRvizPolynomialPoints(const std::vector<interface::Point>& points) {
        visualization_msgs::MarkerArray msg;
        visualization_msgs::Marker marker;

        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = "world";
        marker.ns = "polynomial_points";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 0.1f;       
        
        marker.color.r = 1.0f;
        marker.color.g = 0.78f;
        marker.color.b = 0.68f;
        marker.color.a = 0.7f;
        marker.lifetime = ros::Duration(0.1);

        for (const auto& point : points) {
            geometry_msgs::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;

            marker.points.push_back(p);
        }

        msg.markers.push_back(marker);

        return msg;        
    }

    autohyu_msgs::PlanningInputOutput UpdatePlanningInputOutput(const interface::PlanningSpace& planning_space,
                                                                const interface::Trajectory& trajectory) {
        autohyu_msgs::PlanningInputOutput msg;
        msg.planning_space = UpdatePlanningSpace(planning_space);
        msg.trajectory = UpdateTrajectory(trajectory);

        return msg;
    }

    autohyu_msgs::DAgger UpdateDAggerResult(const double& start_time,
                                            const bool& record_on) {
        autohyu_msgs::DAgger msg;
        msg.header.stamp = ros::Time(start_time);
        msg.record_on.data = record_on;

        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizDAggerResult(const std::vector<std::pair<interface::Point, interface::Point>> compare_points,
                                                           const bool& is_collect_data,
                                                           const interface::VehicleState& vehicle_state,
                                                           const interface::Trajectory& rule_trajectory,
                                                           const interface::Trajectory& e2e_trajectory) {

        visualization_msgs::MarkerArray msg;
        visualization_msgs::Marker rule_marker;
        visualization_msgs::Marker e2e_marker;

        rule_marker.header = ros_bridge::UpdateHeader(vehicle_state.header);
        rule_marker.header.frame_id = "world";
        rule_marker.ns = "rule_trajectory";
        rule_marker.id = 0;
        rule_marker.type = visualization_msgs::Marker::LINE_STRIP;
        rule_marker.action = visualization_msgs::Marker::ADD;
        rule_marker.pose.orientation.w = 1.0;
        rule_marker.scale.x = 0.25;
        rule_marker.color.r = 0.0;
        rule_marker.color.g = 1.0;
        rule_marker.color.b = 1.0;
        rule_marker.color.a = 1.0;
        rule_marker.lifetime = ros::Duration(0.1);

        e2e_marker = rule_marker;
        e2e_marker.header = ros_bridge::UpdateHeader(vehicle_state.header);
        e2e_marker.header.frame_id = "world";
        e2e_marker.ns = "e2e_trajectory";
        e2e_marker.scale.x = 0.25;
        e2e_marker.color.r = 1.0;
        e2e_marker.color.g = 0.0;
        e2e_marker.color.b = 1.0;
        
        for (const auto& point : rule_trajectory.point) {
            geometry_msgs::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = 0.0;
            rule_marker.points.push_back(p);
        }

        for (const auto& point : e2e_trajectory.point) {
            geometry_msgs::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = 0.0;
            e2e_marker.points.push_back(p);
        }

        int id = 0;
        for (const auto& point : compare_points) {
            visualization_msgs::Marker compare_marker;
            compare_marker = rule_marker;
            compare_marker.ns = "compare_points";
            compare_marker.id = id++;
            compare_marker.scale.x = 0.15;

            if (is_collect_data) {
                compare_marker.color.r = 1.0;
                compare_marker.color.g = 0.0;
                compare_marker.color.b = 0.0;
            }
            else {
                compare_marker.color.r = 0.0;
                compare_marker.color.g = 1.0;
                compare_marker.color.b = 0.0;
            }

            compare_marker.points.clear();
            geometry_msgs::Point p;
            p.x = point.first.x;
            p.y = point.first.y;
            p.z = 0.0;
            compare_marker.points.push_back(p);
            p.x = point.second.x;
            p.y = point.second.y;
            p.z = 0.0;
            compare_marker.points.push_back(p);

            msg.markers.push_back(compare_marker);
        }

        msg.markers.push_back(rule_marker);
        msg.markers.push_back(e2e_marker);

        return msg;
    }

    jsk_rviz_plugins::OverlayText UpdateRvizE2EEvaluationResult(const std::string& evaluation_result,
                                                                const interface::VehicleState& vehicle_state,
                                                                const bool& collect_data,
                                                                const bool& collect_data_ongoing,
                                                                const double& collect_start_time) {
        jsk_rviz_plugins::OverlayText msg;
        
        msg.bg_color.r = 0.0;
        msg.bg_color.g = 0.0;
        msg.bg_color.b = 0.0;
        msg.bg_color.a = 0.0;
        
        if (collect_data && collect_data_ongoing) {
            msg.fg_color.r = 1.0;
            msg.fg_color.g = 0.0;
            msg.fg_color.b = 0.0;
        }
        else if (!collect_data && collect_data_ongoing) {
            msg.fg_color.r = 1.0;
            msg.fg_color.g = 0.0;
            msg.fg_color.b = 1.0;
        }
        else if (collect_data && !collect_data_ongoing) {
            msg.fg_color.r = 1.0;
            msg.fg_color.g = 0.5;
            msg.fg_color.b = 0.0;
        }
        else {
            msg.fg_color.r = 0.0;
            msg.fg_color.g = 1.0;
            msg.fg_color.b = 0.0;
        }
        msg.fg_color.a = 1.0;

        msg.line_width = 2;
        msg.text_size = 10.0;

        std::string collect_status = collect_data_ongoing ? "Collecting Data" : "Data Collection Finished";
        std::string collect_time = collect_data_ongoing ? "Collect Start Time: " + util_function::to_string(collect_start_time, 3) + " [s]" : "";
        std::string collect_duration = collect_data_ongoing ? "Collect Duration: " + util_function::to_string(vehicle_state.header.stamp - collect_start_time, 3) + " [s]" : "";

        msg.text = evaluation_result;
        msg.text += collect_status + "\n";
        msg.text += collect_time + "\n";
        msg.text += collect_duration + "\n";
        msg.text += "\n==============================";

        return msg;
    }

    jsk_rviz_plugins::OverlayText UpdateRvizE2EScenarioEvaluationResult(const std::string& evaluation_result) {
        jsk_rviz_plugins::OverlayText msg;
        
        msg.bg_color.r = 0.0;
        msg.bg_color.g = 0.0;
        msg.bg_color.b = 0.0;
        msg.bg_color.a = 0.0;        
        
        msg.fg_color.r = 1.0;
        msg.fg_color.g = 1.0;
        msg.fg_color.b = 0.0;
        msg.fg_color.a = 1.0;

        msg.line_width = 2;
        msg.text_size = 10.0;

        msg.text = evaluation_result;

        return msg;
    }

} // namespace ros_bridge

#endif  // __ROS_BRIDGE_TRAJECTORIES__