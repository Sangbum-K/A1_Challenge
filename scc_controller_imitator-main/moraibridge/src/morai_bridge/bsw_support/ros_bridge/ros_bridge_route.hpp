/**
 * @file        ros_bridge_route.hpp
 * @brief       ROS bridge for route
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 * 
 */

#ifndef __ROS_BRIDGE_ROUTE__
#define __ROS_BRIDGE_ROUTE__
#pragma once

// ROS header
#include <ros/ros.h>
#include <tf/tf.h>

// ROS Message Header
#include <autohyu_msgs/LaneletPath.h>
#include <autohyu_msgs/LaneletRoute.h>
#include <autohyu_msgs/LaneletLanePoint.h>
#include <autohyu_msgs/GoalPoints.h>

// Interface Header
#include "interface_route.hpp"

// Bridge Header
#include "ros_bridge_std_msgs.hpp"
#include "ros_bridge_geometry_msgs.hpp"

namespace ros_bridge {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    interface::GoalPoint GetGoalPoint(const autohyu_msgs::GoalPoint& msg){
        interface::GoalPoint point;
        point.x = msg.x;
        point.y = msg.y;
        point.z = msg.z;
        point.yaw = msg.yaw;

        return point;
    }

    interface::GoalPoint GetGoalPoint(const geometry_msgs::PoseStamped& msg){
        interface::GoalPoint point;
        point.x = msg.pose.position.x;
        point.y = msg.pose.position.y;
        point.z = msg.pose.position.z;

        tf::Quaternion quat;
        tf::quaternionMsgToTF(msg.pose.orientation, quat);
        double roll, pitch, yaw;
        tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        point.yaw = yaw;

        return point;
    }

    interface::GoalPoints GetGoalPoints(const autohyu_msgs::GoalPoints& msg){
        interface::GoalPoints goal_points;
        for (int i = 0 ; i < msg.point.size(); i++){
            interface::GoalPoint point = ros_bridge::GetGoalPoint(msg.point[i]);
            goal_points.point.push_back(point);
        }
        return goal_points;
    }

    interface::LaneletPath GetLaneletPath(const autohyu_msgs::LaneletPath& msg) {
        interface::LaneletPath lanelet_path;
        lanelet_path.id_ego = msg.ll_id_ego;
        lanelet_path.id_path = msg.ll_id_path;
        lanelet_path.goal_point = ros_bridge::GetPoint(msg.goal_point);

        return lanelet_path;
    }

    interface::LaneletRelation GetLaneletRelation(const autohyu_msgs::LaneletRelation& msg) {
        interface::LaneletRelation lanelet_relation;
        lanelet_relation.id = msg.id;
        lanelet_relation.type = static_cast<interface::LaneletRelationType>(msg.type);

        return lanelet_relation;
    }

    interface::LaneletTrafficSign GetLaneletTrafficSign(const autohyu_msgs::LaneletTrafficSign& msg) {
        interface::LaneletTrafficSign lanelet_traffic_sign;
        lanelet_traffic_sign.id = msg.id;
        lanelet_traffic_sign.element_type = msg.element_type;
        
        for (const auto& point : msg.point) {
            lanelet_traffic_sign.point.push_back(ros_bridge::GetPoint(point));
        }

        return lanelet_traffic_sign;
    }

    interface::LaneletTrafficLight GetLaneletTrafficLight(const autohyu_msgs::LaneletTrafficLight& msg) {
        interface::LaneletTrafficLight lanelet_traffic_light;
        lanelet_traffic_light.sign_id = msg.sign_id;
        lanelet_traffic_light.intersection_id = msg.intersection_id;
        lanelet_traffic_light.light_type = msg.light_type;
        
        for (const auto& point : msg.stop_point) {
            lanelet_traffic_light.stop_point.push_back(ros_bridge::GetPoint(point));
        }

        return lanelet_traffic_light;
    }

    interface::LaneletRightOfWay GetLaneletRightOfWay(const autohyu_msgs::LaneletRightOfWay& msg) {
        interface::LaneletRightOfWay lanelet_right_of_way;
        lanelet_right_of_way.left_boundary_id = msg.left_boundary_id;
        lanelet_right_of_way.right_boundary_id = msg.right_boundary_id;
        lanelet_right_of_way.center_lane_id = msg.center_lane_id;

        for (const auto& point : msg.stop_point) {
            lanelet_right_of_way.stop_point.push_back(ros_bridge::GetPoint(point));
        }

        return lanelet_right_of_way;
    }

    interface::Lanelet GetLanelet(const autohyu_msgs::Lanelet& msg) {
        interface::Lanelet lanelet;
        lanelet.id                      = msg.id;
        lanelet.is_ego                  = msg.is_ego;
        lanelet.is_goal                 = msg.is_goal;
        lanelet.is_shortest_path        = msg.is_shortest_path;
        lanelet.left_boundary_line_id   = msg.left_boundary_line_id;
        lanelet.right_boundary_line_id  = msg.right_boundary_line_id;

        for (const auto& relation : msg.relation) {
            lanelet.relation.push_back(ros_bridge::GetLaneletRelation(relation));
        }

        for (const auto& traffic_sign : msg.traffic_sign) {
            lanelet.traffic_sign.push_back(ros_bridge::GetLaneletTrafficSign(traffic_sign));
        }

        for (const auto& turn_signal : msg.turn_signal) {
            lanelet.turn_signal.push_back(ros_bridge::GetLaneletTrafficSign(turn_signal));
        }

        for (const auto& traffic_light : msg.traffic_light) {
            lanelet.traffic_light.push_back(ros_bridge::GetLaneletTrafficLight(traffic_light));
        }

        for (const auto& right_of_way : msg.right_of_way) {
            lanelet.right_of_way.push_back(ros_bridge::GetLaneletRightOfWay(right_of_way));
        }

        return lanelet;
    }

    interface::LaneletLine GetLaneletLine(const autohyu_msgs::LaneletLine& msg) {
        interface::LaneletLine lanelet_line;
        lanelet_line.id         = msg.id;
        lanelet_line.line_type  = msg.line_type;
        lanelet_line.line_color = msg.line_color;
        for (const auto& point : msg.point) {
            lanelet_line.point.push_back(ros_bridge::GetPoint(point));
        }

        return lanelet_line;
    }

    interface::LaneletLane GetLaneletLane(const autohyu_msgs::LaneletLane& msg) {
        interface::LaneletLane lanelet_lane;
        lanelet_lane.id        = msg.id;
        lanelet_lane.lane_num  = msg.lane_num;
        lanelet_lane.max_speed = msg.max_speed;

        for (const auto& p : msg.point) {
            interface::Point point;
            point.x           = p.x;
            point.y           = p.y;
            point.z           = p.z;
            point.vx_mps      = p.vx_mps;
            point.ax_mps2     = p.ax_mps2;
            point.kappa_radpm = p.kappa_radpm;
            point.psi_rad     = p.psi_rad;
            point.s_m         = p.s_m;
            lanelet_lane.point.push_back(point);
        }

        return lanelet_lane;
    }

    interface::LaneletRoute GetLaneletRoute(const autohyu_msgs::LaneletRoute& msg){
        interface::LaneletRoute lanelet_route;
        lanelet_route.header = ros_bridge::GetHeader(msg.header);

        for (const auto& lanelet : msg.lanelet) {
            lanelet_route.lanelet.push_back(ros_bridge::GetLanelet(lanelet));
        }

        for (const auto& boundary_line : msg.boundary_line) {
            lanelet_route.boundary_line.push_back(ros_bridge::GetLaneletLine(boundary_line));
        }

        for (const auto& center_lane : msg.center_lane) {
            lanelet_route.center_lane.push_back(ros_bridge::GetLaneletLane(center_lane));
        }

        return lanelet_route;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    autohyu_msgs::GoalPoint UpdateGoalPoint(const interface::GoalPoint& point) {
        autohyu_msgs::GoalPoint msg;
        msg.x = point.x;
        msg.y = point.y;
        msg.z = point.z;
        msg.yaw = point.yaw;

        return msg;
    }

    autohyu_msgs::GoalPoints UpdateGoalPoints(const interface::GoalPoints& points) {
        autohyu_msgs::GoalPoints msg;
        for (int i = 0 ; i < points.point.size(); i++){
            msg.point.push_back(ros_bridge::UpdateGoalPoint(points.point[i]));
        }
        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizGoalPoints(const interface::GoalPoints& points,
                                                        const std::string& ns,
                                                        const double& color_r, 
                                                        const double& color_g, 
                                                        const double& color_b, 
                                                        const double& color_a) {
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "world";
        msg.ns = ns; 
        msg.id = 0;
        msg.type = visualization_msgs::Marker::SPHERE_LIST;
        msg.action = visualization_msgs::Marker::ADD;
        msg.scale.x = 2.0;
        msg.scale.y = 2.0;
        msg.scale.z = 2.0;
        msg.color.r = color_r;
        msg.color.g = color_g;
        msg.color.b = color_b;
        msg.color.a = color_a;
        msg.lifetime = ros::Duration(1);

        for (const auto& point : points.point) {
            geometry_msgs::Point point_msg;
            point_msg.x = point.x;
            point_msg.y = point.y;
            point_msg.z = point.z;
            msg.points.push_back(point_msg);
        }
        marker_array.markers.push_back(msg);

        return marker_array;
    }

    visualization_msgs::MarkerArray UpdateRvizRoutingGoalPoints(const interface::GoalPoints& points,
                                                                 const std::string& ns,
                                                                 const double& color_r, 
                                                                 const double& color_g, 
                                                                 const double& color_b, 
                                                                 const double& color_a) {
        visualization_msgs::MarkerArray marker_array;

        for (size_t i = 0; i < points.point.size(); ++i) {
            visualization_msgs::Marker msg;

            // msg.header = UpdateHeader(points[i].header);
            msg.header.seq = i;
            msg.header.stamp = ros::Time::now();
            msg.header.frame_id = "world";
            msg.ns = ns + "_" + std::to_string(i + 1); 
            msg.id = i;
            msg.type = visualization_msgs::Marker::ARROW;
            msg.action = visualization_msgs::Marker::ADD;
            msg.pose.position.x = points.point[i].x;
            msg.pose.position.y = points.point[i].y;
            msg.pose.position.z = points.point[i].z;
            
            double yaw = points.point[i].yaw;
            tf::Quaternion quat = tf::createQuaternionFromYaw(yaw);
            msg.pose.orientation.x = quat.x();
            msg.pose.orientation.y = quat.y();
            msg.pose.orientation.z = quat.z();
            msg.pose.orientation.w = quat.w();
            msg.scale.x = 2.0;
            msg.scale.y = 1.0;
            msg.scale.z = 1.0;
            msg.color.r = color_r;
            msg.color.g = color_g;
            msg.color.b = color_b;
            msg.color.a = color_a;
            msg.lifetime = ros::Duration(1);

            marker_array.markers.push_back(msg);
        }

        return marker_array;
    }
    autohyu_msgs::LaneletPath UpdateLaneletPath(const interface::LaneletPath& lanelet_path) {
        autohyu_msgs::LaneletPath msg;
        msg.ll_id_ego = lanelet_path.id_ego;
        msg.ll_id_path = lanelet_path.id_path;
        msg.goal_point = ros_bridge::UpdatePoint(lanelet_path.goal_point);

        return msg;
    }

    autohyu_msgs::LaneletRelation UpdateLaneletRelation(const interface::LaneletRelation& lanelet_relation) {
        autohyu_msgs::LaneletRelation msg;
        msg.id = lanelet_relation.id;
        msg.type = static_cast<uint8_t>(lanelet_relation.type);

        return msg;
    }

    autohyu_msgs::LaneletTrafficSign UpdateLaneletTrafficSign(const interface::LaneletTrafficSign& lanelet_traffic_sign) {
        autohyu_msgs::LaneletTrafficSign msg;
        msg.id = lanelet_traffic_sign.id;
        msg.element_type = lanelet_traffic_sign.element_type;

        for (const auto& point : lanelet_traffic_sign.point) {
            msg.point.push_back(ros_bridge::UpdatePoint(point));
        }

        return msg;
    }

    autohyu_msgs::LaneletTrafficLight UpdateLaneletTrafficLight(const interface::LaneletTrafficLight& lanelet_traffic_light) {
        autohyu_msgs::LaneletTrafficLight msg;
        msg.sign_id = lanelet_traffic_light.sign_id;
        msg.intersection_id = lanelet_traffic_light.intersection_id;
        msg.light_type = lanelet_traffic_light.light_type;

        for (const auto& point : lanelet_traffic_light.stop_point) {
            msg.stop_point.push_back(ros_bridge::UpdatePoint(point));
        }

        return msg;
    }

    autohyu_msgs::LaneletRightOfWay UpdateLaneletRightOfWay(const interface::LaneletRightOfWay& lanelet_right_of_way) {
        autohyu_msgs::LaneletRightOfWay msg;
        msg.left_boundary_id = lanelet_right_of_way.left_boundary_id;
        msg.right_boundary_id = lanelet_right_of_way.right_boundary_id;
        msg.center_lane_id = lanelet_right_of_way.center_lane_id;

        for (const auto& point : lanelet_right_of_way.stop_point) {
            msg.stop_point.push_back(ros_bridge::UpdatePoint(point));
        }

        return msg;
    }

    autohyu_msgs::Lanelet UpdateLanelet(const interface::Lanelet& lanelet) {
        autohyu_msgs::Lanelet msg;
        msg.id                      = lanelet.id;
        msg.is_ego                  = lanelet.is_ego;
        msg.is_goal                 = lanelet.is_goal;
        msg.is_shortest_path        = lanelet.is_shortest_path;
        msg.left_boundary_line_id   = lanelet.left_boundary_line_id;
        msg.right_boundary_line_id  = lanelet.right_boundary_line_id;

        for (const auto& relation : lanelet.relation) {
            msg.relation.push_back(ros_bridge::UpdateLaneletRelation(relation));
        }

        for (const auto& traffic_sign : lanelet.traffic_sign) {
            msg.traffic_sign.push_back(ros_bridge::UpdateLaneletTrafficSign(traffic_sign));
        }

        for (const auto& turn_signal : lanelet.turn_signal) {
            msg.turn_signal.push_back(ros_bridge::UpdateLaneletTrafficSign(turn_signal));
        }

        for (const auto& traffic_light : lanelet.traffic_light) {
            msg.traffic_light.push_back(ros_bridge::UpdateLaneletTrafficLight(traffic_light));
        }

        for (const auto& right_of_way : lanelet.right_of_way) {
            msg.right_of_way.push_back(ros_bridge::UpdateLaneletRightOfWay(right_of_way));
        }

        return msg;
    }

    autohyu_msgs::LaneletLine UpdateLaneletLine(const interface::LaneletLine& lanelet_line) {
        autohyu_msgs::LaneletLine msg;
        
        msg.id         = lanelet_line.id;
        msg.line_type  = lanelet_line.line_type;
        msg.line_color = lanelet_line.line_color;
        for (const auto& point : lanelet_line.point) {
            msg.point.push_back(ros_bridge::UpdatePoint(point));
        }

        return msg;
    }

    autohyu_msgs::LaneletLane UpdateLaneletLane(const interface::LaneletLane& lanelet_lane) {
        autohyu_msgs::LaneletLane msg;

        msg.id        = lanelet_lane.id;
        msg.lane_num  = lanelet_lane.lane_num;
        msg.max_speed = lanelet_lane.max_speed;
        for (const auto& point : lanelet_lane.point) {
            autohyu_msgs::LaneletLanePoint p;
            p.x           = point.x;
            p.y           = point.y;
            p.z           = point.z;
            p.vx_mps      = point.vx_mps;
            p.ax_mps2     = point.ax_mps2;
            p.kappa_radpm = point.kappa_radpm;
            p.psi_rad     = point.psi_rad;
            p.s_m         = point.s_m;
            msg.point.push_back(p);
        }

        return msg;
    }

    autohyu_msgs::LaneletRoute UpdateLaneletRoute(const interface::LaneletRoute& lanelet_route) {
        autohyu_msgs::LaneletRoute msg;
        msg.header = ros_bridge::UpdateHeader(lanelet_route.header);

        for (const auto& lanelet : lanelet_route.lanelet) {
            msg.lanelet.push_back(ros_bridge::UpdateLanelet(lanelet));
        }

        for (const auto& boundary_line : lanelet_route.boundary_line) {
            msg.boundary_line.push_back(ros_bridge::UpdateLaneletLine(boundary_line));
        }

        for (const auto& center_lane : lanelet_route.center_lane) {
            msg.center_lane.push_back(ros_bridge::UpdateLaneletLane(center_lane));
        }

        return msg;
    }

    visualization_msgs::MarkerArray UpdateRvizRoutePoints(const interface::LaneletRoute route_info){
        visualization_msgs::MarkerArray marker_array;

        for(int i=0; i< route_info.boundary_line.size(); i++){
            visualization_msgs::Marker marker;

            marker.header.stamp = ros::Time::now();
            marker.header.frame_id = "world";
            marker.ns = "route boundary";
            marker.id = i;
            marker.action = visualization_msgs::Marker::ADD;
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.lifetime = ros::Duration(1.2);        
            marker.scale.x = 0.1;
            marker.scale.y = 0.1;
            marker.scale.z = 0.1;
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 1.0;
            marker.color.a = 1.0;

            for (int j = 0; j < route_info.boundary_line[i].point.size(); j++){
                geometry_msgs::Point point;
                
                point.x = route_info.boundary_line[i].point[j].x;
                point.y = route_info.boundary_line[i].point[j].y;
                point.z = 0.0;
                marker.points.push_back(point);         
            }        
            marker_array.markers.push_back(marker);    
        }

        for(int i=0; i< route_info.center_lane.size(); i++){
            visualization_msgs::Marker marker;

            marker.header.stamp = ros::Time::now();
            marker.header.frame_id = "world";
            marker.ns = "route center";
            marker.id = i;
            marker.action = visualization_msgs::Marker::ADD;
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.lifetime = ros::Duration(1.2);   
            marker.scale.x = 0.3;
            marker.scale.y = 0.3;
            marker.scale.z = 0.3;
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            for (int j = 0; j < route_info.center_lane[i].point.size(); j++){
                geometry_msgs::Point point;
                
                point.x = route_info.center_lane[i].point[j].x;
                point.y = route_info.center_lane[i].point[j].y;
                point.z = 0.0;
                marker.points.push_back(point);         
            }        
            marker_array.markers.push_back(marker);    
        }
        
        return marker_array;
    }    

    // update rviz route, input : center x,y point of lanelet and lanelet_route_id
    visualization_msgs::MarkerArray UpdateRvizRouteNum(const std::vector<std::pair<float, float>> route_center_point,
                                                       std::vector<int64_t> ll_id_route){
        visualization_msgs::MarkerArray marker_array;
        for(int i=0; i< route_center_point.size(); i++){
            visualization_msgs::Marker marker;
            marker.header.stamp = ros::Time::now();
            marker.header.frame_id = "world";
            marker.ns = "route id";
            marker.id = i;
            marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            marker.action = visualization_msgs::Marker::ADD;
            marker.scale.x = 1.0;
            marker.scale.y = 1.0;
            marker.scale.z = 1.0;
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 1.0;
            marker.color.a = 0.9;
            marker.lifetime = ros::Duration(1.2);
            // geometry_msgs::Point point;
            // point.x = waypoint.first[i];
            // point.y = waypoint.second[i];
            // point.z = 1;
            //marker.points.push_back(point);
            marker.pose.position.x = route_center_point[i].first;
            marker.pose.position.y = route_center_point[i].second;
            marker.pose.position.z = 2.0;
            marker.pose.orientation.w = 1.0;\
            std::stringstream ss;
            ss << i <<" th route: " << ll_id_route[i]; 
            marker.text = ss.str();

            marker_array.markers.push_back(marker);
        }
        return marker_array;
    }

    visualization_msgs::MarkerArray UpdateRvizRawRoutePoints(const interface::LaneletRoute route_info) {
        visualization_msgs::MarkerArray marker_array;

        // Boundary line을 위한 SPHERE 시각화
        for (int i = 0; i < route_info.boundary_line.size(); i++) {
            for (int j = 0; j < route_info.boundary_line[i].point.size(); j++) {
                visualization_msgs::Marker marker;

                marker.header.stamp = ros::Time::now();
                marker.header.frame_id = "world";
                marker.ns = "route_boundary";
                marker.id = i * 1000 + j;  // 각 점마다 고유한 ID를 부여
                // marker.action = visualization_msgs::Marker::ADD;
                marker.type = visualization_msgs::Marker::SPHERE;  // SPHERE 형태로 설정
                marker.lifetime = ros::Duration(1.2);        
                marker.scale.x = 0.5;  // 구의 크기를 적절히 설정
                marker.scale.y = 0.5;
                marker.scale.z = 0.5;
                marker.color.r = 0.0;
                marker.color.g = 0.0;
                marker.color.b = 1.0;
                marker.color.a = 1.0;

                geometry_msgs::Point point;
                point.x = route_info.boundary_line[i].point[j].x;
                point.y = route_info.boundary_line[i].point[j].y;
                point.z = 1.0;
                marker.pose.position = point;  // 위치를 지정

                marker_array.markers.push_back(marker);    
            }
        }

        // Center line을 위한 SPHERE 시각화
        for (int i = 0; i < route_info.center_lane.size(); i++) {
            for (int j = 0; j < route_info.center_lane[i].point.size(); j++) {
                visualization_msgs::Marker marker;

                marker.header.stamp = ros::Time::now();
                marker.header.frame_id = "world";
                marker.ns = "route_center";
                marker.id = i * 1000 + j;  // 각 점마다 고유한 ID를 부여
                // marker.action = visualization_msgs::Marker::ADD;
                marker.type = visualization_msgs::Marker::SPHERE;  // SPHERE 형태로 설정
                marker.lifetime = ros::Duration(1.2);        
                marker.scale.x = 0.5;  // 구의 크기를 적절히 설정
                marker.scale.y = 0.5;
                marker.scale.z = 0.5;
                marker.color.r = 0.0;
                marker.color.g = 0.0;
                marker.color.b = 1.0;
                marker.color.a = 1.0;

                geometry_msgs::Point point;
                point.x = route_info.center_lane[i].point[j].x;
                point.y = route_info.center_lane[i].point[j].y;
                point.z = 1.0;
                marker.pose.position = point;  // 위치를 지정

                marker_array.markers.push_back(marker);    
            }
        }
        
        return marker_array;
    }          

    // autohyu_msgs::LaneletRoute UpdateLaneletRoute(const interface::LaneletRoute& lanelet_route) {
    //     autohyu_msgs::LaneletRoute msg;
    //     msg.ll_id_ego = lanelet_route.id_ego;
    //     msg.ll_id_route = lanelet_route.id_route;
    //     msg.goal_point = ros_bridge::UpdatePoint(lanelet_route.goal_point);

    //     return msg;
    // }


} // namespace ros_bridge

#endif  // __ROS_BRIDGE_ROUTE__