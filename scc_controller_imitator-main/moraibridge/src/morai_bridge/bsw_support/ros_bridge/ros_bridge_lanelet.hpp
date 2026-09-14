/**
 * @file        ros_bridge_lanelet.hpp
 * @brief       ROS bridge for lanelet
 * 
 * @authors     JunheeLee (998jun@gmail.com)          
 * 
 * @date        2024-05-01 created by Junhee Lee
 *              2024-06-20 updated by Yuseung Na: add useful functions
 * 
 */

#ifndef __ROS_BRIDGE_LANELET__
#define __ROS_BRIDGE_LANELET__
#pragma once

// ROS header
#include <ros/ros.h>
#include <tf/tf.h>

// ROS Message Header
#include <geometry_msgs/PolygonStamped.h>
#include <autohyu_msgs/MapBin.h>
#include <autohyu_msgs/MapBinSrv.h>
#include <visualization_msgs/MarkerArray.h>
#include <jsk_recognition_msgs/PolygonArray.h>

// Interface Header
#include "interface_lanelet.hpp"
#include "interface_vehicle_state.hpp"

// Util Function Header
#include "function_lanelet.hpp"

// Bridge Header
#include "ros_bridge_std_msgs.hpp"
#include "ros_bridge_geometry_msgs.hpp"

namespace ros_bridge {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    interface::MapBin GetLaneletMapBin(const autohyu_msgs::MapBin& msg) {
        interface::MapBin map_data;
        map_data.header = ros_bridge::GetHeader(msg.header);
        map_data.data = msg.data;

        return map_data;
    }

    lanelet::LaneletMapPtr GetLaneletMap(const autohyu_msgs::MapBin& msg) {
        std::vector<int8_t> data = msg.data;
        return util_function::DecodeLaneletMapBin(data);
    }    
    
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    autohyu_msgs::MapBin UpdateLaneletMapBin(const interface::MapBin& map_data) {
        autohyu_msgs::MapBin msg;
        msg.header = ros_bridge::UpdateHeader(map_data.header);
        msg.data = map_data.data;
        
        return msg;
    }    

    visualization_msgs::MarkerArray UpdateRvizMap(const lanelet::LaneletMapPtr& ptr_lanelet_map,
                                                  const bool& is_ref_origin = false,
                                                  const ros::Time& current_time = ros::Time::now()) {
        lanelet::LaneletMap &map = *ptr_lanelet_map;
        visualization_msgs::MarkerArray msg;

        for (auto line_string : map.lineStringLayer) {
            visualization_msgs::Marker marker;
            double x, r, g, b, a, yaw;

            std::string marker_type = "";

            // Left / Right boundary

            if ((std::string)line_string.attributeOr("type","-") == "virtual") {
                marker_type = "line_strip";
                x = 0.05;
                r = 0.7, g = 0.7, b = 0.7, a = 0.1;
                // continue;
            }
            else if((std::string)line_string.attributeOr("subtype","-") == "dashed") {
                marker_type = "line_strip";
                x = 0.2;
                r = 0.7, g = 0.7, b = 0.7, a = 0.3;
            }
            else if((std::string)line_string.attributeOr("subtype","-") == "solid") {
                marker_type = "line_strip";
                x = 0.5;
                r = 0.7, g = 0.7, b = 0.7, a = 0.5;
            }
            else if((std::string)line_string.attributeOr("subtype","-") == "solid_dashed"
                    || (std::string)line_string.attributeOr("subtype","-") == "dashed_solid") {
                marker_type = "line_strip";
                x = 0.4;
                r = 0.7, g = 0.7, b = 0.7, a = 0.4;
            }
            // Regulatory Elements
            else if((std::string)line_string.attributeOr("subtype","-") == "stop_line") {
                marker_type = "line_strip";
                x = 1.0;
                r = 0.7, g = 0.7, b = 0.7, a = 0.5;
            }
            else if((std::string)line_string.attributeOr("subtype","-") == "speed_bump") {
                marker_type = "line_strip";
                x = 1.0;
                r = 0.7, g = 0.7, b = 0.0, a = 0.5;
            }
            else if((std::string)line_string.attributeOr("type","-") == "traffic_light") {
                marker_type = "cube";
                x = 0.0;
                for (auto i_ls = 1; i_ls < line_string.size(); i_ls++) {    
                    double prev_x = line_string[i_ls-1].x();
                    double prev_y = line_string[i_ls-1].y();
                    double curr_x = line_string[i_ls].x();
                    double curr_y = line_string[i_ls].y();                
                    x += sqrt(pow(curr_x-prev_x, 2) + pow(curr_y-prev_y, 2));
                }
                yaw = atan2(line_string.back().y() - line_string.front().y(), line_string.back().x() - line_string.front().x());
                r = 0.7, g = 0.7, b = 0.7, a = 0.5;
            }
            else if ((std::string)line_string.attributeOr("subtype","-") == "turn_signal") {
                marker_type = "arrow";
                for (auto i_ls = 1; i_ls < line_string.size(); i_ls++) {    
                    double prev_x = line_string[i_ls-1].x();
                    double prev_y = line_string[i_ls-1].y();
                    double curr_x = line_string[i_ls].x();
                    double curr_y = line_string[i_ls].y();                
                    x += sqrt(pow(curr_x-prev_x, 2) + pow(curr_y-prev_y, 2));
                }
                yaw = atan2(line_string.back().y() - line_string.front().y(), line_string.back().x() - line_string.front().x());
                r = 0.0, g = 1.0, b = 0.0, a = 0.5;
            }
            else if ((std::string)line_string.attributeOr("subtype","-") == "de206") {
                marker_type = "line_strip";
                x = 0.5;
                r = 1.0, g = 1.0, b = 1.0, a = 0.5;
            }
            // Center Line or Unknown
            else{
                marker_type = "line_strip";
                x = 0.1;
                r = 0.0, g = 1.0, b = 0.0, a = 0.2;
            }

            if(is_ref_origin == false) {                
                r = 1.0, g = 0.1, b = 0.57, a = 1.0;    
            }

            if (marker_type == "line_strip") {
                marker.header.stamp    = current_time;
                marker.header.frame_id = "world";
                marker.ns              = "line";
                marker.id              = (int32_t)line_string.id();
                marker.type            = visualization_msgs::Marker::LINE_STRIP;
                marker.action          = visualization_msgs::Marker::ADD;
                marker.scale.x         = x;
                marker.color.r         = r;
                marker.color.g         = g;
                marker.color.b         = b;
                marker.color.a         = a;
                marker.lifetime        = ros::Duration(1.0);

                for (auto i_ls = 0; i_ls < line_string.size(); i_ls++) {
                    geometry_msgs::Point point;
                    point.x = line_string[i_ls].x();
                    point.y = line_string[i_ls].y();
                    point.z = line_string[i_ls].z();
                    marker.points.push_back(point);
                }
                msg.markers.push_back(marker);
            }
            else if (marker_type == "cube") {
                marker.header.stamp    = current_time;
                marker.header.frame_id = "world";
                marker.ns              = "cube";
                marker.id              = (int32_t)line_string.id();
                marker.type            = visualization_msgs::Marker::CUBE;
                marker.action          = visualization_msgs::Marker::ADD;
                marker.scale.x         = x;
                marker.scale.y         = 0.2;
                marker.scale.z         = 0.2;
                marker.color.r         = r;
                marker.color.g         = g;
                marker.color.b         = b;
                marker.color.a         = a;
                marker.lifetime        = ros::Duration(1.0);

                geometry_msgs::Point point;
                point.x = (line_string.front().x() + line_string.back().x()) / 2.0;
                point.y = (line_string.front().y() + line_string.back().y()) / 2.0;
                point.z = (line_string.front().z() + line_string.back().z()) / 2.0;
                marker.pose.position = point;
                marker.pose.orientation = tf::createQuaternionMsgFromYaw(yaw);
                msg.markers.push_back(marker);
            }
            else if (marker_type == "arrow") {
                marker.header.stamp    = current_time;
                marker.header.frame_id = "world";
                marker.ns              = "arrow";
                marker.id              = (int32_t)line_string.id();
                marker.type            = visualization_msgs::Marker::ARROW;
                marker.action          = visualization_msgs::Marker::ADD;
                marker.scale.x         = 0.5;
                marker.scale.y         = 0.2;
                marker.scale.z         = 0.2;
                marker.color.r         = r;
                marker.color.g         = g;
                marker.color.b         = b;
                marker.color.a         = a;
                marker.lifetime        = ros::Duration(1.0);

                geometry_msgs::Point point;
                point.x = (line_string.front().x() + line_string.back().x()) / 2.0;
                point.y = (line_string.front().y() + line_string.back().y()) / 2.0;
                point.z = (line_string.front().z() + line_string.back().z()) / 2.0;
                marker.pose.position = point;
                marker.pose.orientation = tf::createQuaternionMsgFromYaw(yaw);
                msg.markers.push_back(marker);
            }
        }    

        return msg;
    }

    // update rviz path, input : lanelet::routing::LaneletPath 
    visualization_msgs::MarkerArray UpdateRvizPath(const lanelet::routing::LaneletPath &path){
        using namespace lanelet;
        visualization_msgs::MarkerArray marker_array;
        for (auto lanelet : path) {
            auto centerline = lanelet.centerline();
            visualization_msgs::Marker marker_center;
            marker_center.header.stamp    = ros::Time::now();
            marker_center.header.frame_id = "world";
            marker_center.ns              = "shortest_path";
            marker_center.id              = (int32_t)lanelet.id();
            marker_center.type            = visualization_msgs::Marker::LINE_STRIP;
            marker_center.action          = visualization_msgs::Marker::ADD;
            marker_center.scale.x         = 1.0;
            marker_center.color.r         = 0.6;
            marker_center.color.g         = 0.98;
            marker_center.color.b         = 0.98;
            marker_center.color.a         = 0.2;
            marker_center.lifetime        = ros::Duration(1.2);
            for (auto i_cl = 0; i_cl < centerline.size(); i_cl++) {
                geometry_msgs::Point point;
                point.x = centerline[i_cl].x();
                point.y = centerline[i_cl].y();
                point.z = 0.2;
                marker_center.points.push_back(point);
            }
            marker_array.markers.push_back(marker_center);
        }
        return marker_array;
    }

    // update rviz route, input : lanelet::LaneletSubmapConstPtr 
    visualization_msgs::MarkerArray UpdateRvizRoute(const lanelet::LaneletSubmapConstPtr& sub_map, const int64_t &ego_id){
        using namespace lanelet;
        visualization_msgs::MarkerArray marker_array;
        //std::cout << "here 0" << std::endl;
        for (auto lanelet : sub_map->laneletLayer) {
            //std::cout << "here -1" << std::endl;
            auto centerline = lanelet.centerline();

            //std::cout << "here 1" << std::endl;
            visualization_msgs::Marker marker_center;
            marker_center.header.stamp    = ros::Time::now();
            marker_center.header.frame_id = "world";
            marker_center.ns              = "route";
            marker_center.id              = (int32_t)lanelet.id();
            marker_center.type            = visualization_msgs::Marker::LINE_STRIP;
            marker_center.action          = visualization_msgs::Marker::ADD;
            marker_center.scale.x         = 1.5;
            marker_center.color.r         = 0.6;
            marker_center.color.g         = 0.98;
            marker_center.color.b         = 0.98;
            marker_center.color.a         = 0.1;
            if(ego_id == lanelet.id()){
                marker_center.color.a         = 0.3;
            }
            marker_center.lifetime        = ros::Duration(1.2);
            for (auto i_cl = 0; i_cl < centerline.size(); i_cl++) {
                geometry_msgs::Point point;
                point.x = centerline[i_cl].x();
                point.y = centerline[i_cl].y();
                point.z = 0.0;
                marker_center.points.push_back(point);
            }      
            marker_array.markers.push_back(marker_center);
        }
        return marker_array;   
    }

    // update vector information
    visualization_msgs::MarkerArray UpdateRvizLaneletDirections(const std::vector<std::pair<double, double>>& v_lanelet_directions,int ego_ll_idx, const std::string& frame_id, const interface::VehicleState& vehicle_state) {
    
        visualization_msgs::MarkerArray marker_array;

        int id = 0; // 각 마커의 고유 ID
        for (const auto& direction : v_lanelet_directions) {
            visualization_msgs::Marker arrow;
            arrow.header.frame_id = frame_id; // 차량 좌표계 기준
            arrow.header.stamp = ros::Time::now();
            arrow.ns = "lanelet_directions";
            
            arrow.type = visualization_msgs::Marker::ARROW;
            arrow.action = visualization_msgs::Marker::ADD;

            arrow.scale.x = 0.4; // 화살표의 샤프트 굵기
            arrow.scale.y = 0.5; // 화살표의 헤드 크기
            arrow.scale.z = 1.1; // 화살표의 높이는 필요 없음

            if(id == ego_ll_idx){
                arrow.color.r = 0.0f;
                arrow.color.g = 0.0f;
                arrow.color.b = 1.0f;
                arrow.color.a = 1.0f; // 불투명
            }
            else{
                arrow.color.r = 1.0f;
                arrow.color.g = 0.0f;
                arrow.color.b = 0.0f;
                arrow.color.a = 1.0f; // 불투명
            }
            
            geometry_msgs::Point start, end;

            // 시작점 (0, 0)
            start.x = vehicle_state.x;
            start.y = vehicle_state.y;
            start.z = 1.1;

            // 종료점 (pair<double, double>)
            end.x = vehicle_state.x+direction.first;
            end.y = vehicle_state.y+direction.second;
            end.z = 1.1;

            arrow.points.push_back(start);
            arrow.points.push_back(end);
            arrow.id = id++;
            arrow.lifetime = ros::Duration(0.1);
            marker_array.markers.push_back(arrow);
        }

        return marker_array;
    }

    visualization_msgs::Marker UpdateRvizEgoDirection(const std::pair<double, double>& v_ego_direction, const std::string& frame_id,  const interface::VehicleState& vehicle_state) {
        visualization_msgs::Marker arrow;
        // Header 설정
        arrow.header.frame_id = frame_id;   // 차량 좌표계 기준
        arrow.header.stamp = ros::Time::now();
        arrow.ns = "ego_direction";
        arrow.id = 0;  // 단일 화살표이므로 ID는 고정

        // Marker 타입 및 액션
        arrow.type = visualization_msgs::Marker::ARROW;
        arrow.action = visualization_msgs::Marker::ADD;

        // Marker 크기
        arrow.scale.x = 0.4; // 화살표의 샤프트 굵기
        arrow.scale.y = 0.5; // 화살표의 헤드 크기
        arrow.scale.z = 1.6; // 화살표의 높이는 필요 없음

        // Marker 색상
        arrow.color.r = 0.0f; // 빨간색
        arrow.color.g = 1.0f;
        arrow.color.b = 1.0f;
        arrow.color.a = 1.0f; // 불투명

        // 화살표의 시작점과 끝점 설정
        geometry_msgs::Point start, end;

        // 시작점 (0, 0)
        start.x = vehicle_state.x;
        start.y = vehicle_state.y;
        start.z = 1.6;

        // 종료점 (v_ego_direction의 x, y)
        end.x = vehicle_state.x + v_ego_direction.first;
        end.y = vehicle_state.y + v_ego_direction.second;
        end.z = 1.6;

        // Points 설정
        arrow.points.push_back(start);
        arrow.points.push_back(end);
        arrow.lifetime = ros::Duration(0.1);
        return arrow;
    }



} // namespace ros_bridge

#endif  // __ROS_BRIDGE_LANELET__