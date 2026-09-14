/**
 * @file        interface_route.hpp
 * @brief       interface for route
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)
 * 
 * @date        2024-06-20 created by Yuseung Na
 *  
 */

#ifndef __INTERFACE_ROUTE_HPP__
#define __INTERFACE_ROUTE_HPP__
#pragma once

// STD Header
#include <stdint.h>
#include <map>
#include <utility>
#include <vector>
#include <string>

// Interface Header
#include "interface_std_msgs.hpp"
#include "interface_geometry_msgs.hpp"

namespace interface {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // enum
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //     
    typedef enum {
        LL_NONE = 0,
        LL_SUCCESSOR = 1,
        LL_LEFT = 2,
        LL_RIGHT = 3,
        LL_ADJACENT_LEFT = 4,
        LL_ADJACENT_RIGHT = 5,
        LL_CONFLICTING = 6,
        LL_AREA = 7,
        LL_DIVERGE_STRAIGHT = 8,
        LL_DIVERGE_LEFT = 9,
        LL_DIVERGE_RIGHT = 10,
        LL_MERGE_STRAIGHT = 11,
        LL_MERGE_LEFT = 12,
        LL_MERGE_RIGHT = 13,
    } LaneletRelationType;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // structs
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //    
    typedef struct {
        std::string link_id{""};
        double x;
        double y;
        double z;
        double yaw;
    } GoalPoint;
    inline bool operator==(const GoalPoint& a, const GoalPoint& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z && a.yaw == b.yaw;
    }

    typedef struct {
        std::vector<GoalPoint> point;
    } GoalPoints;
    inline bool operator==(const GoalPoints& a, const GoalPoints& b) {
        if (a.point.size() != b.point.size()) {
            return false;
        }
        for (size_t i = 0; i < a.point.size(); ++i) {
            if (!(a.point[i] == b.point[i])) {
                return false;
            }
        }
        return true; 
    }

    typedef struct{
        Header                  header;
        int64_t                 id_ego;
        std::vector<int64_t>    id_path;
        interface::Point        goal_point;
    } LaneletPath;

    typedef struct {
        int64_t id;
        LaneletRelationType type;
    } LaneletRelation;

    typedef struct {
        int64_t id;
        std::string element_type;   // turn_signal, speed_bump, etc.
        std::string value;
        std::vector<interface::Point> point;
    } LaneletTrafficSign;

    typedef struct {
        int64_t sign_id;
        int64_t intersection_id;    
        std::string light_type;     // straight, left, right, uturn
        std::vector<interface::Point> stop_point;
    } LaneletTrafficLight;

    typedef struct {
        int64_t left_boundary_id;
        int64_t right_boundary_id;
        int64_t center_lane_id;
        std::vector<interface::Point> stop_point;
    } LaneletRightOfWay;

    typedef struct {
        std::vector<LaneletTrafficSign> traffic_sign;
        std::vector<LaneletTrafficSign> turn_signal;
        std::vector<LaneletTrafficLight> traffic_light;
        std::vector<LaneletRightOfWay> right_of_way;
    } LaneletRegulatoryElements;

    typedef struct {
        int64_t id;
        bool is_ego;
        bool is_goal;
        bool is_shortest_path;
        int64_t left_boundary_line_id;
        int64_t right_boundary_line_id;
        std::vector<LaneletRelation> relation;
        std::vector<LaneletTrafficSign> traffic_sign;
        std::vector<LaneletTrafficSign> turn_signal;
        std::vector<LaneletTrafficLight> traffic_light;
        std::vector<LaneletRightOfWay> right_of_way;
    } Lanelet;

    typedef struct {
        int64_t     id;
        std::string line_type;
        std::string line_color;
        std::vector<interface::Point> point;
    } LaneletLine;

    typedef struct {
        int64_t id;
        int64_t lane_num{-1};
        double  max_speed;
        std::vector<interface::Point> point;
    } LaneletLane;

    typedef struct {
        Header                      header;
        std::vector<Lanelet>        lanelet;
        std::vector<LaneletLine>    boundary_line;
        std::vector<LaneletLane>    center_lane;
    } LaneletRoute;
} // namespace interface

#endif // __INTERFACE_ROUTE_HPP__
