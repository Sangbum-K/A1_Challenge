/**
 * @file        interface_trajectories.hpp
 * @brief       interface for trajectories
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-01 updated by Yuseung Na: Refactoring 
 */

#ifndef __INTERFACE_TRAJECTORIES_HPP__
#define __INTERFACE_TRAJECTORIES_HPP__
#pragma once

// STD Header
#include <stdint.h>
#include <map>
#include <utility>
#include <vector>
#include <string>

// Interface Header
#include "interface_route.hpp"
#include "interface_std_msgs.hpp"
#include "interface_geometry_msgs.hpp"

namespace interface {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // enum
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //   
    typedef enum {
        LANE_KEEPING = 0,
        NO_OVERTAKING = 1,
        CAR_FOLLOWING = 2,
        OVERTAKING = 3,
        OVERTAKE_SLOW_VEHICLE = 4,
        OVERTAKE_STATIC_VEHICLE = 5,
        LAP_TIME = 6,
        STOP = 7,
        COME_BACK = 8,
    } BehaviorType;

    typedef enum{
        CENTER_LANE  = 0,
        RIGHT_LANE   = 1,
        LEFT_LANE    = 2,
    } LaneType;
    typedef enum{
        SINGLE_LANE  = 0,
        NEXT_RIGHT   = 1,
        NEXT_LEFT    = 2,
        WRONG        = 3,
    } CenterLaneType;

    typedef enum{
        CENTER_LINE  = 0,
        RIGHT_LINE   = 1,
        LEFT_LINE    = 2,
    } LineType;

    typedef enum {
        FORWARD = 0,
        BACKWARD = 1,
    } TrajectoryDirection;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // structs
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    typedef struct {
        int64_t lane_num;
        double  max_speed;
        double  x;
        double  y;
        double  z;
    } LanePoint;

    typedef struct {
        std::string line_type;
        std::string line_color;
        double  x;
        double  y;
        double  z;
    } LinePoint;

    typedef struct {
        bool is_final;
        Lanelet lanelet;

        std::vector<LanePoint> center_lane_point;
        std::vector<LinePoint> left_line_point;
        std::vector<LinePoint> right_line_point;        
    } RouteSegment;

    typedef struct {
        uint16_t on_ego; // 0: ego, 1: left, 2: right
        std::vector<RouteSegment> segment;
    } RouteCandidate;

    typedef struct {
        RouteCandidate ego_route;
        std::vector<RouteCandidate> left_routes;
        std::vector<RouteCandidate> right_routes;
    } RouteCandidates;
    
    typedef struct {
        double time;
        double distance;
        double x;
        double y;
        double z;
        double yaw;
        double curvature;
        double speed;
        double max_speed;
        bool   on_shortest_path;
        bool   on_goalpoint;
        int16_t lane_num;
    } BehaviorTrajectoryPoint;

    typedef struct {
        float x;
        float y;
        float z;
        float s;
        float n;
    } TrajectoryBoundaryPoint;

    typedef struct {
        std::vector<TrajectoryBoundaryPoint> point;
    } TrajectoryBoundary;

    typedef struct {
        float x;
        float y;
        float z;
        bool  on_shortest_path;
        bool  on_goalpoint;
        int16_t lane_num;
    } SpatialTrajectoryPoint;

    typedef struct {
        std::vector<SpatialTrajectoryPoint> point;
    } SpatialTrajectory;

    typedef struct {
        float time;
        float distance;
        float curvature;
        float speed;
        float ref_speed;
    } TemporalTrajectoryPoint;

    typedef struct {
        std::vector<TemporalTrajectoryPoint> point;
    } TemporalTrajectory;

    typedef struct {
        PointArray left_boundary;
        PointArray right_boundary;
        PointArray center_line;
        std::vector<Point> stop_point;
    } RightOfWayTrajectory;

    typedef struct {
        Header header;
        uint32_t id;     // 0: available, 1: selected, 2: unavailable
        uint16_t on_ego; // 0: ego, 1: left, 2: right

        std::vector<LaneletTrafficSign>  traffic_sign;
        std::vector<LaneletTrafficSign>  turn_signal;
        std::vector<LaneletTrafficLight> traffic_light;
        std::vector<RightOfWayTrajectory> right_of_way;

        SpatialTrajectory spatial_trajectory;        
        TemporalTrajectory temporal_trajectory;

        TrajectoryBoundary left_boundary;
        TrajectoryBoundary right_boundary;        
        std::vector<BehaviorTrajectoryPoint> point;
    } BehaviorTrajectory;

    typedef struct {
        Header header;
        BehaviorTrajectory lk_trajectory;
        std::vector<BehaviorTrajectory> llc_trajectories;
        std::vector<BehaviorTrajectory> rlc_trajectories;
    } BehaviorTrajectories;

    typedef struct {
        bool is_valid{false};
        bool is_danger{false};
        bool is_dynamic_object{false};
        bool is_static_object{false};

        double ttc_front;
        double ttc_rear;

        double distance_front;
        double distance_rear;
    } LaneState;

    typedef struct {        
        Header header;
        
        float x;
        float y;
        float z;
        float yaw;
        float curvature;
        float speed;
        float acceleration;

        float s;
        float ds;
        float dds;
        float n;
        float dn;
        float ddn;
    } StartState;

    typedef struct {
        float x;
        float y;
        float z;  
        float yaw;
        float curvature;
        float distance;
        float speed;
        float acceleration;
        float time;
        float s;
        float n;
        float ds;
        float dn;
        float dds;
        float ddn;
    } HybridTrajectoryPoint;

    typedef struct {
        Header       header;
        BehaviorType behavior_type;
        int          behavior_decision;
        bool         is_collision;
        double       cost;
        
        std::vector<HybridTrajectoryPoint> point;
    } HybridTrajectory;

    typedef struct {
        std::vector<HybridTrajectory> trajectory;
    } HybridTrajectories;

    typedef struct {
        float               time;
        float               distance;
        TrajectoryDirection direction;
        float               x;
        float               y;
        float               z;
        float               yaw;
        float               curvature;
        float               speed;
        float               ref_speed;
        float               acceleration;
        float               s;
        float               n;
        float               ds;
        float               dn;
        float               dds;
        float               ddn;
        float               yaw_sn;
    } TrajectoryPoint;

    typedef struct {
        Header       header;
        uint16_t     id;
        BehaviorType behavior_type;
        float        lateral_offset;

        BehaviorType behavior_zone;
        BehaviorType next_behavior_zone;
        BehaviorType decision_behavior;
        double       behavior_distance;
        bool         is_target;

        TrajectoryBoundary           left_boundary;
        TrajectoryBoundary           right_boundary;
        TrajectoryBoundary           center_boundary;
        std::vector<TrajectoryPoint> point;
    } Trajectory;

    typedef struct {
        float time;
        float distance;
        float x;
        float y;
        float z;
        float curvature;
        float speed;
        float max_speed;
    } PathPoint;

    typedef struct {
        Header       header;
        uint16_t     id;
        BehaviorType behavior_type;
        float        lateral_offset;        
        
        TrajectoryBoundary           left_boundary;
        TrajectoryBoundary           right_boundary;
        std::vector<PathPoint> point;
    } Path;

    typedef struct {
        Header header;
        std::vector<Path> path;
    } Paths;

    typedef struct {
        Header header;
        std::vector<Trajectory> trajectory;
    } Trajectories;
    
    typedef struct {
        bool on_ego;
        bool collision;
        double x;
        double y;
        double r;
    } CollisionBoundary;
    
    typedef struct{
        Header header;
        std::vector<CollisionBoundary> boundary;
    } CollisionBoundaries;

}

#endif // __INTERFACE_TRAJECTORIES_HPP__
