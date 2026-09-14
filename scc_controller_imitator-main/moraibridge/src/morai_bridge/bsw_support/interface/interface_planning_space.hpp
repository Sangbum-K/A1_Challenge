/**
 * @file        interface_trajectories.hpp
 * @brief       interface for trajectories
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-01 updated by Yuseung Na: Refactoring 
 */

#ifndef __INTERFACE_PLANNING_SPACE_HPP__
#define __INTERFACE_PLANNING_SPACE_HPP__
#pragma once

// STD Header
#include <stdint.h>
#include <map>
#include <utility>
#include <vector>

// Interface Header
#include "interface_std_msgs.hpp"
#include "interface_geometry_msgs.hpp"
#include "interface_trajectories.hpp"
#include "interface_objects.hpp"

namespace interface {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // enum
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //     
    

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // structs
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //  
    typedef struct {
        double stamp{0.0};
        double x{0.0};
        double y{0.0};
        double heading{0.0};
        double vx{0.0};
        double vy{0.0};
        double ax{0.0};
        double ay{0.0};
        double tire_angle{0.0};
    } PlanningSpaceState;
    
    typedef struct {
        std::vector<PlanningSpaceState> state;
    } PlanningSpaceEgo;

    typedef struct {
        uint64_t id;
        double   length;
        double   width;
        double   height;
        uint16_t classification;
    
        std::vector<PlanningSpaceState> state;
    } PlanningSpaceObject;

    typedef struct {
        double      x{0.0};
        double      y{0.0};
        double      heading{0.0};
        std::string traffic_light_id{""};
        uint16_t    traffic_light_color{0};
        double      speed_limit{0.0};
    } PlanningSpacePoint;

    typedef struct {
        Header header;
        std::string id{""};
        uint16_t    type{0};
        uint16_t    color{0};
        std::string check_state{""}; // left, straight, right, etc.
        std::vector<Point> stop_line_points;

    } PlanningSpaceTrafficLight;

    typedef struct {
        int64_t id;
        bool is_dividing_lane;
        std::vector<PlanningSpacePoint> point;
        std::vector<PlanningSpacePoint> left_boundary;
        std::vector<PlanningSpacePoint> right_boundary;
    } PlanningSpaceVectorMap;

    typedef struct {
        Header header;

        // Ego
        PlanningSpaceEgo ego;

        // Objects
        std::vector<PlanningSpaceObject> objects;
        
        // Predicted Objects
        PredictObjects predict_objects;

        // Vector Map
        std::vector<PlanningSpaceVectorMap> lanes;
        std::vector<PlanningSpaceVectorMap> routes;
        std::vector<PlanningSpaceVectorMap> crosswalks;
        std::vector<PlanningSpaceVectorMap> goal_routes;
        
        // Drivable Area
        interface::Trajectory drivable_area;

        // Imitation Trajectory
        interface::Trajectory imitation_trajectory;

        // Vocab
        PlanningSpaceTrafficLight traffic_light;
    } PlanningSpace;
}

#endif // __INTERFACE_PLANNING_SPACE_HPP__
