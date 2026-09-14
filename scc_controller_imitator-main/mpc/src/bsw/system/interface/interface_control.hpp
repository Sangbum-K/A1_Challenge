/**
 * @file        interface_control.hpp
 * @brief       interface for control
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-01 updated by Yuseung Na: Refactoring 
 */

#ifndef __INTERFACE_CONTROL_HPP__
#define __INTERFACE_CONTROL_HPP__
#pragma once

// STD Header
#include <stdint.h>
#include <map>
#include <utility>
#include <vector>

// Interface Header
#include "interface_std_msgs.hpp"

namespace interface {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // enum
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //     

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // structs
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //  
    typedef struct{
        // prediction info
        double x;
        double y;
        double yaw;
        double curvature;
        double vx;
        double time;
        double distance;

        // command info
        double steer;        
        double dsteer;        
        double ax;
        double jerkx;
        double fx;
        double trq;
        double accel_pedal;
        double brake_pedal;
    } ControlPoint;

    typedef struct {
        Header header;
        std::vector<ControlPoint> control_point;
    } ControlTrajectory;

    typedef struct {
        float steering_tire_angle;
        float ax;
        float trq;
        float accel_pedal;
        float brake_pedal;
        ControlTrajectory control_trajectory;
    } ControlCommand;

    typedef struct {        
        double cross_track_error;   // [m]
        double yaw_error;           // [deg]
        double speed_error;         // [kph]
        double speed_compensation_error;         // [kph]
        double accel_error;         // [mps2]
        double target_speed;        // [kph]
        double current_speed;       // [kph]
        double opt_cost;            
        double opt_time;            // [ms]
        double solve_time;          // [ms]
        double total_time;          // [ms]
        double sqp_iter;
        double qp_iter;
        double compensation_longitudinal;
        double compensation_lateral;
    } PathTrackingInfos;

    typedef struct { 
        Header header;
        double steering_angle;
        double front_tire_angle;
        double speed;
        double accel;
        double gas;
        double brake;
        double torque;
    } VehicleCmd;
} // namespace interface

#endif // __INTERFACE_CONTROL_HPP__
