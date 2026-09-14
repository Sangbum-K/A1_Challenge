/**
 * @file        interface_planning_control.hpp
 * @brief       interface for planning and control
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-01 updated by Yuseung Na: Refactoring 
 */

 #ifndef __INTERFACE_PLANNING_CONTROL_HPP__
 #define __INTERFACE_PLANNING_CONTROL_HPP__
 #pragma once
 
 // STD Header
 #include <stdint.h>
 #include <map>
 #include <utility>
 #include <vector>
 
 // Interface Header
 #include "interface_std_msgs.hpp"
 #include "interface_vehicle_state.hpp"
 #include "interface_trajectories.hpp"
 
 namespace interface {
     // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
     // enum
     // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //     
     
 
     // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
     // structs
     // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //  
    typedef struct {
        Header header;

        interface::VehicleState vehicle_state;
        interface::Trajectory trajectory;
        interface::Trajectory solved_trajectory;

        std::vector<double> command_steer;
        std::vector<double> command_accel_pedal;
        std::vector<double> command_brake_pedal;
        std::vector<double> command_speed;
        std::vector<double> command_acceleration;

    } ControlInputOutput;
 }
 
 #endif // __INTERFACE_PLANNING_CONTROL_HPP__
 