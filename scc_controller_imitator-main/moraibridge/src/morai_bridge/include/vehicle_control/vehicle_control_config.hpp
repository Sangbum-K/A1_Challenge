/**
 * @file        vehicle_control_config.hpp
 * @brief       configuration hpp file for morai vehicle control node
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)         
 * 
 * @date        2024-05-16 created by Yuseung Na
 * 
 */

#ifndef __MORAI_VEHICLE_CONTROL_CONFIG_HPP__
#define __MORAI_VEHICLE_CONTROL_CONFIG_HPP__
#pragma once

// STD Header
#include <string>

typedef struct {  
    // Node Config
    double task_period;

    // Simulator
    int morai_simulator_version; // 1: Old unity MORAI simulator, 2: New Unreal MORAI simulator
    std::string communication_control_type;
    std::string udp_source_ip;
    std::string udp_destination_ip;

    // Vehicle Control
    double max_accel{5.0};
    double max_decel{-10.0};
    int udp_ego_ctl_source_port;
    int udp_ego_ctl_destination_port;
    int udp_turn_signal_lamp_source_port;
    int udp_turn_signal_lamp_destination_port;
    bool udp_morai_ros_multi_publish;
    int udp_morai_autonomous_mode;
    int sim_steer_display_type;
    double sim_steer_range_max;
    int control_type; // 1: Throttle control, 2: Velocity control, 3: Acceleration control
} MoraiVehicleControlConfig;

#endif // __MORAI_VEHICLE_CONTROL_CONFIG_HPP__