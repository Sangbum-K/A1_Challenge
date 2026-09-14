/**
 * @file        vehicle_driver_config.hpp
 * @brief       configuration hpp file for morai vehicle driver node
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)
 * @authors     Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * 
 * @date        2024-05-16 created by Yuseung Na
 * @date        2024-12-26 Add Morai UDP configurations by Jiwon Seok
 * 
 */

#ifndef __MORAI_VEHICLE_DRIVER_CONFIG_HPP__
#define __MORAI_VEHICLE_DRIVER_CONFIG_HPP__
#pragma once

// STD Header
#include <string>

typedef struct {  
    // Node Config
    double task_period;
    double ref_lat;
    double ref_lon;
    std::string location;
    std::string projection_mode;

    // Simulator
    int morai_simulator_version; // 1: Old unity MORAI simulator, 2: New Unreal MORAI simulator
    std::string communication_pubsubser_type;
    std::string udp_source_ip; // Simulator IP ("127.0.0.1")
    std::string udp_destination_ip; // Ego IP ("127.0.0.1")

    // Morai Vehicle Driver
    bool use_noisy_vehicle;
    bool use_morai_enu_position;

    int udp_vehicle_info_source_port;
    int udp_vehicle_info_destination_port;
    int udp_vehicle_noisy_info_source_port;
    int udp_vehicle_noisy_info_destination_port;
    int udp_gps_source_port;
    int udp_gps_destination_port;
    bool udp_morai_ros_republish;

    int sim_steer_display_type;
    double sim_steer_range_max;
    double cg_position_x_from_rear;
} MoraiVehicleDriverConfig;

#endif // __MORAI_VEHICLE_DRIVER_CONFIG_HPP__
