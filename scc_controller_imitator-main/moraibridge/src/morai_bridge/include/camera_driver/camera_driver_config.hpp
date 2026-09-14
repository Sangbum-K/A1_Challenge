/**
 * @file        camera_driver_config.hpp
 * @brief       configuration hpp file for morai camera driver node
 * 
 * @authors     Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * 
 * @date        2025-01-17 created by Jiwon Seok
 * 
 */

#ifndef __MORAI_CAMERA_DRIVER_CONFIG_HPP__
#define __MORAI_CAMERA_DRIVER_CONFIG_HPP__
#pragma once

#include <string>

#define MAX_NUM_OF_CAMERA 50

typedef struct {
    // Node Config
    double task_period;

    // Simulator
    int morai_simulator_version; // 1: Old unity MORAI simulator, 2: New Unreal MORAI simulator
    // std::string communication_type; // Only UDP
    std::string udp_source_ip;
    std::string udp_destination_ip;

    // Morai Camera Driver
    int num_of_camera;
    bool publish_raw_image;
    bool publish_compressed_image;
    int udp_camera_driver_source_ports[MAX_NUM_OF_CAMERA];
    int udp_camera_driver_destination_ports[MAX_NUM_OF_CAMERA];
    std::string camera_topic_name_list[MAX_NUM_OF_CAMERA];
    std::string camera_topic_frame_id_list[MAX_NUM_OF_CAMERA];
} CameraDriverConfig;

#endif // __MORAI_CAMERA_DRIVER_CONFIG_HPP__
