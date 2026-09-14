/**
 * @file        camera_driver_node.hpp
 * @brief       camera driver node header file
 * @authors     Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * 
 * @date        2025-01-17 created by Jiwon Seok
 * 
 */

#ifndef __MORAI_CAMERA_DRIVER_NODE_HPP__
#define __MORAI_CAMERA_DRIVER_NODE_HPP__
#pragma once

// STD Header
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

// System Header
#include "task_manager.hpp"
#include "ini_parser.h"

// Bridge Header
#include "ros_bridge_simulator.hpp"

// Morai UDP Header
#include "sensors/camera.hpp"
#include "sensors/camera2.hpp"
#include "sensor_msgs/CompressedImage.h"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// Parameter Header
#include "camera_driver_config.hpp"

// Algorithm Header

// Namespace
using namespace ros;
using namespace tf;
using namespace std;

class CameraDriver : public TaskManager {
public:
    // Constructor
    explicit CameraDriver(std::string node_name, double period);
    // Destructor
    virtual ~CameraDriver();

    void Run();
    void Publish();
    void ProcessINI();
    void ProcessRosparam();

private:
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Functions

    // Callback functions for subscribe variables
    void CallbackMoraiUdpCamera(const MoraiCppUdp::Camera::CameraData& msg, int camera_index);
    void CallbackMoraiUdpCamera2(const MoraiCppUdp::Camera2::CameraData& msg, int camera_index);

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Variables

    // Node Handler
    ros::NodeHandle nh_;

    // Subscriber

    // Morai UDP Receiver
    std::vector<std::unique_ptr<MoraiCppUdp::Camera>> morai_udp_camera_receivers_;
    std::vector<std::unique_ptr<MoraiCppUdp::Camera2>> morai_udp_camera2_receivers_;

    // Morai UDP Inputs
    std::vector<MoraiCppUdp::Camera::CameraData> i_morai_udp_camera_;
    std::vector<MoraiCppUdp::Camera2::CameraData> i_morai_udp_camera2_;

    // Publisher
    std::vector<ros::Publisher> p_cameras_;
    std::vector<ros::Publisher> p_compressed_cameras_;
    bool b_is_publisher_initialized_ = false;

    // Output

    // Algorithm

    // Util and Configuration
    IniParser util_ini_parser_;
    CameraDriverConfig cfg_;

    // Flags

    // Global Variables
    // ADD GLOBAL VARIABLES HERE
};

#endif // __MORAI_CAMERA_DRIVER_NODE_HPP__
