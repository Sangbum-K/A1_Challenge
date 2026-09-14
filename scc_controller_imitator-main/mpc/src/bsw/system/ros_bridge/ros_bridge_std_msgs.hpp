/**
 * @file        ros_bridge_std_msgs.hpp
 * @brief       ROS bridge for std_msgs
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)      
 * 
 * @date        2023-07-14 created by Yuseung Na
 * 
 */

#ifndef __ROS_BRIDGE_STD_MSGS__
#define __ROS_BRIDGE_STD_MSGS__
#pragma once

// ROS header
#include <ros/ros.h>
#include <tf/tf.h>

// ROS Message Header
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>

// Interface Header
#include "interface_std_msgs.hpp"

// Bridge Header

namespace ros_bridge {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    double GetTimeStamp(const ros::Time& stamp) {
        return (double)stamp.sec + (double)stamp.nsec * 1e-9;
    }

    interface::Header GetHeader(const std_msgs::Header& msg) {
        interface::Header header;

        header.seq      = msg.seq;
        header.stamp    = ros_bridge::GetTimeStamp(msg.stamp);
        header.frame_id = msg.frame_id;

        return header;
    }

    double GetFloat32(const std_msgs::Float32& msg) {
        return msg.data;
    }

    int GetInt32(const std_msgs::Int32& msg) {
        return msg.data;
    }

    bool GetBool(const std_msgs::Bool& msg) {
        return msg.data;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    std_msgs::Header UpdateHeader(const interface::Header& header) {
        std_msgs::Header msg;

        msg.seq      = header.seq;
        msg.stamp    = ros::Time(header.stamp);
        msg.frame_id = header.frame_id;

        return msg;
    }

    std_msgs::Float32 UpdateFloat32(const double& data) {
        std_msgs::Float32 msg;
        msg.data = data;

        return msg;
    }

    std_msgs::Int32 UpdateInt32(const int& data) {
        std_msgs::Int32 msg;
        msg.data = data;

        return msg;
    }

    std_msgs::Bool UpdateBool(const bool& data) {
        std_msgs::Bool msg;
        msg.data = data;

        return msg;
    }

} // namespace ros_bridge

#endif  // __ROS_BRIDGE_STD_MSGS__
