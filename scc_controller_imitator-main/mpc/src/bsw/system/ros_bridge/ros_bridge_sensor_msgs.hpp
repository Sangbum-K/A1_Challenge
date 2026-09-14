/**
 * @file        ros_bridge_sensor_msgs.hpp
 * @brief       ROS bridge for sensor_msgs
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)   
 *              Seheon Ha (seheonha@hanyang.ac.kr)          
 * 
 * @date        2024-04-24 created by Yuseung Na
 *              2024-04-24 updated by Seheon ha: Add PointCloud2
 *              2024-04-24 updated by Seongjae Jeong: Add Image
 */

#ifndef __ROS_BRIDGE_SENSOR_MSGS__
#define __ROS_BRIDGE_SENSOR_MSGS__
#pragma once

// ROS header
#include <ros/ros.h>
#include <tf/tf.h>

// ROS Message Header
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>

// Interface Header
#include "interface_sensor_msgs.hpp"

// Bridge Header
#include "ros_bridge_std_msgs.hpp"

namespace ros_bridge {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    interface::PointCloud2 GetPointCloud2(const sensor_msgs::PointCloud2& msg) {
        interface::PointCloud2 pointcloud2;
        pointcloud2.header = ros_bridge::GetHeader(msg.header);

        pointcloud2.height = msg.height;
        pointcloud2.width  = msg.width;

        for (auto field : msg.fields) {
            interface::PointField point_field;

            point_field.name        = field.name;
            point_field.offset      = field.offset;
            point_field.datatype    = static_cast<interface::PointFieldDatatype>(field.datatype);
            point_field.count       = field.count;

            pointcloud2.fields.push_back(point_field);
        }

        pointcloud2.is_bigendian = msg.is_bigendian;
        pointcloud2.point_step   = msg.point_step;
        pointcloud2.row_step     = msg.row_step;
        pointcloud2.data         = msg.data;

        pointcloud2.is_dense = msg.is_dense;

        return pointcloud2;
    }

    interface::Image GetImage(const sensor_msgs::Image& msg) {
        interface::Image image;
        image.header = ros_bridge::GetHeader(msg.header);

        image.height        = msg.height;
        image.width         = msg.width;

        image.encoding      = msg.encoding;
        
        image.is_bigendian  = msg.is_bigendian;
        image.step          = msg.step;
        image.data          = msg.data;

        return image;
    }
} // namespace ros_bridge

#endif  // __ROS_BRIDGE_SENSOR_MSGS__