/**
 * @file        ros_bridge_simulator.hpp
 * @brief       ROS bridge for simulator
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)
 * @authors     Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * 
 * @date        2024-07-17 created by Yuseung Na
 *              2025-01-02 updated by Jiwon Seok: Add GetMoraiWGS84FromMoraiENU & GetMoraiReference
 */

#ifndef __ROS_BRIDGE_EVALUATION_HPP__
#define __ROS_BRIDGE_EVALUATION_HPP__
#pragma once

// ROS Header
#include <ros/ros.h>
#include <tf/tf.h>

// ROS Message Header
#include <morai_msgs/GetTrafficLightStatus.h>
#include <morai_msgs/CollisionData.h>
#include <autohyu_msgs/TrafficLight.h>
#include <visualization_msgs/MarkerArray.h>
#include <jsk_rviz_plugins/OverlayText.h>

// Interface Header
#include "interface_objects.hpp"

// Function Header

// Bridge Header

// Utility header

namespace ros_bridge {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    interface::TrackObjects GetMoraiCollisionData(const morai_msgs::CollisionData& msg) {
        interface::TrackObjects collision_object;
        collision_object.header = ros_bridge::GetHeader(msg.header);
        collision_object.header.frame_id = "world";

        for (const auto& i_object : msg.collision_object) {
            interface::TrackObject object;
            object.id = (uint32_t)i_object.unique_id;

            if (i_object.type == 0) {
                object.classification = interface::ObjectClass::PEDESTRIAN;
            }
            else if (i_object.type == 1) {
                object.classification = interface::ObjectClass::CAR;
            }
            else if (i_object.type == 2) {
                object.classification = interface::ObjectClass::STATIC_UNKNOWN;
            }

            collision_object.object.push_back(object);
        }

        return collision_object;
    }

    interface::TrackObjects GetCollisionObejct(const autohyu_msgs::TrackObjects& msg) {
        interface::TrackObjects object;
        object.header = ros_bridge::GetHeader(msg.header);
        
        for (const auto& i_object : msg.object) {
            interface::TrackObject collision_object;

            collision_object.classification = static_cast<interface::ObjectClass>(i_object.classification);
            collision_object.id = i_object.id;

            object.object.push_back(collision_object);
        }

        return object;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //   
    autohyu_msgs::TrackObjects UpdateCollisionObject(const interface::TrackObjects& objects) {
        autohyu_msgs::TrackObjects collision_object;
        collision_object.header = ros_bridge::UpdateHeader(objects.header);
        collision_object.header.frame_id = "world";

        for (const auto& i_object : objects.object) {
            autohyu_msgs::TrackObject object;

            object.id = i_object.id;
            if (i_object.classification == interface::ObjectClass::PEDESTRIAN) {
                object.classification = (uint8_t)interface::ObjectClass::PEDESTRIAN;
            }
            else if (i_object.classification == interface::ObjectClass::CAR) {
                object.classification = (uint8_t)interface::ObjectClass::CAR;
            }
            else if (i_object.classification == interface::ObjectClass::STATIC_UNKNOWN) {
                object.classification = (uint8_t)interface::ObjectClass::STATIC_UNKNOWN;
            }

            collision_object.object.push_back(object);
        }
        
        return collision_object;
    }

    jsk_rviz_plugins::OverlayText UpdateEvaluationResult(const std::string& result_text) {
        jsk_rviz_plugins::OverlayText msg;

        double text_rgb[] = {0.9f,0.9f,0.9f,0.7f};
        //Bounding Box
        msg.width = 128;
        msg.height = 128;
        msg.left = 128;
        msg.top = 128;
        msg.bg_color.r = 0.0f;
        msg.bg_color.g = 0.0f;
        msg.bg_color.b = 0.0f;
        msg.bg_color.a = 0.0f;
        //Text
        msg.line_width = 1;
        msg.text_size = 9.0;
        msg.fg_color.r = 0.9;
        msg.fg_color.r = 0.9;
        msg.fg_color.r = 0.9;
        msg.fg_color.r = 0.7;

        msg.text = result_text;

        return msg;
    }

    
} // namespace ros_bridge

#endif // __ROS_BRIDGE_EVALUATION_HPP__