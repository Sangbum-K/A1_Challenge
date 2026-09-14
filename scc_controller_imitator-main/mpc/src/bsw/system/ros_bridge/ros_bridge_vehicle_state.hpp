/**
 * @file        ros_bridge_vehicle_state.hpp
 * @brief       ROS bridge for vehicle state
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-01 updated by Yuseung Na: Refactoring 
 */

#ifndef __ROS_BRIDGE_VEHICLE_STATE__
#define __ROS_BRIDGE_VEHICLE_STATE__
#pragma once

// ROS header
#include <ros/ros.h>
#include <tf/tf.h>

// ROS Message Header
#include <autohyu_msgs/VehicleState.h>
#include <autohyu_msgs/Reference.h>
#include <morai_msgs/EgoVehicleStatus.h>
#include <morai_msgs/EgoNoisyStatus.h>

// Interface Header
#include "interface_constants.hpp"
#include "interface_vehicle_state.hpp"

// Bridge Header
#include "ros_bridge_std_msgs.hpp"

namespace ros_bridge {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    interface::VehicleCAN GetVehicleCAN(const autohyu_msgs::VehicleCAN& msg) {
        interface::VehicleCAN vehicle_can;
        vehicle_can.header = GetHeader(msg.header);

        vehicle_can.steering_wheel_angle    = msg.steering_wheel_angle;
        vehicle_can.steering_tire_angle     = msg.steering_tire_angle;
        vehicle_can.steering_state          = static_cast<interface::MdpsState>(msg.steering_state);

        vehicle_can.accel_position          = msg.accel_position;
        vehicle_can.brake_position          = msg.brake_position;
        vehicle_can.gear_select             = static_cast<interface::Gear>(msg.gear_select);

        vehicle_can.operation_mode          = static_cast<interface::OperationMode>(msg.operation_mode);
        vehicle_can.lateral_ad_mode         = static_cast<interface::AutonomousMode>(msg.lateral_ad_mode);
        vehicle_can.longitudinal_ad_mode    = static_cast<interface::AutonomousMode>(msg.longitudinal_ad_mode);

        return vehicle_can;
    }

    interface::WGS84 GetWGS84(const autohyu_msgs::WGS84& msg) {
        interface::WGS84 wgs84;
        wgs84.latitude  = msg.latitude;
        wgs84.longitude = msg.longitude;
        wgs84.altitude  = msg.altitude;

        return wgs84;
    }

    interface::Reference GetReference(const autohyu_msgs::Reference& msg) {
        interface::Reference reference;
        reference.projection        = msg.projection;
        reference.wgs84.latitude    = msg.wgs84.latitude;
        reference.wgs84.longitude   = msg.wgs84.longitude;
        reference.wgs84.altitude    = msg.wgs84.altitude;

        return reference;
    }

    interface::VehicleState GetVehicleState(const autohyu_msgs::VehicleState& msg) {
        interface::VehicleState vehicle_state;
        vehicle_state.header    = GetHeader(msg.header);

        vehicle_state.gnss          = GetWGS84(msg.gnss);

        vehicle_state.x = msg.x;
        vehicle_state.y = msg.y;
        vehicle_state.z = msg.z;

        vehicle_state.vx = msg.vx;
        vehicle_state.vy = msg.vy;
        vehicle_state.vz = msg.vz;

        vehicle_state.ax = msg.ax;
        vehicle_state.ay = msg.ay;
        vehicle_state.az = msg.az;

        vehicle_state.roll = msg.roll;
        vehicle_state.pitch = msg.pitch;
        vehicle_state.yaw = msg.yaw;

        vehicle_state.roll_vel = msg.roll_vel;
        vehicle_state.pitch_vel = msg.pitch_vel;
        vehicle_state.yaw_vel = msg.yaw_vel;

        vehicle_state.link_id = msg.link_id;

        vehicle_state.vehicle_can = GetVehicleCAN(msg.vehicle_can);

        return vehicle_state;
    }

    interface::VehicleState GetMoraiEgoStatus(const morai_msgs::EgoVehicleStatus& msg) {
        interface::VehicleState vehicle_state;
        vehicle_state.header = GetHeader(msg.header);

        vehicle_state.gnss.latitude  = -99.9;   // cannot get latlon by /Ego_topic
        vehicle_state.gnss.longitude = -99.9;   // cannot get latlon by /Ego_topic
        vehicle_state.gnss.altitude  = -99.9;   // cannot get latlon by /Ego_topic

        vehicle_state.x = msg.position.x;
        vehicle_state.y = msg.position.y;
        vehicle_state.z = msg.position.z;

        vehicle_state.vx = msg.velocity.x;
        vehicle_state.vy = msg.velocity.y;
        vehicle_state.vz = msg.velocity.z;

        vehicle_state.ax = msg.acceleration.x;
        vehicle_state.ay = msg.acceleration.y;
        vehicle_state.az = msg.acceleration.z;

        vehicle_state.roll = 0.0;               // cannot get by /Ego_topic
        vehicle_state.pitch = 0.0;              // cannot get by /Ego_topic
        vehicle_state.yaw = msg.heading * interface::DEG2RAD;

        vehicle_state.roll_vel = 0.0;           // cannot get by /Ego_topic
        vehicle_state.pitch_vel = 0.0;          // cannot get by /Ego_topic
        vehicle_state.yaw_vel = 0.0;            // cannot get by /Ego_topic

        // init vehicle_cat
        interface::VehicleCAN vehicle_can;
        vehicle_can.header = GetHeader(msg.header);

        vehicle_can.steering_wheel_angle    = 0.0;      // cannot get by /Ego_topic
        vehicle_can.steering_tire_angle     = msg.wheel_angle;
        vehicle_can.steering_state          = interface::MdpsState::MDPS_READY;

        vehicle_can.accel_position          = msg.accel;
        vehicle_can.brake_position          = msg.brake;
        vehicle_can.gear_select             = interface::Gear::GEAR_D;

        vehicle_can.operation_mode          = interface::OperationMode::AUTONOMOUS;
        vehicle_can.lateral_ad_mode         = interface::AutonomousMode::READY;
        vehicle_can.longitudinal_ad_mode    = interface::AutonomousMode::READY;

        vehicle_state.vehicle_can = vehicle_can;

        return vehicle_state;
    }
    
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    autohyu_msgs::VehicleCAN UpdateVehicleCAN(const interface::VehicleCAN& vehicle_can) {
        autohyu_msgs::VehicleCAN msg;
        msg.header = UpdateHeader(vehicle_can.header);

        msg.steering_wheel_angle    = vehicle_can.steering_wheel_angle;
        msg.steering_tire_angle     = vehicle_can.steering_tire_angle;
        msg.steering_state          = static_cast<uint8_t>(vehicle_can.steering_state);
        
        msg.accel_position          = vehicle_can.accel_position;
        msg.brake_position          = vehicle_can.brake_position;
        msg.gear_select             = static_cast<uint8_t>(vehicle_can.gear_select);
        
        msg.operation_mode          = static_cast<uint8_t>(vehicle_can.operation_mode);
        msg.lateral_ad_mode         = static_cast<uint8_t>(vehicle_can.lateral_ad_mode);
        msg.longitudinal_ad_mode    = static_cast<uint8_t>(vehicle_can.longitudinal_ad_mode);

        return msg;
    }

    autohyu_msgs::WGS84 UpdateWGS84(const interface::WGS84& wgs84) {
        autohyu_msgs::WGS84 msg;
        msg.latitude = wgs84.latitude;
        msg.longitude = wgs84.longitude;
        msg.altitude = wgs84.altitude;

        return msg;
    }

    autohyu_msgs::Reference UpdateReference(const interface::Reference& reference) {
        autohyu_msgs::Reference msg;
        msg.projection = reference.projection;
        msg.wgs84 = UpdateWGS84(reference.wgs84);

        return msg;
    }

    autohyu_msgs::VehicleState UpdateVehicleState(const interface::VehicleState& vehicle_state) {
        autohyu_msgs::VehicleState msg;
        msg.header = UpdateHeader(vehicle_state.header);

        msg.gnss        = UpdateWGS84(vehicle_state.gnss);

        msg.x = vehicle_state.x;
        msg.y = vehicle_state.y;
        msg.z = vehicle_state.z;

        msg.vx = vehicle_state.vx;
        msg.vy = vehicle_state.vy;
        msg.vz = vehicle_state.vz;

        msg.ax = vehicle_state.ax;
        msg.ay = vehicle_state.ay;
        msg.az = vehicle_state.az;

        msg.roll = vehicle_state.roll;
        msg.pitch = vehicle_state.pitch;
        msg.yaw = vehicle_state.yaw;

        msg.roll_vel = vehicle_state.roll_vel;
        msg.pitch_vel = vehicle_state.pitch_vel;
        msg.yaw_vel = vehicle_state.yaw_vel;

        msg.link_id = vehicle_state.link_id;

        msg.vehicle_can = UpdateVehicleCAN(vehicle_state.vehicle_can);

        return msg;
    }

} // namespace ros_bridge

#endif  // __ROS_BRIDGE_VEHICLE_STATE__