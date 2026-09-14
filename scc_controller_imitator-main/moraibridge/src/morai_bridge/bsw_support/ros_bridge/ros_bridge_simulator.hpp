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

#ifndef __ROS_BRIDGE_SIMULATOR_HPP__
#define __ROS_BRIDGE_SIMULATOR_HPP__
#pragma once

// ROS Header
#include <ros/ros.h>
#include <tf/tf.h>

// ROS Message Header
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>

#include <morai_msgs/EgoVehicleStatus.h>
#include <morai_msgs/GPSMessage.h>
#include <morai_msgs/ObjectStatus.h>
#include <morai_msgs/ObjectStatusList.h>
#include <morai_msgs/CtrlCmd.h>
#include <morai_msgs/GetTrafficLightStatus.h>
#include <morai_msgs/Lamps.h>
#include <morai_msgs/MultiEgoSetting.h>

#include <morai_msgs/ScenarioLoad.h>    
#include <morai_msgs/MoraiScenarioLoadSrv.h>
#include <morai_msgs/MoraiSrvResponse.h>
#include <morai_msgs/MoraiSimProcStatus.h>

#include <autohyu_msgs/Reference.h>
#include <autohyu_msgs/VehicleState.h>
#include <autohyu_msgs/TrackObjects.h>
#include <autohyu_msgs/TrafficLight.h>
#include <visualization_msgs/MarkerArray.h>

// Interface Header
#include "interface_vehicle_state.hpp"
#include "interface_sensor_msgs.hpp"
#include "interface_objects.hpp"
#include "interface_trajectories.hpp"

// Function Header
#include "function_geometry.hpp"
#include "function_print.hpp"

// Bridge Header
#include "ros_bridge_vehicle_state.hpp"

// Utility header
#include <lanelet2_core/geometry/LaneletMap.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_io/Io.h>
#include <lanelet2_projection/LocalCartesian.h>
#include <lanelet2_projection/UTM.h>

// MORAI_CPP_NETWORK_MODULE
#include <actuator/ego_ctl.hpp>
#include <actuator/turn_signal_lamp_control.hpp>
#include <sensors/camera.hpp>
#include <sensors/gps.hpp>
#include <sensors/imu.hpp>
#include <sensors/object_info.hpp>
#include <sensors/traffic_light.hpp>
#include <sensors/vehicle_state.hpp>
#include <simulator/multi_ego_setting.hpp>

namespace ros_bridge {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    interface::VehicleState GetMoraiEgoVehicleStatus(const morai_msgs::EgoVehicleStatus& msg) {
        interface::VehicleState vehicle_state;
        vehicle_state.header = ros_bridge::GetHeader(msg.header);
        vehicle_state.header.frame_id = "world";

        // Gnss
        vehicle_state.gnss.latitude  = 0.0; // Not Used in Vehicle Status
        vehicle_state.gnss.longitude = 0.0; // Not Used in Vehicle Status
        vehicle_state.gnss.altitude  = 0.0; // Not Used in Vehicle Status

        // VehicleState
        vehicle_state.x = msg.position.x;
        vehicle_state.y = msg.position.y;
        vehicle_state.z = msg.position.z;

        vehicle_state.vx = msg.velocity.x;
        vehicle_state.vy = msg.velocity.y;
        vehicle_state.vz = msg.velocity.z;

        vehicle_state.ax = msg.acceleration.x;
        vehicle_state.ay = msg.acceleration.y;
        vehicle_state.az = msg.acceleration.z;

        vehicle_state.roll  = 0.0; // Not Used in Vehicle Status
        vehicle_state.pitch = 0.0; // Not Used in Vehicle Status
        vehicle_state.yaw   = msg.heading * interface::DEG2RAD;

        vehicle_state.roll_vel  = 0.0; // Not Used in Vehicle Status
        vehicle_state.pitch_vel = 0.0; // Not Used in Vehicle Status
        vehicle_state.yaw_vel   = 0.0; // Not Used in Vehicle Status

        // VehicleCAN
        vehicle_state.vehicle_can.header = ros_bridge::GetHeader(msg.header);
        vehicle_state.vehicle_can.header.frame_id = "can";

        vehicle_state.vehicle_can.steering_wheel_angle = msg.wheel_angle * interface::DEG2RAD * interface::STEERING_RATIO;
        vehicle_state.vehicle_can.steering_tire_angle  = msg.wheel_angle * interface::DEG2RAD;
        vehicle_state.vehicle_can.steering_state       = interface::MdpsState::MDPS_ACTIVATE;

        vehicle_state.vehicle_can.accel_position       = msg.accel * 100.0;
        vehicle_state.vehicle_can.brake_position       = msg.brake * 100.0;
        vehicle_state.vehicle_can.gear_select          = interface::Gear::GEAR_D;

        vehicle_state.vehicle_can.operation_mode       = interface::OperationMode::AUTONOMOUS;
        vehicle_state.vehicle_can.lateral_ad_mode      = interface::AutonomousMode::RUN;
        vehicle_state.vehicle_can.longitudinal_ad_mode = interface::AutonomousMode::RUN;

        return vehicle_state;
    }

    interface::VehicleState GetMoraiGPSMessage(const morai_msgs::GPSMessage& msg,
                                               const autohyu_msgs::Reference& reference) {
        interface::VehicleState vehicle_state;
        vehicle_state.header = ros_bridge::GetHeader(msg.header);
        vehicle_state.header.frame_id = "world";

        vehicle_state.gnss.latitude  = msg.latitude;
        vehicle_state.gnss.longitude = msg.longitude;
        vehicle_state.gnss.altitude  = msg.altitude;

        lanelet::GPSPoint i_gps_point;
        i_gps_point.lat = msg.latitude;
        i_gps_point.lon = msg.longitude;
        i_gps_point.ele = 0.0;

        lanelet::Origin origin({reference.wgs84.latitude, reference.wgs84.longitude});

        if (i_gps_point.lat != 0.0 && i_gps_point.lon != 0.0) {
            if (reference.projection == "utm") {
                lanelet::projection::UtmProjector projector(origin);
                lanelet::BasicPoint3d proj_position = projector.forward(i_gps_point);
                vehicle_state.x = proj_position.x();
                vehicle_state.y = proj_position.y();
                vehicle_state.z = 0.0;
            }
            else if (reference.projection == "local_cartesian") {
                lanelet::projection::LocalCartesianProjector projector(origin);
                lanelet::BasicPoint3d proj_position = projector.forward(i_gps_point);
                vehicle_state.x = proj_position.x();
                vehicle_state.y = proj_position.y();
                vehicle_state.z = 0.0;
            }
        }
        else {
            vehicle_state.x = 0.0;
            vehicle_state.y = 0.0;
            vehicle_state.z = 0.0;
        }

        return vehicle_state;
    }

    double GetRearOverhang(const std::string& model, const std::unordered_map<std::string, double>& rear_overhang_map) {
        auto it = rear_overhang_map.find(model);
        if (it != rear_overhang_map.end()) {
            return it->second;
        }

        util_function::DebugPrintError(
            "Overhang not found for model: " + model
        );
        return 0.0;
    }

    interface::TrackObjects GetMoraiObjects(const morai_msgs::ObjectStatusList& msg,
                                            const interface::Reference& reference,
                                            const std::unordered_map<std::string, double>& rear_overhang_map) {
        interface::TrackObjects objects;
        objects.header = GetHeader(msg.header);
        objects.header.frame_id = "world";

        lanelet::Origin origin({reference.wgs84.latitude, reference.wgs84.longitude});
        lanelet::projection::UtmProjector utm_projector(origin);
        lanelet::projection::LocalCartesianProjector local_cartesian_projector(origin);

        for (const auto& i_object : msg.npc_list) {
            interface::TrackObject object;
            object.id = i_object.unique_id;
            object.classification = interface::ObjectClass::CAR;
            object.dynamic_state = interface::ObjectDynamicState::DYNAMIC;

            object.dimension.length = i_object.size.x;
            object.dimension.width  = i_object.size.y;
            object.dimension.height = i_object.size.z;

            object.state.header = GetHeader(msg.header);

            // Convert ego_yaw to radians
            double theta = i_object.heading * interface::DEG2RAD;

            // Rotation matrix elements
            double cos_theta = cos(theta);
            double sin_theta = sin(theta);

            // Perform rotation
            double global_vx = i_object.velocity.x * interface::KPH2MPS * cos_theta 
                                - i_object.velocity.y * interface::KPH2MPS * sin_theta;
            double global_vy = i_object.velocity.x * interface::KPH2MPS * sin_theta 
                                + i_object.velocity.y * interface::KPH2MPS * cos_theta;

            double global_ax = i_object.acceleration.x * cos_theta 
                                - i_object.acceleration.y * sin_theta;
            double global_ay = i_object.acceleration.x * sin_theta 
                                + i_object.acceleration.y * cos_theta;

            object.state.header = objects.header;
            object.state.yaw    = i_object.heading * interface::DEG2RAD;
            object.state.vx    = global_vx;
            object.state.vy    = global_vy;
            object.state.ax    = global_ax;
            object.state.ay    = global_ay;

            // 1. Morai Postion to GPS (reverse projection)
            lanelet::BasicPoint3d object_morai_position(i_object.position.x,i_object.position.y,i_object.position.z);
            lanelet::GPSPoint object_gps_point = utm_projector.reverse(object_morai_position);
            
            // 2. GPS to World Position (forward projection)
            lanelet::BasicPoint3d object_autohyu_projpos;
            if(reference.projection == "utm") {
                object_autohyu_projpos = utm_projector.forward(object_gps_point);
            }
            else if(reference.projection == "local_cartesian") {
                object_autohyu_projpos = local_cartesian_projector.forward(object_gps_point);
            }

            double rear_overhang = GetRearOverhang(i_object.name, rear_overhang_map);
            double offset = object.dimension.length/2.0 - rear_overhang;
            
            object.state.x = object_autohyu_projpos.x() + offset * cos(object.state.yaw);
            object.state.y = object_autohyu_projpos.y() + offset * sin(object.state.yaw);

            objects.object.push_back(object);
        }

        for (const auto& i_object : msg.pedestrian_list) {
            interface::TrackObject object;
            object.id = i_object.unique_id;
            object.classification = interface::ObjectClass::PEDESTRIAN;

            object.dimension.length = i_object.size.x;
            object.dimension.width  = i_object.size.y;
            object.dimension.height = i_object.size.z;

            object.state.header = GetHeader(msg.header);

            // Convert ego_yaw to radians
            double theta = i_object.heading * interface::DEG2RAD;

            // Rotation matrix elements
            double cos_theta = cos(theta);
            double sin_theta = sin(theta);

            // Perform rotation
            double global_vx = i_object.velocity.x * interface::KPH2MPS * cos_theta 
                                - i_object.velocity.y * interface::KPH2MPS * sin_theta;
            double global_vy = i_object.velocity.x * interface::KPH2MPS * sin_theta 
                                + i_object.velocity.y * interface::KPH2MPS * cos_theta;
            
            double global_v = sqrt(global_vx * global_vx + global_vy * global_vy);
            if (global_v * interface::MPS2KPH > 1.0) {
                object.dynamic_state = interface::ObjectDynamicState::DYNAMIC;
            }
            else {
                object.dynamic_state = interface::ObjectDynamicState::STATIC;
            }

            double global_ax = i_object.acceleration.x * cos_theta 
                                - i_object.acceleration.y * sin_theta;
            double global_ay = i_object.acceleration.x * sin_theta 
                                + i_object.acceleration.y * cos_theta;

            object.state.header = objects.header;
            object.state.yaw    = i_object.heading * interface::DEG2RAD;
            object.state.vx    = global_vx;
            object.state.vy    = global_vy;
            object.state.ax    = global_ax;
            object.state.ay    = global_ay;

            // 1. Morai Postion to GPS (reverse projection)
            lanelet::BasicPoint3d object_morai_position(i_object.position.x,i_object.position.y,i_object.position.z);
            lanelet::GPSPoint object_gps_point = utm_projector.reverse(object_morai_position);
            
            // 2. GPS to World Position (forward projection)
            lanelet::BasicPoint3d object_autohyu_projpos;
            if(reference.projection == "utm") {
                object_autohyu_projpos = utm_projector.forward(object_gps_point);
            }
            else if(reference.projection == "local_cartesian") {
                object_autohyu_projpos = local_cartesian_projector.forward(object_gps_point);
            }
            
            object.state.x = object_autohyu_projpos.x();
            object.state.y = object_autohyu_projpos.y();

            objects.object.push_back(object);
        }

        for (const auto& i_object : msg.obstacle_list) {
            interface::TrackObject object;
            double offset = 0.0;
            object.id = i_object.unique_id;
            if (i_object.name == "Bollard01") {
                object.classification = interface::ObjectClass::STATIC_BOLLARD;
                object.dynamic_state = interface::ObjectDynamicState::STATIC;
            }
            else if (i_object.name == "Rubber_Cone") {
                object.classification = interface::ObjectClass::STATIC_CONE;
                object.dynamic_state = interface::ObjectDynamicState::STATIC;
            }
            else if (i_object.name == "CargoBox") {
                object.classification = interface::ObjectClass::STATIC_BOX;
                object.dynamic_state = interface::ObjectDynamicState::STATIC;
            }
            else if (i_object.name == "Concrete_Barricade01") {
                object.classification = interface::ObjectClass::STATIC_BARRICADE_CONCRETE;
                object.dynamic_state = interface::ObjectDynamicState::STATIC;
            }
            else if (i_object.name == "PE_Barricade_Orange") {
                object.classification = interface::ObjectClass::STATIC_BARRICADE_PLASTIC;
                object.dynamic_state = interface::ObjectDynamicState::STATIC;
            }
            else if (i_object.name == "Rockslide") {
                object.classification = interface::ObjectClass::STATIC_ROCK;
                object.dynamic_state = interface::ObjectDynamicState::STATIC;
            }
            else if (i_object.name == "obj_volvo_excavator") {
                object.classification = interface::ObjectClass::STATIC_CONSTRUCTION;
                object.dynamic_state = interface::ObjectDynamicState::STATIC;

                double rear_overhang = GetRearOverhang(i_object.name, rear_overhang_map);
                offset = i_object.size.x/2.0 - rear_overhang;
            }
            else if (i_object.name == "OBJ_Hyundai_Xcient_Mixer") {
                object.classification = interface::ObjectClass::STATIC_MIXER;
                object.dynamic_state = interface::ObjectDynamicState::STATIC;

                double rear_overhang = GetRearOverhang(i_object.name, rear_overhang_map);
                offset = i_object.size.x/2.0 - rear_overhang;
            }
            else if (i_object.name == "OBJ_Hyundai_Universe_High") {
                object.classification = interface::ObjectClass::STATIC_BUS;
                object.dynamic_state = interface::ObjectDynamicState::STATIC;

                double rear_overhang = GetRearOverhang(i_object.name, rear_overhang_map);
                offset = i_object.size.x/2.0 - rear_overhang;
            }
            else {
                object.classification = interface::ObjectClass::STATIC_UNKNOWN;
                object.dynamic_state = interface::ObjectDynamicState::STATIC;
            }

            object.dimension.length = i_object.size.x;
            object.dimension.width  = i_object.size.y;
            object.dimension.height = i_object.size.z;

            object.state.header = GetHeader(msg.header);

            // Convert ego_yaw to radians
            double theta = i_object.heading * interface::DEG2RAD;

            // Perform rotation
            double global_vx = 0.0;
            double global_vy = 0.0;

            double global_ax = 0.0;
            double global_ay = 0.0;

            object.state.header = objects.header;
            object.state.yaw    = i_object.heading * interface::DEG2RAD;
            object.state.vx    = global_vx;
            object.state.vy    = global_vy;
            object.state.ax    = global_ax;
            object.state.ay    = global_ay;

            // 1. Morai Postion to GPS (reverse projection)
            lanelet::BasicPoint3d object_morai_position(i_object.position.x,i_object.position.y,i_object.position.z);
            lanelet::GPSPoint object_gps_point = utm_projector.reverse(object_morai_position);
            
            // 2. GPS to World Position (forward projection)
            lanelet::BasicPoint3d object_autohyu_projpos;
            if(reference.projection == "utm") {
                object_autohyu_projpos = utm_projector.forward(object_gps_point);
            }
            else if(reference.projection == "local_cartesian") {
                object_autohyu_projpos = local_cartesian_projector.forward(object_gps_point);
            } 

            object.state.x = object_autohyu_projpos.x() + offset * cos(object.state.yaw);
            object.state.y = object_autohyu_projpos.y() + offset * sin(object.state.yaw);

            objects.object.push_back(object);
        }

        return objects;
    }

    interface::TrafficLight GetMoraiTrafficLight(const morai_msgs::GetTrafficLightStatus& msg) {
        interface::TrafficLight traffic_light;
        traffic_light.header.stamp = ros_bridge::GetTimeStamp(msg.header.stamp);
        traffic_light.header.frame_id = "world";

        traffic_light.id    = msg.trafficLightIndex;
        traffic_light.type  = static_cast<interface::TrafficLightType>(msg.trafficLightType);
        traffic_light.color = static_cast<interface::TrafficLightColor>(msg.trafficLightStatus);

        return traffic_light;
    }

    interface::WGS84 GetMoraiWGS84FromMoraiENU(const interface::Reference& morai_ref, const interface::Point& enu) {
        return util_function::GetWgs84(morai_ref, enu);
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //   
    autohyu_msgs::VehicleState UpdateVehicleState(const interface::VehicleState& vehicle_state,
                                                  const interface::VehicleState& gps,
                                                  const interface::Header& header) {
        autohyu_msgs::VehicleState msg;
        msg.header = ros_bridge::UpdateHeader(header);

        msg.gnss   = ros_bridge::UpdateWGS84(gps.gnss);
        
        msg.x = gps.x;
        msg.y = gps.y;
        msg.z = gps.z;

        msg.vx = vehicle_state.vx;
        msg.vy = vehicle_state.vy;
        msg.vz = vehicle_state.vz;

        msg.ax = vehicle_state.ax;
        msg.ay = vehicle_state.ay;
        msg.az = vehicle_state.az;

        msg.roll  = 0.0;    // Not Included in Vehicle State
        msg.pitch = 0.0;    // Not Included in Vehicle State
        msg.yaw   = vehicle_state.yaw;

        msg.roll_vel  = 0.0;    // Not Included in Vehicle State
        msg.pitch_vel = 0.0;    // Not Included in Vehicle State
        msg.yaw_vel   = 0.0;    // Not Included in Vehicle State

        msg.vehicle_can = ros_bridge::UpdateVehicleCAN(vehicle_state.vehicle_can);
        
        return msg;
    }

    morai_msgs::CtrlCmd UpdateControlCommand(const int & command_type, const double& command_steer, const double& command_pedal_accel, const double& command_pedal_brake, const double& command_velocity, const double& command_accel) {
        morai_msgs::CtrlCmd msg;
        msg.longlCmdType = command_type;
        
        msg.steering     = command_steer * interface::DEG2RAD;

        if(command_type == 1) { // Throttle control
            msg.accel = command_pedal_accel;
            msg.brake = command_pedal_brake;
        } else if(command_type == 2) { // Velocity control
            msg.velocity = command_velocity;
        } else if(command_type == 3) { // Acceleration control
            msg.acceleration = command_accel;
        }

        return msg;
    }

    morai_msgs::Lamps UpdateTurnSignalLamp(const bool& left, 
                                           const bool& right, 
                                           const bool& emergency) {
        morai_msgs::Lamps msg;

        msg.header.stamp = ros::Time::now();

        msg.turnSignal = 0;         // 0: None, 1: Left, 2: Right
        msg.emergencySignal = 0;    // 0: None, 1: Hazard

        if (left) {
            msg.turnSignal = 1;
        }
        else if (right) {
            msg.turnSignal = 2;
        }
        else if (emergency) {
            msg.emergencySignal = 1;
        }

        return msg;
    }
} // namespace ros_bridge

#endif // __ROS_BRIDGE_SIMULATOR_HPP__