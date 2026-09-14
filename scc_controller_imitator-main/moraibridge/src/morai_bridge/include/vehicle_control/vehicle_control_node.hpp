/**
 * @file        vehicle_control_node.hpp
 * @brief       morai vehicle control node hpp file
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)      
 * 
 * @date        2024-05-16 created by Yuseung Na
 * 
 */

#ifndef __MORAI_VEHICLE_CONTROL_HPP__
#define __MORAI_VEHICLE_CONTROL_HPP__
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
#include "actuator/ego_ctl.hpp"
#include "actuator/turn_signal_lamp_control.hpp"

#include "actuator/ego_ctl2.hpp"

// Parameter Header
#include "vehicle_control_config.hpp"

// Algorithm Header

// Namespace
using namespace ros;
using namespace std;

class VehicleControl : public TaskManager {
    public:
        // Constructor
        explicit VehicleControl(std::string node_name, double period);
        // Destructor
        virtual ~VehicleControl();

        void Run();
        void Publish();
        void ProcessINI();
        void ProcessRosparam();

    private:
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Functions

        // Callback functions for subscribe variables
        inline void CallbackCommandSteer(const std_msgs::Float32::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_command_steer_);
            i_command_steer_ = ros_bridge::GetFloat32(*msg);
            b_is_command_lateral_ = true;
        }
        inline void CallbackCommandPedalAccel(const std_msgs::Float32::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_command_pedal_accel_);
            i_command_pedal_accel_ = ros_bridge::GetFloat32(*msg);
            b_is_command_longitudinal_ = true;
        }
        inline void CallbackCommandPedalBrake(const std_msgs::Float32::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_command_pedal_brake_);
            i_command_pedal_brake_ = ros_bridge::GetFloat32(*msg);
            b_is_command_longitudinal_ = true;
        }
        inline void CallbackPlaTargetSpeed(const std_msgs::Float32::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_pla_target_speed_);
            i_pla_target_speed_ = ros_bridge::GetFloat32(*msg);
            b_is_command_longitudinal_ = true;            
        }
        inline void CallbackCommandAccel(const std_msgs::Float32::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_command_accel_);
            i_command_accel_ = ros_bridge::GetFloat32(*msg);
            if (i_command_accel_ > cfg_.max_accel) {
                i_command_accel_ = cfg_.max_accel;
            }
            else if (i_command_accel_ < cfg_.max_decel) {
                i_command_accel_ = cfg_.max_decel;
            }
            b_is_command_longitudinal_ = true;
        }

        // Update UDP Control Signal
        MoraiCppUdp::EgoControl::ControlCommand UpdateUdpControlSignal(const int & command_type, const double& command_steer, const double& command_pedal_accel, const double& command_pedal_brake, const double& command_velocity, const double& command_accel);
        MoraiCppUdp::TurnSignalLampControl::LampCommand UpdateUdpTurnSignalLamp(const bool& left_on, const bool& right_on, const bool& hazard_on);
        MoraiCppUdp::EgoControl2::ControlCommand2 UpdateUdpControlSignal2(const double& command_pedal_accel, const double& command_pedal_brake, const double& command_steer);

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Variables     

        // Node Handler
        ros::NodeHandle nh_;

        // Subscriber
        ros::Subscriber s_command_steer_;
        ros::Subscriber s_command_pedal_accel_;
        ros::Subscriber s_command_pedal_brake_;
        ros::Subscriber s_pla_target_speed_;
        ros::Subscriber s_command_accel_;

        // Input
        double i_command_steer_;
        double i_command_pedal_accel_;
        double i_command_pedal_brake_;
        double i_pla_target_speed_;
        double i_command_accel_;

        // Mutex
        std::mutex mutex_command_steer_;
        std::mutex mutex_command_pedal_accel_;
        std::mutex mutex_command_pedal_brake_;
        std::mutex mutex_pla_target_speed_;
        std::mutex mutex_command_accel_;

        // Publisher
        ros::Publisher p_control_signal_;
        ros::Publisher p_turn_signal_lamp_;
        bool b_is_publisher_initialized_ = false;

        // UDP Sender
        std::unique_ptr<MoraiCppUdp::EgoControl> morai_udp_ego_ctl_sender_;
        std::unique_ptr<MoraiCppUdp::TurnSignalLampControl> morai_udp_lamps_sender_;
        bool b_is_udp_sender_initialized_ = false;

        std::unique_ptr<MoraiCppUdp::EgoControl2> morai_udp_ego_ctl2_sender_;
        bool b_is_udp_sender2_initialized_ = false;

        // Output
        morai_msgs::CtrlCmd o_control_signal_;
        morai_msgs::Lamps o_turn_signal_lamp_;
        MoraiCppUdp::EgoControl::ControlCommand o_control_signal_udp_;
        MoraiCppUdp::TurnSignalLampControl::LampCommand o_turn_signal_lamp_udp_;
        MoraiCppUdp::EgoControl2::ControlCommand2 o_control_signal_udp2_;

        // Algorithm

        // Util and Configuration
        IniParser util_ini_parser_;
        MoraiVehicleControlConfig cfg_;

        // Flags
        bool b_is_command_lateral_;
        bool b_is_command_longitudinal_;

        // Global Variables
        // ADD GLOBAL VARIABLES HERE
};

#endif  // __MORAI_VEHICLE_CONTROL_HPP__