/**
 * @file        lateral_control.hpp
 * @brief       lateral control node
 * 
 * @authors     Junhee Lee (998jun@gmail.com)         
 */

#ifndef __LATERAL_CONTROL_NODE_HPP__
#define __LATERAL_CONTROL_NODE_HPP__
#pragma once

// STD Header
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

// ROS Header
#include <ros/ros.h>

// ROS Message Header
#include <autohyu_msgs/VehicleState.h>
#include <autohyu_msgs/Trajectory.h>
#include <autohyu_msgs/ControlInfo.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/MarkerArray.h>

#include <morai_msgs/EgoVehicleStatus.h>

// System Header
#include "task_manager.hpp"
#include "ini_parser.h"

// Bridge Header
#include "ros_bridge_control.hpp"
#include "ros_bridge_trajectories.hpp"
#include "ros_bridge_vehicle_state.hpp"

// Parameter Header
#include "lateral_control_config.hpp"

// Algorithm Header
#include "spline.h"
#include "lateral_control/lateral_control.hpp"

// Namespace
using namespace ros;
using namespace tf;
using namespace std;

class LateralControlNode : public TaskManager {
    public:
        // Constructor
        explicit LateralControlNode(std::string task_node, double period);
        // Destructor
        virtual ~LateralControlNode();

        void Run();
        void Publish();
        void ProcessINI();

    private:
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Functions

        // Callback functions for subscribe variables        
        inline void CallbackVehicleState(const autohyu_msgs::VehicleState::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_vehicle_state_);
            i_vehicle_state_ = ros_bridge::GetVehicleState(*msg);
            b_is_vehicle_state_ = true;
        }
        inline void CallbackTrajectory(const autohyu_msgs::Trajectory::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_trajectory_);
            i_trajectory_ = ros_bridge::GetTrajectory(*msg);
            b_is_trajectory_ = true;
        }
        inline void CallbackMoraiState(const morai_msgs::EgoVehicleStatus::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_vehicle_state_);
            i_vehicle_state_ = ros_bridge::GetMoraiEgoStatus(*msg);
            b_is_vehicle_state_ = true;
        }


        // Raceline CTE (world-frame, closest-point method)
        double ComputeRacelineCTE(const interface::VehicleState& vehicle_state,
                                  const interface::Trajectory&   trajectory);

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Variables
        
        // Subscriber
        ros::Subscriber s_vehicle_state_;
        ros::Subscriber s_trajectory_;

        // Input
        interface::VehicleState     i_vehicle_state_;
        interface::Trajectory       i_trajectory_;

        // Mutex
        std::mutex mutex_vehicle_state_;
        std::mutex mutex_trajectory_;
        
        // Publisher
        ros::Publisher p_command_steer_;
        ros::Publisher p_control_trajectory_;
        ros::Publisher p_reference_path_;
        ros::Publisher p_control_info_;
        ros::Publisher p_current_understeer_gradient_;
        ros::Publisher p_control_io_;
        ros::Publisher p_cross_track_error_;

        // Output
        std_msgs::Float32 o_command_steer_;
        autohyu_msgs::ControlInfo o_control_info_;
        visualization_msgs::MarkerArray o_control_trajectory_;
        visualization_msgs::MarkerArray o_reference_path_;
        std_msgs::Float32 o_current_understeer_gradient_;
        autohyu_msgs::ControlInputOutput o_control_io_;
        std_msgs::Float32 o_cross_track_error_;

        // Algorithm
        std::unique_ptr<LateralControl> ptr_lateral_control_;

        // Util and Configuration
        IniParser util_ini_parser_;

        // Flags
        bool b_is_vehicle_state_ = false;        
        bool b_is_trajectory_ = false;

        // Global Variables
        bool use_morai_ego_ = false;
        bool use_gt_state_ = false;


        // ML-LatLonControlML
        bool ml_controller_enable_ = false;
};

#endif  // __LATERAL_CONTROL_NODE_HPP__