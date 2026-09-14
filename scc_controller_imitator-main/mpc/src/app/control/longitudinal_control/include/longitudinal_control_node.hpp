/**
 * @file        longitudinal_control.hpp
 * @brief       longitudinal control node
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)    
 *              Junhee Lee (998jun@gmail.com)      
 * 
 * @date        2023-02-02 Yuseung Na - Created. (Longitudinal MPC using CVXGEN)
 *              2023-03-10 Junhee Lee - Add dynamic MPC model
 *              2023-04-12 Junhee Lee - Add dynamic MPC du model (HPIPM)
 *              2023-08-06 Junhee Lee - Code Cleanup
 */

#ifndef __LONGITUDINAL_CONTROL_NODE_HPP__
#define __LONGITUDINAL_CONTROL_NODE_HPP__
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
#include <autohyu_msgs/PredictObjects.h>
#include <autohyu_msgs/Trajectory.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/MarkerArray.h>
#include <morai_msgs/EgoVehicleStatus.h>

// System Header
#include "task_manager.hpp"
#include "ini_parser.h"

// Bridge Header
#include "ros_bridge_control.hpp"
#include "ros_bridge_objects.hpp"
#include "ros_bridge_trajectories.hpp"
#include "ros_bridge_vehicle_state.hpp"

// Parameter Header

// Algorithm Header
#include "longitudinal_control/longitudinal_control.hpp"

// Namespace
using namespace ros;
using namespace interface;
using namespace std;

class LongitudinalControlNode : public TaskManager {
    public:
        // Constructor
        explicit LongitudinalControlNode(std::string task_node, double period);
        // Destructor
        virtual ~LongitudinalControlNode();

        void Run();
        void Publish();
        void ProcessINI();
        void ProcessRosparam(const ros::NodeHandle& nh);

    private:
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Functions

        // Callback functions for subscribe variables        
        inline void CallbackVehicleState(const autohyu_msgs::VehicleState::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_vehicle_state_);
            i_vehicle_state_ = ros_bridge::GetVehicleState(*msg);
            b_is_vehicle_state_ = true;
        }
        inline void CallbackPredictObjects(const autohyu_msgs::PredictObjects::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_predict_objects_);
            i_predict_objects_ = ros_bridge::GetPredictObjects(*msg);
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

        // Update functions for publish variables
        void UpdateControlCommand(const ControlCommand& cmd);
        void UpdateSpeedTrackingInfos(const PathTrackingInfos& speed_tracking_errors,
                            const VehicleState& vehicle_state,
                            const ControlTrajectory& trajectory);
        
        void UpdateRvizPredictedSpeedProfile(const ControlTrajectory& control_trajectory);

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Variables
        
        // Subscriber
        ros::Subscriber s_vehicle_state_;
        ros::Subscriber s_predict_objects_;
        ros::Subscriber s_trajectory_;

        // Input
        interface::VehicleState    i_vehicle_state_;
        interface::PredictObjects  i_predict_objects_;
        interface::Trajectory      i_trajectory_;

        // Mutex
        std::mutex mutex_vehicle_state_;
        std::mutex mutex_predict_objects_;
        std::mutex mutex_trajectory_;
        
        // Publisher
        ros::Publisher p_command_torque_;
        ros::Publisher p_command_acceleration_;
        ros::Publisher p_command_accel_pedal_;
        ros::Publisher p_command_brake_pedal_;
        
        ros::Publisher p_target_speed_;
        ros::Publisher p_target_ax_;
        ros::Publisher p_current_speed_;
        ros::Publisher p_current_speederror_;
        ros::Publisher p_current_axerror_;
        ros::Publisher p_compensated_torque_;
        ros::Publisher p_speed_compensation_error_;
        ros::Publisher p_predicted_speed_profile_;
        ros::Publisher p_control_io_;

        // Output
        std_msgs::Float32 o_command_torque_;
        std_msgs::Float32 o_command_acceleration_;
        std_msgs::Float32 o_accel_pedal_;
        std_msgs::Float32 o_brake_pedal_;
 
        std_msgs::Float32 o_target_speed_;
        std_msgs::Float32 o_target_ax_;        
        std_msgs::Float32 o_current_speed_;        
        std_msgs::Float32 o_current_speederror_;        
        std_msgs::Float32 o_current_axerror_;
        std_msgs::Float32 o_compensated_torque_;        
        std_msgs::Float32 o_speed_compensation_error_;  
        autohyu_msgs::ControlInputOutput o_control_io_;

        visualization_msgs::MarkerArray o_predicted_speed_profile_;

        // Algorithm
        unique_ptr<LongitudinalControl> ptr_longitudinal_control_;

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

#endif  // __LONGITUDINAL_CONTROL_NODE_HPP__