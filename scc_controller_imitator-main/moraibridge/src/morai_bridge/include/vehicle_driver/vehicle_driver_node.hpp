/**
 * @file        vehicle_driver_node.hpp
 * @brief       morai vehicle driver node hpp file
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)
 * @authors     Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * 
 * @date        2024-05-16 created by Yuseung Na
 * @date        2024-12-26 modified by Jiwon Seok: Update for UDP communication
 * 
 */

#ifndef __MORAI_VEHICLE_DRIVER_HPP__
#define __MORAI_VEHICLE_DRIVER_HPP__
#pragma once

// STD Header
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <random>
#include <unordered_map>

// System Header
#include "task_manager.hpp"
#include "ini_parser.h"

// Bridge Header
#include "ros_bridge_simulator.hpp"
#include "ros_bridge_vehicle_state.hpp"

// Function Header
#include "function_geometry.hpp"
#include "function_common_math.hpp"
#include "function_vehicle_state.hpp"

// Parameter Header
#include "vehicle_driver_config.hpp"

// Morai Header
#include "sensors/vehicle_state.hpp"
#include "sensors/vehicle_noisy_state.hpp"
#include "sensors/gps.hpp"

#include "sensors/vehicle_state2.hpp"

// Algorithm Header

// Namespace
using namespace ros;
using namespace tf;
using namespace std;

class VehicleDriver : public TaskManager {
    public:
        // Constructor
        explicit VehicleDriver(std::string node_name, double period);
        // Destructor
        virtual ~VehicleDriver();

        void Run();
        void Publish();
        void ProcessINI();
        void ProcessRosparam();

    private:
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Functions
        

        // Callback functions for subscribe variables
        inline void CallbackMoraiEgoVehicleStatus(const morai_msgs::EgoVehicleStatus::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_morai_ego_vehicle_status_); {
                i_morai_ego_vehicle_status_gt_ = ros_bridge::GetMoraiEgoVehicleStatus(*msg);

                // Convert UTM to WGS84 using reference point
                // Ego to GPS
                interface::Point morai_point;   
                morai_point.x = msg->position.x;
                morai_point.y = msg->position.y;
                morai_point.z = msg->position.z;
                interface::Reference morai_ref;
                morai_ref.wgs84.latitude  = cfg_.ref_lat;
                morai_ref.wgs84.longitude = cfg_.ref_lon;
                morai_ref.wgs84.altitude  = 0.0;
                morai_ref.projection      = "UTM";
                interface::WGS84 morai_wgs84 = ros_bridge::GetMoraiWGS84FromMoraiENU(morai_ref, morai_point);

                i_morai_ego_vehicle_status_gt_.gnss.latitude = morai_wgs84.latitude;
                i_morai_ego_vehicle_status_gt_.gnss.longitude = morai_wgs84.longitude;
                i_morai_ego_vehicle_status_gt_.gnss.altitude = morai_wgs84.altitude;

                if(cfg_.use_morai_enu_position == true) {
                    i_morai_ego_vehicle_status_gt_.x = msg->position.x;
                    i_morai_ego_vehicle_status_gt_.y = msg->position.y;
                    i_morai_ego_vehicle_status_gt_.z = msg->position.z;
                } else {
                    // Position (WGS84 to ENU coordinates)
                    // Convert WGS84 to ENU using reference point
                    interface::Reference reference;
                    reference.wgs84.latitude  = cfg_.ref_lat;
                    reference.wgs84.longitude = cfg_.ref_lon;
                    reference.projection      = cfg_.projection_mode;
                    interface::Point enu_position = util_function::GetEnu(reference, morai_wgs84);

                    i_morai_ego_vehicle_status_gt_.x = enu_position.x;
                    i_morai_ego_vehicle_status_gt_.y = enu_position.y;
                    i_morai_ego_vehicle_status_gt_.z = enu_position.z;
                }

                // Yaw rate estimation
                static double prev_yaw = msg->heading;
                static double prev_time = msg->header.stamp.toSec();

                double curr_time = msg->header.stamp.toSec();
                double dt = curr_time - prev_time;
                dt = std::max(dt, 0.001);
                double yaw_rate_deg = (msg->heading - prev_yaw) / dt;
                double yaw_rate = yaw_rate_deg * interface::DEG2RAD;

                prev_yaw = msg->heading;
                prev_time = curr_time;

                // Angle normalize
                yaw_rate = util_function::NormalizeAngleRad(yaw_rate);

                // Update yaw rate
                i_morai_ego_vehicle_status_gt_.yaw_vel = yaw_rate;

                b_is_ego_vehicle_status = true;
            }

            // if (b_is_gps) {
            o_gt_vehicle_state_ = ros_bridge::UpdateVehicleState(i_morai_ego_vehicle_status_gt_);
            // if (!cfg_.use_noisy_vehicle) {
            //     p_vehicle_state_.publish(o_gt_vehicle_state_);
            // }
            p_vehicle_state_gt_.publish(o_gt_vehicle_state_);
            // }
        }
        inline void CallbackMoraiEgoNoisyVehicleStatus(const morai_msgs::EgoNoisyStatus::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_morai_ego_noisy_status_); {
                i_morai_ego_vehicle_status_ = GetMoraiEgoNoisyStatus(*msg);
                b_is_ego_vehicle_status = true;

                // if (cfg_.use_noisy_vehicle) {
                    o_vehicle_state_ = ros_bridge::UpdateVehicleState(i_morai_ego_vehicle_status_);
                    p_vehicle_state_.publish(o_vehicle_state_);
                // }
            }
        };
        inline void CallbackMoraiGPSMessage(const morai_msgs::GPSMessage::ConstPtr& msg) {
            std::lock_guard<std::mutex> lock(mutex_morai_gps_); {
                i_morai_gps_      = ros_bridge::GetMoraiGPSMessage(*msg, o_reference_point_);
                b_is_gps = true;
            }

            // if (b_is_ego_vehicle_status) {
            //     o_vehicle_state_ = ros_bridge::UpdateVehicleState(i_morai_ego_vehicle_status_, 
            //                                                       i_morai_gps_,
            //                                                       i_morai_gps_.header);

            //     // p_vehicle_state_.publish(o_vehicle_state_);
            // }
        }

        void CallbackMoraiUdpVehicleState(const MoraiCppUdp::VehicleState::VehicleData& msg);
        void CallbackMoraiUdpNoisyVehicleState(const MoraiCppUdp::VehicleNoisyState::VehicleData& msg);
        void CallbackMoraiUdpVehicleState2(const MoraiCppUdp::VehicleState2::VehicleData& msg);
        
        void CallbackMoraiUdpGPS(const MoraiCppUdp::GPS::GPSData& msg);

        morai_msgs::EgoVehicleStatus CvtMoraiUdpToMoraiRosVehicleState(const MoraiCppUdp::VehicleState::VehicleData& msg);
        morai_msgs::EgoNoisyStatus CvtMoraiUdpToMoraiRosNoisyVehicleState(const MoraiCppUdp::VehicleNoisyState::VehicleData& msg);
        morai_msgs::EgoVehicleStatus CvtMoraiUdpToMoraiRosVehicleState2(const MoraiCppUdp::VehicleState2::VehicleData& msg);
        
        morai_msgs::GPSMessage CvtMoraiUdpToMoraiRosGPS(const MoraiCppUdp::GPS::GPSData& msg);
        
        autohyu_msgs::VehicleState CvtMoraiUdpToAutohyuVehicleState(const MoraiCppUdp::VehicleState::VehicleData& msg);
        autohyu_msgs::VehicleState CvtMoraiUdpNoisyToAutohyuVehicleState(const MoraiCppUdp::VehicleNoisyState::VehicleData& msg);
        autohyu_msgs::VehicleState CvtMoraiUdpToAutohyuVehicleState2(const MoraiCppUdp::VehicleState2::VehicleData& msg);
        
        interface::VehicleState GetMoraiEgoNoisyStatus(const morai_msgs::EgoNoisyStatus& msg);

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Variables
        // Morai
        const double STEERING_WHEEL_RATIO  = 45.0 / 4.0;
        const double STEERING_WHEEL_RATIO2 = 18.0;

        // Node Handler
        ros::NodeHandle nh_;

        // Subscriber
        std::unique_ptr<ros::Subscriber> s_morai_ego_vehicle_status_;
        std::unique_ptr<ros::Subscriber> s_morai_ego_noisy_status_;
        std::unique_ptr<ros::Subscriber> s_morai_gps_;

        // Morai UDP Receiver
        std::unique_ptr<MoraiCppUdp::VehicleState> morai_udp_vehicle_state_receiver_;
        std::unique_ptr<MoraiCppUdp::VehicleNoisyState> morai_udp_noisy_vehicle_state_receiver_;
        std::unique_ptr<MoraiCppUdp::GPS> morai_udp_gps_receiver_;
        std::unique_ptr<MoraiCppUdp::VehicleState2> morai_udp_vehicle_state2_receiver_;
        
        // Morai UDP Inputs
        MoraiCppUdp::VehicleState::VehicleData i_morai_udp_ego_vehicle_status_;
        MoraiCppUdp::VehicleNoisyState::VehicleData i_morai_udp_ego_noisy_status_;
        MoraiCppUdp::GPS::GPSData i_morai_udp_gps_;
        MoraiCppUdp::VehicleState2::VehicleData i_morai_udp_ego_vehicle_status2_;
        
        // Input
        interface::VehicleState i_morai_ego_vehicle_status_gt_;
        interface::VehicleState i_morai_ego_vehicle_status_;
        interface::VehicleState i_morai_gps_;

        // Mutex
        std::mutex mutex_morai_ego_vehicle_status_;
        std::mutex mutex_morai_ego_noisy_status_;
        std::mutex mutex_morai_gps_;

        // Publisher
        ros::Publisher p_vehicle_state_;
        ros::Publisher p_vehicle_state_gt_;
        ros::Publisher p_reference_;
        ros::Publisher p_morai_udp_ego_vehicle_status_;
        ros::Publisher p_morai_udp_ego_noisy_info_;

        // Output
        autohyu_msgs::VehicleState o_gt_vehicle_state_;
        autohyu_msgs::VehicleState o_vehicle_state_;
        autohyu_msgs::Reference    o_reference_point_;

        // Algorithm

        // Util and Configuration
        IniParser util_ini_parser_;
        MoraiVehicleDriverConfig cfg_;

        // Flags
        bool b_is_ego_vehicle_status = false;
        bool b_is_gps = false;

        // Global Variables
        // ADD GLOBAL VARIABLES HERE
};

#endif  // __MORAI_VEHICLE_DRIVER_HPP__