/**
 * @file        function_control.hpp
 * @brief       util functions for control
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-11 updated by Yuseung Na: Refactoring 
 */

#ifndef __FUNCTION_CONTROL_HPP__
#define __FUNCTION_CONTROL_HPP__
#pragma once

// STD header
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

// Eigen header
#include <Eigen/Dense>

// Utility header
#include <spline.h>

// Interface Header
#include "interface_constants.hpp"
#include "interface_control.hpp"
#include "interface_trajectories.hpp"
#include "function_print.hpp"

using namespace interface;

namespace morai_ioniq5 {
    const double ACCEL_NEUTRAL = -0.43;      // Acceleration Neutral
    const double GAS_1 = 0.329793;
    const double BRAKE_1 = 0.095538;
}

namespace util_function {
    inline void ConvertAccelerationToPedal(const std::vector<double>& acceleration, 
                                           std::vector<double>& accel_pedal, 
                                           std::vector<double>& brake_pedal) {
        double gas, brake;
        for (int i = 0; i < acceleration.size(); i++) {
            if(acceleration[i] >= morai_ioniq5::ACCEL_NEUTRAL) {       
                gas = std::max(0.0, std::min(1.0, morai_ioniq5::GAS_1*acceleration[i]));
                brake = 0.0;
                accel_pedal.push_back(gas);
                brake_pedal.push_back(brake);
            }
            else if(acceleration[i] < morai_ioniq5::ACCEL_NEUTRAL) {
                gas = 0.0;
                brake = std::max(0.0, std::min(1.0, (-1.)*morai_ioniq5::BRAKE_1*acceleration[i]));
                accel_pedal.push_back(gas);
                brake_pedal.push_back(brake);
            }
        }
        return;
    }

    inline Trajectory TransformTrajectory(const Trajectory& trajectory, const VehicleState& vehicle_state,
                                          const std::string& frame_out = "ego_frame"){
        Trajectory ego_trajectory = trajectory; 
        ego_trajectory.point.clear();
        
        for(auto point : trajectory.point){
            TrajectoryPoint ego_point;
            Eigen::Affine3d point_global_tf, point_local_tf, ego_frame_tf;
            point_global_tf = Eigen::Affine3d::Identity();
            point_local_tf = Eigen::Affine3d::Identity();
            ego_frame_tf = Eigen::Affine3d::Identity();

            point_global_tf.translate(Eigen::Vector3d(point.x, point.y, 0.0));
            point_global_tf.rotate( Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) * 
                                    Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(point.yaw, Eigen::Vector3d::UnitZ()));

            ego_frame_tf.translation()[0] = vehicle_state.x;
            ego_frame_tf.translation()[1] = vehicle_state.y;
            ego_frame_tf.translation()[2] = 0.0;

            ego_frame_tf = Eigen::Translation3d(ego_frame_tf.translation()) *
                            (Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) *
                            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(vehicle_state.yaw, Eigen::Vector3d::UnitZ()));

            point_local_tf = ego_frame_tf.inverse() * point_global_tf;
            Eigen::Vector3d ego_point_rpy = point_local_tf.rotation().eulerAngles(0,1,2);

            ego_point     = point;
            ego_point.x   = std::round(point_local_tf.translation().x()*1000.0)/1000.0;
            ego_point.y   = std::round(point_local_tf.translation().y()*1000.0)/1000.0;
            ego_point.yaw = std::round(ego_point_rpy(2)*1000.0)/1000.0;

            // Translate x w.r.t target frame (default is rear axle)
            if(frame_out == "CG"){
                ego_point.x -= VEHICLE_REAR_AXLE_TO_CG;
            }
            else if(frame_out == "front_axle"){
                ego_point.x -= VEHICLE_WHEEL_BASE;
            }
            
            ego_trajectory.point.push_back(ego_point);
        }

        // return x,y,(yaw),speed only trajectory
        return ego_trajectory;

    }

    // MA F4 GGV (longitudinal ax_max vs speed in m/s), sourced from
    // tools/raceline_optimizer/config/vehicles/ggv/ggv_ma_f4_jys_0409.csv.
    // Wheel torque is derived as T = m * ax * r so the upstream API
    // (CalculateMaxWheelTorque) stays intact for SK/ACC/CA MPC solvers.
    inline double GetMaxLongitudinalAccel(const double& speed_mps) {
        // 1 m/s spacing, 0..64 m/s; clamp ends.
        static const double kAxMax[] = {
            15.0, 15.0, 15.0, 14.8, 14.8, 13.7, 11.5, 11.2, 9.3, 8.8,
            8.8,  8.7,  8.7,  8.7,  8.3,  8.2,  8.2,  8.2,  8.1, 8.1,
            8.1,  8.0,  8.0,  8.0,  7.9,  7.9,  7.8,  7.8,  7.7, 7.7,
            7.6,  7.6,  7.5,  7.5,  7.4,  7.3,  7.3,  7.2,  7.1, 7.1,
            7.0,  6.9,  6.9,  6.8,  6.7,  6.6,  6.6,  6.5,  6.4, 6.3,
            6.2,  6.1,  6.0,  6.0,  5.9,  5.8,  5.7,  5.6,  5.5, 5.4,
            5.3,  5.2,  5.1,  5.0,  5.0
        };
        constexpr int kN = sizeof(kAxMax) / sizeof(kAxMax[0]);
        if (speed_mps <= 0.0) return kAxMax[0];
        if (speed_mps >= (double)(kN - 1)) return kAxMax[kN - 1];
        const int i  = (int)std::floor(speed_mps);
        const double a = speed_mps - (double)i;
        return (1.0 - a) * kAxMax[i] + a * kAxMax[i + 1];
    }

    inline double CalculateMaxWheelTorque(const double& speed) {
        // T_wheel [Nm] = m * a * r. Keeps the rest of the pipeline's u_max
        // arithmetic (max_force / (VEHICLE_WHEEL_RADIUS * VEHICLE_MASS))
        // collapsing back to the GGV ax_max directly.
        return VEHICLE_MASS * GetMaxLongitudinalAccel(speed) * VEHICLE_WHEEL_RADIUS;
    }
} // namespace util_function

#endif  // __FUNCTION_CONTROL_HPP__