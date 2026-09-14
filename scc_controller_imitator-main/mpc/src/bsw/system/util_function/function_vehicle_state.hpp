/**
 * @file        function_vehicle_state.hpp
 * @brief       util functions for vehicle state
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-08-03 created by Yuseung Na
 *              2024-04-11 updated by Yuseung Na: Refactoring 
 */

#ifndef __FUNCTION_VEHICLE_STATE_HPP__
#define __FUNCTION_VEHICLE_STATE_HPP__
#pragma once

// STD header
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

// Utility header
#include <spline.h>

// Interface Header
#include "interface_constants.hpp"
#include "interface_vehicle_state.hpp"
#include "function_print.hpp"

using namespace interface;

namespace util_function {
    inline FrenetVehicleState ConvertFrenetState(const VehicleState& vehicle_state,
                                                 tk::Map& road_map) {
        
        FrenetVehicleState frenet_state;
        frenet_state.header = vehicle_state.header;

        // Convert Cartesian to Frenet coordinate
        // std::vector<double> frenet_position = road_map.ToFrenet(vehicle_state.x, vehicle_state.y);
        // frenet_state.s = frenet_position[0];
        // frenet_state.n = frenet_position[1];

        // double vx_global = vehicle_state.vx*cos(vehicle_state.yaw) - vehicle_state.vy*sin(vehicle_state.yaw);
        // double vy_global = vehicle_state.vx*sin(vehicle_state.yaw) + vehicle_state.vy*cos(vehicle_state.yaw);
        // std::vector<double> frenet_speed = road_map.ToFrenetVelocity(vx_global, vy_global, frenet_position[0]);
        // frenet_state.ds = frenet_speed[0];
        // frenet_state.dn = frenet_speed[1];

        // double ax_global = vehicle_state.ax*cos(vehicle_state.yaw) - vehicle_state.ay*sin(vehicle_state.yaw);
        // double ay_global = vehicle_state.ax*sin(vehicle_state.yaw) + vehicle_state.ay*cos(vehicle_state.yaw);
        // std::vector<double> frenet_accel = road_map.ToFrenetVelocity(ax_global, ay_global, frenet_position[0]);
        // frenet_state.dds = frenet_accel[0];
        // frenet_state.ddn = frenet_accel[1];

        // Using ToFrenetAllT (x, y, yaw, curvature, velocity, acceleration)
        std::vector<double> cartesian_state;
        cartesian_state.push_back(vehicle_state.x);
        cartesian_state.push_back(vehicle_state.y);
        cartesian_state.push_back(vehicle_state.yaw);
        cartesian_state.push_back(tan(vehicle_state.vehicle_can.steering_tire_angle)/interface::VEHICLE_WHEEL_BASE);
        cartesian_state.push_back(vehicle_state.vx);
        cartesian_state.push_back(vehicle_state.ax);

        std::vector<double> frenet_state_all = road_map.ToFrenetAllT(cartesian_state);
        frenet_state.s = frenet_state_all[0];
        frenet_state.ds = frenet_state_all[1];
        frenet_state.dds = frenet_state_all[2];
        frenet_state.n = frenet_state_all[3];
        frenet_state.dn = frenet_state_all[4];
        frenet_state.ddn = frenet_state_all[5];
    
        return frenet_state;
    }

    inline double TireAngleToWheelAngle(const double& tire_angle) {        
        double a3 = -0.003527; //-0.0009287;
        double a2 = -0.001528; //-0.0009204;
        double a1 = 16.06; //15.17;
        double a0 = 0.0;
        
        double wheel_angle = a3 * pow(tire_angle, 3) + a2 * pow(tire_angle, 2) + a1 * tire_angle + a0;

        return wheel_angle;
    }

    inline double WheelAngleToTireAngle(const double& wheel_angle) {
        double a3 = 8.013e-08;
        double a2 = 7.092e-07; 
        double a1 = 0.06124; 
        double a0 = 0.0;
        
        double tire_angle = a3 * pow(wheel_angle, 3) + a2 * pow(wheel_angle, 2) + a1 * wheel_angle + a0;

        return tire_angle;
    }

    inline double SlerpTwoRad(double upper_rad, double lower_rad, double ratio){
        double interpolated_v, interpolated_u;

        interpolated_v = ratio * sin(upper_rad) + (1.0 - ratio) * sin(lower_rad);
        interpolated_u = ratio * cos(upper_rad) + (1.0 - ratio) * cos(lower_rad);

        if (fabs(interpolated_u) < 1e-10) interpolated_u = 1e-10;

        return atan2(interpolated_v, interpolated_u);
    }
}

#endif // __FUNCTION_VEHICLE_STATE_HPP__