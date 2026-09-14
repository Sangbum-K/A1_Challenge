/**
 * @file        interface_constants.hpp
 * @brief       interface for constants
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-01 updated by Yuseung Na: Refactoring 
 */

#ifndef __INTERFACE_CONSTANTS_HPP__
#define __INTERFACE_CONSTANTS_HPP__
#pragma once

// STD Header
#include <stdint.h>
#include <map>
#include <utility>
#include <vector>
#include <cmath>

namespace interface {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // constants
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    const double DEG2RAD = M_PI/180.0;
    const double RAD2DEG = 180.0/M_PI;
    const double MPS2KPH = 3600.0/1000.0;
    const double KPH2MPS = 1000.0/3600.0;
    const double EARTH_RADIUS_EQUA = 6378137.0;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // vehicle parameters — MA Formula 4
    // ref: config/control.ini [Vehicle]
    //
    // NOTE: Phase 1 (Pure Pursuit + PID) 에서는 아래 값들을 직접 사용하지 않음.
    //       Phase 2 MPC 전환 전에 MA F4 실측값으로 반드시 검증 필요.
    //       [*] 표시 항목은 현재 미검증 추정값.
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    const double VEHICLE_WIDTH  = 1.890;                // [m] [*] Phase 2 전에 MA F4 실측 필요
    const double VEHICLE_LENGTH = 4.635;                // [m] [*] Phase 2 전에 MA F4 실측 필요
    const double VEHICLE_HEIGHT = 1.605;                // [m] [*] Phase 2 전에 MA F4 실측 필요
    const double VEHICLE_WHEEL_BASE = 2.35;             // [m] control.ini 와 일치
    const double VEHICLE_FRONT_AXLE_TO_CG = 1.4479;    // [m] = wheel_base - rear_axle_to_cg
    const double VEHICLE_REAR_AXLE_TO_CG = 0.9021;     // [m] control.ini 와 일치
    const double VEHICLE_REAR_OVERHANG = 0.82;          // [m] [*] Phase 2 전에 MA F4 실측 필요
    const double VEHICLE_FRONT_OVERHANG = 0.845;        // [m] [*] Phase 2 전에 MA F4 실측 필요
    const double VEHICLE_MASS = 630.0;                  // [kg] sprung(550) + wheels(80), control.ini 와 일치
    const double VEHICLE_MOMENT_OF_INERTIA = 3637.526;  // [kgm²] [*] Phase 2 MPC 전에 MA F4 실측 필요
    const double VEHICLE_WHEEL_RADIUS = 0.280;          // [m]
    const double VEHICLE_FRONT_AREA = 1.2;              // [m^2]

    const double STEERING_RATIO = 14.0625;              // [ratio] [*] Phase 2 MPC 전에 MA F4 실측 필요
    const double MAX_STEERING_TIRE_ANGLE = 25.0;        // [Deg]
    const double MAX_STEERING_TIRE_SPEED = 40.0;        // [Deg/s]

    const double TIRE_TRACTION = 1.0;                   // [ratio]
    const double FRONT_CORNERING_STIFFNESS = 98047;     // [N/rad] [*] Phase 2 MPC 전에 MA F4 실측 필요
    const double REAR_CORNERING_STIFFNESS  = 164000;    // [N/rad] [*] Phase 2 MPC 전에 MA F4 실측 필요
    const double AIR_DENSITY = 1.205;
    const double AIR_DRAG_COEFF = 0.7;
    const double GRAVITY_COEFF = 9.806;
    const double ROLLING_RESISTANCE = 0.015;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // control parameters
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - // 
    
    // Torque Limit
    const double T_UL  = 6443.0;                    // Torque Max
    const double T_LL  = -10000.0;                   // Torque Min    
    
} // namespace interface

#endif // __INTERFACE_CONSTANTS_HPP__