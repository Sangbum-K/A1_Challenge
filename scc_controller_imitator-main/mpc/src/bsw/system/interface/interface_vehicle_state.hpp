/**
 * @file        interface_vehicle_state.hpp
 * @brief       interface for vehicle state
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-01 updated by Yuseung Na: Refactoring 
 */

#ifndef __INTERFACE_VEHICLE_STATE_HPP__
#define __INTERFACE_VEHICLE_STATE_HPP__
#pragma once

// STD Header
#include <stdint.h>
#include <map>
#include <utility>
#include <vector>

// Interface Header
#include "interface_std_msgs.hpp"

namespace interface {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // enum
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    typedef enum {
        MANUAL     = 0,
        AUTONOMOUS = 1,
        FAIL       = 2,
    } OperationMode;

    typedef enum {
        READY = 0,
        RUN   = 1,
    } AutonomousMode;

    typedef enum {
        GEAR_P = 0,
        GEAR_R = 1,
        GEAR_N = 2,
        GEAR_D = 3,
    } Gear;

    typedef enum {
        TURNSIG_OFF    = 0,
        TURNSIG_LEFT   = 1,
        TURNSIG_RIGHT  = 2,
        TURNSIG_HAZARD = 3,
    } TurnSignal;

    typedef enum {
        FLAG_READY   = 0,
        FLAG_START   = 1,
        FLAG_STOP    = 2,
        FLAG_WARN    = 3,
        FLAG_DEFAULT = 4,
    } EFlag;

    typedef enum {
        MDPS_INIT             = 1,
        MDPS_READY            = 2,
        MDPS_STANDBY          = 3,
        MDPS_ACTIVATION_START = 4,
        MDPS_ACTIVATE         = 5,
        MDPS_ERROR            = 6,
        MDPS_ABORTED          = 7,
    } MdpsState;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // structs
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    typedef struct {
        double latitude;
        double longitude;
        double altitude;
    } WGS84;

    typedef struct {
        WGS84 wgs84;
        std::string projection;
    } Reference;

    typedef struct {
        Header          header;

        double          steering_wheel_angle;   // Vehicle steering wheel angle [rad]
        double          steering_tire_angle;    // Tire direction angle [rad]
        MdpsState       steering_state;         // MDPS steering state
        
        double          accel_position;         // Acceleration padal position
        double          brake_position;         // Brake fluid presure [Pa]
        Gear            gear_select;            // Gear position P|R|N|D

        OperationMode   operation_mode;         // Operation mode [manual | AUTONOMOUS | FAIL]
        AutonomousMode  lateral_ad_mode;        // Lateral autonomous mode [READY | RUN]
        AutonomousMode  longitudinal_ad_mode;   // Longitudinal autonomous mode [READY | RUN]
    } VehicleCAN;

    // Race-flag input (dormant on A1_Rule — no publisher wired yet).
    // Preserved so autoku-r derived behavior_planning logic compiles unchanged.
    // Header-based shape chosen for ROS2 compatibility (frame_id + stamp).
    typedef struct {
        Header header;
        EFlag  e_flag{EFlag::FLAG_DEFAULT};
    } RaceFlag;

    // Minimal cartesian position — used by behavior_planning speed_limiter
    // to carry a transient red-flag stop target across ticks.
    typedef struct {
        double x{0.0};
        double y{0.0};
        double z{0.0};
    } Position;

    typedef struct {
        Header                  header;

        // GNSS
        WGS84                   gnss;
        
        // position and motion
        double x;               // East position in ENU frame [m]
        double y;               // North position in ENU frame [m]
        double z;               // Up position in ENU frame [m]

        double vx;              // Vehicle longitudinal velocity [m/s]
        double vy;              // Vehicle lateral velocity [m/s]
        double vz;              // Vehicle up velocity [m/s]

        double ax;              // Vehicle longitudinal acceleration [m/s2]
        double ay;              // Vehicle lateral acceleration [m/s2]
        double az;              // Vehicle up acceleration [m/s2]

        double roll;            // World to vehicle roll [rad]
        double pitch;           // World to vehicle pitch [rad]
        double yaw;             // World to vehicle yaw [rad]

        double roll_vel;        // Vehicle x axis rotation rate [rad/s]
        double pitch_vel;       // Vehicle y axis rotation rate [rad/s]
        double yaw_vel;         // Vehicle z axis rotation rate [rad/s]

        std::string link_id;         // Link ID of the vehicle

        // vehicle can
        VehicleCAN vehicle_can;
        
    } VehicleState;

    typedef struct {
        //double time_stamp{0.0};
        Header header;

        double s;
        double ds;
        double dds;

        double n;
        double dn;
        double ddn;

        double mu;
        double dmu;
        double ddmu;
    } FrenetVehicleState;
} // namespace interface

#endif // __INTERFACE_VEHICLE_STATE_HPP__
