/**
 * @file        interface_geometry_msgs.hpp
 * @brief       interface for geometry_msgs
 * 
 * @authors     Yuseung Na (yuseungna@hayang.ac.kr)
 *              Seheon Ha (seheonha@hanyang.ac.kr)      
 * 
 * @date        2024-04-24 created by Yuseung Na
 *              2024-04-29 updated by Seheon Ha: Add operator==
 */

#ifndef __INTERFACE_GEOMETRY_MSGS_HPP__
#define __INTERFACE_GEOMETRY_MSGS_HPP__
#pragma once

// STD Header
#include <stdint.h>
#include <map>
#include <utility>
#include <vector>

#include "interface_std_msgs.hpp"

namespace interface {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // enum
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //     

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // structs
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    typedef struct {
        double x;
        double y;
        double z;
        double vx_mps      = 0.0;  // IQP optimized race speed (m/s)
        double ax_mps2     = 0.0;  // IQP optimized longitudinal acceleration (m/s^2)
        double kappa_radpm = 0.0;  // curvature (rad/m)
        double psi_rad     = 0.0;  // heading (rad)
        double s_m         = 0.0;  // arc-length along raceline (m)
    } Point;
    inline bool operator==(const Point a, const Point b){
        return a.x == b.x && a.y == b.y && a.z == b.z && a.vx_mps == b.vx_mps;
    }

    typedef struct{
        std::vector<Point> point;
    } PointArray;

    typedef struct {
        interface::Header header;
        Point point;
    } PointStamped;

    typedef struct {
        double x;
        double y;
        double z;
        double w;
    } Quaternion;

    typedef struct{
        Point position;
        Quaternion orientation;
    } Pose;

    typedef struct {
        interface::Header header;
        Pose pose;
    } PoseStamped;

    typedef struct {
        double x;
        double y;
        double z;
    } Vector3;

    typedef struct{
        interface::Header header;
        std::vector<Pose> poses;
    } PoseArray;

} // namespace interface

#endif // __INTERFACE_GEOMETRY_MSGS_HPP__
