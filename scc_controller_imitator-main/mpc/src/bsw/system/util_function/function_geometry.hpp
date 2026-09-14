/**
 * @file        function_geometry.hpp
 * @brief       util functions for geometry
 * 
 * @authors     Jiwon Seok (jiwonseok@hanyang.ac.kr)          
 * 
 * @date        2025-01-02 created by Jiwon Seok
 */

#ifndef __FUNCTION_GEOMETRY_HPP__
#define __FUNCTION_GEOMETRY_HPP__
#pragma once

// Interface Header
#include "interface_geometry_msgs.hpp"

// Utility Header
#include <lanelet2_core/geometry/LaneletMap.h>
#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_io/Io.h>
#include <lanelet2_projection/LocalCartesian.h>
#include <lanelet2_projection/UTM.h>

using namespace interface;

namespace util_function {
    interface::WGS84 GetWgs84(const interface::Reference& reference, const interface::Point& enu) {
        interface::WGS84 wgs84;
        // Ego to GPS
        lanelet::Origin origin({reference.wgs84.latitude, reference.wgs84.longitude, reference.wgs84.altitude});
        lanelet::projection::UtmProjector utm_projector(origin);
        lanelet::projection::LocalCartesianProjector local_projector(origin);
        lanelet::BasicPoint3d position(enu.x, enu.y, enu.z);
        lanelet::GPSPoint gps_point;
        if(reference.projection == "utm" || reference.projection == "UTM" || reference.projection == "Utm") {
            gps_point = utm_projector.reverse(position);
        } else if(reference.projection == "local_cartesian" || reference.projection == "LocalCartesian") {
            gps_point = local_projector.reverse(position);
        }
        else {
            std::string error_msg = "Invalid projection mode in GetWgs84 (function_geometry.hpp), current mode: " + reference.projection;
            throw std::runtime_error(error_msg.c_str());
        }
        wgs84.latitude = gps_point.lat;
        wgs84.longitude = gps_point.lon;
        wgs84.altitude = gps_point.ele;
        return wgs84;
    }

    interface::Point GetEnu(const interface::Reference& reference, const interface::WGS84& wgs84) {
        interface::Point enu;
        // WGS84 to ENU
        lanelet::Origin origin({reference.wgs84.latitude, reference.wgs84.longitude, reference.wgs84.altitude});
        lanelet::projection::UtmProjector utm_projector(origin);
        lanelet::projection::LocalCartesianProjector local_projector(origin);
        lanelet::GPSPoint gps_point{wgs84.latitude, wgs84.longitude, wgs84.altitude};
        lanelet::BasicPoint3d position;
        if(reference.projection == "utm" || reference.projection == "UTM" || reference.projection == "Utm") {
            position = utm_projector.forward(gps_point);
        } else if(reference.projection == "local_cartesian" || reference.projection == "LocalCartesian") {
            position = local_projector.forward(gps_point);
        }
        else {
            std::string error_msg = "Invalid projection mode in GetEnu (function_geometry.hpp), current mode: " + reference.projection;
            throw std::runtime_error(error_msg.c_str());
        }
        enu.x = position.x();
        enu.y = position.y();
        enu.z = position.z();
        return enu;
    }
}

#endif