/**
 * @file        interface_lanelet.hpp
 * @brief       interface for binary-lanelet
 * 
 * @authors     Junhee Lee (998jun@gmail.com)      
 * 
 * @date        2024-04-12 created by Junhee Lee
 *  
 */

#ifndef __INTERFACE_LANELET_HPP__
#define __INTERFACE_LANELET_HPP__
#pragma once


// STD Header
#include <stdint.h>
#include <map>
#include <utility>
#include <vector>

// Interface Header
#include "interface_std_msgs.hpp"
#include "interface_geometry_msgs.hpp"

namespace interface {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // enum
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // structs
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //  
    typedef struct {  
        std::string map_file_path;
        std::string projector;
        double ref_lat;
        double ref_lon;
        bool is_ref_update = false;
    } MapInfo;

    typedef struct {
        Header                  header;
        std::vector<int8_t>     data;
    } MapBin;
} // namespace interface

#endif // __INTERFACE_LANELET_HPP__
