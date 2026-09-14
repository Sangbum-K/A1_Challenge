/**
 * @file        interface_sensor_msgs.hpp
 * @brief       interface for sensor
 * 
 * @authors     Yuseung Na (yuseungna@hayang.ac.kr)
 *              Seheon Ha (seheonha@hanyang.ac.kr)
 *              Seongjae Jeong (sjeong99@hanyang.ac.kr)
 * 
 * @date        2024-04-24 created by Yuseung Na
 *              2024-04-24 updated by Seheon Ha: Add PointCloud2
 *              2024-04-24 updated by Seongjae Jeong: Add Image
 */

#ifndef __INTERFACE_SENSOR_HPP__
#define __INTERFACE_SENSOR_HPP__
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
    typedef enum {
        INT8    = 1,
        UINT8   = 2,
        INT16   = 3,
        UINT16  = 4,
        INT32   = 5,
        UINT32  = 6,
        FLOAT32 = 7,
        FLOAT64 = 8        
    } PointFieldDatatype;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // structs
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    typedef struct {
        Header header;

        interface::Quaternion orientation;
        interface::Vector3    orientation_RPY;
        interface::Vector3    angular_velocity;
        interface::Vector3    linear_acceleration;
    } Imu;

    typedef struct {
        std::string         name;
        uint32_t            offset;
        PointFieldDatatype  datatype;
        uint32_t            count;
    } PointField;

    typedef struct {
        Header                      header;

        // 2D structure of the point cloud. If the cloud is unordered, height is
        // 1 and width is the length of the point cloud.        
        uint32_t                    height;
        uint32_t                    width;

        // Describes the channels and their layout in the binary data blob.
        std::vector<PointField>     fields;

        bool                        is_bigendian;   // Is this data bigendian?
        uint32_t                    point_step;     // Length of a point in bytes
        uint32_t                    row_step;       // Length of a row in bytes
        std::vector<uint8_t>        data;           // Actual point data, size is (row_step*height)
            
        bool                        is_dense;       // True if there are no invalid points
    } PointCloud2;

    typedef struct {
        Header               header;
        
        uint32_t             height;        // image height, that is, number of rows
        uint32_t             width;         // image width, that is, number of columns
        
        std::string          encoding;      // Encoding of pixels -- channel meaning, ordering, size
                                            // taken from the list of strings in include/sensor_msgs/image_encodings.h

        uint8_t              is_bigendian;  // is this data bigendian?
        uint32_t             step;          // Full row length in bytes
        std::vector<uint8_t> data;          // actual matrix data, size is (step * rows)
    } Image;
} // namespace interface

#endif // __INTERFACE_SENSOR_HPP__
