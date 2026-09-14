/**
 * @file        interface_std_msgs.hpp
 * @brief       interface for std_msgs
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)      
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              
 */

#ifndef __INTERFACE_STD_MSGS_HPP__
#define __INTERFACE_STD_MSGS_HPP__
#pragma once

// STD Header
#include <stdint.h>
#include <map>
#include <utility>
#include <vector>
#include <string>

// 순환 참조 방지를 위해 제거

namespace interface {
    typedef struct{
        uint32_t    seq;
        double      stamp{0.0};
        std::string frame_id;
    } Header;

    // PoseArray는 interface_geometry_msgs.hpp로 이동
} // namespace interface

#endif // __INTERFACE_STD_MSGS_HPP__