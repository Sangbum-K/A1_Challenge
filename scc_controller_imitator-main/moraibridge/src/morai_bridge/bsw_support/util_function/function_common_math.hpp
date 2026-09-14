/**
 * @file        function_common_math.hpp
 * @brief       util functions for common math
 * 
 * @authors     Seheon Ha (seheon@hanyang.ac.kr)          
 * 
 * @date        2024-08-16 created by Seheon Ha
 */

#ifndef __FUNCTION_COMMON_MATH_HPP__
#define __FUNCTION_COMMON_MATH_HPP__
#pragma once

// STD header
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <cmath>
// Utility header


// Interface Header


using namespace interface;

namespace util_function {
    inline double GetTwoPointsDistance(const double& x1, const double& y1, const double& x2, const double& y2){
        return sqrt(pow(x1 - x2 , 2) + pow(y1 - y2 , 2));
    }
    
    inline double VectorDotProduct2D(const double& x1, const double& y1 , const double& x2, const double& y2){
        double v1_size = sqrt(pow(x1,2)+pow(y1,2));
        double v2_size = sqrt(pow(x2,2)+pow(y2,2));
        double v1_x = x1 / v1_size;
        double v1_y = y1 / v1_size;
        double v2_x = x2 / v2_size;
        double v2_y = y2 / v2_size; 

        return v1_x * v2_x + v1_y * v2_y;
    }

    inline double NormalizeAngleDiffRad(double rad_a, double rad_b) {
        double diff = rad_a - rad_b;
        while (diff > M_PI) {
            diff -= 2 * M_PI;
        }
        while (diff < -M_PI) {
            diff += 2 * M_PI;
        }
        return diff;
    }

    inline double NormalizeAngleRad(double i_rad){
        while (i_rad > M_PI) {
            i_rad -= 2 * M_PI;
        }
        while (i_rad < -M_PI) {
            i_rad += 2 * M_PI;
        }
        return i_rad;  
    }
}

#endif // __FUNCTION_COMMON_MATH_HPP__