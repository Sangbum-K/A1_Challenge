/**
 * @file        function_trajectories.hpp
 * @brief       util functions for trajectories
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2023-07-14 created by Yuseung Na
 *              2024-04-11 updated by Yuseung Na: Refactoring 
 */

#ifndef __FUNCTION_TRAJECTORIES_HPP__
#define __FUNCTION_TRAJECTORIES_HPP__
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
#include "interface_trajectories.hpp"
#include "function_print.hpp"

using namespace interface;

namespace util_function {
    template <typename T>
    inline bool GenerateRoadMap(const T& trajectory, tk::Map& road_map, const double& distance_threshold = 0.1) {
        if ((uint16_t)trajectory.point.size() < 4) {            
            util_function::DebugPrintError(
                "Cannot generate road map to spline!"
            );
            return false;
        } else {
            std::vector<double> sx, sy, ss, sdx, sdy;
            double sum_s = 0.0;
            double prev_sum_s = 0.0;
            for (uint16_t i = 0; i < (uint16_t)trajectory.point.size(); i++) {
                double x = trajectory.point[i].x;
                double y = trajectory.point[i].y;
                double dx = i == 0 ? 0.0 : x - trajectory.point[i - 1].x;
                double dy = i == 0 ? 0.0 : y - trajectory.point[i - 1].y;
                sum_s += sqrt(dx * dx + dy * dy);

                if(sum_s - prev_sum_s > distance_threshold) {                    
                    sx.push_back(x);
                    sy.push_back(y);
                    ss.push_back(sum_s);
                    sdx.push_back(dx);
                    sdy.push_back(dy);
                    prev_sum_s = sum_s;
                }
            }

            if (sx.size() < 4) {
                return false;
            }
            
            road_map = tk::Map(sx, sy, ss, sdx, sdy);
            return true;
        }
    }

    template <typename T>
    inline tk::Map GenerateRoadMap(const T& trajectory, const double& ego_yaw = 0.0, const double& distance_threshold = 0.1) {
        tk::Map road_map;
        if ((uint16_t)trajectory.point.size() < 4) {            
            util_function::DebugPrintError(
                "Cannot generate road map to spline!"
            );
        } else {
            std::vector<double> sx, sy, ss, sdx, sdy;
            double sum_s = 0.0;
            double prev_sum_s = 0.0;
            for (uint16_t i = 0; i < (uint16_t)trajectory.point.size(); i++) {
                double x = trajectory.point[i].x;
                double y = trajectory.point[i].y;
                double dx = i == 0 ? 0.0 : x - trajectory.point[i - 1].x;
                double dy = i == 0 ? 0.0 : y - trajectory.point[i - 1].y;
                sum_s += sqrt(dx * dx + dy * dy);

                if(sum_s - prev_sum_s > distance_threshold) {                    
                    sx.push_back(x);
                    sy.push_back(y);
                    ss.push_back(sum_s);
                    sdx.push_back(dx);
                    sdy.push_back(dy);
                    prev_sum_s = sum_s;
                }
            }

            if (sx.size() < 4) {
                util_function::DebugPrintError(
                    "Trajectory point is not enough! Generate as a straight line."
                );
                
                sum_s = 0;
                sx.clear();
                sy.clear();
                ss.clear();
                sdx.clear();
                sdy.clear();

                sx.push_back(trajectory.point.front().x);
                sy.push_back(trajectory.point.front().y);
                ss.push_back(sum_s);
                sdx.push_back(0.0);
                sdy.push_back(0.0);
                for (uint16_t i = 0; i < (uint16_t)trajectory.point.size(); i++) {
                    double x = sx.back() + cos(ego_yaw) * 0.1;
                    double y = sy.back() + sin(ego_yaw) * 0.1;
                    double dx = x - sx.back();
                    double dy = y - sy.back();
                    sum_s += sqrt(dx * dx + dy * dy);
                    sx.push_back(x);
                    sy.push_back(y);
                    ss.push_back(sum_s);
                    sdx.push_back(dx);
                    sdy.push_back(dy);
                }
            }
            
            road_map = tk::Map(sx, sy, ss, sdx, sdy);
        }

        return road_map;
    }

    inline tk::Map GenerateBoundaryMap(const TrajectoryBoundary& trajectory_boundary) {
        tk::Map road_map;
        if ((uint16_t)trajectory_boundary.point.size() < 4) {
            util_function::DebugPrintError(
                "Cannot generate boundary road map to spline!"
            );
            return road_map;
        }

        std::vector<double> sx, sy, ss, sdx, sdy;
        double sum_s = 0.0;
        for (uint16_t i = 0; i < (uint16_t)trajectory_boundary.point.size(); i++) {
            double x = trajectory_boundary.point[i].x;
            double y = trajectory_boundary.point[i].y;
            double dx = i == 0 ? 0.0 : x - trajectory_boundary.point[i - 1].x;
            double dy = i == 0 ? 0.0 : y - trajectory_boundary.point[i - 1].y;
            sum_s += sqrt(dx * dx + dy * dy);
            sx.push_back(x);
            sy.push_back(y);
            ss.push_back(sum_s);
            sdx.push_back(dx);
            sdy.push_back(dy);
        }

        road_map = tk::Map(sx, sy, ss, sdx, sdy);
        return road_map;
    }

    template <typename T>
    inline tk::Map GeneratePointMap(const T& points, const double& distance_threshold = 0.01) {
        tk::Map road_map;
        if ((uint16_t)points.size() < 4) {
            util_function::DebugPrintError(
                "Cannot generate road map to spline!"
            );
        } else {
            std::vector<double> sx, sy, ss, sdx, sdy;
            double sum_s = 0.0;
            double prev_sum_s = 0.0;
            for (uint16_t i = 0; i < (uint16_t)points.size(); i++) {
                double x = points[i].x;
                double y = points[i].y;
                double dx = i == 0 ? 0.0 : x - points[i - 1].x;
                double dy = i == 0 ? 0.0 : y - points[i - 1].y;
                sum_s += sqrt(dx * dx + dy * dy);

                if(sum_s - prev_sum_s > distance_threshold) {                    
                    sx.push_back(x);
                    sy.push_back(y);
                    ss.push_back(sum_s);
                    sdx.push_back(dx);
                    sdy.push_back(dy);
                    prev_sum_s = sum_s;
                }
            }

            road_map = tk::Map(sx, sy, ss, sdx, sdy);
        }

        return road_map;
    }

    // reference_line: 0 left, 1 center, 2 right, 3 raceline (Trajectory.point)
    // Pulls (sx, sy, ss, sdx, sdy) into the output vectors so callers can build
    // a tk::Map. Ported from autoku-r function_trajectories.hpp.
    inline bool GetPredictionFrenetReference(const Trajectory& reference,
                                             const int& reference_line,
                                             std::vector<double>& sx,
                                             std::vector<double>& sy,
                                             std::vector<double>& ss,
                                             std::vector<double>& sdx,
                                             std::vector<double>& sdy) {
        if (reference.point.size() < 4) {
            util_function::DebugPrintValueError(
                "Size of reference_map point is not enough to calculate frenet reference!\n"
                "Size of reference_map point: ", (int)reference.point.size()
            );
            return false;
        }

        if (reference_line == 0 || reference_line == 1 || reference_line == 2) {
            TrajectoryBoundary boundary;
            if (reference_line == 0)      boundary = reference.left_boundary;
            else if (reference_line == 1) boundary = reference.center_boundary;
            else                          boundary = reference.right_boundary;

            if ((int)boundary.point.size() < 4) {
                util_function::DebugPrintValueError(
                    "Size of boundary point is not enough to calculate frenet reference!\n"
                    "Size of boundary point: ", (int)boundary.point.size()
                );
                return false;
            }

            double sum_s = 0.0;
            for (uint16_t i = 0; i < (uint16_t)boundary.point.size(); i++) {
                double x = boundary.point[i].x;
                double y = boundary.point[i].y;
                double dx = i == 0 ? 0.0 : x - boundary.point[i - 1].x;
                double dy = i == 0 ? 0.0 : y - boundary.point[i - 1].y;
                sum_s += std::sqrt(dx * dx + dy * dy);
                sx.push_back(x);
                sy.push_back(y);
                ss.push_back(sum_s);
                sdx.push_back(dx);
                sdy.push_back(dy);
            }
        }
        else {
            // Race line / optimal path
            double sum_s = 0.0;
            for (uint16_t i = 0; i < (uint16_t)reference.point.size(); i++) {
                double x = reference.point[i].x;
                double y = reference.point[i].y;
                double dx = i == 0 ? 0.0 : x - reference.point[i - 1].x;
                double dy = i == 0 ? 0.0 : y - reference.point[i - 1].y;
                sum_s += std::sqrt(dx * dx + dy * dy);
                sx.push_back(x);
                sy.push_back(y);
                ss.push_back(sum_s);
                sdx.push_back(dx);
                sdy.push_back(dy);
            }
        }

        if ((int)ss.size() < 4) {
            util_function::DebugPrintValueError(
                "Size of output frenet point is not enough!\n"
                "Size of output frenet point:", (int)ss.size()
            );
            return false;
        }
        return true;
    }

    template <typename T>
    inline bool GenerateBoundarySpline(const T& reference,
                                       tk::Map& road_map,
                                       tk::spline& left_boundary,
                                       tk::spline& right_boundary) {
        if (reference.left_boundary.point.size() < 4 || reference.right_boundary.point.size() < 4) {
            util_function::DebugPrintValueError(
                "Boundary points are not enough!",
                (uint16_t)reference.left_boundary.point.size()
            );
            return false;
        }
        // tk::spline::set_points requires STRICTLY increasing x; both linear and
        // cubic paths read x[n-2]/x[n-1] near the tail unconditionally, so
        // x.size() < 3 dereferences out-of-bounds (SIGSEGV in release builds).
        // Equal x values divide by zero in the linear coefficient loop and
        // poison every subsequent spline query with NaN. Both surface when the
        // referenced trajectory differs from the boundaries' shape (e.g. Mode 1
        // PIT segment with .osm boundaries from MAIN — many MAIN points project
        // to the same s on the short PIT road_map → dedup collapse + duplicates).
        constexpr double kBoundaryDedupEps = 1e-3;  // 1 mm in s-space
        auto project_monotonic = [&](const auto& src,
                                     std::vector<double>& s_out,
                                     std::vector<double>& n_out) {
            for (auto bp : src) {
                std::vector<double> sn = road_map.ToFrenet(bp.x, bp.y);
                if (!s_out.empty() && sn.at(0) <= s_out.back() + kBoundaryDedupEps) continue;
                s_out.push_back(sn.at(0));
                n_out.push_back(sn.at(1));
            }
        };

        std::vector<double> left_s, left_n, right_s, right_n;
        project_monotonic(reference.left_boundary.point,  left_s,  left_n);
        project_monotonic(reference.right_boundary.point, right_s, right_n);

        if (left_s.size() < 4 || right_s.size() < 4) {
            util_function::DebugPrintValueError(
                "Boundary points are not enough after monotonic projection!",
                (uint16_t)left_s.size()
            );
            return false;
        }

        left_boundary.set_points(left_s, left_n, false);
        right_boundary.set_points(right_s, right_n, false);
        return true;
    }

    inline void GenerateBoundarySplineCenter(const Trajectory& reference,
                                             tk::Map& center_map,
                                             tk::spline& left_boundary,
                                             tk::spline& right_boundary) {
        if (reference.left_boundary.point.size() < 4 || reference.right_boundary.point.size() < 4) {
            util_function::DebugPrintValueError(
                "Boundary points are not enough!",
                (uint16_t)reference.left_boundary.point.size()
            );
            return;
        }
        std::vector<double> left_s, left_n, right_s, right_n;
        for (auto lp : reference.left_boundary.point) {
            std::vector<double> bp_sn = center_map.ToFrenet(lp.x, lp.y);
            left_s.push_back(bp_sn.at(0));
            left_n.push_back(bp_sn.at(1));
        }
        for (auto rp : reference.right_boundary.point) {
            std::vector<double> bp_sn = center_map.ToFrenet(rp.x, rp.y);
            right_s.push_back(bp_sn.at(0));
            right_n.push_back(bp_sn.at(1));
        }
        left_boundary.set_points(left_s, left_n);
        right_boundary.set_points(right_s, right_n);
    }

    inline Trajectory ConvertTrajectoryByCenter(const Trajectory& trajectory,
                                                tk::Map& center_map) {
        Trajectory converted_trajectory = trajectory;
        for (auto& point : converted_trajectory.point) {
            std::vector<double> bp_sn = center_map.ToFrenet(point.x, point.y);
            point.s = (float)bp_sn.at(0);
            point.n = (float)bp_sn.at(1);
            point.distance = point.s;
        }
        return converted_trajectory;
    }

    inline Trajectory RestrictLateralAccel(const Trajectory& trajectory,
                                           const float& max_lateral_accel,
                                           const float& max_longitudinal_accel,
                                           const float& dt) {
        Trajectory restricted_trajectory = trajectory;  

        float max_restricted_speed = std::numeric_limits<float>::max();

        for(uint32_t i = 1; i < restricted_trajectory.point.size(); i++) {
            // Apply speed limit
            float restricted_speed = sqrt(fabs(max_lateral_accel / restricted_trajectory.point[i].curvature));

            if ((restricted_speed - restricted_trajectory.point[i-1].speed)/dt <= -max_longitudinal_accel) {
                restricted_speed = restricted_trajectory.point[i-1].speed - max_longitudinal_accel * dt;
            }
            restricted_trajectory.point[i].speed = std::min(restricted_trajectory.point[i].speed, restricted_speed);
        }

        // Forward smoothing
        for (uint32_t i = 0; i < restricted_trajectory.point.size(); i++) {
            // Calculate acceleration
            double v = restricted_trajectory.point.at(i).speed;
            double v_old = i == 0 ? v : restricted_trajectory.point.at(i-1).speed;
            double a = (pow(v, 2) - pow(v_old, 2)) / (2);

            // Smoothen the speed not to exceed the maximum acceleration
            if (a > max_longitudinal_accel) {
                restricted_trajectory.point.at(i).speed = sqrt(pow(v_old, 2) + 2 * max_longitudinal_accel);
            }
        }

        // Backward smoothing
        for (uint32_t i = restricted_trajectory.point.size() - 1; i > 0; i--) {
            // Calculate acceleration
            double v = restricted_trajectory.point.at(i).speed;
            double v_old = restricted_trajectory.point.at(i-1).speed;
            double a = (pow(v, 2) - pow(v_old, 2)) / (2);

            // Backward smoothing according to minimum acceleration
            if (a < -max_longitudinal_accel) {  
                restricted_trajectory.point.at(i-1).speed = sqrt(pow(v, 2) - 2 * (-max_longitudinal_accel));
            }
        }

        // Recalculate time
        for(uint32_t i = 1; i < restricted_trajectory.point.size(); i++){
            TrajectoryPoint& current_point = restricted_trajectory.point[i];
            const TrajectoryPoint& previous_point = restricted_trajectory.point[i-1];

            // time = distance / speed
            double distance = current_point.distance - previous_point.distance;
            double speed = (current_point.speed + previous_point.speed) / 2.0;

            if (speed >= 0.1) {
                current_point.time = previous_point.time + (distance / speed);
            }
            else {
                current_point.time = previous_point.time + 0.1;
            }
        }

        // Recalculate acceleration
        for(uint32_t i = 1; i < restricted_trajectory.point.size(); i++){
            TrajectoryPoint& current_point = restricted_trajectory.point[i];
            const TrajectoryPoint& previous_point = restricted_trajectory.point[i-1];

            // v = v0 + at
            double dt = current_point.time - previous_point.time;
            current_point.acceleration = (current_point.speed - previous_point.speed) / dt;
        }

        return restricted_trajectory;
    }

    inline bool GenerateSparseRoadMap(const std::vector<interface::Point> points,
                                      tk::Map& road_map,
                                      double distance_threshold) {
        if ((uint16_t)points.size() < 4) {            
            util_function::DebugPrintError(
                "Cannot generate road map to spline!"
            );
            return false;
        } else {
            std::vector<double> sx, sy, ss, sdx, sdy;
            double sum_s = 0.0;
            for (uint16_t i = 0; i < (uint16_t)points.size(); i++) {
                double x = points[i].x;
                double y = points[i].y;
                double dx = i == 0 ? 0.0 : x - points[i - 1].x;
                double dy = i == 0 ? 0.0 : y - points[i - 1].y;
                double ds = sqrt(dx * dx + dy * dy);
                if(ds > distance_threshold){
                    sum_s += ds;
                    sx.push_back(x);
                    sy.push_back(y);
                    ss.push_back(sum_s);
                    sdx.push_back(dx);
                    sdy.push_back(dy);
                }else{
                    continue;
                }
            }

            if((uint16_t)ss.size() < 4){
                // util_function::DebugPrintError(
                //     "Cannot generate road map to spline!"
                // );
                return false;
            }
            
            road_map = tk::Map(sx, sy, ss, sdx, sdy);
            return true;
        }
    }

    inline double GetDistToPath(const double& position_x, const double& position_y, const double& yaw,
                                const double& length, const double& width, tk::Map& road_map) {

        std::vector<double> object_center = {position_x, position_y};
        std::vector<double> object_center_frenet = road_map.ToFrenet(object_center.at(0), object_center.at(1));
        std::vector<double> object_left_front = {object_center.at(0) + length/2.0 * cos(yaw) - width/2.0 * sin(yaw),
                                                 object_center.at(1) + length/2.0 * sin(yaw) + width/2.0 * cos(yaw)};
        std::vector<double> object_left_front_frenet = road_map.ToFrenet(object_left_front.at(0), object_left_front.at(1));        
        std::vector<double> object_right_front = {object_center.at(0) + length/2.0 * cos(yaw) + width/2.0 * sin(yaw),
                                                  object_center.at(1) + length/2.0 * sin(yaw) - width/2.0 * cos(yaw)};
        std::vector<double> object_right_front_frenet = road_map.ToFrenet(object_right_front.at(0), object_right_front.at(1));
        std::vector<double> object_left_rear = {object_center.at(0) - length/2.0 * cos(yaw) - width/2.0 * sin(yaw),
                                                object_center.at(1) - length/2.0 * sin(yaw) + width/2.0 * cos(yaw)};
        std::vector<double> object_left_rear_frenet = road_map.ToFrenet(object_left_rear.at(0), object_left_rear.at(1));        
        std::vector<double> object_right_rear = {object_center.at(0) - length/2.0 * cos(yaw) + width/2.0 * sin(yaw),
                                                 object_center.at(1) - length/2.0 * sin(yaw) - width/2.0 * cos(yaw)};
        std::vector<double> object_right_rear_frenet = road_map.ToFrenet(object_right_rear.at(0), object_right_rear.at(1));

        double dist_to_path = fabs(object_center_frenet.at(1));

        dist_to_path = std::min({dist_to_path,
                            fabs(object_left_front_frenet.at(1)),
                            fabs(object_right_front_frenet.at(1)),
                            fabs(object_left_rear_frenet.at(1)),
                            fabs(object_right_rear_frenet.at(1))});

        if (object_left_front_frenet.at(1) * object_right_rear_frenet.at(1) < 0.0) {
            dist_to_path = 0.0;
        }
        if (object_left_rear_frenet.at(1) * object_right_front_frenet.at(1) < 0.0) {
            dist_to_path = 0.0;
        }
        if (object_left_front_frenet.at(1) * object_left_rear_frenet.at(1) < 0.0) {
            dist_to_path = 0.0;
        }
        if (object_right_front_frenet.at(1) * object_right_rear_frenet.at(1) < 0.0) {
            dist_to_path = 0.0;
        }

        return dist_to_path;
    }

    inline bool CheckObjectInLaneBoundary(const double& position_x, const double& position_y, const double& yaw,
                                          const double& length, const double& width, tk::Map& road_map,
                                          tk::spline& left_boundary, tk::spline& right_boundary) {
        std::vector<double> object_center = {position_x, position_y};
        std::vector<double> object_center_frenet = road_map.ToFrenet(object_center.at(0), object_center.at(1));
        std::vector<double> object_left_front = {object_center.at(0) + length/2.0 * cos(yaw) - width/2.0 * sin(yaw),
                                                 object_center.at(1) + length/2.0 * sin(yaw) + width/2.0 * cos(yaw)};
        std::vector<double> object_left_front_frenet = road_map.ToFrenet(object_left_front.at(0), object_left_front.at(1));        
        std::vector<double> object_right_front = {object_center.at(0) + length/2.0 * cos(yaw) + width/2.0 * sin(yaw),
                                                  object_center.at(1) + length/2.0 * sin(yaw) - width/2.0 * cos(yaw)};
        std::vector<double> object_right_front_frenet = road_map.ToFrenet(object_right_front.at(0), object_right_front.at(1));
        std::vector<double> object_left_rear = {object_center.at(0) - length/2.0 * cos(yaw) - width/2.0 * sin(yaw),
                                                object_center.at(1) - length/2.0 * sin(yaw) + width/2.0 * cos(yaw)};
        std::vector<double> object_left_rear_frenet = road_map.ToFrenet(object_left_rear.at(0), object_left_rear.at(1));        
        std::vector<double> object_right_rear = {object_center.at(0) - length/2.0 * cos(yaw) + width/2.0 * sin(yaw),
                                                 object_center.at(1) - length/2.0 * sin(yaw) - width/2.0 * cos(yaw)};
        std::vector<double> object_right_rear_frenet = road_map.ToFrenet(object_right_rear.at(0), object_right_rear.at(1));
        
        // Filter if the lanelet is diverging
        if (left_boundary(0.0) * right_boundary(0.0) >= 0.0) {
            if (object_center_frenet.at(0) + length < 0.0) {
                // util_function::DebugPrintError(
                //     "Left Boundary: " + std::to_string(left_boundary(0.0)) + "\n"
                //     "Right Boundary: " + std::to_string(right_boundary(0.0)) + "\n"
                //     "Start S: " + std::to_string(start_s)
                // );
                return false;
            }
        }        

        if (left_boundary(object_center_frenet.at(0)) > object_center_frenet.at(1) &&
            right_boundary(object_center_frenet.at(0)) < object_center_frenet.at(1)) {
            // util_function::DebugPrintWarn(
            //     "Left Boundary: " + std::to_string(left_boundary(object_center_frenet.at(0))) + "\n"
            //     "Right Boundary: " + std::to_string(right_boundary(object_center_frenet.at(0))) + "\n"
            //     "Object Center Frenet: " + std::to_string(object_center_frenet.at(1)) + "\n"
            // );
            return true;
        }
        else if (left_boundary(object_left_front_frenet.at(0)) > object_left_front_frenet.at(1) &&
            right_boundary(object_left_front_frenet.at(0)) < object_left_front_frenet.at(1)) {
            // util_function::DebugPrintWarn(
            //     "Left Boundary: " + std::to_string(left_boundary(object_left_front_frenet.at(0))) + "\n"
            //     "Right Boundary: " + std::to_string(right_boundary(object_left_front_frenet.at(0))) + "\n"
            //     "Object Left Front Frenet: " + std::to_string(object_left_front_frenet.at(1)) + "\n"
            // );
            return true;
        }
        else if (left_boundary(object_right_front_frenet.at(0)) > object_right_front_frenet.at(1) &&
            right_boundary(object_right_front_frenet.at(0)) < object_right_front_frenet.at(1)) {
            // util_function::DebugPrintWarn(
            //     "Left Boundary: " + std::to_string(left_boundary(object_right_front_frenet.at(0))) + "\n"
            //     "Right Boundary: " + std::to_string(right_boundary(object_right_front_frenet.at(0))) + "\n"
            //     "Object Right Front Frenet: " + std::to_string(object_right_front_frenet.at(1)) + "\n"
            // );
            return true;
        }
        else if (left_boundary(object_left_rear_frenet.at(0)) > object_left_rear_frenet.at(1) &&
            right_boundary(object_left_rear_frenet.at(0)) < object_left_rear_frenet.at(1)) {
            // util_function::DebugPrintWarn(
            //     "Left Boundary: " + std::to_string(left_boundary(object_left_rear_frenet.at(0))) + "\n"
            //     "Right Boundary: " + std::to_string(right_boundary(object_left_rear_frenet.at(0))) + "\n"
            //     "Object Left Rear Frenet: " + std::to_string(object_left_rear_frenet.at(1)) + "\n"
            // );
            return true;
        }
        else if (left_boundary(object_right_rear_frenet.at(0)) > object_right_rear_frenet.at(1) &&
            right_boundary(object_right_rear_frenet.at(0)) < object_right_rear_frenet.at(1)) {
            // util_function::DebugPrintWarn(
            //     "Left Boundary: " + std::to_string(left_boundary(object_right_rear_frenet.at(0))) + "\n"
            //     "Right Boundary: " + std::to_string(right_boundary(object_right_rear_frenet.at(0))) + "\n"
            //     "Object Right Rear Frenet: " + std::to_string(object_right_rear_frenet.at(1)) + "\n"
            // );
            return true;
        }
        else if (object_left_front_frenet.at(1) * object_right_rear_frenet.at(1) < 0.0
                 && left_boundary(object_left_front_frenet.at(0)) > object_left_front_frenet.at(1)
                 && left_boundary(object_right_rear_frenet.at(0)) < object_right_rear_frenet.at(1)) {
            // util_function::DebugPrintWarn(
            //     "Left Front Frenet: " + std::to_string(object_left_front_frenet.at(1)) + "\n"
            //     "Right Rear Frenet: " + std::to_string(object_right_rear_frenet.at(1)) + "\n"
            // );
            return true;
        }
        else if (object_left_rear_frenet.at(1) * object_right_front_frenet.at(1) < 0.0
                 && left_boundary(object_left_rear_frenet.at(0)) > object_left_rear_frenet.at(1)
                 && left_boundary(object_right_front_frenet.at(0)) < object_right_front_frenet.at(1)) {
            // util_function::DebugPrintWarn(
            //     "Left Rear Frenet: " + std::to_string(object_left_rear_frenet.at(1)) + "\n"
            //     "Right Front Frenet: " + std::to_string(object_right_front_frenet.at(1)) + "\n"
            // );
            return true;
        }
        else if (object_left_front_frenet.at(1) * object_left_rear_frenet.at(1) < 0.0
                 && left_boundary(object_left_front_frenet.at(0)) > object_left_front_frenet.at(1)
                 && left_boundary(object_left_rear_frenet.at(0)) < object_left_rear_frenet.at(1)) {
            // util_function::DebugPrintWarn(
            //     "Left Front Frenet: " + std::to_string(object_left_front_frenet.at(1)) + "\n"
            //     "Left Rear Frenet: " + std::to_string(object_left_rear_frenet.at(1)) + "\n"
            // );
            return true;
        }
        else if (object_right_front_frenet.at(1) * object_right_rear_frenet.at(1) < 0.0
                 && left_boundary(object_right_front_frenet.at(0)) > object_right_front_frenet.at(1)
                 && left_boundary(object_right_rear_frenet.at(0)) < object_right_rear_frenet.at(1)) {
            // util_function::DebugPrintWarn(
            //     "Right Front Frenet: " + std::to_string(object_right_front_frenet.at(1)) + "\n"
            //     "Right Rear Frenet: " + std::to_string(object_right_rear_frenet.at(1)) + "\n"
            // );
            return true;
        }
        else {
            // util_function::DebugPrintError(
            //     "Object is not in lane boundary!"
            // );
            return false;
        }
    }

    inline std::pair<std::vector<double>, double> GetBuffer(const double& position_x, const double& position_y, const double& yaw,
                                                            const double& length, const double& width,
                                                            tk::Map& road_map, tk::spline& left_boundary, tk::spline& right_boundary) {
        std::vector<double> object_center = {position_x, position_y};
        std::vector<double> object_center_frenet = road_map.ToFrenet(object_center.at(0), object_center.at(1));
        std::vector<double> object_left_front = {object_center.at(0) + length/2.0 * cos(yaw) - width/2.0 * sin(yaw),
                                                 object_center.at(1) + length/2.0 * sin(yaw) + width/2.0 * cos(yaw)};
        std::vector<double> object_left_front_frenet = road_map.ToFrenet(object_left_front.at(0), object_left_front.at(1));        
        std::vector<double> object_right_front = {object_center.at(0) + length/2.0 * cos(yaw) + width/2.0 * sin(yaw),
                                                  object_center.at(1) + length/2.0 * sin(yaw) - width/2.0 * cos(yaw)};
        std::vector<double> object_right_front_frenet = road_map.ToFrenet(object_right_front.at(0), object_right_front.at(1));
        std::vector<double> object_left_rear = {object_center.at(0) - length/2.0 * cos(yaw) - width/2.0 * sin(yaw),
                                                object_center.at(1) - length/2.0 * sin(yaw) + width/2.0 * cos(yaw)};
        std::vector<double> object_left_rear_frenet = road_map.ToFrenet(object_left_rear.at(0), object_left_rear.at(1));        
        std::vector<double> object_right_rear = {object_center.at(0) - length/2.0 * cos(yaw) + width/2.0 * sin(yaw),
                                                 object_center.at(1) - length/2.0 * sin(yaw) - width/2.0 * cos(yaw)};
        std::vector<double> object_right_rear_frenet = road_map.ToFrenet(object_right_rear.at(0), object_right_rear.at(1));

        tk::spline reference_boundary, near_boundary;
        if (object_center_frenet.at(1) > 0.0) {
            reference_boundary  = right_boundary;
            near_boundary       = left_boundary;
        }
        else {
            reference_boundary  = left_boundary;
            near_boundary       = right_boundary;
        }

        double buffer = 0.0;
        double center_buffer = 0.0;
        double left_front_buffer = 0.0;
        double right_front_buffer = 0.0;
        double left_rear_buffer = 0.0;
        double right_rear_buffer = 0.0;

        if (object_center_frenet.at(1) > 0.0) {
            if ((object_center_frenet.at(1)         - near_boundary(object_center_frenet.at(0))) > 0.0
                && (object_left_front_frenet.at(1)  - near_boundary(object_left_front_frenet.at(0))) > 0.0
                && (object_right_front_frenet.at(1) - near_boundary(object_right_front_frenet.at(0))) > 0.0
                && (object_left_rear_frenet.at(1)   - near_boundary(object_left_rear_frenet.at(0))) > 0.0
                && (object_right_rear_frenet.at(1)  - near_boundary(object_right_rear_frenet.at(0))) > 0.0) {
                // util_function::DebugPrintError(
                //     "Object is not in lane boundary!"
                // );
                return std::make_pair(object_center_frenet, HUGE_VAL);
            }

            center_buffer       = object_center_frenet.at(1)      - reference_boundary(object_center_frenet.at(0));
            left_front_buffer   = object_left_front_frenet.at(1)  - reference_boundary(object_left_front_frenet.at(0));
            right_front_buffer  = object_right_front_frenet.at(1) - reference_boundary(object_right_front_frenet.at(0));
            left_rear_buffer    = object_left_rear_frenet.at(1)   - reference_boundary(object_left_rear_frenet.at(0));
            right_rear_buffer   = object_right_rear_frenet.at(1)  - reference_boundary(object_right_rear_frenet.at(0));
            
            buffer = std::min({center_buffer, left_front_buffer, right_front_buffer, left_rear_buffer, right_rear_buffer});

            // util_function::DebugPrintInfo(
            //     "Object is on the left side"
            // );
        }
        else {
            if ((object_center_frenet.at(1)         - near_boundary(object_center_frenet.at(0))) < 0.0
                && (object_left_front_frenet.at(1)  - near_boundary(object_left_front_frenet.at(0))) < 0.0
                && (object_right_front_frenet.at(1) - near_boundary(object_right_front_frenet.at(0))) < 0.0
                && (object_left_rear_frenet.at(1)   - near_boundary(object_left_rear_frenet.at(0))) < 0.0
                && (object_right_rear_frenet.at(1)  - near_boundary(object_right_rear_frenet.at(0))) < 0.0) {
                // util_function::DebugPrintError(
                //     "Object is not in lane boundary!"
                // );
                return std::make_pair(object_center_frenet, HUGE_VAL);
            }

            center_buffer       = reference_boundary(object_center_frenet.at(0))      - object_center_frenet.at(1);
            left_front_buffer   = reference_boundary(object_left_front_frenet.at(0))  - object_left_front_frenet.at(1);
            right_front_buffer  = reference_boundary(object_right_front_frenet.at(0)) - object_right_front_frenet.at(1);
            left_rear_buffer    = reference_boundary(object_left_rear_frenet.at(0))   - object_left_rear_frenet.at(1);
            right_rear_buffer   = reference_boundary(object_right_rear_frenet.at(0))  - object_right_rear_frenet.at(1);
            
            buffer = std::min({center_buffer, left_front_buffer, right_front_buffer, left_rear_buffer, right_rear_buffer});

            // util_function::DebugPrintInfo(
            //     "Object is on the right side"
            // );                        
        }

        std::vector<double> object_frenet = object_center_frenet;
        if (center_buffer == buffer) {
            object_frenet = object_center_frenet;
        }
        else if (left_front_buffer == buffer) {
            object_frenet = object_left_front_frenet;
        }
        else if (right_front_buffer == buffer) {
            object_frenet = object_right_front_frenet;
        }
        else if (left_rear_buffer == buffer) {
            object_frenet = object_left_rear_frenet;
        }
        else if (right_rear_buffer == buffer) {
            object_frenet = object_right_rear_frenet;
        }
        object_frenet.at(0) = std::min({object_center_frenet.at(0), object_left_front_frenet.at(0), object_right_front_frenet.at(0), object_left_rear_frenet.at(0), object_right_rear_frenet.at(0)});

        // util_function::DebugPrintWarn(
        //     "center_buffer:\t " + std::to_string(center_buffer) 
        // );
        // util_function::DebugPrintWarn(
        //     "left_front_buffer:\t " + std::to_string(left_front_buffer) 
        // );
        // util_function::DebugPrintWarn(
        //     "right_front_buffer:\t " + std::to_string(right_front_buffer) 
        // );
        // util_function::DebugPrintWarn(
        //     "left_rear_buffer:\t " + std::to_string(left_rear_buffer) 
        // );
        // util_function::DebugPrintWarn(
        //     "right_rear_buffer:\t " + std::to_string(right_rear_buffer)
        // );
        // util_function::DebugPrintWarn(
        //     "\tbuffer:\t\t " + std::to_string(buffer)
        // );

        // if (object_left_front_frenet.at(1) * object_right_rear_frenet.at(1) < 0.0) {
        //     buffer = 0.0;
        //     object_frenet = object_right_rear_frenet;
        // }
        // if (object_left_rear_frenet.at(1) * object_right_front_frenet.at(1) < 0.0) {
        //     buffer = 0.0;
        //     object_frenet = object_left_rear_frenet;
        // }
        // if (object_left_front_frenet.at(1) * object_left_rear_frenet.at(1) < 0.0) {
        //     buffer = 0.0;
        //     object_frenet = object_left_rear_frenet;
        // }
        // if (object_right_front_frenet.at(1) * object_right_rear_frenet.at(1) < 0.0) {
        //     buffer = 0.0;
        //     object_frenet = object_right_rear_frenet;
        // }

        return std::make_pair(object_frenet, buffer);
    }

    // Generate a dt-parameterized target trajectory from a reference trajectory.
    // Used by both LateralControl and LongitudinalControl.
    //   reference_time_offset : time offset from [Lateral/Longitudinal Control] ini
    //   step_num, step_time   : from ptr_*_controller_->GetControlTrajectoryStep*()
    inline interface::Trajectory GenerateDTTargetTrajectory(
            const interface::VehicleState& vehicle_state,
            const interface::Trajectory&   trajectory,
            tk::Map&                       road_map,
            double                         reference_time_offset,
            int                            step_num,
            double                         step_time) {

        interface::Trajectory interpolated_trajectory;
        if (trajectory.point.empty()) return interpolated_trajectory;

        std::vector<double> time, speed, distance;
        double sum_s = 0.0;
        double sum_time = 0.0;
        for (uint16_t i = 0; i < (uint16_t)trajectory.point.size(); i++) {
            double dx = i == 0 ? 0.0 : trajectory.point[i].x - trajectory.point[i - 1].x;
            double dy = i == 0 ? 0.0 : trajectory.point[i].y - trajectory.point[i - 1].y;
            double ds = std::sqrt(dx * dx + dy * dy);
            // At rest both vx and reference speed can be ~0 which blows up `dt`
            // (ds/0.1) and collapses the (time,distance) spline to a near-flat
            // slope — the control-side time-window then samples almost zero
            // motion. Use the larger of ego vx and reference speed so the
            // time axis stays meaningful when either side is coasting.
            double ref_speed = std::max((double)trajectory.point[i].speed,
                                        (double)vehicle_state.vx);
            double dt = ds / std::max(ref_speed, 1.0);
            speed.push_back(trajectory.point[i].speed);
            sum_s    += ds;
            sum_time += dt;
            distance.push_back(sum_s);
            time.push_back(sum_time);
        }

        if (time.size() < 2 || speed.size() < 2 || distance.size() < 2) {
            return interpolated_trajectory;
        }

        tk::spline st, ts, tv;
        bool use_cubic = vehicle_state.vx >= 0.1;
        st.set_points(distance, time,     use_cubic);
        ts.set_points(time,     distance, use_cubic);
        tv.set_points(time,     speed,    use_cubic);

        std::vector<double> ego_sn = road_map.ToFrenet(vehicle_state.x, vehicle_state.y);
        double time_offset = st(ego_sn[0]) + reference_time_offset;

        double previous_yaw = vehicle_state.yaw;
        double yaw_offset   = 0.0;

        for (int i = 0; i < step_num + 2; i++) {
            interface::TrajectoryPoint dp;
            double target_distance = ts(step_time * i + time_offset);
            std::vector<double> xy = road_map.ToCartesian(target_distance, 0.0);
            dp.x = xy[0];
            dp.y = xy[1];

            double current_yaw = road_map.GetSlope(target_distance);
            double delta_yaw   = current_yaw - previous_yaw;
            if      (delta_yaw >  M_PI) yaw_offset -= 2 * M_PI;
            else if (delta_yaw < -M_PI) yaw_offset += 2 * M_PI;
            dp.yaw       = current_yaw + yaw_offset;
            previous_yaw = current_yaw;

            dp.curvature    = road_map.GetCurvature(target_distance);
            dp.speed        = tv(step_time * i + time_offset);
            dp.time         = step_time * i;
            dp.distance     = target_distance - ts(time_offset);
            dp.acceleration = 0.0;
            interpolated_trajectory.point.push_back(dp);
        }
        return interpolated_trajectory;
    }

    // Resample trajectory by uniform arc-length (distance) step. Ported from
    // autoku-r (trajectory_planning optimal_selector / candidate_generator
    // pipeline). Preserves time/speed/curvature via cubic splines.
    inline interface::Trajectory ResampleTrajectoryDistance(const interface::Trajectory& trajectory,
                                                            const double& resample_distance) {
        interface::Trajectory resampled_trajectory = trajectory;
        if (resampled_trajectory.point.size() < 4) {
            return resampled_trajectory;
        }

        std::vector<double> time, x, y, speed, distance, curvature;
        for (const auto& point : trajectory.point) {
            time.push_back(point.time - trajectory.point.at(0).time);
            x.push_back(point.x);
            y.push_back(point.y);
            speed.push_back(point.speed);
            distance.push_back(point.distance);
            curvature.push_back(point.curvature);
        }

        resampled_trajectory.point.clear();
        tk::spline sx, sy, sv, st, sk;
        sx.set_points(distance, x);
        sy.set_points(distance, y);
        sv.set_points(distance, speed);
        st.set_points(distance, time);
        sk.set_points(distance, curvature);

        int num_total_point = (int)std::ceil((distance.back() - distance.front()) / resample_distance);
        double start_s = distance.front();
        for (int i = 0; i < num_total_point; i++) {
            interface::TrajectoryPoint tp{};
            double s_i  = resample_distance * i + start_s;
            double s_n  = resample_distance * (i + 1) + start_s;
            tp.time      = (float)(st(s_i) + trajectory.point.at(0).time);
            tp.x         = (float)sx(s_i);
            tp.y         = (float)sy(s_i);
            tp.z         = 0.0f;
            tp.speed     = (float)sv(s_i);
            tp.distance  = (float)(s_i - start_s);
            tp.curvature = (float)sk(s_i);
            double dx = sx(s_n) - tp.x;
            double dy = sy(s_n) - tp.y;
            tp.yaw          = (float)atan2(dy, dx);
            tp.acceleration = 0.0f;
            resampled_trajectory.point.push_back(tp);
        }
        return resampled_trajectory;
    }

    // Resample trajectory by uniform time step. Ported from autoku-r.
    inline interface::Trajectory ResampleTrajectoryTime(const interface::Trajectory& trajectory,
                                                        const double& resample_time) {
        interface::Trajectory resampled_trajectory = trajectory;
        if (resampled_trajectory.point.size() < 4) {
            return resampled_trajectory;
        }

        std::vector<double> time, x, y, speed, distance;
        for (const auto& point : resampled_trajectory.point) {
            time.push_back(point.time);
            x.push_back(point.x);
            y.push_back(point.y);
            speed.push_back(point.speed);
            distance.push_back(point.distance);
        }

        resampled_trajectory.point.clear();
        tk::spline tx, ty, tv, ts;
        tx.set_points(time, x);
        ty.set_points(time, y);
        tv.set_points(time, speed);
        ts.set_points(time, distance);

        int num_total_point = (int)std::ceil(time.back() / resample_time);
        double time_offset = 0.0;
        for (int i = 0; i < num_total_point; i++) {
            interface::TrajectoryPoint tp;
            double t_i = resample_time * i + time_offset;
            double t_n = resample_time * (i + 1) + time_offset;
            tp.time         = (float)(resample_time * i);
            tp.x            = (float)tx(t_i);
            tp.y            = (float)ty(t_i);
            tp.z            = 0.0f;
            tp.speed        = (float)tv(t_i);
            tp.distance     = (float)(ts(t_i) - ts(time_offset));
            tp.yaw          = (float)atan2(ty(t_n) - tp.y, tx(t_n) - tp.x);
            tp.curvature    = 0.0f;
            tp.acceleration = 0.0f;
            resampled_trajectory.point.push_back(tp);
        }
        return resampled_trajectory;
    }

} // namespace util_function

#endif  // __FUNCTION_TRAJECTORIES_HPP__