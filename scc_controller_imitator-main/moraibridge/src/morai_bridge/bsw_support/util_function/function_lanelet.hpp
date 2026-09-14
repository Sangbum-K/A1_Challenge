/**
 * @file        function_lanelet.hpp
 * @brief       util functions for lanelet
 * 
 * @authors     Junhee Lee (998jun@gmail.com)          
 *              Seheon Ha (seheonha@hanyang.ac.kr)
 * 
 * @date        2024-04-11 created by Junhee Lee 
 *              2024-06-21 updated by Seheon Ha: add CheckAndBuildRoutingGraph function
 *              2024-11-01 updated by Yuseung Na: Refactoring
 * 
 */
#ifndef __FUNCTION_LANELET_HPP__
#define __FUNCTION_LANELET_HPP__
#pragma once

// STD header
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <sstream>
#include <unordered_map>

// Utility header
#include <lanelet2_core/geometry/LaneletMap.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_core/primitives/GPSPoint.h>
#include <lanelet2_core/utility/Utilities.h>
#include <lanelet2_io/io_handlers/Serialize.h>
#include <lanelet2_io/io_handlers/Factory.h>
#include <lanelet2_io/Io.h>
#include <lanelet2_projection/UTM.h>
#include <lanelet2_projection/LocalCartesian.h>
#include <lanelet2_routing/Route.h>
#include <lanelet2_routing/RoutingCost.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_routing/RoutingGraphContainer.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>

#include <pugixml.hpp>
#include "spline.h"

// Interface Header
#include "interface_lanelet.hpp"
#include "interface_route.hpp"

// Util Function Header
#include "function_print.hpp"
#include "function_common_math.hpp"

using namespace interface;

namespace interface {
    typedef struct {
        double boundary_distance{0.0};
        double center_distance{0.0};
        double heading_diff{0.0};
        lanelet::Lanelet lanelet;
    } LaneletWithDistance;
}

namespace util_function {
    inline std::vector<int8_t> EncodeLaneletMapBin(const lanelet::LaneletMapPtr& map){
        std::stringstream ss;
        boost::archive::binary_oarchive oa(ss);
        oa << *map;
        std::string str = ss.str();
        std::vector<int8_t> data(str.begin(), str.end());
        return data;
    } 

    inline lanelet::LaneletMapPtr DecodeLaneletMapBin(const std::vector<int8_t>& data){
        std::string str(data.begin(), data.end());
        std::stringstream ss(str);
        lanelet::LaneletMapPtr map = std::make_shared<lanelet::LaneletMap>();
        util_function::DebugPrintWarn(
            "Decode Lanelet Map Bin File ... "
        );
        boost::archive::binary_iarchive ia(ss);
        ia >> *map;
        return map;
    } 

    inline lanelet::GPSPoint GetMapCenter(const std::string& map_path){
        //std::pair<double, double> center;
        lanelet::GPSPoint center;
        pugi::xml_document map;
    
        // load the OSM(xml) file
        if (!map.load_file(map_path.c_str())) {
            std::cout <<"origin finder : Cannot open map in the path";
        }
        pugi::xml_node nodes = map.child("osm");

        double max_lat = -90.0;
        double max_lon = -180.0;
        double min_lat = 90.0;
        double min_lon = 180.0;
    
        for (pugi::xml_node_iterator it = nodes.begin(); it != nodes.end(); ++it) {
            for (pugi::xml_attribute_iterator ait = it->attributes_begin(); ait != it->attributes_end(); ++ait) {
                if(std::string(ait->name()) == "lat"){
                    double temp_lat = stod(std::string(ait->value()));
                    if(temp_lat > max_lat) {
                        max_lat = temp_lat;
                    }
                    if(temp_lat < min_lat) {
                        min_lat = temp_lat;
                    }
                }
                if(std::string(ait->name()) == "lon"){
                    double temp_lon = stod(std::string(ait->value()));
                    if(temp_lon > max_lon) {
                        max_lon = temp_lon;
                    }
                    if(temp_lon < min_lon) {
                        min_lon = temp_lon;
                    }
                }
            }
        }

        center.lat = 0.5*(max_lat + min_lat);
        center.lon = 0.5*(max_lon + min_lon);

        return center;
    }

    inline lanelet::routing::RoutingGraphUPtr BuildRoutingGraph(const lanelet::LaneletMapPtr& map){
        lanelet::traffic_rules::TrafficRulesPtr trafficRules =
                lanelet::traffic_rules::TrafficRulesFactory::create(lanelet::Locations::Germany, lanelet::Participants::Vehicle);
        return lanelet::routing::RoutingGraph::build(*map, *trafficRules);
    }

    inline bool CheckAndBuildRoutingGraph(const lanelet::LaneletMapPtr& lanelet_map, 
                                          lanelet::routing::RoutingGraphUPtr& routing_graph,
                                          uint32_t& prev_map_seq,
                                          const uint32_t& map_seq){
        if (map_seq != prev_map_seq) {
            util_function::DebugPrintWarn(
                "Map has been updated. Build Routing Graph"
            );
            routing_graph = BuildRoutingGraph(lanelet_map);
            prev_map_seq = map_seq;

            return true;
        }
        return false;
    }

    inline std::vector<interface::LaneletWithDistance> FindNearestWithCenterlineDistance(const lanelet::LaneletMapPtr& map,
                                                                                         const lanelet::BasicPoint3d& point,
                                                                                         unsigned count = 5) {
        lanelet::BasicPoint2d point_2d(point.x(), point.y());
        // Step 1: Use standard findNearest to get candidate lanelets
        std::vector<std::pair<double, lanelet::Lanelet>> candidates =
            lanelet::geometry::findNearest(map->laneletLayer, point_2d, count);

        // Step 2: Calculate centerline distance for each candidate
        std::vector<interface::LaneletWithDistance> results;

        for (const auto& candidate : candidates) {
            lanelet::ConstLanelet lanelet = map->laneletLayer.get(candidate.second.id());

            // Calculate distance to centerline
            double centerline_distance = lanelet::geometry::distanceToCenterline2d(lanelet, point_2d);

            // Calculate heading difference at the closest point on centerline
            double heading_diff = 0.0;
            auto centerline = lanelet.centerline2d();
            if (centerline.size() >= 2) {
                // Find closest point on centerline to the given point
                double min_dist = std::numeric_limits<double>::max();
                int closest_idx = 0;

                for (int i = 0; i < centerline.size(); i++) {
                    double dist = sqrt(pow(centerline[i].x() - point_2d.x(), 2) +
                                    pow(centerline[i].y() - point_2d.y(), 2));
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_idx = i;
                    }
                }

                // Calculate heading at the closest point
                double lanelet_heading = 0.0;
                if (closest_idx == 0 && centerline.size() > 1) {
                    // At start, use direction to next point
                    double dx = centerline[1].x() - centerline[0].x();
                    double dy = centerline[1].y() - centerline[0].y();
                    lanelet_heading = std::atan2(dy, dx);
                } else if (closest_idx == centerline.size() - 1 && centerline.size() > 1) {
                    // At end, use direction from previous point
                    double dx = centerline[closest_idx].x() - centerline[closest_idx - 1].x();
                    double dy = centerline[closest_idx].y() - centerline[closest_idx - 1].y();
                    lanelet_heading = std::atan2(dy, dx);
                } else if (centerline.size() > 2) {
                    // In middle, use average direction (prev to current, current to next)
                    double dx1 = centerline[closest_idx].x() - centerline[closest_idx - 1].x();
                    double dy1 = centerline[closest_idx].y() - centerline[closest_idx - 1].y();
                    double heading1 = std::atan2(dy1, dx1);

                    double dx2 = centerline[closest_idx + 1].x() - centerline[closest_idx].x();
                    double dy2 = centerline[closest_idx + 1].y() - centerline[closest_idx].y();
                    double heading2 = std::atan2(dy2, dx2);

                    // Calculate proper angle average considering wrapping
                    double angle_diff = util_function::NormalizeAngleRad(heading2 - heading1);

                    // Normalize to [-PI, PI]
                    lanelet_heading   = util_function::NormalizeAngleRad(heading1 + angle_diff / 2.0);                
                }

                // Calculate heading difference (point.z() contains yaw in BasicPoint3d)
                // Normalize ego heading to [-PI, PI] first
                double point_heading = util_function::NormalizeAngleRad(point.z());

                // Normalize heading difference to [-PI, PI]
                heading_diff = util_function::NormalizeAngleDiffRad(lanelet_heading, point_heading);

                // Use absolute value for comparison
                heading_diff = std::abs(heading_diff);
            }

            lanelet::Lanelet non_const_lanelet = const_cast<lanelet::Lanelet&>(
                reinterpret_cast<const lanelet::Lanelet&>(lanelet)
            );

            interface::LaneletWithDistance lanelet_with_distance;
            lanelet_with_distance.boundary_distance = candidate.first;
            lanelet_with_distance.center_distance   = centerline_distance;
            lanelet_with_distance.heading_diff      = heading_diff;
            lanelet_with_distance.lanelet           = non_const_lanelet;

            results.emplace_back(lanelet_with_distance);

            // util_function::DebugPrintInfo(
            //     "Lanelet ID: " + std::to_string(lanelet.id()) +
            //     ", Boundary distance: " + std::to_string(candidate.first) +
            //     ", Centerline distance: " + std::to_string(centerline_distance) +
            //     ", Heading diff: " + std::to_string(heading_diff * 180.0 / M_PI) + " deg"
            // );
        }

        // Step 3: Sort by centerline distance
        std::sort(results.begin(), results.end(),
            [](const interface::LaneletWithDistance& a,
            const interface::LaneletWithDistance& b) {
                return a.center_distance < b.center_distance;
            });

        // Step 4: Return only requested count
        if (results.size() > count) {
            results.resize(count);
        }

        return results;
    }

    inline lanelet::ConstLanelet GetClosestLanelet(const std::vector<interface::LaneletWithDistance>& lanelets,
                                                   double distance_condition = 1.0,
                                                   double heading_condition = 30.0) {
        if (lanelets.empty()) {
            return lanelet::ConstLanelet();
        }

        // If only one lanelet is found, return it
        if (lanelets.size() == 1) {
            return lanelets.front().lanelet;
        }

        // Step 1: Find lanelets with boundary_distance 0.0 (point is inside lanelet)
        std::vector<interface::LaneletWithDistance> inside_lanelet;
        for (const auto& ll_with_dist : lanelets) {
            if (std::fabs(ll_with_dist.boundary_distance) < distance_condition
                && std::fabs(ll_with_dist.heading_diff) < (heading_condition * M_PI / 180.0)) {
                inside_lanelet.push_back(ll_with_dist);
            }
        }

        // Step 2: If we have inside lanelets, process them
        if (!inside_lanelet.empty()) {
            if (inside_lanelet.size() == 1) {
                // Only one inside lanelet, return it
                // util_function::DebugPrintInfo(
                //     "Single inside lanelet found: " 
                //         + std::to_string(inside_lanelet.front().lanelet.id())
                // );
                return inside_lanelet.front().lanelet;
            } else {
                interface::LaneletWithDistance best_lanelet = inside_lanelet.front();
                
                // Multiple inside lanelets, compare headings
                for (const auto& ll : inside_lanelet) {
                    if (ll.heading_diff < best_lanelet.heading_diff) {
                        best_lanelet = ll;
                    }
                }

                return best_lanelet.lanelet;
            }
        }
        // Step 3: No inside lanelets, use score-based selection
        else {
            double best_score = -1.0;
            interface::LaneletWithDistance best_lanelet = lanelets.front();

            for (const auto& ll : lanelets) {
                // Use pre-calculated values
                double distance = ll.center_distance;
                double heading_diff = ll.heading_diff;

                // Calculate scores
                double distance_score = 1.0 / (1.0 + distance);
                double heading_score = 1.0 / (1.0 + heading_diff);

                // Combined score: 60% heading, 40% distance
                double total_score = 0.6 * heading_score + 0.4 * distance_score;

                if (total_score > best_score) {
                    best_score = total_score;
                    best_lanelet = ll;
                }
            }

            return best_lanelet.lanelet;
        }

        // If no candidates within threshold, return the closest lanelet
        return lanelets.front().lanelet;
    }
}

namespace util_function {
    // -----------------------------------------------------------------------
    // RacelinePoint: unified raceline OSM point with all fields
    // -----------------------------------------------------------------------
    struct RacelinePoint {
        double x          = 0.0;
        double y          = 0.0;
        double z          = 0.0;
        double psi_rad    = 0.0;  // heading (rad), computed from atan2 if OSM tag absent
        double vx_mps     = 0.0;  // IQP optimized race speed (m/s)
        double ax_mps2    = 0.0;  // IQP optimized longitudinal acceleration (m/s^2)
        double kappa_radpm = 0.0; // curvature (rad/m)
        double s_m        = 0.0;  // arc-length along raceline (m)
    };

    // -----------------------------------------------------------------------
    // ParseRacelineOsm: unified parser replacing LoadCenterlinePointsFromOsm()
    //   (goalpoint_planning) and LoadCenterlineVxFromOsm() (global_planning).
    //
    //   projector_type : "utm" | "local_cartesian"
    //   use_ref_origin : if true, use ref_lat/ref_lon as projection origin;
    //                    otherwise compute origin from map bounding box center
    // -----------------------------------------------------------------------
    inline bool ParseRacelineOsm(const std::string& osm_path,
                                 const std::string& projector_type,
                                 std::vector<RacelinePoint>& out,
                                 bool   use_ref_origin = false,
                                 double ref_lat        = 0.0,
                                 double ref_lon        = 0.0) {
        out.clear();

        pugi::xml_document doc;
        if (!doc.load_file(osm_path.c_str())) {
            return false;
        }

        pugi::xml_node osm = doc.child("osm");
        if (!osm) {
            return false;
        }

        lanelet::GPSPoint map_origin;
        if (use_ref_origin) {
            map_origin.lat = ref_lat;
            map_origin.lon = ref_lon;
        } else {
            map_origin = util_function::GetMapCenter(osm_path);
        }

        // --- parse nodes ---
        std::unordered_map<long long, RacelinePoint> id_to_point;
        for (pugi::xml_node node = osm.child("node"); node; node = node.next_sibling("node")) {
            long long id = node.attribute("id").as_llong();
            if (id == 0 && !node.attribute("id")) continue;

            lanelet::GPSPoint gps;
            gps.lat = node.attribute("lat").as_double();
            gps.lon = node.attribute("lon").as_double();

            lanelet::BasicPoint3d projected;
            if (projector_type == "utm") {
                projected = lanelet::projection::UtmProjector(lanelet::Origin(map_origin)).forward(gps);
            } else {
                projected = lanelet::projection::LocalCartesianProjector(lanelet::Origin(map_origin)).forward(gps);
            }

            RacelinePoint pt;
            pt.x = projected.x();
            pt.y = projected.y();
            pt.z = projected.z();

            for (pugi::xml_node tag = node.child("tag"); tag; tag = tag.next_sibling("tag")) {
                std::string k = tag.attribute("k").as_string();
                double      v = tag.attribute("v").as_double();
                if      (k == "vx_mps")       pt.vx_mps      = v;
                else if (k == "ax_mps2")      pt.ax_mps2     = v;
                else if (k == "kappa_radpm")  pt.kappa_radpm = v;
                else if (k == "psi_rad")      pt.psi_rad     = v;
                else if (k == "s_m")          pt.s_m         = v;
            }

            id_to_point[id] = pt;
        }

        // --- find longest way (raceline) ---
        std::vector<long long> best_way_refs;
        for (pugi::xml_node way = osm.child("way"); way; way = way.next_sibling("way")) {
            std::vector<long long> refs;
            for (pugi::xml_node nd = way.child("nd"); nd; nd = nd.next_sibling("nd")) {
                refs.push_back(nd.attribute("ref").as_llong());
            }
            if (refs.size() > best_way_refs.size()) {
                best_way_refs = refs;
            }
        }

        if (best_way_refs.size() < 2) {
            return false;
        }

        for (const auto& ref_id : best_way_refs) {
            auto it = id_to_point.find(ref_id);
            if (it != id_to_point.end()) {
                out.push_back(it->second);
            }
        }

        if (out.size() < 2) {
            out.clear();
            return false;
        }

        // --- fallback: compute psi_rad from consecutive points if all zero ---
        bool has_psi = false;
        for (const auto& pt : out) {
            if (pt.psi_rad != 0.0) { has_psi = true; break; }
        }
        if (!has_psi) {
            for (size_t i = 0; i + 1 < out.size(); ++i) {
                double dx = out[i + 1].x - out[i].x;
                double dy = out[i + 1].y - out[i].y;
                out[i].psi_rad = std::atan2(dy, dx);
            }
            out.back().psi_rad = out[out.size() - 2].psi_rad;
        }

        return true;
    }
} // namespace util_function

#endif // __FUNCTION_LANELET_HPP__
