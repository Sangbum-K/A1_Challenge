#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cstdint>
#include <array>
#include <map>
#include <string>
namespace a1::inspection {
Eigen::Matrix3d body_from_lidar(double yaw);
Eigen::Matrix3d nominal_optical_from_body();
bool in_front_half(const Eigen::Vector3d& body_point,double center_rad=0.);
// Measured XY minimum-area rectangle; yaw is a shape axis, not calibration or heading.
struct Footprint { double length{},width{},yaw{}; bool valid{}; };
Footprint measured_footprint(const std::vector<Eigen::Vector3d>& points);
struct Box { Eigen::Vector3d lo,hi; int points{}; Footprint footprint{}; Eigen::Vector3d center()const{return (lo+hi)*.5;} };
bool camera_surface_geometry(const Box&,bool retained_target=false);
// Partial surfaces are allowed; these are geometric bounds, not a vehicle classifier.
bool vehicle_extent(const Box&);
std::vector<std::vector<int>> merge_fragments(const std::vector<Box>&,double gap=.45);
// Autoware.Auto-inspired symmetric range-adaptive XY connectivity; own implementation.
struct ClusterConfig { double near_radius{.45}, start_range{20.}, slope{.01}, max_radius{.85}, max_height_gap{1.2}; int min_points{5}; };
std::vector<std::vector<int>> adaptive_clusters(const std::vector<Eigen::Vector3d>&,const ClusterConfig& config={});
bool full_vehicle_geometry(const Box&);
bool partial_vehicle_geometry(const Box&);
struct BinRegion { int left{},right{}; double peak{}; };
std::vector<BinRegion> camera_regions(const std::array<double,40>& probabilities,
 const std::vector<BinRegion>& previous,double enter,double hold,double edge,int lower=0,int upper=39);
bool in_target_range(const Box&,double minimum,double maximum);
struct Projection { bool valid{}; double x0{},y0{},x1{},y1{}; };
// Plumb-bob lens distortion; all zero means an ideal pinhole.
struct Distortion { double k1{},k2{},p1{},p2{}; };
Projection project(const Box&,const Eigen::Matrix3d& optical_from_body,const Eigen::Vector3d& camera_position,
                   double fx,double fy,double cx,double cy,int width,int height,const Distortion& distortion={});
// Forward region of interest; widens with range so corners and grades keep far vehicles.
bool in_processing_roi(double x,double y,double z,double half_width,double half_angle_tangent,double grade_margin);
std::vector<int> hungarian(const Eigen::MatrixXd& cost);
} // namespace
