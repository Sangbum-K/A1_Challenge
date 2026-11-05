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
struct Track {
  int64_t id{}; Eigen::Vector4d state=Eigen::Vector4d::Zero();
  Eigen::Matrix4d covariance=Eigen::Matrix4d::Identity();
  Box box,raw_box; int hits{},misses{}; double last_seen{}; bool measured{}; bool vehicle_evidence{};
};
// SORT-inspired CV Kalman + global Hungarian assignment, adapted to LiDAR BEV.
// Velocities are apparent velocities in the rotating sensor/body frame, NOT navigation velocities.
struct TrackerConfig { double coast_s{.3},memory_s{.8},max_position_variance{1.0}; };
class Tracker {
 public:
  explicit Tracker(TrackerConfig config={}):config_(config){}
  bool usable(const Track& t,double stamp)const;
  const std::vector<Track>& update(double stamp,const std::vector<Box>& detections);
  void reset(){tracks_.clear();last_=-1;}
 private:
  TrackerConfig config_;
  std::vector<Track> tracks_; int64_t next_{1}; double last_{-1};
};
// Nearest measured eligible target; retain the current primary during bounded coasting.
int64_t select_primary(const std::vector<const Track*>& eligible,int64_t current);
// Camera-supported observed surfaces need not contain a vehicle's full depth.
bool camera_surface_geometry(const Box&,bool retained_target=false);
struct TargetObservation {
  int64_t id{-1}; double range{}; bool usable{},measured{},matched{},ambiguous{};
};
struct TargetSelection {
  int64_t primary{-1},standby{-1}; double camera_age{};
  std::string status{"no_camera_lidar_target"};
};
// No birth without a current unambiguous camera match. A lease is renewed ONLY by
// a new camera observation, never by a LiDAR-only or repeated-camera observation.
class CameraTargetSelector {
 public:
  explicit CameraTargetSelector(double hold=.3,int confirmations=3):hold_(hold),confirmations_(confirmations){}
  TargetSelection update(double stamp,double camera_stamp,const std::vector<TargetObservation>& observations);
  void reset(){primary_=-1;last_=-1;last_matches_.clear();pending_.clear();}
 private:
  double hold_,last_{-1};int confirmations_;int64_t primary_{-1};std::map<int64_t,double> last_matches_;
  std::map<int64_t,std::pair<double,int>> pending_;
};
// Partial surfaces are allowed; these are geometric bounds, not a vehicle classifier.
bool vehicle_extent(const Box&);
std::vector<std::vector<int>> merge_fragments(const std::vector<Box>&,double gap=.45);
// Autoware.Auto-inspired symmetric range-adaptive XY connectivity; own implementation.
struct ClusterConfig { double near_radius{.45}, start_range{20.}, slope{.01}, max_radius{.85}, max_height_gap{1.2}; int min_points{5}; };
std::vector<std::vector<int>> adaptive_clusters(const std::vector<Eigen::Vector3d>&,const ClusterConfig& config={});
bool full_vehicle_geometry(const Box&);
bool partial_vehicle_geometry(const Box&);
const char* vehicle_geometry_status(const Track&);
struct BinRegion { int left{},right{}; double peak{}; };
std::vector<BinRegion> camera_regions(const std::array<double,40>& probabilities,
 const std::vector<BinRegion>& previous,double enter,double hold,double edge,int lower=0,int upper=39);
bool in_target_range(const Box&,double minimum,double maximum);
// Return a region index only for unambiguous one-to-one edges; -1 otherwise.
std::vector<int> unique_region_matches(const Eigen::MatrixXd& overlaps,double threshold);
struct Projection { bool valid{}; double x0{},y0{},x1{},y1{}; };
Projection project(const Box&,const Eigen::Matrix3d& optical_from_body,const Eigen::Vector3d& camera_position,
                   double fx,double fy,double cx,double cy,int width,int height);
double horizontal_iou(const Projection&,double bin_left,double bin_right);
std::vector<int> hungarian(const Eigen::MatrixXd& cost);
} // namespace
