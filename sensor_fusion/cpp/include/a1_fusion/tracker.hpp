#pragma once
#include "a1_fusion/association.hpp"
#include <cstdint>
#include <functional>
#include <limits>
namespace a1::inspection {
// Planar pose of the LiDAR origin in the continuous odometry frame.
struct Pose2 {
  double x{},y{},yaw{};
  Eigen::Vector2d to_world(const Eigen::Vector2d& body) const;
  Eigen::Vector2d to_body(const Eigen::Vector2d& world) const;
};
// Mean point time per azimuth degree. Whole-point deskew is unnecessary for a
// lead vehicle, which spans few degrees; a per-azimuth pose is enough.
class ScanTiming {
 public:
  void add(double azimuth_rad,double time);
  void finish(double fallback_time);
  double at(double azimuth_rad) const;
  bool valid() const { return valid_; }
 private:
  static int cell(double azimuth_rad);
  std::array<double,360> sum_{};std::array<int,360> count_{};std::array<double,360> time_{};
  double first_{std::numeric_limits<double>::infinity()},last_{-std::numeric_limits<double>::infinity()},fallback_{};
  bool valid_{};
};
// Tracks live in the odometry frame only while ego motion is available. Losing it,
// or regaining it stably, changes the frame, so update() returning true means reset.
class EgoLink {
 public:
  bool update(double stamp,bool available);
  bool compensated() const { return compensated_; }
 private:
  bool compensated_{};
  double available_since_{-1};
};
enum class Quality { Reject, Weak, Strong };
Quality classify(const Box&,double range,double weak_min_range);

struct AnchorFit {
  Eigen::Vector2d anchor=Eigen::Vector2d::Zero();  // Rear-face centre.
  double sigma_along{},sigma_lateral{},width{},length{};
  bool rear_visible{};
};
// The rear face is stable when a turning car shows its side; a box centre is not.
AnchorFit rear_anchor(const std::vector<Eigen::Vector2d>& xy,double heading,
                      const Eigen::Vector2d& sensor,double width_prior);

enum class Visibility { Outside, Visible, Occluded, Silent };

struct Track {
  using State=Eigen::Matrix<double,5,1>;  // px, py, v, heading, yaw rate. Continuous frame.
  using Cov=Eigen::Matrix<double,5,5>;
  int64_t id{};
  State x=State::Zero();
  Cov P=Cov::Identity();
  double t{};
  double length{4.5},width{1.9},height{1.5},z{},observed_width{};
  uint16_t seen{};
  int hits{},misses{};
  double last_seen{};
  bool measured{},confirmed{},weak_born{},weak_update{},guided{},rear_visible{true},vehicle_evidence{};
  Box raw_box{Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),0,{}};
  // Camera evidence, owned by camera_evidence.cpp.
  double evidence{},evidence_stamp{-1},camera_score{};
  Visibility visibility{Visibility::Silent};
  // Map structure overlap. NaN means the map could not judge this track.
  double static_ratio{std::numeric_limits<double>::quiet_NaN()};
  int static_updates{};
  bool map_static{},map_covered{};
  // Far-vehicle score and path coordinates, owned by the selection stage.
  double far_score{std::numeric_limits<double>::quiet_NaN()};
  bool far_vehicle{};
  double path_s{std::numeric_limits<double>::quiet_NaN()},path_n{std::numeric_limits<double>::quiet_NaN()};
  int path_hint{-1};
  Eigen::Vector2d anchor() const { return Eigen::Vector2d(x[0],x[1]); }
  double velocity() const { return x[2]; }
  double heading() const { return x[3]; }
  double yaw_rate() const { return x[4]; }
};
Track::State ctrv_predict(const Track::State&,double dt);
Track::Cov ctrv_jacobian(const Track::State&,double dt);

struct Detection {
  std::vector<int> points;  // Indices into Scan point arrays.
  double time{};
  Box body_box{Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),0,{}};
  Quality quality{Quality::Reject};
};
struct Scan {
  double stamp{};
  Pose2 ego;
  std::vector<Eigen::Vector3d> body_points,world_points;
  std::vector<double> point_time;
  std::vector<Detection> detections;
  bool map_ok{};  // Map, localization and structure labels are all trustworthy.
  std::function<double(const Eigen::Vector2d&)> heading_prior;
  std::function<bool(int)> is_structure,is_covered;  // Empty when the map cannot judge.
};
Detection make_detection(const Scan&,const std::vector<int>& points,double weak_min_range);

struct TrackerConfig {
  double coast_s{.5},memory_s{1.5},tentative_s{.5},max_position_variance{1.};
  double sigma_a{3.},sigma_alpha{1.},weak_min_range{40.};
  int max_weak_tracks{30};
  bool guided{true};
};
class Tracker {
 public:
  explicit Tracker(TrackerConfig config={}):config_(config){}
  std::vector<Track>& update(Scan scan);
  std::vector<Track>& tracks() { return tracks_; }
  const std::vector<Track>& tracks() const { return tracks_; }
  bool usable(const Track&,double stamp) const;
  // Pure read of a track at another time; negative dt is allowed for camera frames.
  void peek(const Track&,double t,Track::State& x,Track::Cov& P) const;
  void reset() { tracks_.clear();last_=-1; }
  void set_process_scale(double scale) { process_scale_=scale; }
 private:
  void propagate(Track::State& x,Track::Cov& P,double dt) const;
  void predict_to(Track&,double t) const;
  double heading_for(const Track&,const Scan&) const;
  void gather(Scan&);
  double pair_cost(const Track&,const Detection&,const Scan&,double chi2,double max_distance) const;
  void assign(Scan&,const std::vector<int>& track_ids,const std::vector<int>& detection_ids,
              double chi2,double max_distance,std::vector<bool>& consumed);
  void apply(Track&,const Detection&,const Scan&,const AnchorFit&,double noise_scale,bool guided);
  void observe_map(Track&,const Detection&,const Scan&) const;
  void birth(const Detection&,const Scan&);
  TrackerConfig config_;
  std::vector<Track> tracks_;
  int64_t next_{1};
  double last_{-1},process_scale_{1.};
};
double near_surface_x(const Track&,const Pose2&);
Box track_body_box(const Track&,const Pose2&);
Box track_rear_box(const Track&,const Pose2&);
Eigen::Vector2d relative_velocity(const Track&,const Pose2& ego,const Eigen::Vector2d& ego_velocity_world);
const char* vehicle_geometry_status(const Track&);
}
