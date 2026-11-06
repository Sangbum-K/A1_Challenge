#pragma once
#include <Eigen/Geometry>
#include <array>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace a1 {
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
using Vec = Eigen::Vector3d;
struct Ego {
  double stamp{}, yaw{}, yaw_rate{};
  Vec velocity{Vec::Zero()};  // ENU at vehicle reference point
  Vec acceleration{Vec::Constant(nan)}; // gravity-removed continuous-frame acceleration
  Vec position{Vec::Zero()};  // continuous local integration, never map jumps
  bool valid{};     // Usable values for computation and logging.
  bool verified{};  // Mounting and timing validated; required for a valid request.
  std::string source{"ins_navigation"};
  double forward() const { return std::cos(yaw)*velocity.x()+std::sin(yaw)*velocity.y(); }
};
// Continuous-frame ego samples. Positions integrate velocity, never map-corrected poses.
class EgoHistory {
 public:
  // False for non-monotonic time.
  bool push(const Ego& e);
  // Integrates position from velocity. A gap above max_gap_s restarts the local frame.
  bool push_integrated(Ego e,double max_gap_s);
  void clear() { samples_.clear(); }
  bool empty() const { return samples_.empty(); }
  const Ego& latest() const { return samples_.back(); }
  // Interpolates between samples; extrapolates past the newest by at most max_extrapolation_s.
  std::optional<Ego> at(double t,double max_extrapolation_s) const;
 private:
  std::deque<Ego> samples_;
};
struct ImuConfig {
  bool stationary_start_confirmed{false};
  bool mount_verified{false};
  double calibration_s{2.0}, max_integration_s{10.0}, max_dt{0.05};
  Eigen::Matrix3d base_from_imu{Eigen::Matrix3d::Identity()};
};
// Experimental strapdown estimate from a confirmed stationary start. No absolute speed reference.
class ImuSpeed {
 public:
  explicit ImuSpeed(ImuConfig config): config_(config) {}
  Ego update(double stamp, const Vec& acceleration, const Vec& angular_rate);
  const std::string& reason() const { return reason_; }
  void reset();
 private:
  ImuConfig config_;std::string reason_{"imu_initializing"};
  double first_{-1},last_{-1},ready_at_{-1};int samples_{};
  Vec sum_a_{Vec::Zero()},sum_w_{Vec::Zero()},sum_sq_{Vec::Zero()};
  Vec bias_a_{Vec::Zero()},bias_w_{Vec::Zero()},previous_accel_{Vec::Zero()};
  Eigen::Quaterniond attitude_{Eigen::Quaterniond::Identity()};
  Ego ego_;
};
struct A2 {
  double stamp{};
  std::array<float, 280> raw{}; // bin-major [40,7], unmodified logits
  bool ready{};
};
struct Detection { Vec center, lo, hi; };
struct Track {
  int id{}, hits{};
  double stamp{};
  Vec position, velocity{Vec::Zero()}; // continuous ENU
  Detection box;
  bool measured{}, velocity_valid{};
};
struct Config {
  double max_age{0.25}, pair_slop{0.08}, track_gate{3.0}, timeout{0.5};
  double corridor_half_width{1.8}, bumper_x{0.0}, object_threshold{0.5};
  double camera_fx{640}, camera_cx{640}, camera_width{1280};
  Eigen::Isometry3d camera_from_base{Eigen::Isometry3d::Identity()};
  bool calibration_valid{false}, clock_valid{false};
  bool camera_metric_valid{false};
  double plan_speed{-1}; // negative = actual; otherwise explicit what-if
};
struct Result {
  double stamp{}, valid_until{};
  int track_id{-1}, bin{-1};
  bool valid{}, what_if{};
  std::string reason{"no_input"}, source, gap_definition, ego_source;
  double x{nan}, y{nan}, range{nan}, gap{nan}, relative_speed{nan};
  double ego_speed{nan}, lead_speed{nan}, plan_speed{nan}, gate{nan};
  double raw_p5{nan}, request{nan};
  double request_unverified{nan}; // Logged even when verification flags block the request.
};
struct PrimaryTarget {
  int id{-1};double stamp{},valid_until{};
  Vec center{Vec::Constant(nan)};
  double gap{nan},relative_speed{nan};bool predicted{},valid{};
};
double p5(double plan, double gap, double lead);
double probability(double logit);
Vec to_world(const Vec& p, const Ego& ego);
Vec to_body(const Vec& p, const Ego& ego);
class Tracker {
 public:
  explicit Tracker(Config config): config_(config) {}
  const std::vector<Track>& update(double stamp, const std::vector<Detection>& detections, const Ego& ego);
  void reset() { tracks_.clear(); last_stamp_ = -1; }
 private:
  Config config_;
  std::vector<Track> tracks_;
  int next_id_{1};
  double last_stamp_{-1};
};
class Engine {
 public:
  explicit Engine(Config config): config_(config), tracker_(config) {}
  // input_issue names an upstream verification failure; empty means none.
  Result primary(const PrimaryTarget&,const Ego&,double now,const std::string& input_issue={});
  Result camera(const A2&, const Ego&, double now);
  Result fusion(const A2&, const Ego&, double stamp, const std::vector<Detection>&, double now);
  void reset() { tracker_.reset(); gate_=0; gate_stamp_=-1; }
 private:
  Result base(const A2&, const Ego&, double stamp, double now);
  void finish(Result&, const Ego&);
  Config config_;
  Tracker tracker_;
  double gate_{0}, gate_stamp_{-1};
};
} // namespace a1
