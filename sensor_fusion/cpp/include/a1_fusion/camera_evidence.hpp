#pragma once
#include "a1_fusion/tracker.hpp"
#include <array>
#include <string>
namespace a1::inspection {
// A2 reports presence per 32 px lateral bin, not 2D boxes, so a track reads the
// bins it projects onto instead of competing for thresholded regions.
struct BinSpan { int left{-1},right{-1}; bool valid{}; };
BinSpan bins_for(const Projection&);
double span_score(const std::array<double,40>& probabilities,const BinSpan&,
                  const std::array<bool,40>& owned_by_others,int dilation);
struct TrackView {
  Track* track{};
  double range{};
  Projection projection;
  BinSpan span;
  bool can_own{};  // Only vehicle-like, unmapped objects may hide what is behind them.
  Visibility visibility{Visibility::Outside};
  double score{};
};
// Nearest eligible track owns each bin. True when two tracks could not be ordered.
bool assign_ownership(std::vector<TrackView>& views,const std::array<double,40>& probabilities,int dilation=1);

struct EvidenceConfig {
  double tau{4.},p0{.25},alpha{.15},pos{1.5},neg{-.2},l_min{-3.},l_max{8.},match_score{.4};
};
// One A2 frame. Absence counts weakly: A2 often misses cars it can see, so a car
// seen about 40 % of the time must keep its evidence between detections.
void observe(Track&,double stamp,double score,Visibility,const EvidenceConfig&);

struct FarConfig { double near_limit{35.},promote{3.},release{1.5}; };
struct FarInputs {
  bool map_ok{},ego_compensated{},raceline_active{};
  double speed_sigma{},range{},path_n{};
};
// Cues add up because far cues are often missing. Only validity and map rejection are mandatory.
void update_far_state(Track&,const FarInputs&,const FarConfig&);

struct TargetObservation {
  int64_t id{-1};
  double distance{},range{},evidence{};
  bool usable{},measured{},in_lane{},vehicle_shape{},map_static{},far_vehicle{},camera_matched{};
  Visibility visibility{Visibility::Silent};
};
struct TargetSelection {
  int64_t primary{-1},standby{-1};
  std::string status{"no_target"};
  double camera_age{std::numeric_limits<double>::quiet_NaN()};
};
struct SelectorConfig {
  double l_conf{3.},l_drop{1.},far_entry_range{50.},far_keep_range{35.};
  double target_min{3.},target_max{50.},camera_silent_s{2.},switch_hold_s{.3};
};
class EvidenceSelector {
 public:
  explicit EvidenceSelector(SelectorConfig config={}):config_(config){}
  TargetSelection update(double stamp,double last_camera_stamp,const std::vector<TargetObservation>&);
  void reset();
 private:
  SelectorConfig config_;
  int64_t primary_{-1},pending_{-1};
  double last_{-1},pending_since_{},matched_stamp_{std::numeric_limits<double>::quiet_NaN()};
};
}
