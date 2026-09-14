// scc_speed_replanner.hpp
//
// Trajectory + PredictObjects + VehicleState 입력으로 SCC 속도 re-profile.
// path (x, y, yaw, ...) 는 그대로, point[i].speed 만 갱신.
//
// 로직:
//   1. trajectory road map (cum_s) 생성
//   2. 각 PredictObject 의 state[k] (k=0..N-1, dt=0.1s 가정) 을 Frenet (s, n) 으로 투영
//   3. |n| ≤ lane_half_width 이고 0 ≤ s ≤ s_max → in-lane lead 후보
//   4. trajectory point i 의 예상 도착 시간 t_i (cum_s/v_orig) 에 lead 의 s_at_t_i 보간
//   5. gap_i = s_at_t_i - cum_s[i].  gap_i > 0 인 가장 가까운 lead 선택.
//   6. v_safe = (gap_i - s_min) / t_headway, v_new[i] = min(v_orig[i], max(0, v_safe))
//
#pragma once

#include <cmath>
#include <vector>
#include <utility>

#include "interface_trajectories.hpp"
#include "interface_objects.hpp"
#include "interface_vehicle_state.hpp"

namespace scc {

struct SccConfig {
    double t_headway      = 1.0;   // s
    double s_min          = 3.0;   // m
    double lane_half_width = 2.0;  // m   |n| <= 이 값 → in lane
    double obj_state_dt   = 0.1;   // PredictObject.state 의 간격 (s) — A1_Rule perception 기본 0.1
    double min_v_for_time = 0.5;   // t_i 계산 시 division-by-zero 방지용 floor
    // 최대 감속도 (m/s², 양수). v_safe 적용 후 forward/backward pass 로 |dv/dt| ≤ max_decel
    // 보장. 'soft' = hard limit 으로 강제 (운전성/안전을 위해). 사용자가 값 조정 가능.
    double max_decel      = 15.0;
    // 최대 가속도 (m/s², 양수). 가속 방향 limit. SCC 거동 자체는 감속 위주이므로 영향 작음.
    double max_accel      = 10.0;
};

struct SccDebugInfo {
    std::vector<double> cum_s;            // (N,)  trajectory arc length from origin
    std::vector<double> t_arrival;        // (N,)  estimated arrival time at each wp
    std::vector<double> v_original;       // (N,)
    std::vector<double> v_pre_accel_limit; // (N,)  lead-based limit 만 적용 후, accel limit 전
    std::vector<double> v_replanned;      // (N,)  forward + backward pass 후 최종
    std::vector<int>    in_lane_obj_ids;  // PredictObject.id passing the lane filter
    std::vector<double> closest_lead_s;   // (N,)  lead 의 t_i 시점 s. nan = no lead
    std::vector<double> gap;              // (N,)  closest_lead_s - cum_s[i]. nan = no lead
};

class SccSpeedReplanner {
public:
    explicit SccSpeedReplanner(const SccConfig& cfg = SccConfig{}) : cfg_(cfg) {}

    // path 그대로, speed 만 갱신한 trajectory 반환.
    // lead 없는 frame 은 original 과 동일 (speed 변경 없음).
    interface::Trajectory Replan(
        const interface::Trajectory& original,
        const interface::PredictObjects& objects,
        const interface::VehicleState& ego);

    const SccDebugInfo& last_debug() const { return debug_; }
    const SccConfig&    config()     const { return cfg_; }
    void set_config(const SccConfig& c) { cfg_ = c; }

private:
    SccConfig    cfg_;
    SccDebugInfo debug_;

    // (px, py) → (s, n) via nearest-neighbor on (ref_x, ref_y).
    // synth_k1 의 manual frenet 과 동일 (cross product 로 lateral 부호).
    std::pair<double, double> ProjectToFrenet(
        double px, double py,
        const std::vector<double>& ref_x,
        const std::vector<double>& ref_y,
        const std::vector<double>& cum_s) const;
};

}  // namespace scc
