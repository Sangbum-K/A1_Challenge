// frame_loader.hpp
// 단일 frame (run_id, tick) → interface::VehicleState + interface::Trajectory
// + interface::PredictObjects 생성.
//
// 동작:
//   1. ParseEgoInfo(EGO_INFO/{tick}.txt) → VehicleState (모든 채널 vehicle frame)
//   2. 미래 N tick EGO_INFO 파싱 → cumulative wp_rel → Trajectory (vehicle frame)
//   3. ParseObjectInfo(OBJECT_INFO/object_info_{tick}.txt)
//      → 각 객체 velocity 를 R_w2e @ R_o2w @ v_own 으로 ego-basis 변환 (frame-fix)
//      → PredictObject.state[k]: 위치는 등속 propagation (k = 0..N_state-1)
//
// 좌표 변환 컨벤션:
//   - raw position 은 world frame, position 만 → ego frame 으로 변환 (R_w2e @ (p_world − p_ego))
//   - velocity 변환은 위 frame-fix (R_w2e R_o2w v_own)
//   - jiwon_e2e velocity frame bug 회피
//
#pragma once

#include <string>

#include "interface_trajectories.hpp"
#include "interface_objects.hpp"
#include "interface_vehicle_state.hpp"

namespace scc_batch_synth {

struct FrameLoaderConfig {
    int traj_horizon = 40;          // trajectory wp 수
    int obj_state_horizon = 40;     // PredictObject.state[k] 수
    double dt = 0.1;                // s, trajectory + object state 간격
    double dt_per_tick_ms = 10.0;   // run 의 tick 해상도 (raw 파일 간격 = step_ticks * dt_per_tick_ms)
                                    // 보통 ~10ms/tick. step_ticks = dt*1000 / dt_per_tick_ms = 10
};

struct LoadedFrame {
    interface::VehicleState  vehicle_state;
    interface::Trajectory    trajectory;
    interface::PredictObjects objects;
};

class FrameLoader {
public:
    explicit FrameLoader(const FrameLoaderConfig& cfg = FrameLoaderConfig{});

    // run_dir = ".../sangam_track/MultiAgnet/260513_48"
    // tick = 15395 등. EGO_INFO/{tick}.txt + OBJECT_INFO/object_info_{tick}.txt 읽음.
    // 미래 trajectory 는 tick + i*step_ticks (i=1..horizon) 의 EGO_INFO 들 사용.
    bool Load(const std::string& run_dir, int tick, LoadedFrame& out) const;

    const FrameLoaderConfig& config() const { return cfg_; }

private:
    FrameLoaderConfig cfg_;
};

}  // namespace scc_batch_synth
