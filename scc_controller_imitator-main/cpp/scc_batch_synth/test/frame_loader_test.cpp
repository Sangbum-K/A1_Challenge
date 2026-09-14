// frame_loader_test.cpp
// 알려진 frame (260513_48 / tick=15395) 로딩 + Python 결과와 일관성 확인.
//
// 기대 (Python 검증 결과):
//   ego_vx = 14.36, ego_yaw_world = 172.14 deg
//   3 객체: Car_2(179°), Car_4(172°), Car_3(137°)
//   velocity fix: Car_4 ego_basis = (+22.11, -0.09) [거의 정합]

#include <cassert>
#include <cmath>
#include <cstdio>

#include "scc_batch_synth/frame_loader.hpp"

int main() {
    using namespace scc_batch_synth;

    FrameLoaderConfig cfg;
    cfg.dt_per_tick_ms = 10.168841956578428;   // run 260513_48 의 dt
    cfg.dt = 0.1;
    cfg.traj_horizon = 30;
    cfg.obj_state_horizon = 40;
    FrameLoader loader(cfg);

    const std::string run_dir = "/home/ailab-12/AILabDataset/03_Shared_Repository/jiwon/jiwon_e2e/e2e_racing/data/raw/sangam_track/MultiAgnet/260513_48";
    const int tick = 15395;

    LoadedFrame frame;
    if (!loader.Load(run_dir, tick, frame)) {
        std::fprintf(stderr, "Load failed\n");
        return 1;
    }

    std::printf("=== VehicleState ===\n");
    std::printf("  vx=%.3f vy=%.3f  ax=%.3f ay=%.3f  yaw_vel=%.4f rad/s\n",
                 frame.vehicle_state.vx, frame.vehicle_state.vy,
                 frame.vehicle_state.ax, frame.vehicle_state.ay,
                 frame.vehicle_state.yaw_vel);
    assert(std::abs(frame.vehicle_state.vx - 14.362262) < 1e-3);
    assert(std::abs(frame.vehicle_state.vy - (-0.296632)) < 1e-3);

    std::printf("\n=== Trajectory (%zu wp) ===\n", frame.trajectory.point.size());
    for (size_t i = 0; i < std::min<size_t>(5, frame.trajectory.point.size()); ++i) {
        const auto& p = frame.trajectory.point[i];
        std::printf("  i=%zu  x=%6.2f y=%+6.2f yaw=%+.3f rad  v=%5.2f m/s\n",
                     i, p.x, p.y, p.yaw, p.speed);
    }
    assert(frame.trajectory.point.size() == 31);  // origin + 30 horizon

    std::printf("\n=== Objects (%zu) — velocity fix 적용된 ego_basis vel ===\n",
                 frame.objects.object.size());
    for (const auto& o : frame.objects.object) {
        if (o.state.empty()) continue;
        const auto& s0 = o.state[0];
        std::printf("  id=%u  pos=(%+6.1f, %+6.1f)  vel=(%+5.2f, %+5.2f)  L=%.2f W=%.2f  yaw=%+.3f\n",
                     o.id, s0.x, s0.y, s0.vx, s0.vy,
                     o.dimension.length, o.dimension.width, s0.yaw);
    }
    assert(frame.objects.object.size() == 3);

    // 검증: Car_4 (track_id=4) 의 ego_basis vel ≈ (+22.11, -0.09)
    bool found_car4 = false;
    for (const auto& o : frame.objects.object) {
        if (o.id == 4 && !o.state.empty()) {
            const float vx = o.state[0].vx;
            const float vy = o.state[0].vy;
            std::printf("\n[Car_4 verify] vx=%+.3f (expect +22.11)  vy=%+.3f (expect -0.09)\n",
                         vx, vy);
            assert(std::abs(vx - 22.11) < 0.05);
            assert(std::abs(vy - (-0.09)) < 0.1);
            found_car4 = true;
        }
    }
    assert(found_car4);

    std::printf("\nALL PASS\n");
    return 0;
}
