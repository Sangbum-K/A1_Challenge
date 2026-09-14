// scc_replanner_test.cpp
//
// SccSpeedReplanner 의 4 시나리오 검증 (assert + 콘솔 출력):
//   1. lead 없음 → speed 그대로
//   2. in-lane lead → 감속
//   3. off-lane lead → speed 그대로 (filter 됨)
//   4. mixed (2 객체, 한쪽만 in-lane) → in-lane 만 영향

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "scc_replanner/scc_speed_replanner.hpp"

using interface::Object3DState;
using interface::PredictObject;
using interface::PredictObjects;
using interface::Trajectory;
using interface::TrajectoryPoint;
using interface::VehicleState;
using scc::SccConfig;
using scc::SccSpeedReplanner;

namespace {

Trajectory MakeStraightTrajectory(double speed, int N = 40, double dt = 0.1) {
    Trajectory t;
    t.point.reserve(N);
    for (int i = 0; i < N; ++i) {
        TrajectoryPoint p{};
        p.x = static_cast<float>((i + 1) * speed * dt);
        p.y = 0.0f;
        p.yaw = 0.0f;
        p.speed = static_cast<float>(speed);
        t.point.push_back(p);
    }
    return t;
}

PredictObject MakeObject(int id, double x0, double y0, double vx, double vy,
                          int n_state = 40, double dt = 0.1) {
    PredictObject o{};
    o.id = static_cast<uint32_t>(id);
    o.dimension.length = 4.5f;
    o.dimension.width  = 2.0f;
    o.dimension.height = 1.5f;
    o.state.reserve(n_state);
    for (int k = 0; k < n_state; ++k) {
        Object3DState s{};
        s.x = static_cast<float>(x0 + vx * k * dt);
        s.y = static_cast<float>(y0 + vy * k * dt);
        s.vx = static_cast<float>(vx);
        s.vy = static_cast<float>(vy);
        o.state.push_back(s);
    }
    return o;
}

VehicleState MakeEgo(double vx) {
    VehicleState v{};
    v.vx = static_cast<float>(vx);
    return v;
}

bool ApproxEqual(double a, double b, double tol = 1e-3) {
    return std::abs(a - b) < tol;
}

void PrintSummary(const char* name, const SccSpeedReplanner& r,
                   const Trajectory& replanned, const Trajectory& original) {
    const auto& d = r.last_debug();
    printf("\n[%s]\n", name);
    printf("  in_lane_obj_ids: ");
    for (int id : d.in_lane_obj_ids) printf("%d ", id);
    printf("(%zu)\n", d.in_lane_obj_ids.size());

    double v_orig_min = 1e9, v_new_min = 1e9;
    int    v_change_count = 0;
    for (size_t i = 0; i < original.point.size(); ++i) {
        const double vo = original.point[i].speed;
        const double vn = replanned.point[i].speed;
        v_orig_min = std::min(v_orig_min, vo);
        v_new_min  = std::min(v_new_min, vn);
        if (std::abs(vo - vn) > 0.01) v_change_count++;
    }
    printf("  v_orig min = %.2f  v_new min = %.2f   changed wp = %d / %zu\n",
            v_orig_min, v_new_min, v_change_count, original.point.size());

    // 첫 5개 wp 의 gap 표시
    printf("  i  cum_s  t_i   v_orig  v_new   closest_s   gap\n");
    for (size_t i = 0; i < std::min<size_t>(original.point.size(), 5); ++i) {
        printf("  %2zu  %5.1f  %4.2f  %5.2f   %5.2f   %8.2f   %6.2f\n",
                i, d.cum_s[i], d.t_arrival[i],
                d.v_original[i], d.v_replanned[i],
                d.closest_lead_s[i], d.gap[i]);
    }
}

}  // namespace

int main() {
    SccConfig cfg;
    cfg.t_headway = 1.0;
    cfg.s_min     = 3.0;
    cfg.lane_half_width = 2.0;
    SccSpeedReplanner replanner(cfg);

    // ---- scenario 1: no lead ----
    {
        Trajectory traj = MakeStraightTrajectory(20.0);
        PredictObjects objs;   // 빈 objects
        VehicleState ego = MakeEgo(20.0);

        auto out = replanner.Replan(traj, objs, ego);
        PrintSummary("S1 no lead", replanner, out, traj);

        // 모든 speed 변경 없어야 함
        for (size_t i = 0; i < traj.point.size(); ++i) {
            assert(ApproxEqual(out.point[i].speed, traj.point[i].speed));
        }
        assert(replanner.last_debug().in_lane_obj_ids.empty());
        printf("  S1 PASS\n");
    }

    // ---- scenario 2: in-lane lead at (30m, 0), 같은 속도 ----
    // ego 따라잡지 못함 (lead = ego speed) — gap 줄지 않음 → 일정 gap 유지
    // 단 lead 의 절대 위치 30m, ego 가 trajectory 따라 i*2m 진행
    // wp i=15 일때 ego_s=30m, t_i=1.5s, lead_s_at_t=30+20*1.5=60m, gap=30m → v_safe=27 → 영향 약간
    // 첫 wp 들에서는 gap 작아서 영향 큼
    {
        Trajectory traj = MakeStraightTrajectory(20.0);
        PredictObjects objs;
        objs.object.push_back(MakeObject(/*id=*/1, /*x0=*/30.0, /*y0=*/0.0,
                                          /*vx=*/20.0, /*vy=*/0.0));
        VehicleState ego = MakeEgo(20.0);

        auto out = replanner.Replan(traj, objs, ego);
        PrintSummary("S2 in-lane lead @30m", replanner, out, traj);

        // 첫 wp 의 v_new < v_orig 여야 함 (gap = 30m - 2m = 28m, v_safe=(28-3)/1=25, min(20,25)=20 → 변화 없음 ?)
        // 실제로는 i=0 에서 gap 28m → v_safe 25 → min(20,25) = 20 (변화 없음)
        // 더 가까운 lead 가 필요. 다음 케이스로 검증
        assert(!replanner.last_debug().in_lane_obj_ids.empty());
        assert(replanner.last_debug().in_lane_obj_ids[0] == 1);
        printf("  S2 PASS (lead detected as in-lane)\n");
    }

    // ---- scenario 3: very close in-lane lead → 명확한 감속 ----
    // 단, decel limit 15 m/s² 작동 시: backward pass 로 인접 wp 간 v 차이 ≤ 1.5 m/s (dt=0.1)
    {
        Trajectory traj = MakeStraightTrajectory(20.0);
        PredictObjects objs;
        objs.object.push_back(MakeObject(/*id=*/2, /*x0=*/8.0, /*y0=*/0.0,
                                          /*vx=*/0.0, /*vy=*/0.0));
        VehicleState ego = MakeEgo(20.0);

        auto out = replanner.Replan(traj, objs, ego);
        PrintSummary("S3 close static lead @8m (decel limit 15)", replanner, out, traj);

        // decel limit 적용으로 v_new[0] 은 더 이상 0 가까이 가지 않음 (backward pass 효과)
        // 인접 wp 간 차이 검사
        bool decel_ok = true;
        const auto& d = replanner.last_debug();
        for (size_t i = 1; i < out.point.size(); ++i) {
            const double dv = out.point[i].speed - out.point[i-1].speed;
            const double dt_i = d.t_arrival[i] - d.t_arrival[i-1];
            if (dt_i > 0 && dv < 0) {
                const double decel = -dv / dt_i;
                if (decel > 15.5) {   // 작은 tolerance
                    printf("  ! decel violation at i=%zu: %.3f m/s²\n", i, decel);
                    decel_ok = false;
                }
            }
        }
        assert(decel_ok);
        printf("  S3 PASS (모든 인접 wp 의 decel ≤ 15 m/s²)\n");
    }

    // ---- scenario 4: off-lane lead (lateral +5m) → filter 됨 ----
    {
        Trajectory traj = MakeStraightTrajectory(20.0);
        PredictObjects objs;
        objs.object.push_back(MakeObject(/*id=*/3, /*x0=*/8.0, /*y0=*/5.0,
                                          /*vx=*/0.0, /*vy=*/0.0));
        VehicleState ego = MakeEgo(20.0);

        auto out = replanner.Replan(traj, objs, ego);
        PrintSummary("S4 off-lane static lead @(8, +5)", replanner, out, traj);

        assert(replanner.last_debug().in_lane_obj_ids.empty());
        for (size_t i = 0; i < traj.point.size(); ++i) {
            assert(ApproxEqual(out.point[i].speed, traj.point[i].speed));
        }
        printf("  S4 PASS (off-lane lead filtered, no speed change)\n");
    }

    // ---- scenario 5: mixed — id=4 in lane, id=5 off lane ----
    {
        Trajectory traj = MakeStraightTrajectory(20.0);
        PredictObjects objs;
        objs.object.push_back(MakeObject(/*id=*/4, /*x0=*/8.0, /*y0=*/0.5,
                                          /*vx=*/0.0, /*vy=*/0.0));  // in-lane
        objs.object.push_back(MakeObject(/*id=*/5, /*x0=*/5.0, /*y0=*/5.0,
                                          /*vx=*/0.0, /*vy=*/0.0));  // off-lane (closer in s but |n|>2)
        VehicleState ego = MakeEgo(20.0);

        auto out = replanner.Replan(traj, objs, ego);
        PrintSummary("S5 mixed (id=4 in, id=5 off)", replanner, out, traj);

        const auto& ids = replanner.last_debug().in_lane_obj_ids;
        assert(ids.size() == 1);
        assert(ids[0] == 4);
        // id=5 (off-lane, 더 가까움) 가 무시되고 id=4 만 영향
        assert(out.point[0].speed < 5.0);
        printf("  S5 PASS (only id=4 affects, id=5 filtered)\n");
    }

    printf("\nALL TESTS PASSED\n");
    return 0;
}
