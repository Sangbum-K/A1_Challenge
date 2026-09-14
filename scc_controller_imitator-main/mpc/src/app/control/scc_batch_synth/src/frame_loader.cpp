// frame_loader.cpp
#include "scc_batch_synth/frame_loader.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

#include "scc_batch_synth/ego_parser.hpp"
#include "scc_batch_synth/object_parser.hpp"

namespace scc_batch_synth {

namespace {

constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;

// World point (px, py) → ego frame 좌표 (ex, ey). ego_yaw 는 rad.
inline std::pair<double, double> WorldToEgo(
    double px, double py,
    double ego_x, double ego_y, double ego_yaw_rad)
{
    const double dx = px - ego_x;
    const double dy = py - ego_y;
    const double c = std::cos(-ego_yaw_rad), s = std::sin(-ego_yaw_rad);
    return {c * dx - s * dy, s * dx + c * dy};
}

// Object 의 own vehicle frame velocity → ego basis velocity.
// = R(-ego_yaw) @ R(+obj_yaw) @ v_own  (jiwon_e2e velocity frame bug fix)
inline std::pair<double, double> ObjVelInEgo(
    double vx_own, double vy_own,
    double obj_yaw_rad, double ego_yaw_rad)
{
    const double dyaw = obj_yaw_rad - ego_yaw_rad;
    const double c = std::cos(dyaw), s = std::sin(dyaw);
    return {c * vx_own - s * vy_own, s * vx_own + c * vy_own};
}

std::string TickFile(const std::string& run_dir, const std::string& subdir,
                       const std::string& prefix, int tick) {
    char buf[512];
    if (prefix.empty()) {
        std::snprintf(buf, sizeof(buf), "%s/%s/%d.txt",
                       run_dir.c_str(), subdir.c_str(), tick);
    } else {
        std::snprintf(buf, sizeof(buf), "%s/%s/%s%d.txt",
                       run_dir.c_str(), subdir.c_str(), prefix.c_str(), tick);
    }
    return std::string(buf);
}

}  // namespace

FrameLoader::FrameLoader(const FrameLoaderConfig& cfg) : cfg_(cfg) {}

bool FrameLoader::Load(const std::string& run_dir, int tick, LoadedFrame& out) const
{
    // 1) 현재 ego
    EgoInfo ego_now;
    if (!ParseEgoInfo(TickFile(run_dir, "EGO_INFO", "", tick), ego_now)) {
        return false;
    }
    const double ego_x0 = ego_now.position[0];
    const double ego_y0 = ego_now.position[1];
    const double ego_yaw0 = ego_now.orientation[2] * DEG2RAD;  // yaw deg → rad

    // ---- 1a) VehicleState 채우기 (world frame, controller 가 transform 수행) ----
    auto& vs = out.vehicle_state;
    vs = interface::VehicleState{};
    vs.x = static_cast<float>(ego_x0);
    vs.y = static_cast<float>(ego_y0);
    vs.yaw = static_cast<float>(ego_yaw0);
    vs.vx = static_cast<float>(ego_now.velocity[0]);
    vs.vy = static_cast<float>(ego_now.velocity[1]);
    vs.vz = static_cast<float>(ego_now.velocity[2]);
    vs.ax = static_cast<float>(ego_now.acceleration[0]);
    vs.ay = static_cast<float>(ego_now.acceleration[1]);
    vs.az = static_cast<float>(ego_now.acceleration[2]);
    vs.roll  = static_cast<float>(ego_now.orientation[0] * DEG2RAD);
    vs.pitch = static_cast<float>(ego_now.orientation[1] * DEG2RAD);
    // VehicleState.yaw_vel 등 추가 채널은 interface 구조 따라 부분만 채움 (필요시 확장).
    vs.yaw_vel = static_cast<float>(ego_now.angular_vel[2] * DEG2RAD);
    // 조향 / 페달
    vs.vehicle_can.steering_tire_angle = static_cast<float>(ego_now.steer * DEG2RAD);
    // operation_mode = AUTONOMOUS 필수 — controller 들 이 != AUTONOMOUS 면 init 모드로 진입.
    vs.vehicle_can.operation_mode = interface::OperationMode::AUTONOMOUS;

    // ---- 2) 미래 horizon 의 EGO_INFO → Trajectory ----
    auto& traj = out.trajectory;
    traj = interface::Trajectory{};
    // raw step (target dt 까지 가야할 tick 수). 단 EGO_INFO 파일은 file_step=5 tick 간격으로만
    // 존재하므로, file_step 배수로 반올림 해야 실제 파일 매칭.
    const int FILE_STEP = 5;   // jiwon_e2e MultiAgnet 모두 5 확인됨
    const double raw_step = cfg_.dt * 1000.0 / cfg_.dt_per_tick_ms;
    int step_ticks = static_cast<int>(std::round(raw_step / FILE_STEP)) * FILE_STEP;
    if (step_ticks <= 0) step_ticks = FILE_STEP;
    if (step_ticks <= 0) {
        std::fprintf(stderr, "[FrameLoader] invalid step_ticks (%d) from dt=%.3f dt_per_tick_ms=%.3f\n",
                      step_ticks, cfg_.dt, cfg_.dt_per_tick_ms);
        return false;
    }
    // 첫 wp 는 ego 자체 (world frame)
    {
        interface::TrajectoryPoint p{};
        p.x = static_cast<float>(ego_x0);
        p.y = static_cast<float>(ego_y0);
        p.yaw = static_cast<float>(ego_yaw0);
        p.speed = static_cast<float>(ego_now.velocity[0]);
        p.time = 0.0f;
        traj.point.push_back(p);
    }
    int missing_future = 0;
    for (int i = 1; i <= cfg_.traj_horizon; ++i) {
        const int t = tick + i * step_ticks;
        EgoInfo ego_f;
        if (!ParseEgoInfo(TickFile(run_dir, "EGO_INFO", "", t), ego_f)) {
            missing_future++;
            if (traj.point.empty()) break;
            traj.point.push_back(traj.point.back());
            continue;
        }
        interface::TrajectoryPoint p{};
        p.x = static_cast<float>(ego_f.position[0]);
        p.y = static_cast<float>(ego_f.position[1]);
        p.yaw = static_cast<float>(ego_f.orientation[2] * DEG2RAD);
        p.speed = static_cast<float>(ego_f.velocity[0]);
        p.time = static_cast<float>(i * cfg_.dt);
        traj.point.push_back(p);
    }
    if (missing_future > 0) {
        std::fprintf(stderr, "[FrameLoader] tick=%d: %d/%d future EGO_INFO 누락\n",
                      tick, missing_future, cfg_.traj_horizon);
        // half 이상 누락 = trajectory degenerate. LMPC crash 방지를 위해 fail 반환.
        if (missing_future * 2 >= cfg_.traj_horizon) {
            std::fprintf(stderr, "[FrameLoader] tick=%d: too many missing, skip\n", tick);
            return false;
        }
    }

    // ---- 3) OBJECT_INFO + velocity frame fix + state propagation ----
    auto& objs = out.objects;
    objs = interface::PredictObjects{};
    std::vector<RawObject> raw_objs;
    ParseObjectInfo(TickFile(run_dir, "OBJECT_INFO", "object_info_", tick), raw_objs);

    for (const auto& ro : raw_objs) {
        interface::PredictObject po{};
        po.id = static_cast<uint32_t>(ro.track_id);
        po.dimension.length = static_cast<float>(ro.size[0]);
        po.dimension.width  = static_cast<float>(ro.size[1]);
        po.dimension.height = static_cast<float>(ro.size[2]);

        // object 위치 = world frame 그대로
        const double ox0 = ro.position[0];
        const double oy0 = ro.position[1];
        const double obj_yaw_rad = ro.orientation[2] * DEG2RAD;

        // velocity: world frame 으로 변환 (raw 는 obj 자기 frame 이므로 R_o2w 곱)
        const double c = std::cos(obj_yaw_rad), s = std::sin(obj_yaw_rad);
        const double vwx = c * ro.velocity_own[0] - s * ro.velocity_own[1];
        const double vwy = s * ro.velocity_own[0] + c * ro.velocity_own[1];

        // 등속 propagation: state[k] (k=0..N_state-1, k*dt 초 후), world frame
        po.state.reserve(cfg_.obj_state_horizon);
        for (int k = 0; k < cfg_.obj_state_horizon; ++k) {
            const double t = static_cast<double>(k) * cfg_.dt;
            interface::Object3DState st{};
            st.x = static_cast<float>(ox0 + vwx * t);
            st.y = static_cast<float>(oy0 + vwy * t);
            st.z = 0.0f;
            st.yaw = static_cast<float>(obj_yaw_rad);
            st.vx = static_cast<float>(vwx);
            st.vy = static_cast<float>(vwy);
            st.vz = 0.0f;
            po.state.push_back(st);
        }
        objs.object.push_back(std::move(po));
    }

    return true;
}

}  // namespace scc_batch_synth
