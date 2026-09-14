// synth_batch.cpp
// Batch GT synthesis main loop:
//   raw EGO_INFO / OBJECT_INFO → FrameLoader → SccSpeedReplanner →
//   LateralControl::Run + LongitudinalControl::Run → 결과 dump.
//
// frame-list 형식 (1 줄 1 frame, 공백 구분):
//   <run_id> <tick> <dt_per_tick_ms> <K> <bank_file_or_->
//     K=1, bank=- : logged 만 (4-col 도 호환)
//     K>1, bank=path : logged + (K-1) random (binary bank file)
//   Bank file format: int32 K, int32 N_wp, float32[K * N_wp * 4]  (x,y,yaw,speed) world.
//
// 병렬 처리: --shard <idx>/<total>  → 자신의 shard 만 처리
//   (i % total == idx) 인 line 만 처리. 16 workers 면 --shard 0/16 ... 15/16.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "scc_batch_synth/frame_loader.hpp"
#include "scc_replanner/scc_speed_replanner.hpp"

#include "lateral_control/lateral_control.hpp"
#include "longitudinal_control/longitudinal_control.hpp"

namespace {

struct FrameSpec {
    std::string run_id;
    int         tick;
    double      dt_per_tick_ms;
    int         K = 1;           // total trajectories: 1 logged + (K-1) random
    std::string bank_file;        // empty/"-" → no random
};

bool ReadFrameList(const std::string& path, std::vector<FrameSpec>& out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "frame-list 열기 실패: %s\n", path.c_str());
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        FrameSpec fs;
        if (!(iss >> fs.run_id >> fs.tick >> fs.dt_per_tick_ms)) continue;
        if (!(iss >> fs.K)) fs.K = 1;
        if (!(iss >> fs.bank_file)) fs.bank_file = "";
        out.push_back(fs);
    }
    return true;
}

// Bank file 로드. 출력: trajs (K_rand, N_wp, 4).
bool LoadBank(const std::string& path, int& K_rand, int& N_wp,
              std::vector<float>& data /* size K_rand*N_wp*4 */) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    int32_t h[2];
    f.read(reinterpret_cast<char*>(h), sizeof(h));
    if (!f) return false;
    K_rand = h[0]; N_wp = h[1];
    if (K_rand <= 0 || N_wp <= 0) return false;
    const size_t n = static_cast<size_t>(K_rand) * N_wp * 4;
    data.resize(n);
    f.read(reinterpret_cast<char*>(data.data()), n * sizeof(float));
    return static_cast<size_t>(f.gcount()) == n * sizeof(float);
}

void Usage(const char* prog) {
    std::fprintf(stderr,
        "사용: %s --frame-list <file> --raw-base <dir> --out <dir>\n"
        "      [--t-headway 1.0] [--s-min 3.0] [--lane-half 2.0]\n"
        "      [--shard idx/total] [--skip-existing]\n"
        "\n"
        "frame-list 1 줄: <run_id> <tick> <dt_per_tick_ms> <K> <bank_file_or_->\n",
        prog);
}

bool FileExists(const std::string& p) {
    std::ifstream f(p); return f.good();
}

void WriteOutput(const std::string& outpath,
                 const FrameSpec& fs, const std::string& tag,
                 const scc_batch_synth::LoadedFrame& frame,
                 const interface::Trajectory& scc_traj,
                 const scc::SccDebugInfo& dbg,
                 const interface::ControlCommand& lat_cmd,
                 const interface::ControlCommand& long_cmd) {
    std::ofstream f(outpath);
    f << "# run=" << fs.run_id << " tick=" << fs.tick
      << " dt_per_tick_ms=" << fs.dt_per_tick_ms << " tag=" << tag << "\n";
    f << "in_lane_obj_ids:";
    for (int id : dbg.in_lane_obj_ids) f << " " << id;
    f << "\nN=" << frame.trajectory.point.size() << "\n";
    f << "# i  cum_s  t_i  v_orig  v_scc  lead_s  gap\n";
    for (size_t i = 0; i < frame.trajectory.point.size(); ++i) {
        f << i << " " << dbg.cum_s[i] << " " << dbg.t_arrival[i] << " "
          << dbg.v_original[i] << " " << dbg.v_replanned[i] << " "
          << dbg.closest_lead_s[i] << " " << dbg.gap[i] << "\n";
    }
    f << "# instant command (t=0)\n";
    f << "lat_steer_tire_deg=" << lat_cmd.steering_tire_angle << "\n";
    f << "long_ax=" << long_cmd.ax << "\n";
    f << "long_trq=" << long_cmd.trq << "\n";
    f << "long_accel_pedal=" << long_cmd.accel_pedal << "\n";
    f << "long_brake_pedal=" << long_cmd.brake_pedal << "\n";
    f << "long_traj_N=" << long_cmd.control_trajectory.control_point.size() << "\n";
    f << "# i  time  vx  ax  steer  trq  accel_pedal  brake_pedal\n";
    for (size_t i = 0; i < long_cmd.control_trajectory.control_point.size(); ++i) {
        const auto& cp = long_cmd.control_trajectory.control_point[i];
        f << i << " " << cp.time << " " << cp.vx << " " << cp.ax << " "
          << cp.steer << " " << cp.trq << " " << cp.accel_pedal << " " << cp.brake_pedal << "\n";
    }
    f << "lat_traj_N=" << lat_cmd.control_trajectory.control_point.size() << "\n";
    f << "# i  time  vx  ax  steer\n";
    for (size_t i = 0; i < lat_cmd.control_trajectory.control_point.size(); ++i) {
        const auto& cp = lat_cmd.control_trajectory.control_point[i];
        f << i << " " << cp.time << " " << cp.vx << " " << cp.ax << " " << cp.steer << "\n";
    }
}

// trajectory unique wp >= 10 인지 검증 (LMPC crash 방지)
bool TrajOk(const interface::Trajectory& tr) {
    size_t unique_wp = 0;
    double prev_x = 1e9, prev_y = 1e9;
    for (const auto& p : tr.point) {
        const double d = std::hypot(p.x - prev_x, p.y - prev_y);
        if (d > 0.1) { unique_wp++; prev_x = p.x; prev_y = p.y; }
    }
    return unique_wp >= 10;
}

bool RunOneSample(const FrameSpec& fs, const std::string& tag,
                  scc_batch_synth::LoadedFrame& frame,
                  scc::SccSpeedReplanner& replanner,
                  LateralControl& lat_ctrl, LongitudinalControl& long_ctrl,
                  const std::string& outpath,
                  int n_warmup = 5) {
    if (!TrajOk(frame.trajectory)) return false;
    auto scc_traj = replanner.Replan(frame.trajectory, frame.objects, frame.vehicle_state);
    const auto& dbg = replanner.last_debug();
    interface::ControlCommand lat_cmd, long_cmd;
    interface::ControlTrajectory lat_traj, long_traj;
    interface::PathTrackingInfos lat_info, long_info;
    try {
        // [scc_imitator] 매 sample 전에 cold-start (이전 sample 의 optimal_states_
        // 가 이 sample 의 vehicle_state/trajectory 와 misalign 됨 → reset 필수).
        // 그 후 warm-chain N 회 반복 → optimal_states_ 가 이 sample 의 input 에
        // 대해 정상상태에 수렴. ROS warm-chain (100Hz cycle) 흉내.
        lat_ctrl.Reset();
        long_ctrl.Reset();
        for (int w = 0; w < n_warmup; ++w) {
            (void) lat_ctrl.Run(frame.vehicle_state, scc_traj);
            (void) long_ctrl.Run(frame.vehicle_state, frame.objects, scc_traj);
        }
        std::tie(lat_cmd, lat_traj, lat_info) = lat_ctrl.Run(frame.vehicle_state, scc_traj);
        std::tie(long_cmd, long_traj, long_info) = long_ctrl.Run(
            frame.vehicle_state, frame.objects, scc_traj);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  skip %s/%d/%s: %s\n",
                      fs.run_id.c_str(), fs.tick, tag.c_str(), e.what());
        return false;
    }
    WriteOutput(outpath, fs, tag, frame, scc_traj, dbg, lat_cmd, long_cmd);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string frame_list_path, raw_base, out_dir;
    double t_headway = 1.0, s_min = 3.0, lane_half = 2.0;
    int n_warmup_cli = 5;
    int shard_idx = 0, shard_total = 1;
    bool skip_existing = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* k) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s 값 누락\n", k); std::exit(1); }
            return argv[++i];
        };
        if      (a == "--frame-list") frame_list_path = need("--frame-list");
        else if (a == "--raw-base")   raw_base = need("--raw-base");
        else if (a == "--out")        out_dir = need("--out");
        else if (a == "--t-headway")  t_headway = std::atof(need("--t-headway"));
        else if (a == "--s-min")      s_min = std::atof(need("--s-min"));
        else if (a == "--lane-half")  lane_half = std::atof(need("--lane-half"));
        else if (a == "--skip-existing") skip_existing = true;
        else if (a == "--warmup")     n_warmup_cli = std::atoi(need("--warmup"));
        else if (a == "--shard") {
            std::string s = need("--shard");
            auto slash = s.find('/');
            if (slash == std::string::npos) { std::fprintf(stderr, "--shard 형식: idx/total\n"); return 1; }
            shard_idx = std::atoi(s.substr(0, slash).c_str());
            shard_total = std::atoi(s.substr(slash + 1).c_str());
        }
        else if (a == "-h" || a == "--help") { Usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); Usage(argv[0]); return 1; }
    }
    if (frame_list_path.empty() || raw_base.empty() || out_dir.empty()) {
        Usage(argv[0]); return 1;
    }

    std::vector<FrameSpec> frames;
    if (!ReadFrameList(frame_list_path, frames)) return 1;

    // shard filter
    std::vector<FrameSpec> my_frames;
    for (size_t i = 0; i < frames.size(); ++i) {
        if (static_cast<int>(i) % shard_total == shard_idx) my_frames.push_back(frames[i]);
    }
    std::printf("[shard %d/%d] %zu/%zu frames\n",
                 shard_idx, shard_total, my_frames.size(), frames.size());

    scc::SccConfig scfg;
    scfg.t_headway = t_headway;
    scfg.s_min = s_min;
    scfg.lane_half_width = lane_half;
    scc::SccSpeedReplanner replanner(scfg);

    LateralControl lat_ctrl;
    LongitudinalControl long_ctrl;
    if (!lat_ctrl.Init()) { std::fprintf(stderr, "LateralControl::Init 실패\n"); return 1; }
    if (!long_ctrl.Init()) { std::fprintf(stderr, "LongitudinalControl::Init 실패\n"); return 1; }

    { std::string cmd = "mkdir -p '" + out_dir + "'"; std::system(cmd.c_str()); }

    int n_ok = 0, n_fail = 0, n_skipped = 0;
    char buf[1024];
    for (size_t fi = 0; fi < my_frames.size(); ++fi) {
        const auto& fs = my_frames[fi];
        scc_batch_synth::FrameLoaderConfig lcfg;
        lcfg.dt = 0.1; lcfg.dt_per_tick_ms = fs.dt_per_tick_ms;
        lcfg.traj_horizon = 40; lcfg.obj_state_horizon = 40;
        scc_batch_synth::FrameLoader loader(lcfg);

        scc_batch_synth::LoadedFrame frame;
        const std::string run_dir = raw_base + "/" + fs.run_id;
        if (!loader.Load(run_dir, fs.tick, frame)) {
            n_fail++; continue;
        }

        // (1) logged
        std::snprintf(buf, sizeof(buf), "%s/%s__t%07d__logged.scc.txt",
                       out_dir.c_str(), fs.run_id.c_str(), fs.tick);
        std::string out_logged = buf;
        if (skip_existing && FileExists(out_logged)) { n_skipped++; }
        else if (RunOneSample(fs, "logged", frame, replanner, lat_ctrl, long_ctrl, out_logged, n_warmup_cli)) n_ok++;
        else n_fail++;

        // (2) random K-1 — bank 가 있으면
        if (fs.K > 1 && !fs.bank_file.empty() && fs.bank_file != "-") {
            int K_rand = 0, N_wp = 0;
            std::vector<float> bank_data;
            if (!LoadBank(fs.bank_file, K_rand, N_wp, bank_data)) {
                std::fprintf(stderr, "  bank load fail: %s\n", fs.bank_file.c_str());
                n_fail++; continue;
            }
            // 저장: 원본 trajectory 보관 (override 후 복원)
            auto saved_traj = frame.trajectory;
            const float* p_base = bank_data.data();
            for (int k = 0; k < K_rand; ++k) {
                std::snprintf(buf, sizeof(buf), "%s/%s__t%07d__rand_%05d.scc.txt",
                               out_dir.c_str(), fs.run_id.c_str(), fs.tick, k);
                std::string outpath = buf;
                if (skip_existing && FileExists(outpath)) { n_skipped++; continue; }
                // trajectory override
                frame.trajectory.point.clear();
                frame.trajectory.point.reserve(N_wp);
                for (int j = 0; j < N_wp; ++j) {
                    const float* p = p_base + (k * N_wp + j) * 4;
                    interface::TrajectoryPoint tp{};
                    tp.x = p[0]; tp.y = p[1]; tp.yaw = p[2]; tp.speed = p[3];
                    tp.time = static_cast<float>(j) * static_cast<float>(lcfg.dt);
                    frame.trajectory.point.push_back(tp);
                }
                if (RunOneSample(fs, "rand_" + std::to_string(k), frame,
                                  replanner, lat_ctrl, long_ctrl, outpath, n_warmup_cli)) n_ok++;
                else n_fail++;
            }
            frame.trajectory = saved_traj;  // 복원 (사용 안 하지만 정리)
        }

        if ((fi + 1) % 50 == 0 || (fi + 1) == my_frames.size()) {
            std::printf("[shard %d] [%zu/%zu] ok=%d fail=%d skip=%d\n",
                         shard_idx, fi + 1, my_frames.size(), n_ok, n_fail, n_skipped);
        }
    }

    std::printf("[shard %d/%d] done: ok=%d fail=%d skip=%d\n",
                 shard_idx, shard_total, n_ok, n_fail, n_skipped);
    return 0;
}
