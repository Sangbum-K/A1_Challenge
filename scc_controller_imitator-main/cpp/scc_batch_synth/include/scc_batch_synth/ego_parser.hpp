// ego_parser.hpp
// raw EGO_INFO/{tick}.txt 텍스트 → 구조체
//
// 형식 (jiwon_e2e ego_parser.py 와 동일):
//   position: x y z              # MORAI map local Cartesian [m]
//   orientation: roll pitch yaw  # deg, yaw 는 world heading
//   velocity: vx vy vz           # m/s, **vehicle frame**
//   angularVelocity: wx wy wz    # deg/s, vehicle frame
//   acceleration: ax ay az       # m/s², vehicle frame
//   accel:  scalar               # 0..1 gas pedal
//   brake:  scalar               # 0..1
//   steer:  scalar               # degrees
//   linkid: str / trafficlightid: str
//
#pragma once

#include <array>
#include <string>

namespace scc_batch_synth {

struct EgoInfo {
    std::array<double, 3> position{};     // x, y, z (world m)
    std::array<double, 3> orientation{};  // roll, pitch, yaw (deg)
    std::array<double, 3> velocity{};     // vx, vy, vz (m/s, vehicle frame)
    std::array<double, 3> angular_vel{};  // wx, wy, wz (deg/s)
    std::array<double, 3> acceleration{}; // ax, ay, az (m/s², vehicle frame)
    double accel{0.0};
    double brake{0.0};
    double steer{0.0};       // deg
    std::string linkid;
    std::string trafficlightid;
};

bool ParseEgoInfo(const std::string& path, EgoInfo& out);

}  // namespace scc_batch_synth
