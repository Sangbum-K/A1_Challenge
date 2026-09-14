// object_parser.hpp
// raw OBJECT_INFO/object_info_{tick}.txt 텍스트 → 구조체 리스트
//
// 형식 (1 line per object, 14 fields):
//   entity_id  pos_x pos_y pos_z  roll pitch yaw  L W H  vel_x vel_y vel_z  track_id
//
// 중요: vel_xyz 는 **각 객체의 자기 vehicle frame** (decision: scc_controller_imitator/
// memory/jiwon_e2e_velocity_frame_bug.md 참조).
//
#pragma once

#include <array>
#include <string>
#include <vector>

namespace scc_batch_synth {

struct RawObject {
    std::string entity_id;
    std::array<double, 3> position{};     // world m
    std::array<double, 3> orientation{};  // roll, pitch, yaw (deg)
    std::array<double, 3> size{};         // length, width, height (m)
    std::array<double, 3> velocity_own{}; // **object own vehicle frame** (m/s)
    int track_id{0};
};

bool ParseObjectInfo(const std::string& path, std::vector<RawObject>& out);

}  // namespace scc_batch_synth
