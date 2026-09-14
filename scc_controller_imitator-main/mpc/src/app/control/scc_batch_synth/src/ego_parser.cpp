// ego_parser.cpp
#include "scc_batch_synth/ego_parser.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace scc_batch_synth {

namespace {

std::vector<double> ParseNumbers(const std::string& rest) {
    std::vector<double> nums;
    std::istringstream iss(rest);
    double v;
    while (iss >> v) nums.push_back(v);
    return nums;
}

}  // namespace

bool ParseEgoInfo(const std::string& path, EgoInfo& out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    out = EgoInfo{};

    std::string line;
    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = line.substr(0, colon);
        const std::string rest = line.substr(colon + 1);

        if (key == "position") {
            auto nums = ParseNumbers(rest);
            if (nums.size() >= 3) out.position = {nums[0], nums[1], nums[2]};
        } else if (key == "orientation") {
            auto nums = ParseNumbers(rest);
            if (nums.size() >= 3) out.orientation = {nums[0], nums[1], nums[2]};
        } else if (key == "velocity") {
            auto nums = ParseNumbers(rest);
            if (nums.size() >= 3) out.velocity = {nums[0], nums[1], nums[2]};
        } else if (key == "angularVelocity") {
            auto nums = ParseNumbers(rest);
            if (nums.size() >= 3) out.angular_vel = {nums[0], nums[1], nums[2]};
        } else if (key == "acceleration") {
            auto nums = ParseNumbers(rest);
            if (nums.size() >= 3) out.acceleration = {nums[0], nums[1], nums[2]};
        } else if (key == "accel") {
            std::istringstream(rest) >> out.accel;
        } else if (key == "brake") {
            std::istringstream(rest) >> out.brake;
        } else if (key == "steer") {
            std::istringstream(rest) >> out.steer;
        } else if (key == "linkid") {
            out.linkid = rest;
        } else if (key == "trafficlightid") {
            out.trafficlightid = rest;
        }
        // unknown keys ignored
    }
    return true;
}

}  // namespace scc_batch_synth
