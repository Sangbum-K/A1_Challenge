// object_parser.cpp
#include "scc_batch_synth/object_parser.hpp"

#include <fstream>
#include <sstream>
#include <string>

namespace scc_batch_synth {

namespace {

int ParseTrackIdSuffix(const std::string& id) {
    const auto pos = id.rfind('_');
    if (pos == std::string::npos) return 0;
    try {
        return std::stoi(id.substr(pos + 1));
    } catch (...) {
        return 0;
    }
}

}  // namespace

bool ParseObjectInfo(const std::string& path, std::vector<RawObject>& out) {
    out.clear();
    std::ifstream f(path);
    if (!f.is_open()) {
        // OBJECT_INFO 파일이 없는 frame 도 흔함 — 빈 결과로 OK
        return true;
    }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        RawObject o;
        if (!(iss >> o.entity_id)) continue;
        double pos[3], ori[3], size[3], vel[3];
        if (!(iss >> pos[0] >> pos[1] >> pos[2]
                  >> ori[0] >> ori[1] >> ori[2]
                  >> size[0] >> size[1] >> size[2]
                  >> vel[0] >> vel[1] >> vel[2])) continue;
        for (int i = 0; i < 3; ++i) {
            o.position[i]     = pos[i];
            o.orientation[i]  = ori[i];
            o.size[i]         = size[i];
            o.velocity_own[i] = vel[i];
        }
        int tid = 0;
        if (iss >> tid) {
            o.track_id = tid;
        } else {
            o.track_id = ParseTrackIdSuffix(o.entity_id);
        }
        out.push_back(o);
    }
    return true;
}

}  // namespace scc_batch_synth
