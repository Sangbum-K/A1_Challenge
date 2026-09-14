#include "simulator/transform_control2.hpp"

using namespace MoraiCppUdp;

TransformControl2::TransformControl2(const std::string& ip, int port) : UDPSender(ip, port) {}

bool TransformControl2::SendTransformControl2(const TransformControl2Packet& packet) {
    uint8_t buffer[28] = {0};
    std::memcpy(&buffer[0], &packet.ego_state.x, sizeof(float));
    std::memcpy(&buffer[4], &packet.ego_state.y, sizeof(float));
    std::memcpy(&buffer[8], &packet.ego_state.z, sizeof(float));
    std::memcpy(&buffer[12], &packet.ego_state.roll, sizeof(float));
    std::memcpy(&buffer[16], &packet.ego_state.pitch, sizeof(float));
    std::memcpy(&buffer[20], &packet.ego_state.yaw, sizeof(float));
    std::memcpy(&buffer[24], &packet.ego_state.steer_angle, sizeof(float));
    return Send(reinterpret_cast<const char*>(buffer), 28);
}
