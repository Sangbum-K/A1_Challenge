/**
 * @file transform_control2.hpp
 * @brief TransformControl2 class definition
 * @details This file contains the definition of the TransformControl2 class, which is used to send the TransformControl2 data to the Morai simulator.
 * 
 * @author Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * @date 2026-03-04
 */
#pragma once
#ifndef __TRANSFORM_CONTROL2_HPP__
#define __TRANSFORM_CONTROL2_HPP__

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "network/udp_sender.hpp"

namespace MoraiCppUdp {
class TransformControl2 : public UDPSender {
public:
#pragma pack(push, 1) // 1바이트 정렬 시작

    // 차량의 상태를 나타내는 구조체 (32 bytes)
    struct EgoState {
        float x;
        float y;
        float z;
        float roll;
        float pitch;
        float yaw;
        float steer_angle;
    };

    // 전체 패킷 구조체
    struct TransformControl2Packet {
        EgoState ego_state; // 차량 상태 데이터
    };

#pragma pack(pop) // 이전 정렬로 복원

public:
    TransformControl2(const std::string& ip, int port);
    ~TransformControl2() = default;

    // 패킷 전송
    bool SendTransformControl2(const TransformControl2Packet& packet);
};
} // namespace MoraiCppUdp

#endif // __TRANSFORM_CONTROL2_HPP__