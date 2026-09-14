#ifndef __VEHICLE_NOISY_STATE_HPP__
#define __VEHICLE_NOISY_STATE_HPP__

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../network/udp_receiver.hpp"

namespace MoraiCppUdp {

/**
 * @class VehicleNoisyState
 * @brief MORAI 시뮬레이터의 차량 상태(Noise 포함) 데이터를 수신하고 처리하는 클래스
 * @details UDP를 통해 차량의 상태 정보를 수신하고 파싱합니다.
 *          전체 패킷 크기: 124 Bytes (14 + 4 + 12 + 4 + 4 + (21*4) + 2)
 */
class VehicleNoisyState : public UDPReceiver {
    const static size_t PACKET_SIZE = 124; // 14 + 4 + 12 + 4 + 4 + (21*4) + 2

public:
    /**
     * @struct VehicleData
     * @brief 노이즈가 포함된 차량 상태 데이터를 저장하는 구조체
     */
    struct VehicleData {
        int32_t seconds;       ///< sec
        int32_t nanoseconds;   ///< nsec
        float noisy_pos_e;     ///< ENU position e (m)
        float noisy_pos_n;     ///< ENU position n (m)
        float noisy_pos_u;     ///< ENU position u (m)
        float noisy_ori_r;     ///< ENU roll (deg)
        float noisy_ori_p;     ///< ENU pitch (deg)
        float noisy_ori_y;     ///< ENU yaw (deg)
        float noisy_vel_e;     ///< ENU velocity e (km/h)
        float noisy_vel_n;     ///< ENU velocity n (km/h)
        float noisy_vel_u;     ///< ENU velocity u (km/h)
        float noisy_vel_x;     ///< Body velocity x (km/h)
        float noisy_vel_y;     ///< Body velocity y (km/h)
        float noisy_vel_z;     ///< Body velocity z (km/h)
        float noisy_roll_rate; ///< Body roll rate (rad/s)
        float noisy_pitch_rate;///< Body pitch rate (rad/s)
        float noisy_yaw_rate;  ///< Body yaw rate (rad/s)
        float noisy_acc_x;     ///< Body acc x (m/s^2)
        float noisy_acc_y;     ///< Body acc y (m/s^2)
        float noisy_acc_z;     ///< Body acc z (m/s^2)
        float accel_input;     ///< Pedal accel (0~1)
        float brake_input;     ///< Pedal brake (0~1)
        float steering;        ///< Steering (deg)
    };

    union VehicleNoisyStatePacketStruct {
        char data[PACKET_SIZE];
        struct {
            char header[14];          // 14 bytes
            uint32_t length;          // 4 bytes
            uint8_t AuxData[12];      // 12 bytes
            VehicleData vehicle_data; // timestamp + 21 floats
            uint8_t tail[2];          // 2 bytes
        } packet;
    };

    /**
     * @brief 콜백 함수 타입 정의
     */
    using VehicleNoisyStateCallback = std::function<void(const VehicleData&)>;

    /**
     * @brief VehicleState 클래스의 생성자
     * @param ip_address UDP 수신을 위한 IP 주소
     * @param port UDP 수신을 위한 포트 번호
     */
    VehicleNoisyState(const std::string& ip_address, uint16_t port);

    /**
     * @brief VehicleState 클래스의 생성자
     * @param ip_address UDP 수신을 위한 IP 주소
     * @param port UDP 수신을 위한 포트 번호
     * @param callback VehicleState 데이터를 처리할 콜백 함수
     */
    VehicleNoisyState(const std::string& ip_address, uint16_t port, VehicleNoisyStateCallback callback);

    /**
     * @brief VehicleState 클래스의 소멸자
     */
    ~VehicleNoisyState();

    /**
     * @brief 차량 상태 데이터를 가져옵니다
     * @param[out] data 차량 상태 데이터를 저장할 구조체
     * @return 데이터 수신 성공 여부
     */
    bool GetVehicleState(VehicleData& data);

    /**
     * @brief 콜백 등록 함수
     * @param callback 콜백 함수
     */
    void RegisterCallback(VehicleNoisyStateCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        vehicle_state_callback_ = callback;
    }

public:
    /**
     * @brief UDP 수신 스레드 함수
     */
    void ThreadVehicleStateReceiver();

    /**
     * @brief 수신된 데이터를 파싱합니다
     * @param buffer 수신된 원본 데이터 버퍼
     * @param size 버퍼의 크기
     * @param[out] data 파싱된 차량 상태 데이터를 저장할 구조체
     * @return 파싱 성공 여부
     */
    bool ParseVehicleState(const char* buffer, size_t size, VehicleNoisyStatePacketStruct& data);

    std::thread thread_vehicle_state_receiver_; ///< 차량 상태 수신 스레드
    std::mutex callback_mutex_;                 ///< 콜백 함수 보호를 위한 뮤텍스
    std::atomic<bool> is_running_{false};       ///< 스레드 실행 상태
    VehicleNoisyStatePacketStruct packet_data_;
    VehicleNoisyStateCallback vehicle_state_callback_{[](const VehicleData& data) {
        std::cout << "Note: Vehicle noisy state data received but no callback is registered.\n"
                  << "Noisy Info:\n"
                  << "\tTimestamp: " << data.seconds << "." << data.nanoseconds << "\n"
                  << "\tYaw(deg): " << data.noisy_ori_y << "\n"
                  << "\tSteering(deg): " << data.steering << "\n"
                  << "\tAccel/Brake: " << data.accel_input << "/" << data.brake_input << "\n"
                  << "Consider registering a callback using RegisterCallback()." << std::endl;
    }}; ///< 기본 콜백 함수
};

} // namespace MoraiCppUdp
#endif // __VEHICLE_NOISY_STATE_HPP__
