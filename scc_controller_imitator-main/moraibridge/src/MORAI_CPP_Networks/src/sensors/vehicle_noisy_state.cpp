#include "sensors/vehicle_noisy_state.hpp"

#include <cstring>
#include <iomanip>
#include <iostream>

using namespace MoraiCppUdp;

VehicleNoisyState::VehicleNoisyState(const std::string& ip_address, uint16_t port)
    : UDPReceiver(ip_address, port), is_running_(false) {
    is_running_ = true;
    thread_vehicle_state_receiver_ = std::thread(&VehicleNoisyState::ThreadVehicleStateReceiver, this);
}

VehicleNoisyState::VehicleNoisyState(const std::string& ip_address, uint16_t port, VehicleNoisyStateCallback callback)
    : UDPReceiver(ip_address, port), is_running_(false), vehicle_state_callback_(callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    vehicle_state_callback_ = callback;
    is_running_ = true;
    thread_vehicle_state_receiver_ = std::thread(&VehicleNoisyState::ThreadVehicleStateReceiver, this);
}

VehicleNoisyState::~VehicleNoisyState() {
    is_running_ = false;
    if (thread_vehicle_state_receiver_.joinable()) {
        thread_vehicle_state_receiver_.join();
    }
    Close();
}

bool VehicleNoisyState::GetVehicleState(VehicleData& data) {
    data = packet_data_.packet.vehicle_data;
    return true;
}

void VehicleNoisyState::ThreadVehicleStateReceiver() {
    char packet_buffer[PACKET_SIZE];

    while (is_running_) {
        try {
            size_t received_size = 0;
            if (!Receive(packet_buffer, PACKET_SIZE, received_size)) {
                std::cerr << "Failed to receive vehicle noisy state data" << std::endl;
                continue;
            }

            std::memset(&packet_data_, 0, sizeof(VehicleNoisyStatePacketStruct));
            if (ParseVehicleState(packet_buffer, received_size, packet_data_)) {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (vehicle_state_callback_) {
                    vehicle_state_callback_(packet_data_.packet.vehicle_data);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Vehicle noisy state error: " << e.what() << std::endl;
            continue;
        }
    }
}

bool VehicleNoisyState::ParseVehicleState(const char* buffer, size_t size, VehicleNoisyStatePacketStruct& data) {
    if (size != PACKET_SIZE) {
        return false;
    }

    size_t offset = 0;
    // Header (14 bytes)
    std::memcpy(&data.packet.header, buffer + offset, sizeof(char[14]));
    offset += sizeof(char[14]);

    // Length (4 bytes)
    std::memcpy(&data.packet.length, buffer + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Aux (12 bytes)
    std::memcpy(&data.packet.AuxData, buffer + offset, sizeof(uint8_t[12]));
    offset += sizeof(uint8_t[12]);

    // sec, nsec
    std::memcpy(&data.packet.vehicle_data.seconds, buffer + offset, sizeof(int32_t));
    offset += sizeof(int32_t);
    std::memcpy(&data.packet.vehicle_data.nanoseconds, buffer + offset, sizeof(int32_t));
    offset += sizeof(int32_t);

    // Noisy fields (21 floats)
    std::memcpy(&data.packet.vehicle_data.noisy_pos_e, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_pos_n, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_pos_u, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_ori_r, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_ori_p, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_ori_y, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_vel_e, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_vel_n, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_vel_u, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_vel_x, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_vel_y, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_vel_z, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_roll_rate, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_pitch_rate, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_yaw_rate, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_acc_x, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_acc_y, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.noisy_acc_z, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.accel_input, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.brake_input, buffer + offset, sizeof(float));
    offset += sizeof(float);
    std::memcpy(&data.packet.vehicle_data.steering, buffer + offset, sizeof(float));
    offset += sizeof(float);

    // Tail (2 bytes)
    std::memcpy(&data.packet.tail, buffer + offset, sizeof(uint8_t[2]));
    offset += sizeof(uint8_t[2]);

    return true;
}


