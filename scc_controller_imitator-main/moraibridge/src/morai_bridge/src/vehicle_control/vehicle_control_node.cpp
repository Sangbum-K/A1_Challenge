/**
 * @file        vehicle_control_node.cpp
 * @brief       morai vehicle control node cpp file
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)      
 * 
 * @date        2024-05-16 created by Yuseung Na
 * 
 */

#include "vehicle_control/vehicle_control_node.hpp"

VehicleControl::VehicleControl(std::string node_name, double period)
    : TaskManager(node_name, period),
    i_command_steer_(0.0),
    i_command_pedal_accel_(0.0),
    i_command_pedal_brake_(0.0),
    i_pla_target_speed_(0.0),
    i_command_accel_(0.0),
    b_is_command_lateral_(false),
    b_is_command_longitudinal_(false) {
    // Initialize
    ROS_WARN_STREAM("[" << node_name << "] Initialize node (Period: " << 1/period << " Hz)");

    // Ini init
    std::string dir(getenv("PWD"));
    std::string ini_dir("/config/simulation.ini");
    util_ini_parser_.Init((dir + ini_dir).c_str());

    // ROS param init
    ProcessINI();
    ProcessRosparam();
    cfg_.task_period = period;

    // Subscriber init
    s_command_steer_ = nh_.subscribe(
        "app/con/command_steer", 10, &VehicleControl::CallbackCommandSteer, this);    
    s_command_pedal_accel_ = nh_.subscribe(
        "app/con/command_pedal_accel", 10, &VehicleControl::CallbackCommandPedalAccel, this);
    s_command_pedal_brake_ = nh_.subscribe(
        "app/con/command_pedal_brake", 10, &VehicleControl::CallbackCommandPedalBrake, this);
    s_pla_target_speed_ = nh_.subscribe(
        "app/pla/target_speed", 10, &VehicleControl::CallbackPlaTargetSpeed, this);
    s_command_accel_ = nh_.subscribe(
        "app/con/command_accel", 10, &VehicleControl::CallbackCommandAccel, this);

    // Publisher init
    if (cfg_.communication_control_type == "ros") {
        p_control_signal_ = nh_.advertise<morai_msgs::CtrlCmd>(
            "ctrl_cmd", 10);
        p_turn_signal_lamp_ = nh_.advertise<morai_msgs::Lamps>(
            "lamps", 10);
        b_is_publisher_initialized_ = true;
        b_is_udp_sender_initialized_ = false;
    }
    if (cfg_.communication_control_type == "udp") {
        morai_udp_ego_ctl_sender_ = std::make_unique<MoraiCppUdp::EgoControl>(cfg_.udp_source_ip, cfg_.udp_ego_ctl_source_port);
        morai_udp_lamps_sender_ = std::make_unique<MoraiCppUdp::TurnSignalLampControl>(cfg_.udp_source_ip, cfg_.udp_turn_signal_lamp_source_port);
        b_is_udp_sender_initialized_ = true;
        b_is_publisher_initialized_ = false;
    }
}

VehicleControl::~VehicleControl() {}

void VehicleControl::Run() {
    ProcessINI();
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get subscribe variables
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Check if command is received
    if (!b_is_command_lateral_) {
        ROS_ERROR_THROTTLE(1.0,"Wait for Command Lateral (Steering)...");
        return;
    }
    if (!b_is_command_longitudinal_) {
        ROS_ERROR_THROTTLE(1.0,"Wait for Command Longitudinal (Throttle or Velocity or Acceleration)...");
        return;
    }

    // Get command variables
    double command_steer = 0.0; {
        std::lock_guard<std::mutex> lock(mutex_command_steer_);
        command_steer = i_command_steer_;
    }
    double command_pedal_brake = 0.0; {
        std::lock_guard<std::mutex> lock(mutex_command_pedal_brake_);
        command_pedal_brake = i_command_pedal_brake_;
    }
    double command_pedal_accel = 0.0; {
        std::lock_guard<std::mutex> lock(mutex_command_pedal_accel_);
        command_pedal_accel = i_command_pedal_accel_;
    }
    double command_velocity = 0.0; {
        std::lock_guard<std::mutex> lock(mutex_pla_target_speed_);
        command_velocity = i_pla_target_speed_;
    }
    double command_accel = 0.0; {
        std::lock_guard<std::mutex> lock(mutex_command_accel_);
        command_accel = i_command_accel_;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Algorithm
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update output
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //    
    if (cfg_.communication_control_type == "ros" || cfg_.udp_morai_ros_multi_publish == true) {
        // ROS Publisher On
        if(!p_control_signal_) {
            p_control_signal_ = nh_.advertise<morai_msgs::CtrlCmd>(
                "ctrl_cmd", 10);
            b_is_publisher_initialized_ = true;
        }

        // Update control signal
        o_control_signal_ = ros_bridge::UpdateControlCommand(cfg_.control_type, command_steer, command_pedal_accel, command_pedal_brake, command_velocity, command_accel);
        o_turn_signal_lamp_ = ros_bridge::UpdateTurnSignalLamp(false, false, false);
    }

    if(cfg_.morai_simulator_version == 1 && cfg_.communication_control_type != "udp" ||
       cfg_.morai_simulator_version == 2) {
        // Morai1 UDP shutdown
        if(morai_udp_ego_ctl_sender_ != nullptr) {
            b_is_udp_sender_initialized_ = false;
            morai_udp_ego_ctl_sender_.reset();
        }
    }

    if (cfg_.morai_simulator_version == 1 && cfg_.communication_control_type == "udp") {
        // Morai2 UDP disable
        if(morai_udp_ego_ctl2_sender_ != nullptr) {
            b_is_udp_sender2_initialized_ = false;
            morai_udp_ego_ctl2_sender_.reset();
        }
        
        // Morai UDP Boot
        if(morai_udp_ego_ctl_sender_ == nullptr) {
            morai_udp_ego_ctl_sender_ = std::make_unique<MoraiCppUdp::EgoControl>(cfg_.udp_source_ip, cfg_.udp_ego_ctl_source_port);
            morai_udp_lamps_sender_ = std::make_unique<MoraiCppUdp::TurnSignalLampControl>(cfg_.udp_source_ip, cfg_.udp_turn_signal_lamp_source_port);
            b_is_udp_sender_initialized_ = true;
        }

        // Update control signal
        o_control_signal_udp_ = UpdateUdpControlSignal(cfg_.control_type, command_steer, command_pedal_accel, command_pedal_brake, command_velocity, command_accel);
        o_turn_signal_lamp_udp_ = UpdateUdpTurnSignalLamp(false, false, false);
    }

    if (cfg_.morai_simulator_version == 1 && cfg_.communication_control_type != "ros" && cfg_.udp_morai_ros_multi_publish == false) {
        // ROS Publisher shutdown
        if(p_control_signal_) {
            b_is_publisher_initialized_ = false;
            p_control_signal_.shutdown();
        }
    }

    if (cfg_.morai_simulator_version == 2) {
        // Morai1 Udp shutdown
        if(morai_udp_ego_ctl_sender_ != nullptr) {
            b_is_udp_sender_initialized_ = false;
            morai_udp_ego_ctl_sender_.reset();
        }
        if(morai_udp_lamps_sender_ != nullptr) {
            b_is_udp_sender_initialized_ = false;
            morai_udp_lamps_sender_.reset();
        }

        // Morai2 UDP enable
        if(morai_udp_ego_ctl2_sender_ == nullptr) {
            morai_udp_ego_ctl2_sender_ = std::make_unique<MoraiCppUdp::EgoControl2>(cfg_.udp_source_ip, cfg_.udp_ego_ctl_destination_port);
            b_is_udp_sender2_initialized_ = true;
        }

        // Update control signal
        // EgoControl2 protocol: accel[0~1], brake[0~1], steering=steering_wheel_deg
        o_control_signal_udp2_ = UpdateUdpControlSignal2(command_pedal_accel, command_pedal_brake, command_steer);

    }
}

void VehicleControl::Publish() {
    if (!b_is_command_lateral_ || !b_is_command_longitudinal_) {
        return;
    }

    if (cfg_.morai_simulator_version == 1 && (cfg_.communication_control_type == "ros" || cfg_.udp_morai_ros_multi_publish == true)) {
        if(b_is_publisher_initialized_ == true) {
            p_control_signal_.publish(o_control_signal_);
            if(cfg_.communication_control_type == "ros") {
                p_turn_signal_lamp_.publish(o_turn_signal_lamp_);
            }
        }
    } 
    if (cfg_.morai_simulator_version == 1 && cfg_.communication_control_type == "udp") {
        if(b_is_udp_sender_initialized_ == true) {
            morai_udp_ego_ctl_sender_->SendControlCommand(o_control_signal_udp_);
            morai_udp_lamps_sender_->SendLampCommand(o_turn_signal_lamp_udp_);
        }
    }
    if (cfg_.morai_simulator_version == 2) {
        if(b_is_udp_sender2_initialized_ == true) {
            if(cfg_.udp_morai_autonomous_mode == 1) {
                morai_udp_ego_ctl2_sender_->SendControlCommand(o_control_signal_udp2_);
            } else {
                std::cout << "[MoraiBridge Vehicle Control] Autonomous mode is disabled. Keyboard mode is enabled." << std::endl;
            }
        }
    }
}

void VehicleControl::ProcessRosparam() {}

void VehicleControl::ProcessINI() {
    if (util_ini_parser_.IsFileUpdated()) {
        util_ini_parser_.ParseConfig("MoraiBridge", "morai_simulator_version", cfg_.morai_simulator_version);
        util_ini_parser_.ParseConfig("MoraiBridge", "communication_control_type", cfg_.communication_control_type);
        util_ini_parser_.ParseConfig("MoraiBridge", "udp_source_ip", cfg_.udp_source_ip);
        util_ini_parser_.ParseConfig("MoraiBridge", "udp_destination_ip", cfg_.udp_destination_ip);
        util_ini_parser_.ParseConfig("MoraiVehicleControl", "max_accel", cfg_.max_accel);  
        util_ini_parser_.ParseConfig("MoraiVehicleControl", "max_decel", cfg_.max_decel);
        util_ini_parser_.ParseConfig("MoraiVehicleControl", "udp_ego_ctl_source_port", cfg_.udp_ego_ctl_source_port);
        util_ini_parser_.ParseConfig("MoraiVehicleControl", "udp_ego_ctl_destination_port", cfg_.udp_ego_ctl_destination_port);
        util_ini_parser_.ParseConfig("MoraiVehicleControl", "udp_turn_signal_lamp_source_port", cfg_.udp_turn_signal_lamp_source_port);
        util_ini_parser_.ParseConfig("MoraiVehicleControl", "udp_turn_signal_lamp_destination_port", cfg_.udp_turn_signal_lamp_destination_port);
        util_ini_parser_.ParseConfig("MoraiVehicleControl", "udp_morai_ros_multi_publish", cfg_.udp_morai_ros_multi_publish);
        util_ini_parser_.ParseConfig("MoraiVehicleControl", "udp_morai_autonomous_mode", cfg_.udp_morai_autonomous_mode);
        util_ini_parser_.ParseConfig("MoraiVehicleControl", "sim_steer_display_type", cfg_.sim_steer_display_type);
        util_ini_parser_.ParseConfig("MoraiVehicleControl", "sim_steer_range_max", cfg_.sim_steer_range_max);
        ROS_WARN("[Vehicle Control] Ini file is updated!");
        util_ini_parser_.ParseConfig("MoraiVehicleControl", "control_type", cfg_.control_type);
    }
}

MoraiCppUdp::EgoControl::ControlCommand VehicleControl::UpdateUdpControlSignal(const int & command_type, const double& command_steer, const double& command_pedal_accel, const double& command_pedal_brake, const double& command_velocity, const double& command_accel) {
    MoraiCppUdp::EgoControl::ControlCommand control_command;
    memset(&control_command, 0, sizeof(control_command));
    // ControlCommand.ctrl_mode = MoraiCppUdp::EgoControl::ControlMode::AUTO_MODE;
    if(cfg_.udp_morai_autonomous_mode == 1) {
        control_command.ctrl_mode = MoraiCppUdp::EgoControl::ControlMode::AUTO_MODE;
    } else {
        control_command.ctrl_mode = MoraiCppUdp::EgoControl::ControlMode::KEYBOARD;
    }
    control_command.gear = MoraiCppUdp::EgoControl::GearCommand::DRIVE;
    if(command_type == 1) { // Throttle control
        control_command.long_cmd_type = MoraiCppUdp::EgoControl::LongitudinalCommandType::THROTTLE;
        control_command.accel = command_pedal_accel;
        control_command.brake = command_pedal_brake;
    } else if(command_type == 2) { // Velocity control
        control_command.long_cmd_type = MoraiCppUdp::EgoControl::LongitudinalCommandType::VELOCITY;
        control_command.velocity = command_velocity;
    } else if(command_type == 3) { // Acceleration control
        control_command.long_cmd_type = MoraiCppUdp::EgoControl::LongitudinalCommandType::ACCELERATION;
        control_command.acceleration = command_accel;
    }
    // tire_angle [-25, 25 deg] → normalized [-1, 1] via linear mapping (MORAI_v2)
    control_command.steering = command_steer / interface::MAX_STEERING_TIRE_ANGLE;
    return control_command;
}

MoraiCppUdp::TurnSignalLampControl::LampCommand VehicleControl::UpdateUdpTurnSignalLamp(const bool& left_on, const bool& right_on, const bool& hazard_on) {
    MoraiCppUdp::TurnSignalLampControl::LampCommand lamp_command;
    memset(&lamp_command, 0, sizeof(lamp_command));
    if(left_on == false && right_on == false && hazard_on == false) {
        lamp_command.turn_signal = MoraiCppUdp::TurnSignalLampControl::TurnSignal::NONE;
        lamp_command.emergency_signal = MoraiCppUdp::TurnSignalLampControl::EmergencySignal::OFF;
    } else if(left_on == true && right_on == false && hazard_on == false) {
        lamp_command.turn_signal = MoraiCppUdp::TurnSignalLampControl::TurnSignal::LEFT;
        lamp_command.emergency_signal = MoraiCppUdp::TurnSignalLampControl::EmergencySignal::OFF;
    } else if(left_on == false && right_on == true && hazard_on == false) {
        lamp_command.turn_signal = MoraiCppUdp::TurnSignalLampControl::TurnSignal::RIGHT;
        lamp_command.emergency_signal = MoraiCppUdp::TurnSignalLampControl::EmergencySignal::OFF;
    } else if(left_on == true && right_on == true || hazard_on == true) {
        lamp_command.turn_signal = MoraiCppUdp::TurnSignalLampControl::TurnSignal::NONE;
        lamp_command.emergency_signal = MoraiCppUdp::TurnSignalLampControl::EmergencySignal::ON;
    }
    return lamp_command;
}

MoraiCppUdp::EgoControl2::ControlCommand2 VehicleControl::UpdateUdpControlSignal2(const double& command_pedal_accel, const double& command_pedal_brake, const double& command_steer) {
    MoraiCppUdp::EgoControl2::ControlCommand2 control_command2;
    memset(&control_command2, 0, sizeof(control_command2));
    control_command2.accel_input = command_pedal_accel;
    control_command2.brake_input = command_pedal_brake;
    
    double steering_tire_angle = 0.0;
    if (cfg_.sim_steer_display_type == 0) { // normalized range(-1.0 ~ 1.0)
        steering_tire_angle = command_steer / cfg_.sim_steer_range_max;
    } else if (cfg_.sim_steer_display_type == 1) { // tire steering angle (rad)
        steering_tire_angle = command_steer * interface::DEG2RAD;
    } else if (cfg_.sim_steer_display_type == 2) { // tire steering angle (deg)
        steering_tire_angle = command_steer;
    }
    control_command2.steering    = steering_tire_angle;
    return control_command2;
}

int main(int argc, char** argv) {
    std::string node_name = "vehicle_control";
    ros::init(argc, argv, node_name);
    ros::NodeHandle nh;

    double period;
    if (!nh.getParam("task_period/period_vehicle_control", period)){
        period = 0.05;
    }

    VehicleControl main_task(node_name, period);
    main_task.Exec();

    return 0;
}