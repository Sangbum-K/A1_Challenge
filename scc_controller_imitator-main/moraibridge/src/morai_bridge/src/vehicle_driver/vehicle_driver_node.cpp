/**
 * @file        vehicle_driver_node.cpp
 * @brief       morai vehicle driver node cpp file
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)
 *              Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * 
 * @date        2024-05-16 created by Yuseung Na
 * @date        2024-12-27 modified by Jiwon Seok: Add Morai UDP
 * 
 */

#include "vehicle_driver/vehicle_driver_node.hpp"

VehicleDriver::VehicleDriver(std::string node_name, double period)
    : TaskManager(node_name, period) {
    // Initialize
    ROS_WARN_STREAM("[" << node_name << "] Initialize node (Period: " << 1/period << " Hz)");

    // Ini init
    std::string dir(getenv("PWD"));
    std::string ini_dir("/config/simulation.ini");
    util_ini_parser_.Init((dir + ini_dir).c_str());

    // Parameter init
    ProcessINI();
    ProcessRosparam();
    cfg_.task_period = period;

    // Publisher init
    p_vehicle_state_ = nh_.advertise<autohyu_msgs::VehicleState>(
        "app/loc/vehicle_state", 10);
    p_reference_ = nh_.advertise<autohyu_msgs::Reference>(
        "app/loc/reference", 10);
    p_vehicle_state_gt_ = nh_.advertise<autohyu_msgs::VehicleState>(
        "app/loc/gt_vehicle_state", 10);

    // Subscriber init
    if(cfg_.communication_pubsubser_type == "ros") {
        s_morai_ego_vehicle_status_ = std::make_unique<ros::Subscriber>(
            nh_.subscribe("Ego_topic", 10, &VehicleDriver::CallbackMoraiEgoVehicleStatus, this));
        s_morai_ego_noisy_status_ = std::make_unique<ros::Subscriber>(
            nh_.subscribe("Ego_noisy_topic", 10, &VehicleDriver::CallbackMoraiEgoNoisyVehicleStatus, this));
        s_morai_gps_ = std::make_unique<ros::Subscriber>(
            nh_.subscribe("gps", 10, &VehicleDriver::CallbackMoraiGPSMessage, this));
    }
    else if(cfg_.communication_pubsubser_type == "udp") {
        // Morai udp msg republisher
        p_morai_udp_ego_vehicle_status_ = nh_.advertise<morai_msgs::EgoVehicleStatus>(
            "udp/Ego_topic", 10);
        p_morai_udp_ego_noisy_info_ = nh_.advertise<morai_msgs::EgoNoisyStatus>(
            "udp/Ego_noisy_topic", 10);
            
        // Morai UDP init
        // Always create GT receiver (normal VehicleState)
        MoraiCppUdp::VehicleState::VehicleStateCallback callback_morai_udp_ego_vehicle_status_ = std::bind(
            &VehicleDriver::CallbackMoraiUdpVehicleState, this, std::placeholders::_1);
        morai_udp_vehicle_state_receiver_.reset(new MoraiCppUdp::VehicleState(
            cfg_.udp_destination_ip, cfg_.udp_vehicle_info_destination_port , callback_morai_udp_ego_vehicle_status_));

        // Optionally create Noisy receiver for live vehicle_state
        if(cfg_.use_noisy_vehicle == true) {
            MoraiCppUdp::VehicleNoisyState::VehicleNoisyStateCallback callback_noisy = std::bind(
                &VehicleDriver::CallbackMoraiUdpNoisyVehicleState, this, std::placeholders::_1);
            morai_udp_noisy_vehicle_state_receiver_.reset(new MoraiCppUdp::VehicleNoisyState(
                cfg_.udp_destination_ip, cfg_.udp_vehicle_noisy_info_destination_port, callback_noisy));
        }

        MoraiCppUdp::GPS::GPSCallback callback_morai_udp_gps_ = std::bind(
            &VehicleDriver::CallbackMoraiUdpGPS, this, std::placeholders::_1);
        morai_udp_gps_receiver_.reset(new MoraiCppUdp::GPS(cfg_.udp_destination_ip, 
            cfg_.udp_gps_destination_port, callback_morai_udp_gps_));
    }
    
    // Algorithm init
}

VehicleDriver::~VehicleDriver() {
    // Terminate
}

void VehicleDriver::Run() {
    ProcessINI();
    
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Algorithm 
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    // Check UDP or ROS
    if(cfg_.morai_simulator_version == 1 && cfg_.communication_pubsubser_type == "ros") {
        // Disable UDP
        if(morai_udp_vehicle_state_receiver_ != nullptr) {
            morai_udp_vehicle_state_receiver_.reset();
        }
        if(morai_udp_noisy_vehicle_state_receiver_ != nullptr) {
            morai_udp_noisy_vehicle_state_receiver_.reset();
        }
        if(morai_udp_gps_receiver_ != nullptr) {
            morai_udp_gps_receiver_.reset();
        }

        // Remove morai republisher
        if(p_morai_udp_ego_vehicle_status_) {
            p_morai_udp_ego_vehicle_status_.shutdown();
        }
        if(p_morai_udp_ego_noisy_info_) {
            p_morai_udp_ego_noisy_info_.shutdown();
        }

        // Enable ROS 
        if(s_morai_ego_vehicle_status_ == nullptr) {
            s_morai_ego_vehicle_status_ = std::make_unique<ros::Subscriber>(
                nh_.subscribe("Ego_topic", 10, &VehicleDriver::CallbackMoraiEgoVehicleStatus, this));
        }
        if(s_morai_ego_noisy_status_ == nullptr) {
            s_morai_ego_noisy_status_ = std::make_unique<ros::Subscriber>(
                nh_.subscribe("Ego_noisy_topic", 10, &VehicleDriver::CallbackMoraiEgoNoisyVehicleStatus, this));
        }
        if(s_morai_gps_ == nullptr) {
            s_morai_gps_ = std::make_unique<ros::Subscriber>(
                nh_.subscribe("gps", 10, &VehicleDriver::CallbackMoraiGPSMessage, this));
        }
    }
    else if(cfg_.morai_simulator_version == 1 && cfg_.communication_pubsubser_type == "udp") {
        // Disable ROS
        if(s_morai_ego_vehicle_status_ != nullptr) {
            s_morai_ego_vehicle_status_->shutdown();
            s_morai_ego_vehicle_status_.reset();  // Reset the unique_ptr after shutdown
        }
        if(s_morai_ego_noisy_status_ != nullptr) {
            s_morai_ego_noisy_status_->shutdown();
            s_morai_ego_noisy_status_.reset();  // Reset the unique_ptr after shutdown
        }
        if(s_morai_gps_ != nullptr) {
            s_morai_gps_->shutdown();
            s_morai_gps_.reset();  // Reset the unique_ptr after shutdown
        }

        // Enable UDP publisher if not exists
        if(!p_morai_udp_ego_vehicle_status_ && cfg_.udp_morai_ros_republish == true) {
            p_morai_udp_ego_vehicle_status_ = nh_.advertise<morai_msgs::EgoVehicleStatus>(
                "udp/Ego_topic", 10);
        }
        if(!p_morai_udp_ego_noisy_info_) {
            p_morai_udp_ego_noisy_info_ = nh_.advertise<morai_msgs::EgoNoisyStatus>(
                "udp/Ego_noisy_topic", 10);
        }

        // Enable UDP (runtime switching)
        // Always ensure GT receiver exists
        if(morai_udp_vehicle_state_receiver_ == nullptr) {
            MoraiCppUdp::VehicleState::VehicleStateCallback callback_morai_udp_ego_vehicle_status_ = std::bind(
                &VehicleDriver::CallbackMoraiUdpVehicleState, this, std::placeholders::_1);
            morai_udp_vehicle_state_receiver_.reset(new MoraiCppUdp::VehicleState(
                cfg_.udp_destination_ip, cfg_.udp_vehicle_info_destination_port, callback_morai_udp_ego_vehicle_status_));
        }
        // Create or destroy noisy receiver based on flag
        if(cfg_.use_noisy_vehicle == true) {
            if(morai_udp_noisy_vehicle_state_receiver_ == nullptr) {
                MoraiCppUdp::VehicleNoisyState::VehicleNoisyStateCallback callback_noisy = std::bind(
                    &VehicleDriver::CallbackMoraiUdpNoisyVehicleState, this, std::placeholders::_1);
                morai_udp_noisy_vehicle_state_receiver_.reset(new MoraiCppUdp::VehicleNoisyState(
                    cfg_.udp_destination_ip, cfg_.udp_vehicle_noisy_info_destination_port, callback_noisy));
            }
        } else {
            if(morai_udp_noisy_vehicle_state_receiver_ != nullptr) {
                morai_udp_noisy_vehicle_state_receiver_.reset();
            }
        }
        if(morai_udp_gps_receiver_ == nullptr) {
            MoraiCppUdp::GPS::GPSCallback callback_morai_udp_gps_ = std::bind(
                &VehicleDriver::CallbackMoraiUdpGPS, this, std::placeholders::_1);
            morai_udp_gps_receiver_.reset(new MoraiCppUdp::GPS(cfg_.udp_destination_ip, cfg_.udp_gps_destination_port, callback_morai_udp_gps_));
        }
    }
    else if(cfg_.morai_simulator_version == 2) {
        // Disable morai version1 ROS
        if(s_morai_ego_vehicle_status_ != nullptr) {
            s_morai_ego_vehicle_status_->shutdown();
            s_morai_ego_vehicle_status_.reset();  // Reset the unique_ptr after shutdown
        }
        if(s_morai_ego_noisy_status_ != nullptr) {
            s_morai_ego_noisy_status_->shutdown();
            s_morai_ego_noisy_status_.reset();  // Reset the unique_ptr after shutdown
        }
        if(s_morai_gps_ != nullptr) {
            s_morai_gps_->shutdown();
            s_morai_gps_.reset();  // Reset the unique_ptr after shutdown
        }

        // Disable morai1 UDP receivers
        if(morai_udp_vehicle_state_receiver_ != nullptr) {
            morai_udp_vehicle_state_receiver_.reset();
        }
        if(morai_udp_noisy_vehicle_state_receiver_ != nullptr) {
            morai_udp_noisy_vehicle_state_receiver_.reset();
        }
        if(morai_udp_gps_receiver_ != nullptr) {
            morai_udp_gps_receiver_.reset();
        }

        // Enable UDP publisher if not exists
        if(!p_morai_udp_ego_vehicle_status_ && cfg_.udp_morai_ros_republish == true) {
            p_morai_udp_ego_vehicle_status_ = nh_.advertise<morai_msgs::EgoVehicleStatus>(
                "udp/Ego_topic", 10);
        }
        if(!p_morai_udp_ego_noisy_info_) {
            p_morai_udp_ego_noisy_info_ = nh_.advertise<morai_msgs::EgoNoisyStatus>(
                "udp/Ego_noisy_topic", 10);
        }

        // Enable UDP (runtime switching)
        // Always ensure GT receiver exists
        if(morai_udp_vehicle_state2_receiver_ == nullptr) {
            MoraiCppUdp::VehicleState2::VehicleState2Callback callback_morai_udp_ego_vehicle_status2_ = std::bind(
                &VehicleDriver::CallbackMoraiUdpVehicleState2, this, std::placeholders::_1);
            morai_udp_vehicle_state2_receiver_.reset(new MoraiCppUdp::VehicleState2(
                cfg_.udp_destination_ip, cfg_.udp_vehicle_info_destination_port, callback_morai_udp_ego_vehicle_status2_));
        }
    }

    // This driver is running in the callback function.
}

void VehicleDriver::Publish() {
    p_reference_.publish(o_reference_point_);
}

void VehicleDriver::ProcessRosparam() {
    // Priority: rosparam (set by simulator.launch) > simulation.ini [MoraiSimLocation]
    if (!nh_.getParam("/vehicle_driver/ref_lat", cfg_.ref_lat)) {
        if (!util_ini_parser_.ParseConfig("MoraiSimLocation", "ref_lat", cfg_.ref_lat)) {
            cfg_.ref_lat = 0.0;
        }
    }
    if (!nh_.getParam("/vehicle_driver/ref_lon", cfg_.ref_lon)) {
        if (!util_ini_parser_.ParseConfig("MoraiSimLocation", "ref_lon", cfg_.ref_lon)) {
            cfg_.ref_lon = 0.0;
        }
    }
    if (!nh_.getParam("/vehicle_driver/proj_mode", cfg_.projection_mode)) {
        if (!util_ini_parser_.ParseConfig("MoraiSimLocation", "proj_mode", cfg_.projection_mode)) {
            cfg_.projection_mode = "";
        }
    }
    if (!nh_.getParam("/vehicle_driver/location", cfg_.location)) {
        if (!util_ini_parser_.ParseConfig("MoraiSimLocation", "location", cfg_.location)) {
            cfg_.location = "";
        }
    }
    ROS_WARN_STREAM("Reference Lat: " << cfg_.ref_lat << ", Lon: " << cfg_.ref_lon << ", Location: " << cfg_.location);

    interface::Reference reference;
    reference.wgs84.latitude  = cfg_.ref_lat;
    reference.wgs84.longitude = cfg_.ref_lon;
    reference.projection      = cfg_.projection_mode;
    o_reference_point_ = ros_bridge::UpdateReference(reference);
}

void VehicleDriver::ProcessINI() {
    if (util_ini_parser_.IsFileUpdated()) {
        ROS_WARN("[Vehicle Driver] Ini file is updated!");
        // Simulator
        if(util_ini_parser_.ParseConfig("MoraiBridge", "morai_simulator_version", cfg_.morai_simulator_version) == false) {
            std::cout << "morai_simulator_version is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiBridge", "communication_pubsubser_type", cfg_.communication_pubsubser_type) == false) {
            std::cout << "communication_pubsubser_type is not found in ini file" << std::endl;
        }

        if(util_ini_parser_.ParseConfig("MoraiBridge", "udp_source_ip", cfg_.udp_source_ip) == false) {
            std::cout << "udp_source_ip is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiBridge", "udp_destination_ip", cfg_.udp_destination_ip) == false) {
            std::cout << "udp_destination_ip is not found in ini file" << std::endl;
        }

        // Morai Vehicle Driver
        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "use_noisy_vehicle", cfg_.use_noisy_vehicle) == false) {
            std::cout << "use_noisy_vehicle is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "use_morai_enu_position", cfg_.use_morai_enu_position) == false) {
            std::cout << "use_morai_enu_position is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "udp_vehicle_info_source_port", cfg_.udp_vehicle_info_source_port) == false) {
            std::cout << "udp_vehicle_info_source_port is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "udp_vehicle_info_destination_port", cfg_.udp_vehicle_info_destination_port) == false) {
            std::cout << "udp_vehicle_info_destination_port is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "udp_vehicle_noisy_info_source_port", cfg_.udp_vehicle_noisy_info_source_port) == false) {
            std::cout << "udp_vehicle_noisy_info_source_port is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "udp_vehicle_noisy_info_destination_port", cfg_.udp_vehicle_noisy_info_destination_port) == false) {
            std::cout << "udp_vehicle_noisy_info_destination_port is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "udp_gps_source_port", cfg_.udp_gps_source_port) == false) {
            std::cout << "udp_gps_source_port is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "udp_gps_destination_port", cfg_.udp_gps_destination_port) == false) {
            std::cout << "udp_gps_destination_port is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "udp_morai_ros_republish", cfg_.udp_morai_ros_republish) == false) {
            std::cout << "udp_morai_ros_republish is not found in ini file" << std::endl;
        }

        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "sim_steer_display_type", cfg_.sim_steer_display_type) == false) {
            std::cout << "sim_steer_display_type is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "sim_steer_range_max", cfg_.sim_steer_range_max) == false) {
            std::cout << "sim_steer_range_max is not found in ini file" << std::endl;
        }
        if(util_ini_parser_.ParseConfig("MoraiVehicleDriver", "cg_position_x_from_rear", cfg_.cg_position_x_from_rear) == false) {
            std::cout << "cg_position_x_from_rear is not found in ini file" << std::endl;
        }
    }
}

void VehicleDriver::CallbackMoraiUdpVehicleState(const MoraiCppUdp::VehicleState::VehicleData& msg) {
    i_morai_udp_ego_vehicle_status_ = msg;
    
    // Publish AutoHYU Vehicle State
    autohyu_msgs::VehicleState vehicle_state = CvtMoraiUdpToAutohyuVehicleState(msg);
    // vehicle_state: only when not using noisy source
    if(cfg_.use_noisy_vehicle == false) {
        p_vehicle_state_.publish(vehicle_state);
    }
    p_vehicle_state_gt_.publish(vehicle_state);

    // Publish
    if(cfg_.udp_morai_ros_republish == true && p_morai_udp_ego_vehicle_status_) {
        // Change to morai_msgs::EgoVehicleStatus
        morai_msgs::EgoVehicleStatus ego_vehicle_status = CvtMoraiUdpToMoraiRosVehicleState(msg);
        p_morai_udp_ego_vehicle_status_.publish(ego_vehicle_status);
    }
}

void VehicleDriver::CallbackMoraiUdpNoisyVehicleState(const MoraiCppUdp::VehicleNoisyState::VehicleData& msg) {
    i_morai_udp_ego_noisy_status_ = msg;

    // Convert to internal VehicleState and publish
    if(cfg_.use_noisy_vehicle == true) {
        autohyu_msgs::VehicleState vehicle_state = CvtMoraiUdpNoisyToAutohyuVehicleState(msg);
        p_vehicle_state_.publish(vehicle_state);
    }

    // Republish as EgoNoisyStatus on /Ego_noisy_info
    if(cfg_.udp_morai_ros_republish == true && p_morai_udp_ego_noisy_info_) {
        morai_msgs::EgoNoisyStatus noisy_msg = CvtMoraiUdpToMoraiRosNoisyVehicleState(msg);
        p_morai_udp_ego_noisy_info_.publish(noisy_msg);
    }
}

void VehicleDriver::CallbackMoraiUdpVehicleState2(const MoraiCppUdp::VehicleState2::VehicleData& msg) {
    i_morai_udp_ego_vehicle_status2_ = msg;
    
    // Publish AutoHYU Vehicle State
    autohyu_msgs::VehicleState vehicle_state = CvtMoraiUdpToAutohyuVehicleState2(msg);
    static double prev_x = msg.position.x;
    static double prev_y = msg.position.y;
    static bool is_skiped = false;
    double curr_x = msg.position.x;
    double curr_y = msg.position.y;
    if ((std::abs(curr_x - prev_x) < 1e-6 || std::abs(curr_y - prev_y) < 1e-6)&& is_skiped==false) {
        is_skiped = true;
        return;
    }
    is_skiped = false;
    prev_x = curr_x;
    prev_y = curr_y;
    p_vehicle_state_.publish(vehicle_state); // morai2 does not use noisy vehicle state
    p_vehicle_state_gt_.publish(vehicle_state);

    // Publish Morai Vehicle State2
    if(cfg_.udp_morai_ros_republish == true && p_morai_udp_ego_vehicle_status_) {
        morai_msgs::EgoVehicleStatus ego_vehicle_status = CvtMoraiUdpToMoraiRosVehicleState2(msg);
        p_morai_udp_ego_vehicle_status_.publish(ego_vehicle_status);
    }
}

void VehicleDriver::CallbackMoraiUdpGPS(const MoraiCppUdp::GPS::GPSData& msg) {
    i_morai_udp_gps_ = msg;
    morai_msgs::GPSMessage gps_message = CvtMoraiUdpToMoraiRosGPS(msg);
    std::lock_guard<std::mutex> lock(mutex_morai_gps_);
    i_morai_gps_ = ros_bridge::GetMoraiGPSMessage(gps_message, o_reference_point_);
    b_is_gps = true;
}



morai_msgs::EgoVehicleStatus VehicleDriver::CvtMoraiUdpToMoraiRosVehicleState(const MoraiCppUdp::VehicleState::VehicleData& msg) {
    // Change to morai_msgs::EgoVehicleStatus
    morai_msgs::EgoVehicleStatus morai_msgs_ego_vehicle_status;
    
    // Header
    morai_msgs_ego_vehicle_status.header.stamp.sec = msg.timestamp.seconds;
    morai_msgs_ego_vehicle_status.header.stamp.nsec = msg.timestamp.nanoseconds;
    morai_msgs_ego_vehicle_status.header.frame_id = "MoraiInfo";
    morai_msgs_ego_vehicle_status.unique_id = 0;

    // Acceleration
    morai_msgs_ego_vehicle_status.acceleration.x = msg.acceleration.x;
    morai_msgs_ego_vehicle_status.acceleration.y = msg.acceleration.y;
    morai_msgs_ego_vehicle_status.acceleration.z = msg.acceleration.z;

    // Position
    morai_msgs_ego_vehicle_status.position.x = msg.position.x;
    morai_msgs_ego_vehicle_status.position.y = msg.position.y;
    morai_msgs_ego_vehicle_status.position.z = msg.position.z;

    // Velocity
    morai_msgs_ego_vehicle_status.velocity.x = msg.velocity.x;
    morai_msgs_ego_vehicle_status.velocity.y = msg.velocity.y;
    morai_msgs_ego_vehicle_status.velocity.z = msg.velocity.z;

    // Others
    morai_msgs_ego_vehicle_status.heading = msg.rotation.z;
    morai_msgs_ego_vehicle_status.accel = msg.accel_input;
    morai_msgs_ego_vehicle_status.brake = msg.brake_input;
    morai_msgs_ego_vehicle_status.wheel_angle = msg.steering;
    morai_msgs_ego_vehicle_status.lateral_offset = 0.0;

    // tire_lateral_force
    morai_msgs_ego_vehicle_status.tire_lateral_force_fl = msg.tire_lateral_force_fl;
    morai_msgs_ego_vehicle_status.tire_lateral_force_fr = msg.tire_lateral_force_fr;
    morai_msgs_ego_vehicle_status.tire_lateral_force_rl = msg.tire_lateral_force_rl;
    morai_msgs_ego_vehicle_status.tire_lateral_force_rr = msg.tire_lateral_force_rr;

    // side_slip_angle
    morai_msgs_ego_vehicle_status.side_slip_angle_fl = msg.side_slip_angle_fl;
    morai_msgs_ego_vehicle_status.side_slip_angle_fr = msg.side_slip_angle_fr;
    morai_msgs_ego_vehicle_status.side_slip_angle_rl = msg.side_slip_angle_rl;
    morai_msgs_ego_vehicle_status.side_slip_angle_rr = msg.side_slip_angle_rr;

    // tire_cornering_stiffness
    morai_msgs_ego_vehicle_status.tire_cornering_stiffness_fl = msg.tire_cornering_stiffness_fl;
    morai_msgs_ego_vehicle_status.tire_cornering_stiffness_fr = msg.tire_cornering_stiffness_fr;
    morai_msgs_ego_vehicle_status.tire_cornering_stiffness_rl = msg.tire_cornering_stiffness_rl;
    morai_msgs_ego_vehicle_status.tire_cornering_stiffness_rr = msg.tire_cornering_stiffness_rr;
    return std::move(morai_msgs_ego_vehicle_status);
}


morai_msgs::EgoNoisyStatus VehicleDriver::CvtMoraiUdpToMoraiRosNoisyVehicleState(const MoraiCppUdp::VehicleNoisyState::VehicleData& msg) {
    morai_msgs::EgoNoisyStatus out;

    // Header
    out.header.stamp.sec  = msg.seconds;
    out.header.stamp.nsec = msg.nanoseconds;
    out.header.frame_id   = "MoraiNoisyInfo";

    // ENU noisy position
    out.noisy_position.east  = msg.noisy_pos_e;
    out.noisy_position.north = msg.noisy_pos_n;
    out.noisy_position.up    = msg.noisy_pos_u;

    // Orientation (deg)
    out.noisy_orientation.roll  = msg.noisy_ori_r;
    out.noisy_orientation.pitch = msg.noisy_ori_p;
    out.noisy_orientation.yaw   = msg.noisy_ori_y;

    // ENU velocity (km/h)
    out.noisy_enu_velocity.east  = msg.noisy_vel_e;
    out.noisy_enu_velocity.north = msg.noisy_vel_n;
    out.noisy_enu_velocity.up    = msg.noisy_vel_u;

    // Body velocity (km/h)
    out.noisy_velocity.x = msg.noisy_vel_x;
    out.noisy_velocity.y = msg.noisy_vel_y;
    out.noisy_velocity.z = msg.noisy_vel_z;

    // Body angular velocity (rad/s)
    out.noisy_angularVelocity.roll  = msg.noisy_roll_rate;
    out.noisy_angularVelocity.pitch = msg.noisy_pitch_rate;
    out.noisy_angularVelocity.yaw   = msg.noisy_yaw_rate;

    // Body acceleration (m/s^2)
    out.noisy_acceleration.x = msg.noisy_acc_x;
    out.noisy_acceleration.y = msg.noisy_acc_y;
    out.noisy_acceleration.z = msg.noisy_acc_z;

    // Inputs
    out.accel = msg.accel_input;
    out.brake = msg.brake_input;
    out.steer = msg.steering;

    return out;
}


morai_msgs::EgoVehicleStatus VehicleDriver::CvtMoraiUdpToMoraiRosVehicleState2(const MoraiCppUdp::VehicleState2::VehicleData& msg) {
    morai_msgs::EgoVehicleStatus morai_msgs_ego_vehicle_status;

    // Header
    morai_msgs_ego_vehicle_status.header.stamp.sec = msg.timestamp.sec;
    morai_msgs_ego_vehicle_status.header.stamp.nsec = msg.timestamp.nsec;
    morai_msgs_ego_vehicle_status.header.frame_id = "MoraiInfo";
    morai_msgs_ego_vehicle_status.unique_id = 0;
    
    // Position
    morai_msgs_ego_vehicle_status.position.x = msg.position.x;
    morai_msgs_ego_vehicle_status.position.y = msg.position.y;
    morai_msgs_ego_vehicle_status.position.z = msg.position.z;

    // Velocity
    morai_msgs_ego_vehicle_status.velocity.x = msg.velocity.x;
    morai_msgs_ego_vehicle_status.velocity.y = msg.velocity.y;
    morai_msgs_ego_vehicle_status.velocity.z = msg.velocity.z;

    // Acceleration
    morai_msgs_ego_vehicle_status.acceleration.x = msg.acceleration.x;
    morai_msgs_ego_vehicle_status.acceleration.y = msg.acceleration.y;
    morai_msgs_ego_vehicle_status.acceleration.z = msg.acceleration.z;

    // Others
    morai_msgs_ego_vehicle_status.heading = msg.rotation.z;
    morai_msgs_ego_vehicle_status.accel = msg.accel_input;
    morai_msgs_ego_vehicle_status.brake = msg.brake_input;
    morai_msgs_ego_vehicle_status.wheel_angle = msg.steering;
    morai_msgs_ego_vehicle_status.lateral_offset = 0.0;

    return std::move(morai_msgs_ego_vehicle_status);
}



morai_msgs::GPSMessage VehicleDriver::CvtMoraiUdpToMoraiRosGPS(const MoraiCppUdp::GPS::GPSData& msg) {
    morai_msgs::GPSMessage gps_message;

    // Header
    gps_message.header.stamp = ros::Time(msg.timestamp);
    gps_message.header.frame_id = "GPS-1";

    // LLH
    gps_message.latitude = msg.latitude;
    gps_message.longitude = msg.longitude;
    gps_message.altitude = msg.altitude;

    // UTM (Convert LLH to UTM)
    lanelet::GPSPoint i_gps_point;
    i_gps_point.lat = msg.latitude;
    i_gps_point.lon = msg.longitude;
    i_gps_point.ele = 0.0;

    // // Convert LLH to UTM: TBD
    // lanelet::Origin origin({0.0, 0.0, 0.0});
    // lanelet::projection::UtmProjector projector(origin);
    // lanelet::BasicPoint3d proj_position = projector.forward(i_gps_point);
    // gps_message.eastOffset = proj_position.x();
    // gps_message.northOffset = proj_position.y();
    gps_message.status = msg.is_valid;

    return std::move(gps_message);
}

autohyu_msgs::VehicleState VehicleDriver::CvtMoraiUdpToAutohyuVehicleState(const MoraiCppUdp::VehicleState::VehicleData& msg) {
    autohyu_msgs::VehicleState vehicle_state;

    // Set header
    vehicle_state.header.stamp.sec = msg.timestamp.seconds;
    vehicle_state.header.stamp.nsec = msg.timestamp.nanoseconds;
    vehicle_state.header.frame_id = "world";

    // Convert UTM to WGS84 using reference point
    // Ego to GPS
    interface::Point morai_point;   
    morai_point.x = msg.position.x;
    morai_point.y = msg.position.y;
    morai_point.z = msg.position.z;
    interface::Reference morai_ref;
    morai_ref.wgs84.latitude  = cfg_.ref_lat;
    morai_ref.wgs84.longitude = cfg_.ref_lon;
    morai_ref.wgs84.altitude  = 0.0;
    morai_ref.projection      = "UTM";
    interface::WGS84 morai_wgs84 = ros_bridge::GetMoraiWGS84FromMoraiENU(morai_ref, morai_point);

    vehicle_state.gnss.latitude = morai_wgs84.latitude;
    vehicle_state.gnss.longitude = morai_wgs84.longitude;
    vehicle_state.gnss.altitude = morai_wgs84.altitude;

    if(cfg_.use_morai_enu_position == true) {
        vehicle_state.x = msg.position.x;
        vehicle_state.y = msg.position.y;
        vehicle_state.z = msg.position.z;
    } else {
        // Position (WGS84 to ENU coordinates)
        // Convert WGS84 to ENU using reference point
        interface::Reference reference;
        reference.wgs84.latitude  = cfg_.ref_lat;
        reference.wgs84.longitude = cfg_.ref_lon;
        reference.projection      = cfg_.projection_mode;
        interface::Point enu_position = util_function::GetEnu(reference, morai_wgs84);

        vehicle_state.x = enu_position.x;
        vehicle_state.y = enu_position.y;
        vehicle_state.z = enu_position.z;
    }

    // Vehicle to CG position
    Eigen::Vector3d vehicle_to_cg_position = Eigen::Vector3d::Zero();
    vehicle_to_cg_position.x() = cfg_.cg_position_x_from_rear;
    vehicle_to_cg_position.y() = 0.0;
    vehicle_to_cg_position.z() = 0.0;

    // Velocity (Vehicle coordinates, MORAI v2 UDP already provides m/s) at CG position
    Eigen::Vector3d ang_vel;
    ang_vel.x() = interface::DEG2RAD * msg.angular_velocity.x;
    ang_vel.y() = interface::DEG2RAD * msg.angular_velocity.y;
    ang_vel.z() = interface::DEG2RAD * msg.angular_velocity.z;

    // Rotate velocity vector obserbed in vehicle coordinates to CG position
    Eigen::Vector3d rotated_vel = ang_vel.cross(vehicle_to_cg_position);

    vehicle_state.vx = msg.velocity.x * interface::KPH2MPS - rotated_vel.x();  // km/h to m/s
    vehicle_state.vy = msg.velocity.y * interface::KPH2MPS - rotated_vel.y();
    vehicle_state.vz = msg.velocity.z * interface::KPH2MPS - rotated_vel.z();

    // Acceleration (Vehicle coordinates)
    static Eigen::Vector3d prev_ang_vel = Eigen::Vector3d::Zero();
    static double prev_time = 0.0;

    double curr_time = static_cast<double>(msg.timestamp.seconds) + static_cast<double>(msg.timestamp.nanoseconds) * 1e-9;
    double dt = curr_time - prev_time;

    Eigen::Vector3d ang_acc = Eigen::Vector3d::Zero();
    if(dt > 0.0) {
        ang_acc = (ang_vel - prev_ang_vel) / dt;
    }
    prev_ang_vel = ang_vel;
    prev_time = curr_time;

    // Rotate acceleration vector obserbed in vehicle coordinates to CG position
    Eigen::Vector3d rotated_acc = ang_vel.cross(rotated_vel) + ang_acc.cross(vehicle_to_cg_position);

    // Acceleration (Vehicle coordinates)
    vehicle_state.ax = msg.acceleration.x - rotated_acc.x();
    vehicle_state.ay = msg.acceleration.y - rotated_acc.y();
    vehicle_state.az = msg.acceleration.z - rotated_acc.z();

    // Orientation (convert from degrees to radians)
    vehicle_state.roll = msg.rotation.x * interface::DEG2RAD;
    vehicle_state.pitch = msg.rotation.y * interface::DEG2RAD;
    vehicle_state.yaw = msg.rotation.z * interface::DEG2RAD;

    // Angular velocity (convert from degrees/s to radians/s)
    vehicle_state.roll_vel = msg.angular_velocity.x * interface::DEG2RAD;
    vehicle_state.pitch_vel = msg.angular_velocity.y * interface::DEG2RAD;
    vehicle_state.yaw_vel = msg.angular_velocity.z * interface::DEG2RAD;

    // Link ID
    vehicle_state.link_id = msg.mgeo_link_id;

    // Vehicle CAN data
    vehicle_state.vehicle_can.header = vehicle_state.header;
    vehicle_state.vehicle_can.header.frame_id = "can";

    double steering_tire_angle = 0.0;
    if (cfg_.sim_steer_display_type == 0) { // normalized range(-1.0 ~ 1.0)
        steering_tire_angle = msg.steering * cfg_.sim_steer_range_max * interface::DEG2RAD;
    } else if (cfg_.sim_steer_display_type == 1) { // tire steering angle (rad)
        steering_tire_angle = msg.steering;
    } else if (cfg_.sim_steer_display_type == 2) { // tire steering angle (deg)
        steering_tire_angle = msg.steering * interface::DEG2RAD;
    }

    vehicle_state.vehicle_can.steering_wheel_angle = steering_tire_angle * STEERING_WHEEL_RATIO2;
    vehicle_state.vehicle_can.steering_tire_angle = steering_tire_angle;  // Convert to radians
    vehicle_state.vehicle_can.steering_state = interface::MdpsState::MDPS_ACTIVATE;  // MDPS_READY

    vehicle_state.vehicle_can.accel_position = msg.accel_input * 100.0;
    vehicle_state.vehicle_can.brake_position = msg.brake_input * 100.0;
    
    // Convert gear mode
    switch(msg.gear) {
        case MoraiCppUdp::VehicleState::GearMode::PARKING:
            vehicle_state.vehicle_can.gear_select = interface::Gear::GEAR_P;  // GEAR_P
            break;
        case MoraiCppUdp::VehicleState::GearMode::REVERSE:
            vehicle_state.vehicle_can.gear_select = interface::Gear::GEAR_R;  // GEAR_R
            break;
        case MoraiCppUdp::VehicleState::GearMode::NEUTRAL:
            vehicle_state.vehicle_can.gear_select = interface::Gear::GEAR_N;  // GEAR_N
            break;
        case MoraiCppUdp::VehicleState::GearMode::DRIVE:
            vehicle_state.vehicle_can.gear_select = interface::Gear::GEAR_D;  // GEAR_D
            break;
        default:
            vehicle_state.vehicle_can.gear_select = interface::Gear::GEAR_D;  // Default to GEAR_D
    }

    // Set operation mode based on control mode
    if (msg.ctrl_mode == MoraiCppUdp::VehicleState::ControlMode::AUTO) {
        vehicle_state.vehicle_can.operation_mode = interface::OperationMode::AUTONOMOUS;          // AUTONOMOUS
        vehicle_state.vehicle_can.lateral_ad_mode = interface::AutonomousMode::RUN;         // READY
        vehicle_state.vehicle_can.longitudinal_ad_mode = interface::AutonomousMode::RUN;    // READY
    } else {
        vehicle_state.vehicle_can.operation_mode = interface::OperationMode::MANUAL;          // MANUAL
        vehicle_state.vehicle_can.lateral_ad_mode = interface::AutonomousMode::READY;         // NOT_READY
        vehicle_state.vehicle_can.longitudinal_ad_mode = interface::AutonomousMode::READY;    // NOT_READY
    }

    return std::move(vehicle_state);
}

autohyu_msgs::VehicleState VehicleDriver::CvtMoraiUdpNoisyToAutohyuVehicleState(const MoraiCppUdp::VehicleNoisyState::VehicleData& msg) {
    autohyu_msgs::VehicleState vehicle_state;

    // Header
    vehicle_state.header.stamp.sec  = msg.seconds;
    vehicle_state.header.stamp.nsec = msg.nanoseconds;
    vehicle_state.header.frame_id   = "world";

    // GNSS emulation: ENU -> WGS84 using location config
    interface::Point morai_point;   
    morai_point.x = msg.noisy_pos_e;
    morai_point.y = msg.noisy_pos_n;
    morai_point.z = msg.noisy_pos_u;
    interface::Reference morai_ref;
    morai_ref.wgs84.latitude  = cfg_.ref_lat;
    morai_ref.wgs84.longitude = cfg_.ref_lon;
    morai_ref.wgs84.altitude  = 0.0;
    morai_ref.projection      = "UTM";
    interface::WGS84 morai_wgs84 = ros_bridge::GetMoraiWGS84FromMoraiENU(morai_ref, morai_point);

    vehicle_state.gnss.latitude  = morai_wgs84.latitude;
    vehicle_state.gnss.longitude = morai_wgs84.longitude;
    vehicle_state.gnss.altitude  = morai_wgs84.altitude;

    if(cfg_.use_morai_enu_position == true) {
        // Use ENU from simulator directly
        vehicle_state.x = msg.noisy_pos_e;
        vehicle_state.y = msg.noisy_pos_n;
        vehicle_state.z = msg.noisy_pos_u;
    } else {
        // Recompute ENU from WGS84 with reference settings (consistent with GT path)
        interface::Reference reference;
        reference.wgs84.latitude  = cfg_.ref_lat;
        reference.wgs84.longitude = cfg_.ref_lon;
        reference.projection      = cfg_.projection_mode;
        interface::Point enu_position = util_function::GetEnu(reference, morai_wgs84);

        vehicle_state.x = enu_position.x;
        vehicle_state.y = enu_position.y;
        vehicle_state.z = enu_position.z;
    }

    // Vehicle to CG position
    Eigen::Vector3d vehicle_to_cg_position = Eigen::Vector3d::Zero();
    vehicle_to_cg_position.x() = cfg_.cg_position_x_from_rear;
    vehicle_to_cg_position.y() = 0.0;
    vehicle_to_cg_position.z() = 0.0;

    // Velocity (Vehicle coordinates, convert from km/h to m/s) at CG position
    Eigen::Vector3d ang_vel;
    ang_vel.x() = msg.noisy_roll_rate;
    ang_vel.y() = msg.noisy_pitch_rate;
    ang_vel.z() = msg.noisy_yaw_rate;

    // Rotate velocity vector obserbed in vehicle coordinates to CG position
    Eigen::Vector3d rotated_vel = ang_vel.cross(vehicle_to_cg_position);

    // Velocity (vehicle/body coords) convert km/h -> m/s
    vehicle_state.vx = msg.noisy_vel_x * interface::KPH2MPS - rotated_vel.x();
    vehicle_state.vy = msg.noisy_vel_y * interface::KPH2MPS - rotated_vel.y();
    vehicle_state.vz = msg.noisy_vel_z * interface::KPH2MPS - rotated_vel.z();

    // Acceleration (Vehicle coordinates)
    static Eigen::Vector3d prev_ang_vel = Eigen::Vector3d::Zero();
    static double prev_time = 0.0;

    double curr_time = static_cast<double>(msg.seconds) + static_cast<double>(msg.nanoseconds) * 1e-9;
    double dt = curr_time - prev_time;

    Eigen::Vector3d ang_acc = Eigen::Vector3d::Zero();
    if(dt > 0.1) {
        ang_acc = (ang_vel - prev_ang_vel) / dt;
        prev_ang_vel = ang_vel;
        prev_time = curr_time;
    }

    // Rotate acceleration vector obserbed in vehicle coordinates to CG position
    Eigen::Vector3d rotated_acc = ang_vel.cross(rotated_vel) + ang_acc.cross(vehicle_to_cg_position);

    // Acceleration (body) already m/s^2
    vehicle_state.ax = msg.noisy_acc_x - rotated_acc.x();
    vehicle_state.ay = msg.noisy_acc_y - rotated_acc.y();
    vehicle_state.az = msg.noisy_acc_z - rotated_acc.z();

    // Orientation (deg -> rad)
    vehicle_state.roll  = msg.noisy_ori_r * interface::DEG2RAD;
    vehicle_state.pitch = msg.noisy_ori_p * interface::DEG2RAD;
    vehicle_state.yaw   = msg.noisy_ori_y * interface::DEG2RAD;

    // Angular velocity (already rad/s in spec)
    vehicle_state.roll_vel  = msg.noisy_roll_rate;
    vehicle_state.pitch_vel = msg.noisy_pitch_rate;
    vehicle_state.yaw_vel   = msg.noisy_yaw_rate;

    // Vehicle CAN (inputs)
    vehicle_state.vehicle_can.header = vehicle_state.header;
    vehicle_state.vehicle_can.header.frame_id = "can";

    double steering_tire_angle = 0.0;
    if (cfg_.sim_steer_display_type == 0) { // normalized range(-1.0 ~ 1.0)
        steering_tire_angle = msg.steering * cfg_.sim_steer_range_max * interface::DEG2RAD;
    } else if (cfg_.sim_steer_display_type == 1) { // tire steering angle (rad)
        steering_tire_angle = msg.steering;
    } else if (cfg_.sim_steer_display_type == 2) { // tire steering angle (deg)
        steering_tire_angle = msg.steering * interface::DEG2RAD;
    }

    vehicle_state.vehicle_can.steering_wheel_angle = steering_tire_angle * STEERING_WHEEL_RATIO2;
    vehicle_state.vehicle_can.steering_tire_angle  = steering_tire_angle;
    vehicle_state.vehicle_can.steering_state       = interface::MdpsState::MDPS_ACTIVATE;
    vehicle_state.vehicle_can.accel_position       = msg.accel_input * 100.0;
    vehicle_state.vehicle_can.brake_position       = msg.brake_input * 100.0;
    vehicle_state.vehicle_can.gear_select          = interface::Gear::GEAR_D;
    vehicle_state.vehicle_can.operation_mode       = interface::OperationMode::AUTONOMOUS;
    vehicle_state.vehicle_can.lateral_ad_mode      = interface::AutonomousMode::RUN;
    vehicle_state.vehicle_can.longitudinal_ad_mode = interface::AutonomousMode::RUN;

    return vehicle_state;
}

autohyu_msgs::VehicleState VehicleDriver::CvtMoraiUdpToAutohyuVehicleState2(const MoraiCppUdp::VehicleState2::VehicleData& msg) {
    autohyu_msgs::VehicleState vehicle_state;

    // Set header
    vehicle_state.header.stamp.sec = msg.timestamp.sec;
    vehicle_state.header.stamp.nsec = msg.timestamp.nsec;
    vehicle_state.header.frame_id = "world";

    // Convert UTM to WGS84 using reference point
    // Ego to GPS
    interface::Point morai_point;   
    morai_point.x = msg.position.x;
    morai_point.y = msg.position.y;
    morai_point.z = msg.position.z;
    interface::Reference morai_ref;
    morai_ref.wgs84.latitude  = cfg_.ref_lat;
    morai_ref.wgs84.longitude = cfg_.ref_lon;
    morai_ref.wgs84.altitude  = 0.0;
    morai_ref.projection      = "UTM";
    interface::WGS84 morai_wgs84 = ros_bridge::GetMoraiWGS84FromMoraiENU(morai_ref, morai_point);

    vehicle_state.gnss.latitude = morai_wgs84.latitude;
    vehicle_state.gnss.longitude = morai_wgs84.longitude;
    vehicle_state.gnss.altitude = morai_wgs84.altitude;

    if(cfg_.use_morai_enu_position == true) {
        vehicle_state.x = msg.position.x;
        vehicle_state.y = msg.position.y;
        vehicle_state.z = msg.position.z;
    } else {
        // Position (WGS84 to ENU coordinates)
        // Convert WGS84 to ENU using reference point
        interface::Reference reference;
        reference.wgs84.latitude  = cfg_.ref_lat;
        reference.wgs84.longitude = cfg_.ref_lon;
        reference.projection      = cfg_.projection_mode;
        interface::Point enu_position = util_function::GetEnu(reference, morai_wgs84);

        vehicle_state.x = enu_position.x;
        vehicle_state.y = enu_position.y;
        vehicle_state.z = enu_position.z;
    }

    // Vehicle to CG position
    Eigen::Vector3d vehicle_to_cg_position = Eigen::Vector3d::Zero();
    vehicle_to_cg_position.x() = cfg_.cg_position_x_from_rear;
    vehicle_to_cg_position.y() = 0.0;
    vehicle_to_cg_position.z() = 0.0;

    // Velocity (Vehicle coordinates, convert from km/h to m/s) at CG position
    Eigen::Vector3d ang_vel;
    ang_vel.x() = interface::DEG2RAD * msg.angular_velocity.x;
    ang_vel.y() = interface::DEG2RAD * msg.angular_velocity.y;
    ang_vel.z() = interface::DEG2RAD * msg.angular_velocity.z;

    // Rotate velocity vector obserbed in vehicle coordinates to CG position
    Eigen::Vector3d rotated_vel = ang_vel.cross(vehicle_to_cg_position);

    vehicle_state.vx = msg.velocity.x;// - rotated_vel.x();
    vehicle_state.vy = msg.velocity.y;// - rotated_vel.y();
    vehicle_state.vz = msg.velocity.z;// - rotated_vel.z();

    // Acceleration (Vehicle coordinates)
    static Eigen::Vector3d prev_ang_vel = Eigen::Vector3d::Zero();
    static double prev_time = 0.0;

    double curr_time = static_cast<double>(msg.timestamp.sec) + static_cast<double>(msg.timestamp.nsec) * 1e-9;
    double dt = curr_time - prev_time;

    Eigen::Vector3d ang_acc = Eigen::Vector3d::Zero();
    if(dt > 0.0) {
        ang_acc = (ang_vel - prev_ang_vel) / dt;
    }
    prev_ang_vel = ang_vel;
    prev_time = curr_time;

    // Rotate acceleration vector obserbed in vehicle coordinates to CG position
    Eigen::Vector3d rotated_acc = ang_vel.cross(rotated_vel) + ang_acc.cross(vehicle_to_cg_position);

    // Acceleration (Vehicle coordinates)
    vehicle_state.ax = msg.acceleration.x;//  - rotated_acc.x();
    vehicle_state.ay = msg.acceleration.y;//  - rotated_acc.y();
    vehicle_state.az = msg.acceleration.z;//  - rotated_acc.z();

    // Orientation (convert from degrees to radians)
    vehicle_state.roll = msg.rotation.x * interface::DEG2RAD;
    vehicle_state.pitch = msg.rotation.y * interface::DEG2RAD;
    vehicle_state.yaw = msg.rotation.z * interface::DEG2RAD;

    // Angular velocity (convert from degrees/s to radians/s)
    vehicle_state.roll_vel = msg.angular_velocity.x * interface::DEG2RAD;
    vehicle_state.pitch_vel = msg.angular_velocity.y * interface::DEG2RAD;
    vehicle_state.yaw_vel = msg.angular_velocity.z * interface::DEG2RAD;

    // Link ID
    vehicle_state.link_id = "N/A";

    // Vehicle CAN data
    vehicle_state.vehicle_can.header = vehicle_state.header;
    vehicle_state.vehicle_can.header.frame_id = "can";

    double steering_tire_angle = 0.0;
    if (cfg_.sim_steer_display_type == 0) { // normalized range(-1.0 ~ 1.0)
        steering_tire_angle = msg.steering * cfg_.sim_steer_range_max * interface::DEG2RAD;
    } else if (cfg_.sim_steer_display_type == 1) { // tire steering angle (rad)
        steering_tire_angle = msg.steering;
    } else if (cfg_.sim_steer_display_type == 2) { // tire steering angle (deg)
        steering_tire_angle = msg.steering * interface::DEG2RAD;
    }
  
    vehicle_state.vehicle_can.steering_wheel_angle = steering_tire_angle * STEERING_WHEEL_RATIO2;
    vehicle_state.vehicle_can.steering_tire_angle = steering_tire_angle;  // Convert to radians
    vehicle_state.vehicle_can.steering_state = interface::MdpsState::MDPS_ACTIVATE;  // MDPS_READY

    vehicle_state.vehicle_can.accel_position = msg.accel_input * 100.0;
    vehicle_state.vehicle_can.brake_position = msg.brake_input * 100.0;
    
    // Convert gear mode
    // switch(msg.gear) { // morai2 does not use gear mode
    switch(MoraiCppUdp::VehicleState::GearMode::DRIVE) {
        case MoraiCppUdp::VehicleState::GearMode::PARKING:
            vehicle_state.vehicle_can.gear_select = interface::Gear::GEAR_P;  // GEAR_P
            break;
        case MoraiCppUdp::VehicleState::GearMode::REVERSE:
            vehicle_state.vehicle_can.gear_select = interface::Gear::GEAR_R;  // GEAR_R
            break;
        case MoraiCppUdp::VehicleState::GearMode::NEUTRAL:
            vehicle_state.vehicle_can.gear_select = interface::Gear::GEAR_N;  // GEAR_N
            break;
        case MoraiCppUdp::VehicleState::GearMode::DRIVE:
            vehicle_state.vehicle_can.gear_select = interface::Gear::GEAR_D;  // GEAR_D
            break;
        default:
            vehicle_state.vehicle_can.gear_select = interface::Gear::GEAR_D;  // Default to GEAR_D
    }

    // Set operation mode based on control mode
    vehicle_state.vehicle_can.operation_mode = interface::OperationMode::AUTONOMOUS;          // AUTONOMOUS
    vehicle_state.vehicle_can.lateral_ad_mode = interface::AutonomousMode::RUN;         // READY
    vehicle_state.vehicle_can.longitudinal_ad_mode = interface::AutonomousMode::RUN;    // READY
    
    
    return vehicle_state;
}


interface::VehicleState VehicleDriver::GetMoraiEgoNoisyStatus(const morai_msgs::EgoNoisyStatus& msg) {
    interface::VehicleState noisy_state;
    noisy_state.header = ros_bridge::GetHeader(msg.header);
    noisy_state.header.frame_id = "world";

    // Gnss
    interface::Point morai_point;
    morai_point.x = msg.noisy_position.east;
    morai_point.y = msg.noisy_position.north;
    morai_point.z = msg.noisy_position.up;
    interface::Reference morai_ref;
    morai_ref.wgs84.latitude  = cfg_.ref_lat;
    morai_ref.wgs84.longitude = cfg_.ref_lon;
    morai_ref.wgs84.altitude  = 0.0;
    morai_ref.projection      = "UTM";
    interface::WGS84 morai_wgs84 = ros_bridge::GetMoraiWGS84FromMoraiENU(morai_ref, morai_point);

    noisy_state.gnss.latitude  = morai_wgs84.latitude;
    noisy_state.gnss.longitude = morai_wgs84.longitude;
    noisy_state.gnss.altitude  = morai_wgs84.altitude;

    // Position

    if(cfg_.use_morai_enu_position == true) {
        noisy_state.x = msg.noisy_position.east;
        noisy_state.y = msg.noisy_position.north;
        noisy_state.z = msg.noisy_position.up;
    } else {
        interface::Reference reference;
        reference.wgs84.latitude  = cfg_.ref_lat;
        reference.wgs84.longitude = cfg_.ref_lon;
        reference.projection      = cfg_.projection_mode;
        interface::Point enu_position = util_function::GetEnu(reference, morai_wgs84);

        noisy_state.x = enu_position.x;
        noisy_state.y = enu_position.y;
        noisy_state.z = enu_position.z;
    }

    // Vehicle to CG position
    Eigen::Vector3d vehicle_to_cg_position = Eigen::Vector3d::Zero();
    vehicle_to_cg_position.x() = cfg_.cg_position_x_from_rear;
    vehicle_to_cg_position.y() = 0.0;
    vehicle_to_cg_position.z() = 0.0;

    // Velocity (Vehicle coordinates, convert from km/h to m/s) at CG position
    Eigen::Vector3d ang_vel;
    ang_vel.x() = interface::DEG2RAD * msg.noisy_angularVelocity.roll;
    ang_vel.y() = interface::DEG2RAD * msg.noisy_angularVelocity.pitch;
    ang_vel.z() = interface::DEG2RAD * msg.noisy_angularVelocity.yaw;

    // Rotate velocity vector obserbed in vehicle coordinates to CG position
    Eigen::Vector3d rotated_vel = ang_vel.cross(vehicle_to_cg_position);

    noisy_state.vx = msg.noisy_velocity.x * interface::KPH2MPS - rotated_vel.x();  // km/h to m/s
    noisy_state.vy = msg.noisy_velocity.y * interface::KPH2MPS - rotated_vel.y();
    noisy_state.vz = msg.noisy_velocity.z * interface::KPH2MPS - rotated_vel.z();

    // Acceleration (Vehicle coordinates)
    static Eigen::Vector3d noisy_state_prev_ang_vel = Eigen::Vector3d::Zero();
    static double noisy_state_prev_time = 0.0;

    double curr_time = ros_bridge::GetTimeStamp(msg.header.stamp);
    double dt = curr_time - noisy_state_prev_time;

    Eigen::Vector3d ang_acc = Eigen::Vector3d::Zero();
    if(dt > 0.0) {
        ang_acc = (ang_vel - noisy_state_prev_ang_vel) / dt;
    }
    noisy_state_prev_ang_vel = ang_vel;
    noisy_state_prev_time = curr_time;

    // Rotate acceleration vector obserbed in vehicle coordinates to CG position
    Eigen::Vector3d rotated_acc = ang_vel.cross(rotated_vel) + ang_acc.cross(vehicle_to_cg_position);

    // Acceleration (Vehicle coordinates)
    noisy_state.ax = msg.noisy_acceleration.x - rotated_acc.x();
    noisy_state.ay = msg.noisy_acceleration.y - rotated_acc.y();
    noisy_state.az = msg.noisy_acceleration.z - rotated_acc.z();

    noisy_state.roll  = msg.noisy_orientation.roll * interface::DEG2RAD;
    noisy_state.pitch = msg.noisy_orientation.pitch * interface::DEG2RAD;
    noisy_state.yaw   = msg.noisy_orientation.yaw * interface::DEG2RAD;

    noisy_state.roll_vel  = msg.noisy_angularVelocity.roll;
    noisy_state.pitch_vel = msg.noisy_angularVelocity.pitch;
    noisy_state.yaw_vel   = msg.noisy_angularVelocity.yaw;

    // VehicleCAN
    noisy_state.vehicle_can.header = ros_bridge::GetHeader(msg.header);
    noisy_state.vehicle_can.header.frame_id = "can";

    double steering_angle_deg = msg.steer * 40.0;
    noisy_state.vehicle_can.steering_wheel_angle = steering_angle_deg * interface::DEG2RAD * interface::STEERING_RATIO;
    noisy_state.vehicle_can.steering_tire_angle  = steering_angle_deg * interface::DEG2RAD;
    noisy_state.vehicle_can.steering_state       = interface::MdpsState::MDPS_ACTIVATE;

    noisy_state.vehicle_can.accel_position       = msg.accel * 100.0;
    noisy_state.vehicle_can.brake_position       = msg.brake * 100.0;
    noisy_state.vehicle_can.gear_select          = interface::Gear::GEAR_D;

    noisy_state.vehicle_can.operation_mode       = interface::OperationMode::AUTONOMOUS;
    noisy_state.vehicle_can.lateral_ad_mode      = interface::AutonomousMode::RUN;
    noisy_state.vehicle_can.longitudinal_ad_mode = interface::AutonomousMode::RUN;

    return noisy_state;
}

int main(int argc, char** argv) {
    std::string node_name = "vehicle_driver";
    ros::init(argc, argv, node_name);
    ros::NodeHandle nh;

    double period;
    if (!nh.getParam("task_period/period_vehicle_driver", period)){
        period = 1.0;
    }
    
    VehicleDriver main_task(node_name, period);
    main_task.Exec(2);

    return 0;
}
