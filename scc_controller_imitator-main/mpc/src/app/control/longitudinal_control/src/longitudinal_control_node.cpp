/**
 * @file        longitudinal_control.cpp
 * @brief       longitudinal control node
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)    
 *              Junhee Lee (998jun@gmail.com)      
 * 
 * @date        2023-02-02 Yuseung Na - Created. (Longitudinal MPC using CVXGEN)
 *              2023-03-10 Junhee Lee - Add dynamic MPC model
 *              2023-04-12 Junhee Lee - Add dynamic MPC du model (HPIPM)
 *              2023-08-06 Junhee Lee - Code Cleanup
 */

#include "longitudinal_control_node.hpp"

LongitudinalControlNode::LongitudinalControlNode(std::string task_node, double period)
    : TaskManager(task_node, period) {
    // Initialize
    ROS_WARN_STREAM("[" << task_node << "] Initialize node (Period: " << 1/period << " Hz)");

    // Node init
    ros::NodeHandle nh;

    // Ini init
    const char* pwd_env = getenv("PWD");
    if (pwd_env == nullptr) {
        ROS_FATAL("[LongitudinalControlNode] PWD environment variable is not set. Cannot load config.");
        ros::shutdown();
        return;
    }
    std::string dir(pwd_env);
    std::string ini_dir("/config/control.ini");
    util_ini_parser_.Init((dir + ini_dir).c_str());

    // Parameter init
    ProcessINI();
    ProcessRosparam(nh);

    // Subscriber init
    if (use_morai_ego_ == false) {
        if (use_gt_state_ == true) {
            s_vehicle_state_ = nh.subscribe(
            "app/loc/gt_vehicle_state", 10, &LongitudinalControlNode::CallbackVehicleState, this);
        } else {
            s_vehicle_state_ = nh.subscribe(
            "app/loc/vehicle_state", 10, &LongitudinalControlNode::CallbackVehicleState, this);
        }
        
    } else if (use_morai_ego_ == true) {
        s_vehicle_state_ = nh.subscribe(
            "/Ego_topic", 10, &LongitudinalControlNode::CallbackMoraiState, this);
    }
    
    s_predict_objects_ = nh.subscribe(
        "app/perc/predict_objects", 10, &LongitudinalControlNode::CallbackPredictObjects, this);        
    s_trajectory_ = nh.subscribe(
        "app/pla/trajectory", 10, &LongitudinalControlNode::CallbackTrajectory, this);

    // Publisher init
    p_command_torque_           = nh.advertise<std_msgs::Float32>("app/con/command_torque", 10);
    p_command_acceleration_     = nh.advertise<std_msgs::Float32>("app/con/command_accel", 10);  
    p_command_accel_pedal_      = nh.advertise<std_msgs::Float32>("app/con/command_pedal_accel", 10);
    p_command_brake_pedal_      = nh.advertise<std_msgs::Float32>("app/con/command_pedal_brake", 10);
    p_control_io_               = nh.advertise<autohyu_msgs::ControlInputOutput>("hmi/con/longitudinal_control_io", 10);
    
    p_target_speed_             = nh.advertise<std_msgs::Float32>("hmi/con/target_speed", 10);
    p_target_ax_                = nh.advertise<std_msgs::Float32>("hmi/con/target_ax", 10);
    p_current_speed_            = nh.advertise<std_msgs::Float32>("hmi/con/current_speed", 10);
    p_current_speederror_       = nh.advertise<std_msgs::Float32>("hmi/con/current_speederror", 10);
    p_current_axerror_          = nh.advertise<std_msgs::Float32>("hmi/con/current_axerror", 10);
    p_compensated_torque_       = nh.advertise<std_msgs::Float32>("hmi/con/compensated_torque", 10);
    p_speed_compensation_error_ = nh.advertise<std_msgs::Float32>("hmi/con/speed_comp_error", 10);

    p_predicted_speed_profile_  = nh.advertise<visualization_msgs::MarkerArray>("hmi/con/control_speed_profile", 10);

    // Algorithm init
    ptr_longitudinal_control_ = std::make_unique<LongitudinalControl>();
    if (!ptr_longitudinal_control_->Init()) {
        ROS_FATAL("[LongitudinalControlNode] Init() failed (pedal map load error). Shutting down.");
        ros::shutdown();
    }
}

LongitudinalControlNode::~LongitudinalControlNode() {
    // Terminate
}

void LongitudinalControlNode::Run() {
    ProcessINI();

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get subscribe variables
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    if (!b_is_vehicle_state_) {
        ROS_ERROR("Wait for Vehicle State...");
        return;
    }
    if (!b_is_trajectory_) {
        ROS_ERROR("Wait for Trajectory...");
        return;
    }

    interface::VehicleState vehicle_state; {
        std::lock_guard<std::mutex> lock(mutex_vehicle_state_);
        vehicle_state = i_vehicle_state_;
    }
    interface::PredictObjects predict_objects; {
        std::lock_guard<std::mutex> lock(mutex_predict_objects_);
        predict_objects = i_predict_objects_;
    }
    interface::Trajectory trajectory; {
        std::lock_guard<std::mutex> lock(mutex_trajectory_);
        trajectory = i_trajectory_;
    }    

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Algorithm
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    // Algorithm 1
    const auto [control_command, control_reference, speed_tracking_infos] = ptr_longitudinal_control_->Run (
        vehicle_state, predict_objects, trajectory);

    // Algorithm 2
    // ...
    
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update output
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    UpdateControlCommand(control_command);
    o_predicted_speed_profile_ = ros_bridge::UpdateRvizPredictedSpeedProfile(control_command.control_trajectory);
    UpdateSpeedTrackingInfos(speed_tracking_infos,vehicle_state,control_reference);
    std::vector<double> command_speed, command_acceleration, command_accel_pedal, command_brake_pedal;
    for (int i = 0; i < control_command.control_trajectory.control_point.size(); i++) {
        command_speed.push_back(control_command.control_trajectory.control_point[i].vx);
        command_acceleration.push_back(control_command.control_trajectory.control_point[i].ax);
        command_accel_pedal.push_back(control_command.control_trajectory.control_point[i].accel_pedal);
        command_brake_pedal.push_back(control_command.control_trajectory.control_point[i].brake_pedal);
    }
    o_control_io_ = ros_bridge::UpdateControlInputOutput(vehicle_state, trajectory, control_command.control_trajectory, std::vector<double>(), command_accel_pedal, command_brake_pedal, command_speed, command_acceleration);
}

void LongitudinalControlNode::Publish() {
    if (!b_is_vehicle_state_ || !b_is_trajectory_) {
        return;
    }
    if (ml_controller_enable_ == false) {
        p_command_torque_.publish(o_command_torque_);
        p_command_acceleration_.publish(o_command_acceleration_);
        p_command_accel_pedal_.publish(o_accel_pedal_);
        p_command_brake_pedal_.publish(o_brake_pedal_);
    }

    p_predicted_speed_profile_.publish(o_predicted_speed_profile_);

    p_target_speed_.publish(o_target_speed_);
    p_target_ax_.publish(o_target_ax_);
    p_current_speed_.publish(o_current_speed_);
    p_current_axerror_.publish(o_current_axerror_);
    p_current_speederror_.publish(o_current_speederror_);
    p_compensated_torque_.publish(o_compensated_torque_);
    p_speed_compensation_error_.publish(o_speed_compensation_error_);
    p_control_io_.publish(o_control_io_);
}

void LongitudinalControlNode::ProcessRosparam(const ros::NodeHandle& nh) {
    if (!nh.getParam("/longitudinal_control/use_morai_ego", use_morai_ego_)) {
        use_morai_ego_ = false;
    }
    if (!nh.getParam("/longitudinal_control/use_gt_state", use_gt_state_)) {
        use_gt_state_ = false;
    }
}

void LongitudinalControlNode::ProcessINI() {
    if ( util_ini_parser_.IsFileUpdated() ) {
        ROS_WARN("[Longitudinal control] Ini file is updated!");
        // TODO(Phase 2): re-parse control parameters here for hot-reload support
        //   e.g. longitudinal_control_method, PID gains, speed compensator params
        util_ini_parser_.ParseConfig("ML-LatLonControlML", "ml_controller_enable", ml_controller_enable_);
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Update functions for publish variables
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
void LongitudinalControlNode::UpdateControlCommand(const ControlCommand& cmd){
    o_command_torque_.data       = cmd.trq;    
    o_command_acceleration_.data = cmd.ax;
    o_accel_pedal_.data = cmd.accel_pedal;
    o_brake_pedal_.data = cmd.brake_pedal;
    return;
}

void LongitudinalControlNode::UpdateSpeedTrackingInfos(const PathTrackingInfos& tracking_errors,
                                            const VehicleState& vehicle_state,
                                            const ControlTrajectory& ref) {
    o_current_axerror_.data = tracking_errors.accel_error;
    o_current_speederror_.data = tracking_errors.speed_error;
    o_compensated_torque_.data = tracking_errors.compensation_longitudinal;
    o_speed_compensation_error_.data = tracking_errors.speed_compensation_error;
    o_target_speed_.data       = ref.control_point[0].vx*MPS2KPH;
    o_target_ax_.data       = ref.control_point[0].ax;
    o_current_speed_.data      = vehicle_state.vx * MPS2KPH;
}

int main(int argc, char** argv) {
    std::string node_name = "longitudinal_control";
    ros::init(argc, argv, node_name);
    ros::NodeHandle nh;
    

    double period;
    if ( !nh.getParam("task_period/period_longitudinal_control", period) ) {
        period = 0.01;
    }

    LongitudinalControlNode main_task(node_name, period);
    main_task.Exec();

    return 0;
}