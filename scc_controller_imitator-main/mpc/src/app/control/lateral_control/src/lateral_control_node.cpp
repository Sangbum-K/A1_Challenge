/**
 * @file        lateral_control_node.cpp
 * @brief       lateral_control_node
 * 
 * @authors     Junhee Lee (998jun@gmail.com)      
 * 
 * @date        2023-02-02 Yuseung Na - Created. (ACELAB's Lateral MPC using NLOPT)
 *              2023-05-17 Junhee Lee - Add du kinematic MPC using HPIPM.
 *              2023-07-11 Junhee Lee - Add actuator aware kinematic MPC using HPIPM
 *              2023-07-20 Junhee Lee - Add time-delay considered MPC using HPIPM.
 *              2023-07-23 Junhee Lee - Add error dynamic MPC using HPIPM
 *              2023-07-30 Junhee Lee - Clean up lateral controller!
 *              2023-09-03 Junhee Lee - Undergradient + Yawrate Compensator
 *              2024-06-25 Junhee Lee - Remove Yawrate Compensator
 *              2024-10-25 Junhee Lee - Debugging topics --> Control info msg
 *              2024-10-28 Junhee Lee - Update acados controllers!
 *              2024-10-29 Junhee Lee - Unwarp reference trajectory
 *              2024-12-10 Seounghoon Park - add function of only using morai state topic
 */

#include "lateral_control_node.hpp"

LateralControlNode::LateralControlNode(std::string task_node, double period)
    : TaskManager(task_node, period) {
    // Initialize
    ROS_WARN_STREAM("[" << task_node << "] Initialize node (Period: " << 1/period << " Hz)");

    // Node init
    ros::NodeHandle nh;

    // Ini init
    const char* pwd_env = getenv("PWD");
    if (pwd_env == nullptr) {
        ROS_FATAL("[LateralControlNode] PWD environment variable is not set. Cannot load config.");
        ros::shutdown();
        return;
    }
    std::string dir(pwd_env);
    std::string ini_dir("/config/control.ini");
    util_ini_parser_.Init((dir + ini_dir).c_str());

    // Parameter init
    ProcessINI();

    if (!nh.getParam("/lateral_control/use_morai_ego", use_morai_ego_)) {
        use_morai_ego_ = false;
    }
    if (!nh.getParam("/lateral_control/use_gt_state", use_gt_state_)) {
        use_gt_state_ = false;
    }

    // Subscriber init
    if (use_morai_ego_ == false) {
        if (use_gt_state_ == true) {
            s_vehicle_state_ = nh.subscribe(
                "app/loc/gt_vehicle_state", 10, &LateralControlNode::CallbackVehicleState, this);
        } else {
            // Default vehicle state topic
        s_vehicle_state_ = nh.subscribe(
            "app/loc/vehicle_state", 10, &LateralControlNode::CallbackVehicleState, this); 
        }    

    } else if (use_morai_ego_ == true) {
        s_vehicle_state_ = nh.subscribe(
            "/Ego_topic", 10, &LateralControlNode::CallbackMoraiState, this);
    }
    
    s_trajectory_ = nh.subscribe(
        "app/pla/trajectory", 10, &LateralControlNode::CallbackTrajectory, this);

    // Publisher init
    p_command_steer_    = nh.advertise<std_msgs::Float32>("app/con/command_steer", 10);
    p_control_io_       = nh.advertise<autohyu_msgs::ControlInputOutput>("hmi/con/lateral_control_io", 10);
    
    p_control_info_     = nh.advertise<autohyu_msgs::ControlInfo>("hmi/con/lat/control_info", 10);
    p_control_trajectory_     = nh.advertise<visualization_msgs::MarkerArray>("hmi/con/lat/prediction", 10);
    p_reference_path_   = nh.advertise<visualization_msgs::MarkerArray>("hmi/con/lat/reference", 10);
    p_current_understeer_gradient_ = nh.advertise<std_msgs::Float32>("hmi/con/lat/current_understeer_gradient", 10);
    p_cross_track_error_           = nh.advertise<std_msgs::Float32>("hmi/con/lat/CTE", 10);
    // Algorithm init
    ptr_lateral_control_ = std::make_unique<LateralControl>();
    ptr_lateral_control_->Init();
}

LateralControlNode::~LateralControlNode() {
    // Terminate
}

void LateralControlNode::Run() {
    ProcessINI();

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get subscribe variables
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    if (!b_is_vehicle_state_) {
        ROS_ERROR("[Lateral Control] Wait for Vehicle State...");
        return;
    }
    if (!b_is_trajectory_) {
        ROS_ERROR("[Lateral Control] Wait for Trajectory...");
        return;
    }

    interface::VehicleState vehicle_state; {
        std::lock_guard<std::mutex> lock(mutex_vehicle_state_);
        vehicle_state = i_vehicle_state_;
    }
    interface::Trajectory trajectory; {
        std::lock_guard<std::mutex> lock(mutex_trajectory_);
        trajectory = i_trajectory_;
    }    

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Algorithm
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    const auto [control_command, control_reference, path_tracking_infos] = ptr_lateral_control_->Run(
            vehicle_state, trajectory);
    
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update output
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    o_command_steer_               = ros_bridge::UpdateFloat32(control_command.steering_tire_angle); //Deg
    o_control_trajectory_          = ros_bridge::UpdateRvizPredictedPath(control_command.control_trajectory);
    o_reference_path_              = ros_bridge::UpdateRvizControlReference(control_reference);
    o_control_info_                = ros_bridge::UpdateControlInfos(path_tracking_infos);
    o_current_understeer_gradient_ = ros_bridge::UpdateFloat32(path_tracking_infos.compensation_lateral);
    o_cross_track_error_           = ros_bridge::UpdateFloat32(ComputeRacelineCTE(vehicle_state, trajectory));

    std::vector<double> command_steer;
    for(int i = 0; i < control_command.control_trajectory.control_point.size(); i++){
        command_steer.push_back(control_command.control_trajectory.control_point[i].steer);
    }
    o_control_io_ = ros_bridge::UpdateControlInputOutput(vehicle_state, trajectory, control_command.control_trajectory, command_steer, std::vector<double>(), std::vector<double>(), std::vector<double>(), std::vector<double>());
}

void LateralControlNode::Publish() {
    if (!b_is_vehicle_state_ || !b_is_trajectory_) {
        return;
    }
    
    p_control_info_.publish(o_control_info_);
    p_control_trajectory_.publish(o_control_trajectory_);
    p_reference_path_.publish(o_reference_path_);
    p_current_understeer_gradient_.publish(o_current_understeer_gradient_);
    p_cross_track_error_.publish(o_cross_track_error_);
    p_control_io_.publish(o_control_io_);
    if (ml_controller_enable_ == false) {
        p_command_steer_.publish(o_command_steer_);
    }
}


// Signed lateral deviation from the closest raceline point (world frame).
// Formula mirrors control_error_calculation_by_lap.py::_closest_on_raceline():
//   dx = ref_x - ego_x,  dy = ref_y - ego_y
//   cte = dy*cos(-ref_yaw) - dx*sin(-ref_yaw)
// Positive: ego is to the right of the reference heading (left turn overdue).
double LateralControlNode::ComputeRacelineCTE(const interface::VehicleState& vs,
                                               const interface::Trajectory&   traj) {
    if (traj.point.empty()) return 0.0;

    double min_dist_sq = std::numeric_limits<double>::max();
    int    closest_idx = 0;
    for (int i = 0; i < static_cast<int>(traj.point.size()); ++i) {
        double dx = traj.point[i].x - vs.x;
        double dy = traj.point[i].y - vs.y;
        double d  = dx*dx + dy*dy;
        if (d < min_dist_sq) { min_dist_sq = d; closest_idx = i; }
    }

    double dx      = traj.point[closest_idx].x - vs.x;
    double dy      = traj.point[closest_idx].y - vs.y;
    double ref_yaw = traj.point[closest_idx].yaw;
    return dy * std::cos(-ref_yaw) - dx * std::sin(-ref_yaw);
}

void LateralControlNode::ProcessINI() {
    if ( util_ini_parser_.IsFileUpdated() ) {
        ROS_WARN("[Lateral control] Ini file is updated!");
        // TODO(Phase 2): re-parse control parameters here for hot-reload support
        //   e.g. lateral_control_method, Pure Pursuit gains, MPC weights
        util_ini_parser_.ParseConfig("ML-LatLonControlML", "ml_controller_enable", ml_controller_enable_);
    }
}

int main(int argc, char** argv) {
    std::string node_name = "lateral_control";
    ros::init(argc, argv, node_name);
    ros::NodeHandle nh;

    double period;
    if (!nh.getParam("task_period/period_lateral_control", period)) {
        period = 0.02;
    }

    LateralControlNode main_task(node_name, period);
    main_task.Exec();

    return 0;
}