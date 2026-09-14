#include <lateral_control/lateral_control.hpp>

using namespace std;
using namespace tk;

LateralControl::LateralControl(){
    // Make unique pointer
}

LateralControl::~LateralControl(){
}

bool LateralControl::Init(){
    // Ini initialization
    const char* pwd_env = getenv("PWD");
    if (pwd_env == nullptr) {
        std::cerr << "[LateralControl] PWD environment variable is not set. Cannot load config." << std::endl;
        return false;
    }
    std::string dir(pwd_env);
    std::string ini_dir("/config/control.ini");
    util_ini_parser_.Init((dir+ini_dir).c_str());

    // Parameters initialization
    ProcessINI();
    
    i_vehicle_state_ = std::make_shared<const VehicleState>();
    i_reference_trajectory_ = std::make_shared<const Trajectory>();

    CreateLateralController();
    ptr_lateral_fallback_ = make_unique<LateralControlFallback>();
    ptr_lateral_fallback_->Init();

    return true;
}

std::tuple<ControlCommand, ControlTrajectory, PathTrackingInfos> LateralControl::Run(
                            const VehicleState& vehicle_state, const Trajectory& trajectory)
{
    i_vehicle_state_ = std::make_shared<const VehicleState>(vehicle_state);
    i_reference_trajectory_ = std::make_shared<const Trajectory>(trajectory);
    ProcessINI();
    CreateLateralController();

    tk::Map road_map_dt;
    if (!util_function::GenerateRoadMap(*i_reference_trajectory_, road_map_dt)) {
        util_function::DebugPrintWarn("[LateralControl] Trajectory too short for spline — skip");
        return {ControlCommand{}, ControlTrajectory{}, path_tracking_info_};
    }
    Trajectory dt_trajectory = GenerateDTTargetTrajectory(*i_vehicle_state_, *i_reference_trajectory_, road_map_dt);

    // TF reference trajectory to controller's target frame
    Trajectory ego_trajectory = util_function::TransformTrajectory(dt_trajectory, *i_vehicle_state_);
    // Generate frenet frame based on transformed trajectory
    tk::Map road_map;
    if (!util_function::GenerateRoadMap(ego_trajectory, road_map)) {
        util_function::DebugPrintWarn("[LateralControl] Ego-frame trajectory too short for spline — skip");
        return {ControlCommand{}, ControlTrajectory{}, path_tracking_info_};
    }
    
    // Resample and calculate reference attributes
    ControlTrajectory controllable_trajectory = GenerateTargetTrajectory(ego_trajectory, road_map, ptr_lateral_controller_->GetTargetFrame());
    
    // Assign current vehicle state to controller
    ptr_lateral_controller_->BuildControllerVehicleState(*i_vehicle_state_,road_map);
    
    auto start = chrono::high_resolution_clock::now();
    boost::optional<ControlCommand> control_command = ptr_lateral_controller_->CalculateOptimalTireSteering(controllable_trajectory);
    auto end = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    path_tracking_info_.total_time = duration.count()*0.001;

    path_tracking_info_.solve_time = ptr_lateral_controller_->GetLastSolveTimeMs();
    
    if(control_command){
        if(params_.use_understeer_gradient){
            // Compensate lateral slip for kinematic model based controllers
            control_command->steering_tire_angle += CompensateUndersteerGradient(controllable_trajectory);
        }
        else{
            path_tracking_info_.compensation_lateral = 0.0;
        }
        return {*control_command, controllable_trajectory, path_tracking_info_};
    }
    else{
        // Fallback controller activate when control_command is boost::none
        std::cerr << "[LateralControl] FallBack Controller Activated!" << std::endl;
        ControlCommand backup_cmd = ptr_lateral_fallback_->PurePursuitSteeringAngle( *i_vehicle_state_,*i_reference_trajectory_);
        return {backup_cmd, controllable_trajectory, path_tracking_info_};
    }
}

ControlTrajectory LateralControl::GenerateTargetTrajectory(const Trajectory& trajectory, tk::Map road_map, const std::string& frame_out){
    // We only use x, y and (speed) from reference trajectory for controller independence 
    
    // Convert ego cartesian to frenet coordinate
    std::vector<double> ego_sn = road_map.ToFrenet(0.0, 0.0);
    path_tracking_info_.cross_track_error = ego_sn[1];
    path_tracking_info_.yaw_error = -road_map.GetSlope(ego_sn[0])*RAD2DEG;

    ControlTrajectory interpolated_trajectory;
    interpolated_trajectory.header = trajectory.header;
    interpolated_trajectory.header.frame_id = frame_out;
    vector<double> time, speed, distance;
    double sum_s = 0.;
    double sum_time = 0.;
    for (uint16_t i = 0; i < (uint16_t)trajectory.point.size(); i++) {

        double dx = i == 0 ? 0.0 : trajectory.point[i].x - trajectory.point[i - 1].x;
        double dy = i == 0 ? 0.0 : trajectory.point[i].y - trajectory.point[i - 1].y;
        double ds = max(sqrt(dx * dx + dy * dy), 0.001);
        double dt;

        if(params_.use_reference_speed == true){
            // reference speed based on reference trajectory speed
            // if you need to use longitudinal control please remove the max condition
            dt = i == 0 ? 0.0 : 2*ds/max(trajectory.point[i].speed + trajectory.point[i-1].speed , (float)0.1);
            speed.push_back(trajectory.point[i].speed);

        }
        else{
            // reference speed based on vehicle current speed
            dt = ds/max(i_vehicle_state_->vx,0.1);
            speed.push_back(i_vehicle_state_->vx);
        }

        sum_s += ds;
        sum_time += dt;

        distance.push_back(sum_s);
        time.push_back(sum_time);
    }

    if ((uint16_t)time.size() < 2 || (uint16_t)speed.size() < 2 || (uint16_t)distance.size() < 2) {
        std::cerr << "[LateralControl] Cannot generate target interpolated_trajectory! "
                  << "time=" << time.size() << " speed=" << speed.size() << " dist=" << distance.size() << std::endl;
        return interpolated_trajectory;
    }

    tk::spline st, ts, tv;
    if(i_vehicle_state_->vx < 0.1){
        // linear interpolation at low speed
        st.set_points(distance, time, false);
        ts.set_points(time, distance, false);
        tv.set_points(time, speed, false);
    }
    else{
        st.set_points(distance, time, true);
        ts.set_points(time, distance, true);
        tv.set_points(time, speed, true);
    }
    double time_offset = st(ego_sn[0]) + params_.reference_time_offset;

    int step_num     = ptr_lateral_controller_->GetControlTrajectoryStepNum();
    double step_time = ptr_lateral_controller_->GetControlTrajectoryStepTime();

    // yaw unwrapping (important for initial guess and cost function!)
    double previous_yaw = 0.0; // Initialize with the first raw yaw value
    double offset = 0.0;  // Angle adjustment value

    for ( uint16_t i = 0; i < step_num + 2; i++) {
        ControlPoint dp;
        double target_distance = ts(step_time * i + time_offset);
        std::vector<double> xy = road_map.ToCartesian(target_distance, 0.0);
        dp.x = xy[0];
        dp.y = xy[1];

        // Calculate current yaw value (within -pi to pi range)
        double current_yaw = road_map.GetSlope(target_distance);

        // Compute the difference between current yaw and previous yaw
        double delta_yaw = current_yaw - previous_yaw;

        // Adjust the offset based on the delta to unwrap the yaw
        if (delta_yaw > M_PI) {
            offset -= 2 * M_PI;
        } else if (delta_yaw < -M_PI) {
            offset += 2 * M_PI;
        }

        // Apply the offset to get the unwrapped yaw
        double unwrapped_yaw = current_yaw + offset;
        dp.yaw = unwrapped_yaw;

        // Update previous_yaw with the current raw yaw (before unwrapping)
        previous_yaw = current_yaw;

        dp.curvature = road_map.GetCurvature(target_distance);
        dp.vx = tv(step_time * i + time_offset);
        dp.time = step_time * i;
        dp.distance = target_distance - ts(time_offset);
        interpolated_trajectory.control_point.push_back(dp);
    }
    return interpolated_trajectory;
}

double LateralControl::CompensateUndersteerGradient(const ControlTrajectory& ref){
    static double understeer_gradient = 0.0; // "static local variable"
    static double prev_understeer_gradient = 0.0;
    static auto prev_command_time = std::chrono::system_clock::now();

    // yawrate and velocities are measured from rear axle (ego frame)
    double beta = atan2(i_vehicle_state_->vy,i_vehicle_state_->vx);
    double alpha_f = -beta+i_vehicle_state_->vehicle_can.steering_tire_angle
                     -params_.vehicle_wheel_base*i_vehicle_state_->yaw_vel/i_vehicle_state_->vx;
    double alpha_r = -beta+0.0*i_vehicle_state_->yaw_vel/i_vehicle_state_->vx;
    
    // double Kv = VEHICLE_REAR_AXLE_TO_CG*VEHICLE_MASS/(FRONT_CORNERING_STIFFNESS*VEHICLE_WHEEL_BASE)
    //             -VEHICLE_FRONT_AXLE_TO_CG*VEHICLE_MASS/(REAR_CORNERING_STIFFNESS*VEHICLE_WHEEL_BASE);
    double Kv = params_.understeer_gradient_kv;

    int delay_sample = max(1, (int)(params_.time_delay_for_compensators/ptr_lateral_controller_->GetControlTrajectoryStepTime()));

    double target_curvature = ref.control_point[delay_sample].curvature;
    if(isnan(target_curvature)){
        target_curvature = 0.0;
    }
    double calculated_understeer_gradient = 0.0;
    if(fabs(i_vehicle_state_->vx) > 1.0){
        if(fabs(alpha_f) - fabs(alpha_r) > -1.0){
            // calculated_understeer_gradient = (alpha_f - alpha_r)*RAD2DEG;
            if (fabs(target_curvature) > 0.001){
                double ay = target_curvature*i_vehicle_state_->vx*i_vehicle_state_->vx;
                calculated_understeer_gradient = Kv*ay*RAD2DEG;
            }
        }
    }
    double target_yawrate = i_vehicle_state_->vx*target_curvature;      
    double yaw_rate_error =  target_yawrate - i_vehicle_state_->yaw_vel;

    // lpf understeer_gradient
    understeer_gradient = params_.understeer_gradient_lpf*calculated_understeer_gradient + (1.0-params_.understeer_gradient_lpf)*understeer_gradient;
    // std::cout << "UndersteerGradient "<<  understeer_gradient << "deg"<< std::endl;
    
    auto current_command_time = std::chrono::system_clock::now();
    double dt = std::chrono::duration<double>(current_command_time-prev_command_time).count();
    double understeer_gradient_rate = (understeer_gradient-prev_understeer_gradient)/dt;

    if(understeer_gradient_rate > params_.understeer_gradient_rate_limit){
        understeer_gradient = prev_understeer_gradient + params_.understeer_gradient_rate_limit*dt;
    }
    else if(understeer_gradient_rate < -params_.understeer_gradient_rate_limit){
        understeer_gradient = prev_understeer_gradient - params_.understeer_gradient_rate_limit*dt;
    }
    prev_understeer_gradient = understeer_gradient; 
    prev_command_time = current_command_time;

    path_tracking_info_.compensation_lateral = understeer_gradient;
    return understeer_gradient;
}

void LateralControl::CreateLateralController(){
    if(prev_control_method_ == params_.lateral_control_method){
        return;
    }   
    ptr_lateral_controller_.reset();
    switch (params_.lateral_control_method) {
        case LateralControlMethod::DELAYED_KINEMATIC_BICYCLE_LMPC:
            ptr_lateral_controller_ = std::make_shared<LateralDelayedKinematicBicycleLMPC>();
            break;
        case LateralControlMethod::KINEMATIC_BICYCLE_LMPC:
            ptr_lateral_controller_ = std::make_shared<LateralKinematicBicycleLMPC>();
            break;
        case LateralControlMethod::PURE_PURSUIT:
            ptr_lateral_controller_ = std::make_shared<LateralPurePursuit>();
            break;
        default:
            std::cerr << "[LateralControl] lateral_control_method=" << params_.lateral_control_method
                      << " is not implemented. Falling back to PURE_PURSUIT." << std::endl;
            params_.lateral_control_method = LateralControlMethod::PURE_PURSUIT;
            ptr_lateral_controller_ = std::make_shared<LateralPurePursuit>();
            break;
    }

    prev_control_method_ = params_.lateral_control_method;

}

Trajectory LateralControl::GenerateDTTargetTrajectory(const VehicleState& vehicle_state,
                                                            const Trajectory& trajectory,
                                                            tk::Map road_map){
    return util_function::GenerateDTTargetTrajectory(
        vehicle_state,
        trajectory,
        road_map,
        params_.reference_time_offset,
        ptr_lateral_controller_->GetControlTrajectoryStepNum(),
        ptr_lateral_controller_->GetControlTrajectoryStepTime()
    );
}

void LateralControl::ProcessINI(){
	if (util_ini_parser_.IsFileUpdated()){
		util_ini_parser_.ParseConfig("Lateral Control", "lateral_control_method", params_.lateral_control_method);
		util_ini_parser_.ParseConfig("Lateral Control", "reference_time_offset" , params_.reference_time_offset);
		util_ini_parser_.ParseConfig("Lateral Control", "use_reference_speed"   , params_.use_reference_speed);
		util_ini_parser_.ParseConfig("Lateral Control", "time_delay_for_compensators"   , params_.time_delay_for_compensators);

		util_ini_parser_.ParseConfig("Lateral Control", "use_understeer_gradient"   , params_.use_understeer_gradient);
        util_ini_parser_.ParseConfig("Lateral Control", "understeer_gradient_lpf"   , params_.understeer_gradient_lpf);
        util_ini_parser_.ParseConfig("Lateral Control", "understeer_gradient_rate_limit" , params_.understeer_gradient_rate_limit);
        util_ini_parser_.ParseConfig("Lateral Control", "understeer_gradient_kv"   , params_.understeer_gradient_kv);

        util_ini_parser_.ParseConfig("Lateral Control", "use_vehicle_twisting_compensator"   , params_.use_vehicle_twisting_compensator);
        util_ini_parser_.ParseConfig("Lateral Control", "vehicle_twisting_constant"   , params_.vehicle_twisting_constant);
        util_ini_parser_.ParseConfig("Lateral Control", "max_vehicle_twisting_compensation"   , params_.max_vehicle_twisting_compensation);
        util_ini_parser_.ParseConfig("Vehicle", "wheel_base", params_.vehicle_wheel_base);

		std::cout<<"[Lateral Control] Ini file is updated!"<<std::endl;
    }
}
