/**
 * @file        template_algorithm_algorithm.cpp
 * @brief       template algorithm cpp file for template node algorithm
 * 
 * @authors     Yuseung Na (yuseungna@hanyang.ac.kr)          
 * 
 * @date        2024-04-01 created by Yuseung Na
 * 
 */

#include <longitudinal_control/longitudinal_control.hpp>

LongitudinalControl::LongitudinalControl(){
}

LongitudinalControl::~LongitudinalControl(){
    // terminate
}

bool LongitudinalControl::Init(){
    // Ini initialization
    const char* pwd_env = getenv("PWD");
    if (pwd_env == nullptr) {
        std::cerr << "[LongitudinalControl] PWD environment variable is not set. Cannot load config." << std::endl;
        return false;
    }
    std::string dir(pwd_env);
    std::string ini_dir("/config/control.ini");
    util_ini_parser_.Init((dir+ini_dir).c_str());

    // Parameters initialization
    ProcessINI();
    
    i_vehicle_state_ = std::make_shared<const VehicleState>();
    i_reference_trajectory_ = std::make_shared<const Trajectory>();
    
    CreateLongitudinalController();

    // Pedal Map
    std::string pedal_map_dir = dir + "/resources/pedal_map/" + params_.pedal_map_file_name;
    if (!pedal_map_.loadFromFile(pedal_map_dir)) {
        std::cerr << "[LongitudinalControl] Failed to load pedal map: " << pedal_map_dir << std::endl;
        return false;
    }


    return true;
}


std::tuple<ControlCommand, ControlTrajectory, PathTrackingInfos> LongitudinalControl::Run(
        const interface::VehicleState& vehicle_state,
        const interface::PredictObjects& predict_objects,
        const interface::Trajectory& trajectory)
{
    i_vehicle_state_ = std::make_shared<const VehicleState>(vehicle_state);
    i_reference_trajectory_ = std::make_shared<const Trajectory>(trajectory);

    ProcessINI();
    CreateLongitudinalController();

    // Compensate redundant points
    interface::Trajectory compensated_trajectory = trajectory;
    double sum_s = 0.0;
    for (uint16_t i = 1; i < (uint16_t)compensated_trajectory.point.size(); i++) {
        double x = compensated_trajectory.point[i].x;
        double y = compensated_trajectory.point[i].y;
        double dx = i == 0 ? 0.0 : x - compensated_trajectory.point[i - 1].x;
        double dy = i == 0 ? 0.0 : y - compensated_trajectory.point[i - 1].y;
        double ds = sqrt(dx * dx + dy * dy);
        sum_s += ds;
        if (ds < 0.00001) {
            for (uint16_t j = i; j < (uint16_t)compensated_trajectory.point.size(); j++) {
                // Add 0.0001m to path with yaw
                double yaw = compensated_trajectory.point[j-1].yaw;
                double dx = 0.0001 * cos(yaw);
                double dy = 0.0001 * sin(yaw);
                compensated_trajectory.point[j].x += dx;
                compensated_trajectory.point[j].y += dy;
            }
        }
    }
    bool is_stop = sum_s < 0.2;

    tk::Map road_map_dt = util_function::GenerateRoadMap(*i_reference_trajectory_);
    Trajectory dt_trajectory = GenerateDTTargetTrajectory(*i_vehicle_state_, *i_reference_trajectory_, road_map_dt);

    // TF reference trajectory to controller's target frame
    Trajectory ego_trajectory = util_function::TransformTrajectory(dt_trajectory, *i_vehicle_state_);
    tk::Map road_map = util_function::GenerateRoadMap(ego_trajectory);
    ControlTrajectory controllable_trajectory = GenerateTargetSpeedProfile(ego_trajectory, vehicle_state, road_map, ptr_longitudinal_controller_->GetTargetFrame());
    CalculateSpeedTrackingErrors(controllable_trajectory,*i_vehicle_state_);
    ptr_longitudinal_controller_->BuildControllerVehicleState(*i_vehicle_state_,road_map);
    ControlCommand control_command = ptr_longitudinal_controller_->CalculateOptimalLongitudinalCommand(controllable_trajectory);

    // Pedal Map
    PedalMap::PedalValues pedal_values = pedal_map_.getPedalValuesProgressiveBrake(i_vehicle_state_->vx, control_command.ax, prev_brake_pedal_);

    PedalValuesFilter(pedal_values);

    control_command.accel_pedal = pedal_values.accel_pedal;
    control_command.brake_pedal = pedal_values.brake_pedal;

    if (is_stop){
        control_command.accel_pedal = 0.0;
        control_command.brake_pedal = 1.0;
    }

    prev_brake_pedal_ = pedal_values.brake_pedal;

    for(int i = 0; i < control_command.control_trajectory.control_point.size(); i++){
        double vx = control_command.control_trajectory.control_point[i].vx;
        double ax = control_command.control_trajectory.control_point[i].ax;
        PedalMap::PedalValues pedal_values = pedal_map_.getPedalValuesProgressiveBrake(vx, ax);
        PedalValuesFilter(pedal_values);
        if (is_stop){
            pedal_values.accel_pedal = 0.0;
            pedal_values.brake_pedal = 1.0;
        }
        control_command.control_trajectory.control_point[i].accel_pedal = pedal_values.accel_pedal;
        control_command.control_trajectory.control_point[i].brake_pedal = pedal_values.brake_pedal;
    }

    // Speed Compensator
    if(params_.use_speed_compensator){
        float comp_trq = SpeedCompensator(controllable_trajectory,*i_vehicle_state_,ptr_longitudinal_controller_->GetControlTrajectoryStepTime());
        speed_tracking_infos_.compensation_longitudinal = comp_trq;
        control_command.trq += comp_trq;
        control_command.ax += comp_trq/(params_.vehicle_mass*params_.vehicle_wheel_radius);
    }
    else{
        speed_tracking_infos_.compensation_longitudinal = 0.0;
    }

    // Fault Check: if nan, stop smooth
    FaultCheck(control_command);

    return {control_command, controllable_trajectory, speed_tracking_infos_};
}

void LongitudinalControl::PedalValuesFilter(PedalMap::PedalValues& pedal_values){
    // If both accel and brake are active, prioritize the bigger one for safety
    if (pedal_values.accel_pedal > pedal_values.brake_pedal) {
        pedal_values.brake_pedal = 0.0;
    } else {
        pedal_values.accel_pedal = 0.0;
    }
    return;
}

ControlTrajectory LongitudinalControl::GenerateTargetSpeedProfile(const Trajectory& trajectory, const VehicleState& vehicle_state, tk::Map road_map, const std::string& frame_out){
    // Convert Ego Cartesian to Frenet coordinate , Only need <X, Y, Speed> 
    std::vector<double> ego_sn = road_map.ToFrenet(0.0, 0.0);
    std::vector<double> start = {ego_sn[0], ego_sn[1]};
    cte_error_ = start[1];

    ControlTrajectory interpolated_trajectory;
    interpolated_trajectory.header = trajectory.header;
    interpolated_trajectory.header.frame_id = frame_out;
    vector<double> time, speed, distance;
    double sum_s = start[0];
    double sum_time = 0.;
    
    // bool is_stop = true;
    
    for (uint16_t i = 0; i < (uint16_t)trajectory.point.size(); i++) {

        double dx = i == 0 ? 0.0 : trajectory.point[i].x - trajectory.point[i - 1].x;
        double dy = i == 0 ? 0.0 : trajectory.point[i].y - trajectory.point[i - 1].y;
        double ds = sqrt(dx * dx + dy * dy);
        double dt;


        // reference speed based on vehicle current speed
        // dt = ds/max(vehicle_state.vx, 0.1);
        // speed.push_back(vehicle_state.vx);

        double ref_speed = trajectory.point[i].speed;
        // Skip points that would NaN-poison the downstream tk::spline:
        //   1) NaN/Inf in x/y/speed → propagates through ds, ref_speed
        //   2) ref_speed ≤ 0 → dt = ds/0 = Inf, then ts/tv splines NaN
        //   3) duplicate distance (ds ≈ 0 after first point) → tk::spline
        //      requires strictly-monotonic knots, else returns NaN for every
        //      query (same root cause as raceline-wrap dedup in 2c26ec3).
        // Without this guard, a single bad upstream point poisons the
        // entire control profile → /hmi/con/target_speed = NaN forever.
        if (!std::isfinite(trajectory.point[i].x)
            || !std::isfinite(trajectory.point[i].y)
            || !std::isfinite(ref_speed)
            || ref_speed <= 0.0) continue;
        dt = ds/max(ref_speed, 0.1);
        if (!distance.empty() && (sum_s + ds) <= distance.back() + 1e-6) continue;

        speed.push_back(ref_speed);
        sum_s += ds;
        sum_time += dt;

        distance.push_back(sum_s);
        time.push_back(sum_time);
    }

    if ((uint16_t)time.size() < 2 || (uint16_t)speed.size() < 2){
        std::cerr << "[LongitudinalControl] Cannot generate target interpolated_trajectory! "
                  << "time=" << time.size() << " speed=" << speed.size() << std::endl;
        return interpolated_trajectory;
    }

    tk::spline st, ts, tv;
    st.set_points(distance, time, false);
    ts.set_points(time, distance, false);
    tv.set_points(time, speed, false);
    
    double time_offset = st(ego_sn[0]) + params_.reference_time_offset;
    //st(start[0])+ params_.reference_time_offset;


    int step_num     = ptr_longitudinal_controller_->GetControlTrajectoryStepNum();
    double step_time = ptr_longitudinal_controller_->GetControlTrajectoryStepTime();

    for (uint16_t i = 0; i < step_num+2; i++) {
        ControlPoint dp;
        double target_distance = ts(step_time * i + time_offset);
        vector<double> xy = road_map.ToCartesian(target_distance, 0.0);
        dp.x = xy[0]; 
        dp.y = xy[1];
        // dp.yaw = road_map.GetSlope(target_distance);
        // dp.curvature = road_map.GetCurvature(target_distance);
        dp.vx = tv(step_time * i + time_offset);
        dp.ax = (tv(step_time * (i+1) + time_offset) - dp.vx)/step_time;
        dp.time = step_time * i ;
        // dp.distance = target_distance - ts(time_offset);
        interpolated_trajectory.control_point.push_back(dp);
    }
    return interpolated_trajectory;
}

float LongitudinalControl::SpeedCompensator(const ControlTrajectory& profile, const VehicleState& vehicle_state, const double& dt){
    

    int target_idx = min((int)(params_.look_ahead_time/dt), (int)profile.control_point.size() - 1);
    if (speed_compensator_ref_speed_ == 0.0) {
        speed_compensator_ref_speed_ = i_vehicle_state_->vx;
    }

    double calculated_ref_speed = profile.control_point.at(target_idx).vx;

    if(!isnan(calculated_ref_speed)){
        speed_compensator_ref_speed_ = params_.ref_speed_lpf*calculated_ref_speed + (1.0-params_.ref_speed_lpf)*speed_compensator_ref_speed_;
    }
    
    double current_speed = sqrt(pow((vehicle_state.vx),2) + pow((vehicle_state.vy),2));

    double speed_error = speed_compensator_ref_speed_ - current_speed;
    sum_speed_error_ += speed_error;

    if(prev_speed_error_ * speed_error < 0){
        sum_speed_error_ = 0.0;
    }
    if(sum_speed_error_ > params_.max_i_error){
        sum_speed_error_ = params_.max_i_error;
    }
    else if(sum_speed_error_ < -params_.max_i_error){
        sum_speed_error_ = -params_.max_i_error;
    }

    speed_tracking_infos_.speed_compensation_error  = (speed_compensator_ref_speed_-current_speed)*MPS2KPH;

    float comp_trq    =  params_.vel_kp * speed_error
                        + params_.vel_ki * sum_speed_error_
                        + params_.vel_kd * (speed_error - prev_speed_error_);
    prev_speed_error_ = speed_error; 
    // std::cout << "comped_trq" << comp_trq << std::endl;
    return comp_trq;
}

void LongitudinalControl::FaultCheck(ControlCommand& control_command){
    if(isnan(control_command.trq)){
        // Stop smooth ... if nan
        control_command.trq = params_.emergency_brake_torque;
        control_command.ax  = params_.emergency_decel;
    }
    else if(i_vehicle_state_->vehicle_can.lateral_ad_mode == AutonomousMode::RUN && i_vehicle_state_->vehicle_can.steering_state != MdpsState::MDPS_ACTIVATE){
        if(i_vehicle_state_->vx > 5.0*KPH2MPS){
            control_command.trq = min(control_command.trq, (float)params_.emergency_brake_torque);
            control_command.ax  = min(control_command.ax,  (float)params_.emergency_decel);
        }
        else{
            control_command.ax  = min(control_command.ax, (float)0.5);
            double control_Fx     = ComputeLongitudinalForce(control_command.ax, i_vehicle_state_->vx, i_vehicle_state_->pitch);
            double control_torque = control_Fx * params_.vehicle_wheel_radius;
            control_command.trq   = min(control_command.trq, (float)control_torque);
        }
    }

    if(params_.use_cte_deceleration){
        double target_cte = fabs(cte_error_);
        if(target_cte > params_.deceleration_min_cte){
            double cte_ratio = (params_.deceleration_max_accel - params_.deceleration_min_accel)/(params_.deceleration_max_cte-params_.deceleration_min_cte);

            target_cte = max(target_cte, params_.deceleration_min_cte);
            target_cte = min(target_cte, params_.deceleration_max_cte);

            double target_accel   = params_.deceleration_min_accel + cte_ratio*(target_cte-params_.deceleration_min_cte);
            control_command.ax    = min(control_command.ax, (float)target_accel);
            double control_Fx     = ComputeLongitudinalForce(control_command.ax, i_vehicle_state_->vx, i_vehicle_state_->pitch);
            double control_torque = control_Fx * params_.vehicle_wheel_radius;
            control_command.trq   = min(control_command.trq, (float)control_torque);
        }
    }
}

void LongitudinalControl::CalculateSpeedTrackingErrors(const ControlTrajectory& profile, const VehicleState& vehicle_state){
    if(profile.control_point.size()>0){
        speed_tracking_infos_.speed_error  = (profile.control_point[0].vx - vehicle_state.vx)*MPS2KPH;
        speed_tracking_infos_.accel_error =  profile.control_point[0].ax - vehicle_state.ax;
    }
    else{
        std::cout << "Not enough ref profile to calculate errors" << std::endl;
    }
}

void LongitudinalControl::CreateLongitudinalController(){
    if(prev_control_method_ == params_.longitudinal_control_method){
        return;
    }
    ptr_longitudinal_controller_.reset();

    if (params_.longitudinal_control_method == LongitudinalControlMethod::PID)
        ptr_longitudinal_controller_ = std::make_shared<LongitudinalPID>();
    else if (params_.longitudinal_control_method == LongitudinalControlMethod::DYNAMIC_LMPC)
        ptr_longitudinal_controller_ = std::make_shared<LongitudinalDynamicLMPC>();
    else if (params_.longitudinal_control_method == LongitudinalControlMethod::DELAYED_KINEMATIC_LMPC)
        ptr_longitudinal_controller_ = std::make_shared<LongitudinalDelayedKinematicLMPC>();
    else
        ptr_longitudinal_controller_ = std::make_shared<LongitudinalPID>(); // default
    

    prev_control_method_ = params_.longitudinal_control_method;
}

Trajectory LongitudinalControl::GenerateDTTargetTrajectory(const VehicleState& vehicle_state,
                                                            const Trajectory& trajectory,
                                                            tk::Map road_map){
    return util_function::GenerateDTTargetTrajectory(
        vehicle_state,
        trajectory,
        road_map,
        params_.reference_time_offset,
        ptr_longitudinal_controller_->GetControlTrajectoryStepNum(),
        ptr_longitudinal_controller_->GetControlTrajectoryStepTime()
    );
}

void LongitudinalControl::ProcessINI(){
	if (util_ini_parser_.IsFileUpdated()){
		util_ini_parser_.ParseConfig("Longitudinal Control", "longitudinal_control_method", params_.longitudinal_control_method);
		util_ini_parser_.ParseConfig("Longitudinal Control", "reference_time_offset" , params_.reference_time_offset);
		util_ini_parser_.ParseConfig("Longitudinal Control", "use_user_speed"        , params_.use_user_speed);
		util_ini_parser_.ParseConfig("Longitudinal Control", "user_speed"            , params_.user_speed);

		util_ini_parser_.ParseConfig("Longitudinal Control", "use_speed_compensator" , params_.use_speed_compensator);
		util_ini_parser_.ParseConfig("Longitudinal Control", "look_ahead_time"       , params_.look_ahead_time);
		util_ini_parser_.ParseConfig("Longitudinal Control", "vel_kp"                , params_.vel_kp);
		util_ini_parser_.ParseConfig("Longitudinal Control", "vel_ki"                , params_.vel_ki);
		util_ini_parser_.ParseConfig("Longitudinal Control", "vel_kd"                , params_.vel_kd);
		util_ini_parser_.ParseConfig("Longitudinal Control", "max_i_error"           , params_.max_i_error);
		util_ini_parser_.ParseConfig("Longitudinal Control", "ref_speed_lpf"         , params_.ref_speed_lpf);

		util_ini_parser_.ParseConfig("Longitudinal Control", "zero_speed"            , params_.zero_speed);

		util_ini_parser_.ParseConfig("Longitudinal Control", "use_cte_deceleration"            , params_.use_cte_deceleration);
		util_ini_parser_.ParseConfig("Longitudinal Control", "deceleration_min_cte"            , params_.deceleration_min_cte);
		util_ini_parser_.ParseConfig("Longitudinal Control", "deceleration_min_accel"            , params_.deceleration_min_accel);
		util_ini_parser_.ParseConfig("Longitudinal Control", "deceleration_max_cte"            , params_.deceleration_max_cte);
		util_ini_parser_.ParseConfig("Longitudinal Control", "deceleration_max_accel"            , params_.deceleration_max_accel);

		util_ini_parser_.ParseConfig("Longitudinal Control", "pedal_map_file_name"       , params_.pedal_map_file_name);
        util_ini_parser_.ParseConfig("Longitudinal Control", "emergency_brake_torque"   , params_.emergency_brake_torque);
        util_ini_parser_.ParseConfig("Longitudinal Control", "emergency_decel"          , params_.emergency_decel);
        util_ini_parser_.ParseConfig("Vehicle", "vehicle_mass",         params_.vehicle_mass        );
        util_ini_parser_.ParseConfig("Vehicle", "gravity",              params_.gravity             );
        util_ini_parser_.ParseConfig("Vehicle", "rolling_resistance",   params_.rolling_resistance  );
        util_ini_parser_.ParseConfig("Vehicle", "air_density",          params_.air_density         );
        util_ini_parser_.ParseConfig("Vehicle", "vehicle_front_area",   params_.vehicle_front_area  );
        util_ini_parser_.ParseConfig("Vehicle", "air_drag_coeff",       params_.air_drag_coeff      );
        util_ini_parser_.ParseConfig("Vehicle", "vehicle_wheel_radius", params_.vehicle_wheel_radius);

        params_.user_speed =  params_.user_speed*KPH2MPS;

		std::cout<<"[Longitudinal Control] Ini file is updated!"<<std::endl;
    }
}