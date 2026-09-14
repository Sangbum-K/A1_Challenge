#include <lateral_control/lateral_control_core/lateral_purepursuit/lateral_purepursuit.hpp>

LateralPurePursuit::LateralPurePursuit() :LateralControllerInterface() {
    const char* pwd_env = getenv("PWD");
    if (pwd_env == nullptr) {
        std::cerr << "[LateralPurePursuit] PWD environment variable is not set. Cannot load config." << std::endl;
        std::exit(1);
    }
    std::string dir(pwd_env);
    std::string ini_dir("/config/control.ini");
    util_ini_parser_.Init((dir+ini_dir).c_str());

    // Parameters initialization
    ProcessINI();
    std::cout << "[LateralPurePursuit] Initialized." << std::endl;
}

LateralPurePursuit::~LateralPurePursuit() {
}

void LateralPurePursuit::BuildControllerVehicleState(const interface::VehicleState& vehicle_state, tk::Map road_map){
    ProcessINI();
    i_vehicle_state_ = vehicle_state;
    i_road_map_      = road_map;
}

boost::optional<interface::ControlCommand> LateralPurePursuit::CalculateOptimalTireSteering(const interface::ControlTrajectory& ref){
    interface::ControlCommand control_command;
    if (i_vehicle_state_.vehicle_can.operation_mode != interface::OperationMode::AUTONOMOUS){
        // initialize
        control_command.control_trajectory.control_point.clear();
        control_command.steering_tire_angle = i_vehicle_state_.vehicle_can.steering_tire_angle*interface::RAD2DEG;
    }
    else{
        // double space_offset =i_road_map_.ToFrenet(ref.control_point.front().x, ref.control_point.front().y)[0];
        // std::vector<double> ego_sn = i_road_map_.ToFrenet(space_offset, 0.0); 
        // std::cout << "space_offset : " << space_offset << std::endl;       
        std::vector<double> ego_sn = i_road_map_.ToFrenet(0.0, 0.0);
        double look_ahead_distance = params_.min_look_ahead +  params_.look_ahead_gain*i_vehicle_state_.vx;
        double target_s = ego_sn[0] + look_ahead_distance;
        double target_n = 0.0;
        std::vector<double> target_xy = i_road_map_.ToCartesian(target_s, target_n);
        double target_yaw = i_road_map_.GetSlope(target_s);
        double diff_angle = atan2(target_xy[1], target_xy[0]);
        double target_distance = sqrt(pow(target_xy[0],2) + pow(target_xy[1],2));
        double look_ahead_error = target_distance * sin(diff_angle);

        double steering_tire_angle_rad = atan2(2.0*params_.vehicle_wheel_base*look_ahead_error, pow(target_distance, 2));

        // DR of purepursuit
        control_command.control_trajectory.header = ref.header;
        interface::ControlPoint point;
        point.x     = 0.0;
        point.y     = 0.0;
        point.yaw   = 0.0;
        control_command.control_trajectory.control_point.push_back(point);    

        for (int i = 0; i < params_.step_num; i++){
            point.x   +=  i_vehicle_state_.vx*cos(point.yaw)* params_.step_dt;
            point.y   +=  i_vehicle_state_.vx*sin(point.yaw)* params_.step_dt;
            point.yaw += i_vehicle_state_.vx*(tan(steering_tire_angle_rad)/params_.vehicle_wheel_base)* params_.step_dt;
            control_command.control_trajectory.control_point.push_back(point);   
        }
        control_command.steering_tire_angle = steering_tire_angle_rad*interface::RAD2DEG;
    }
    return control_command;
}

void LateralPurePursuit::ProcessINI() {
    if (util_ini_parser_.IsFileUpdated()){
        util_ini_parser_.ParseConfig("Vehicle",      "wheel_base",       params_.vehicle_wheel_base);
        util_ini_parser_.ParseConfig("Pure pursuit", "min_look_ahead",   params_.min_look_ahead);
        util_ini_parser_.ParseConfig("Pure pursuit", "look_ahead_gain",  params_.look_ahead_gain);
        util_ini_parser_.ParseConfig("Pure pursuit", "step_dt",          params_.step_dt);
        util_ini_parser_.ParseConfig("Pure pursuit", "step_num",         params_.step_num);
    }
}