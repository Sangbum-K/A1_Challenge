#include <longitudinal_control/longitudinal_control_core/longitudinal_pid/longitudinal_pid.hpp>

LongitudinalPID::LongitudinalPID() :LongitudinalControllerInterface() {
    const char* pwd_env = getenv("PWD");
    if (pwd_env == nullptr) {
        std::cerr << "[LongitudinalPID] PWD environment variable is not set. Cannot load config." << std::endl;
        std::exit(1);
    }
    std::string dir(pwd_env);
    std::string ini_dir("/config/control.ini");
    util_ini_parser_.Init((dir+ini_dir).c_str());

    // Parameters initialization
    ProcessINI();
    std::cout << "init PID" << std::endl;
}

LongitudinalPID::~LongitudinalPID() {
}

void LongitudinalPID::BuildControllerVehicleState(const VehicleState& vehicle_state, tk::Map road_map){
    ProcessINI();
    i_vehicle_state_ = vehicle_state;
}

ControlCommand LongitudinalPID::CalculateOptimalLongitudinalCommand(const ControlTrajectory& ref){
    ControlCommand control_command;
    if (i_vehicle_state_.vehicle_can.operation_mode != OperationMode::AUTONOMOUS){
        // initialize
        control_command.control_trajectory.control_point.clear();
        control_command.trq = 0.0;
        control_command.ax  = 0.0;
    }
    else{
        int target_idx = min((int)(params_.look_ahead_time/params_.step_dt), params_.step_num );

        if (ref_speed_ == 0.0) {
            ref_speed_ = i_vehicle_state_.vx;
        }
        double calculated_ref_speed = ref.control_point.at(target_idx).vx;

        if(!isnan(calculated_ref_speed)){
            ref_speed_ = params_.ref_speed_lpf*calculated_ref_speed + (1.0-params_.ref_speed_lpf)*ref_speed_;
        }

        double current_speed = i_vehicle_state_.vx;

        double speed_error = ref_speed_ - current_speed;
        sum_speed_error_ += speed_error;
        if(!i_vehicle_state_.vehicle_can.longitudinal_ad_mode){
            sum_speed_error_ = 0;
        }
        if(prev_speed_error_ * speed_error < 0){
            sum_speed_error_ = 0.0;
        }
        if(sum_speed_error_ > params_.max_i_error){
            sum_speed_error_ = params_.max_i_error;
        }
        else if(sum_speed_error_ < -params_.max_i_error){
            sum_speed_error_ = -params_.max_i_error;
        }
        double velocity_pid_output =  params_.vel_kp * speed_error
                                    + params_.vel_ki * sum_speed_error_
                                    + params_.vel_kd * (speed_error - prev_speed_error_);
        prev_speed_error_ = speed_error; 
        
        
        // if(fabs(i_vehicle_state_.vx) < 40*KPH2MPS && speed_error > 0){
        //     // low speed overshoot compensation
        //     velocity_pid_output =  1.5 * speed_error; 
        // } // erase for morai


        double control_accel  = velocity_pid_output;
        double control_Fx     = ComputeLongitudinalForce(control_accel, i_vehicle_state_.vx, i_vehicle_state_.pitch);
        double control_torque = control_Fx * params_.vehicle_wheel_radius;


        // DR of pid
        control_command.control_trajectory.header = ref.header;
        ControlPoint point;
        point.x     = 0.0;
        point.y     = 0.0;
        point.vx    = current_speed;
        point.ax    = control_accel;
        control_command.control_trajectory.control_point.push_back(point);    

        control_command.ax  = control_accel;
        control_command.trq = control_torque;
    }
    return control_command;
}

void LongitudinalPID::ProcessINI() {
    if (util_ini_parser_.IsFileUpdated()){
        util_ini_parser_.ParseConfig("PID", "look_ahead_time",    params_.look_ahead_time );
        util_ini_parser_.ParseConfig("PID", "step_dt",            params_.step_dt         );
        util_ini_parser_.ParseConfig("PID", "step_num",           params_.step_num        );
        util_ini_parser_.ParseConfig("PID", "vel_kp",             params_.vel_kp          );
        util_ini_parser_.ParseConfig("PID", "vel_ki",             params_.vel_ki          );
        util_ini_parser_.ParseConfig("PID", "vel_kd",             params_.vel_kd          );
        util_ini_parser_.ParseConfig("PID", "max_i_error",        params_.max_i_error     );
        util_ini_parser_.ParseConfig("PID", "ref_speed_lpf",      params_.ref_speed_lpf   );

        util_ini_parser_.ParseConfig("Vehicle", "vehicle_mass",         params_.vehicle_mass        );
        util_ini_parser_.ParseConfig("Vehicle", "gravity",              params_.gravity             );
        util_ini_parser_.ParseConfig("Vehicle", "rolling_resistance",   params_.rolling_resistance  );
        util_ini_parser_.ParseConfig("Vehicle", "air_density",          params_.air_density         );
        util_ini_parser_.ParseConfig("Vehicle", "vehicle_front_area",   params_.vehicle_front_area  );
        util_ini_parser_.ParseConfig("Vehicle", "air_drag_coeff",       params_.air_drag_coeff      );
        util_ini_parser_.ParseConfig("Vehicle", "vehicle_wheel_radius", params_.vehicle_wheel_radius);

        // cout << params_.look_ahead_time << ",\n" <<
        //         params_.step_dt         << ",\n" <<
        //         params_.step_num        << ",\n" <<
        //         params_.vel_kp          << ",\n" <<
        //         params_.vel_ki          << ",\n" <<
        //         params_.vel_kd          << endl;
    }
}