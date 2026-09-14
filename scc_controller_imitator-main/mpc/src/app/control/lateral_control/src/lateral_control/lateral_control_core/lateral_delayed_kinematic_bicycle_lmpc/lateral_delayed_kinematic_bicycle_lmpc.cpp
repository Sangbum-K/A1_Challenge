#include <lateral_control/lateral_control_core/lateral_delayed_kinematic_bicycle_lmpc/lateral_delayed_kinematic_bicycle_lmpc.hpp>

LateralDelayedKinematicBicycleLMPC::LateralDelayedKinematicBicycleLMPC() :LateralControllerInterface() {
    const char* pwd_env = getenv("PWD");
    if (pwd_env == nullptr) {
        std::cerr << "[LateralDelayedKinematicBicycleLMPC] PWD environment variable is not set. Cannot load config." << std::endl;
        std::exit(1);
    }
    std::string dir(pwd_env);
    std::string ini_dir("/config/control.ini");
    util_ini_parser_.Init((dir+ini_dir).c_str());

    // Parameters initialization
    ProcessINI();
    std::cout << "[LateralDelayedKinematicBicycleLMPC] Initialized." << std::endl;
}

LateralDelayedKinematicBicycleLMPC::~LateralDelayedKinematicBicycleLMPC() {
}



void LateralDelayedKinematicBicycleLMPC::BuildControllerVehicleState(const VehicleState& vehicle_state, tk::Map road_map){
        
    std::cout << "LateralDelayedKinematicBicycleLMPC : " << (int)live_cnt_++ << std::endl;
    
    ProcessINI();
    i_vehicle_state_ = vehicle_state;
    i_road_map_      = road_map;
}

boost::optional<ControlCommand> LateralDelayedKinematicBicycleLMPC::CalculateOptimalTireSteering(const ControlTrajectory& ref){
    ControlCommand control_command;
    if (i_vehicle_state_.vehicle_can.operation_mode != OperationMode::AUTONOMOUS || 
        (i_vehicle_state_.vehicle_can.steering_state == MdpsState::MDPS_ACTIVATION_START && is_steering_fault_==true) ){
        
        // initialize
        control_command.control_trajectory.control_point.clear();
        control_command.steering_tire_angle = i_vehicle_state_.vehicle_can.steering_tire_angle*RAD2DEG;
        optimal_states_.clear();
        operation_inputs_.clear();
        is_steering_fault_ = false;
    }
    else{
        vector<State> operation_point = GenerateOperationPoints(ref); 
        vector<OptVariables> opt_vals = SolveHPIPM(operation_point,ref);   

        control_command.control_trajectory.control_point.clear();
        control_command.control_trajectory.header = ref.header;
        for(int i = 0; i<opt_vals.size() ; i++){
            ControlPoint point = ref.control_point[i];
            point.x = opt_vals[i].first(0);
            point.y = opt_vals[i].first(1);
            point.yaw = opt_vals[i].first(2);
            point.steer = opt_vals[i].first(10)*RAD2DEG;
            point.dsteer = opt_vals[i].second(0)*RAD2DEG;

            control_command.control_trajectory.control_point.push_back(point);
        }
        if(opt_vals.size() >= 1){
            control_command.steering_tire_angle = opt_vals[params_.shift_input_step].first(10)*RAD2DEG;
        }
        else{
            control_command.steering_tire_angle = 0.0;
        }
        CheckSolvedStateTwisted(ref, control_command);
        CheckMdpsState();
        
    }
    if(is_solved_ == true){
        return control_command;
    }
    else{
        return boost::none;
    }
}

vector<LateralDelayedKinematicBicycleLMPC::State> LateralDelayedKinematicBicycleLMPC::GenerateOperationPoints(const ControlTrajectory& ref){
    vector<LateralDelayedKinematicBicycleLMPC::State> operation_points;

    if (operation_inputs_.empty()){
        std::cout<<"empty operation input"<<std::endl;
        LateralDelayedKinematicBicycleLMPC::Input operate_input;
        for(uint8_t i=0; i<step_num_+1; i++){
            operate_input.setZero();     
            operation_inputs_.push_back(operate_input);
        } 
    }    

    operation_points.push_back(GetInitState(optimal_states_.empty()));
    for(uint8_t i=0; i<step_num_+1; i++) 
    {
        // MODEL VARIANT
        double ref_v = (double)(ref.control_point[i].vx );
        State curr_state = operation_points.back();
        Input input = operation_inputs_[i];
        State state_derive = GetContinuousModelDeriv(curr_state , input, ref_v);
        State next_state = curr_state + state_derive * step_dt_;
        
        next_state(4) = curr_state(5);
        next_state(5) = curr_state(6);
        next_state(6) = curr_state(7);
        next_state(7) = curr_state(8);
        next_state(8) = curr_state(9);
        next_state(9) = curr_state(10);
        
        operation_points.push_back(next_state);

    }
    
    return operation_points;
}

LateralDelayedKinematicBicycleLMPC::State LateralDelayedKinematicBicycleLMPC::GetReferenceState(const ControlTrajectory& ref, const uint32_t& idx){
    State x_ref;
    x_ref <<    ref.control_point[idx].x,
                ref.control_point[idx].y,
                ref.control_point[idx].yaw,
                0.0,
                0.0,
                
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0;
    return x_ref;
}


LateralDelayedKinematicBicycleLMPC::State LateralDelayedKinematicBicycleLMPC::GetInitState(bool is_prev_state_empty){
    State x0;
    if(is_prev_state_empty){
        x0 <<   0.0, 
                0.0, 
                0.0, 
                i_vehicle_state_.vehicle_can.steering_tire_angle,
                i_vehicle_state_.vehicle_can.steering_tire_angle,

                i_vehicle_state_.vehicle_can.steering_tire_angle,
                i_vehicle_state_.vehicle_can.steering_tire_angle,
                i_vehicle_state_.vehicle_can.steering_tire_angle,
                i_vehicle_state_.vehicle_can.steering_tire_angle,
                i_vehicle_state_.vehicle_can.steering_tire_angle,
                i_vehicle_state_.vehicle_can.steering_tire_angle;
    }
    else{
        x0 <<   0.0, 
                0.0, 
                0.0, 
                optimal_states_.front()(3), //i_vehicle_state_.steering_tire_angle,
                optimal_states_.front()(4),
                optimal_states_.front()(5),
                optimal_states_.front()(6),
                optimal_states_.front()(7),
                optimal_states_.front()(8),
                optimal_states_.front()(9),
                optimal_states_.front()(10);
    }
    return x0;
};

LateralDelayedKinematicBicycleLMPC::State LateralDelayedKinematicBicycleLMPC::GetContinuousModelDeriv(const State &x,const Input &u, const double &v)
{
    // MODEL VARIANT
    State f;
    f(0) = v*std::cos(x(2));
    f(1) = v*std::sin(x(2));
    f(2) = v*std::tan(x(3))/params_.vehicle_wheel_base;
    f(3) = (-x(3) + x(4))/params_.time_constant;
    f(4) = 0;
    f(5) = 0;
    f(6) = 0;
    f(7) = 0;
    f(8) = 0;
    f(9) = 0;
    f(10) = u(0);
    return f;
}

LateralDelayedKinematicBicycleLMPC::LinModelMatrix LateralDelayedKinematicBicycleLMPC::GetDiscreteLinModelMatrix(const State &x, const Input &u, const double &v){

    A_MPC A_d;
    A_d <<  1.0,    0.0,    -v*std::sin(x(2))*step_dt_,       0.0,                                        0.0,                        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,   // X
            0.0,    1.0,     v*std::cos(x(2))*step_dt_,       0.0,                                        0.0,                        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,   // Y
            0.0,    0.0,     1.0,    (v*(pow(tan(x(3)),2.0)+1.0))*step_dt_/params_.vehicle_wheel_base,  0.0,                        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,   // phi
            0.0,    0.0,     0.0,                 1.0-step_dt_/params_.time_constant,       step_dt_/params_.time_constant,           0.0,    0.0,    0.0,    0.0,    0.0,    0.0,   // delta_a
            0.0,    0.0,     0.0,                             0.0,                                        0.0,                        1.0,    0.0,    0.0,    0.0,    0.0,    0.0,   // delta_6
            0.0,    0.0,     0.0,                             0.0,                                        0.0,                        0.0,    1.0,    0.0,    0.0,    0.0,    0.0,   // delta_5
            0.0,    0.0,     0.0,                             0.0,                                        0.0,                        0.0,    0.0,    1.0,    0.0,    0.0,    0.0,   // delta_4
            0.0,    0.0,     0.0,                             0.0,                                        0.0,                        0.0,    0.0,    0.0,    1.0,    0.0,    0.0,   // delta_3
            0.0,    0.0,     0.0,                             0.0,                                        0.0,                        0.0,    0.0,    0.0,    0.0,    1.0,    0.0,   // delta_2
            0.0,    0.0,     0.0,                             0.0,                                        0.0,                        0.0,    0.0,    0.0,    0.0,    0.0,    1.0,   // delta_1
            0.0,    0.0,     0.0,                             0.0,                                        0.0,                        0.0,    0.0,    0.0,    0.0,    0.0,    1.0;   // delta_0
    //      X       Y        phi                              delta_a                                     delta_6                    delta5 delta4   delta3  delta2   delta1  delta0
    B_MPC B_d;
    B_d <<  0.0,    // X
            0.0,    // Y
            0.0,    // phi 
            0.0,    // delta_a
            0.0,    // delta_6
            0.0,    // delta_5
            0.0,    // delta_4
            0.0,    // delta_3
            0.0,    // delta_2
            0.0,    // delta_1
            step_dt_;     // delta_0


    C_MPC C_d;
    C_d <<  step_dt_*(cos(x(2))*(v)+v*(x(2))*sin(x(2))),
            step_dt_*(sin(x(2))*(v)-v*(x(2))*cos(x(2))),
            step_dt_*((tan(x(3))*(v))/params_.vehicle_wheel_base-(v*(x(3))*(pow(tan(x(3)),2.0)+1.0))/params_.vehicle_wheel_base),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0;

   return {A_d,B_d,C_d};
}


vector<LateralDelayedKinematicBicycleLMPC::OptVariables> LateralDelayedKinematicBicycleLMPC::SolveHPIPM(const vector<State>& operation_points, const ControlTrajectory& ref){
    std::vector<hpipm::OcpQp> qp(step_num_+2);
    
    Eigen::Matrix<double,dim_state_,dim_state_> Q;
    Eigen::Matrix<double,dim_input_,dim_input_> R;
    Eigen::Matrix<double,dim_state_,dim_input_> S;

    // cost
    // MODEL VARIANT //
    Q.setZero();  Q.diagonal()  << params_.weight_x,params_.weight_y, params_.weight_yaw, params_.weight_steer, 0 ,0 ,0 ,0, 0, 0, 0;
    S.setZero();
    R.setZero();  R.diagonal()  << params_.weight_dsteer + WeightDeltaSteer(ref.control_point.front().y);

    Eigen::VectorXd x_ref(dim_state_);
    const Eigen::VectorXd r = Eigen::VectorXd::Zero(dim_input_);
    for (int i=0; i<step_num_+1; ++i) {
        x_ref = GetReferenceState(ref,i); 

        qp[i].Q = Q.transpose();
        qp[i].R = R.transpose();
        qp[i].S = S.transpose();
        qp[i].q = (- Q * x_ref).transpose();
        qp[i].r = r;
    }
    {
        x_ref = GetReferenceState(ref,step_num_+1); 
        // put final cost
        qp[step_num_+1].Q = Q.transpose();
        qp[step_num_+1].q = (- Q * x_ref).transpose();

    }
    

    // constraints
    // dynamics
    for (int i=0; i<step_num_+1; ++i) {
        const auto [a_d, b_d, c_d] = GetDiscreteLinModelMatrix(operation_points.at(i),operation_inputs_.at(i), (double)ref.control_point[i].vx);
        qp[i].A = a_d;
        qp[i].B = b_d;
        qp[i].b = c_d;
    }

    // state constraint
    for (int i=1; i<=step_num_+1; ++i) {
        // X Y phi delta
        qp[i].idxbx = {10};
        qp[i].lbx = (Eigen::VectorXd(1) << -MAX_STEERING_TIRE_ANGLE*DEG2RAD).finished();
        qp[i].ubx = (Eigen::VectorXd(1) <<  MAX_STEERING_TIRE_ANGLE*DEG2RAD).finished();
    }

    //input constraint
    for (int i=0; i<step_num_+1; ++i) {
        qp[i].idxbu = {0};
        qp[i].lbu = (Eigen::VectorXd(1) << -MAX_STEERING_TIRE_SPEED*DEG2RAD).finished();
        qp[i].ubu = (Eigen::VectorXd(1) <<  MAX_STEERING_TIRE_SPEED*DEG2RAD).finished();
    }
    ////////////////////

    hpipm::OcpQpIpmSolverSettings solver_settings;
    solver_settings.mode = hpipm::HpipmMode::Robust;
    solver_settings.iter_max = 100;
    solver_settings.alpha_min = 1e-8;
    solver_settings.mu0 = 1e2;
    solver_settings.tol_stat = 1e-04;
    solver_settings.tol_eq = 1e-04;
    solver_settings.tol_ineq = 1e-04;
    solver_settings.tol_comp = 1e-04;
    solver_settings.reg_prim = 1e-12;
    solver_settings.warm_start = 1;
    solver_settings.pred_corr = 1;
    solver_settings.ric_alg = 0;
    solver_settings.split_step = 1;

    std::vector<hpipm::OcpQpSolution> solution(step_num_+2);
    hpipm::OcpQpIpmSolver solver(qp, solver_settings);
    
    Eigen::VectorXd x = Eigen::VectorXd::Zero(dim_state_);
    for (int i=0; i<step_num_+1; ++i) {
        constexpr double u0 = 0.0;
        solution[i].x = x;
        solution[i].u.resize(dim_input_);
        solution[i].u.fill(u0);
    }
    solution[step_num_+1].x = x;

    Eigen::VectorXd x0 = GetInitState(optimal_states_.empty());

    auto opt_sol = solver.solve(x0, qp, solution);
    
    vector<LateralDelayedKinematicBicycleLMPC::OptVariables> opt_vals;
    if(opt_sol != hpipm::HpipmStatus::Success){
        is_solved_ = false;
        std::cerr << "[LateralDelayedKinematicBicycleLMPC] SOLVER FAILED - using previous solution." << std::endl;
        for (int i=0; i<step_num_+1; ++i) {
            LateralDelayedKinematicBicycleLMPC::OptVariables opt_var;
            opt_var.first = optimal_states_[i];
            opt_var.second = operation_inputs_[i];

            opt_vals.push_back(opt_var);           
        }
        optimal_states_.clear();
        operation_inputs_.clear();
        return opt_vals;
    }
    else{
        is_solved_ = true;
        optimal_states_.clear();
        operation_inputs_.clear();
        for (int i=1; i<step_num_+2; ++i) {
            LateralDelayedKinematicBicycleLMPC::OptVariables opt_var;
            opt_var.first = solution[i].x;
            opt_var.second = solution[i-1].u;

            opt_vals.push_back(opt_var);           
            optimal_states_.push_back(opt_var.first);
            operation_inputs_.push_back(opt_var.second);
        }
    }
    
    return opt_vals;
}

double LateralDelayedKinematicBicycleLMPC::WeightDeltaSteer(const double& lateral_error){
    static double dsteer_weight = 0.0;
    static double prev_lateral_distance = lateral_error;
    double d_lateral_distance = fabs(lateral_error - prev_lateral_distance);
    
    if(d_lateral_distance > params_.weight_dsteer_n_thresh){
        dsteer_weight = params_.weight_dsteer_n + params_.weight_dsteer_n * i_vehicle_state_.vx ;
    }
    else if(is_steering_fault_ == true){
        dsteer_weight = params_.weight_dsteer_n;
    }
    else{
        dsteer_weight = (1.0-params_.weight_dsteer_lpf)*dsteer_weight;
    }
    // cout << "d lat distance " <<d_lateral_distance<<"\tw_dsteer " << dsteer_weight << endl;

    prev_lateral_distance = lateral_error;
    return dsteer_weight;

}

void LateralDelayedKinematicBicycleLMPC::CheckSolvedStateTwisted(const ControlTrajectory& ref, const ControlCommand& control_command){
    // Calculate errors
    // 1. distance error
    // 2. yaw error
    double sum_distance_error = 0.0;
    double sum_yaw_error      = 0.0;

    for(int i = 0; i < control_command.control_trajectory.control_point.size(); i++){
        double distance_error = sqrt(pow(ref.control_point.at(i).x-control_command.control_trajectory.control_point.at(i).x,2)
                                    +pow(ref.control_point.at(i).y-control_command.control_trajectory.control_point.at(i).y,2));
        double yaw_error      = ref.control_point.at(i).yaw-control_command.control_trajectory.control_point.at(i).yaw;

        sum_distance_error += distance_error;
        sum_yaw_error += yaw_error*RAD2DEG;
    }

    double avg_distance_error = fabs(sum_distance_error/control_command.control_trajectory.control_point.size());
    double avg_yaw_error = fabs(sum_yaw_error/control_command.control_trajectory.control_point.size());

    // cout << "dist_e " <<  sum_distance_error/control_command.control_trajectory.control_point.size()  << ",\t yaw_e:" << sum_yaw_error/control_command.control_trajectory.control_point.size() << endl;
    
    if(avg_distance_error > params_.avg_distance_error_max && avg_yaw_error > params_.avg_yaw_error_max){
        std::cerr << "[LateralDelayedKinematicBicycleLMPC] Solution diverged (twisted), resetting solver." << std::endl;
        is_solved_ = false;
        optimal_states_.clear();
        operation_inputs_.clear();
    }

    static double prev_timestamp = i_vehicle_state_.header.stamp;
    double curr_timestamp = i_vehicle_state_.header.stamp;
    if(curr_timestamp < prev_timestamp){
        std::cerr << "[LateralDelayedKinematicBicycleLMPC] Timestamp went backward, resetting solver." << std::endl;
        is_solved_ = false;
        optimal_states_.clear();
        operation_inputs_.clear();
    }
    prev_timestamp = curr_timestamp;
}

void LateralDelayedKinematicBicycleLMPC::CheckMdpsState(){
    if( i_vehicle_state_.vehicle_can.steering_state == MdpsState::MDPS_ABORTED || 
        i_vehicle_state_.vehicle_can.steering_state == MdpsState::MDPS_ERROR){
        
        is_steering_fault_ = true;
    }
    else if(i_vehicle_state_.vehicle_can.steering_state == MdpsState::MDPS_ACTIVATE){
        is_steering_fault_ = false;
    }
}

void LateralDelayedKinematicBicycleLMPC::ProcessINI() {
    if (util_ini_parser_.IsFileUpdated()){
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "step_dt",               params_.step_dt                 );
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "step_num",              params_.step_num                );
        
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "shift_input_step",      params_.shift_input_step        );
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "time_constant",         params_.time_constant           );

        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "weight_x",              params_.weight_x                );
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "weight_y",              params_.weight_y                );
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "weight_yaw",            params_.weight_yaw              );
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "weight_steer",          params_.weight_steer            );
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "weight_dsteer",         params_.weight_dsteer           );
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "weight_dsteer_n",       params_.weight_dsteer_n         );
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "weight_dsteer_n_thresh",params_.weight_dsteer_n_thresh  );
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "weight_dsteer_lpf",     params_.weight_dsteer_lpf         );

        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "avg_distance_error_max",params_.avg_distance_error_max  );
        util_ini_parser_.ParseConfig("Lateral Delayed Kinematic Bicycle LMPC", "avg_yaw_error_max",     params_.avg_yaw_error_max       );
        util_ini_parser_.ParseConfig("Vehicle", "wheel_base", params_.vehicle_wheel_base);

        step_num_ = params_.step_num;
        step_dt_ = params_.step_dt;

        cout << "Update Lateral Delayed Kinematic Bicycle LMPC ini" << endl;

    }
}