#include <lateral_control/lateral_control_core/lateral_kinematic_bicycle_lmpc/lateral_kinematic_bicycle_lmpc.hpp>

LateralKinematicBicycleLMPC::LateralKinematicBicycleLMPC() :LateralControllerInterface() {
    const char* pwd_env = getenv("PWD");
    if (pwd_env == nullptr) {
        std::cerr << "[LateralKinematicBicycleLMPC] PWD environment variable is not set. Cannot load config." << std::endl;
        std::exit(1);
    }
    std::string dir(pwd_env);
    std::string ini_dir("/config/control.ini");
    util_ini_parser_.Init((dir+ini_dir).c_str());

    // Parameters initialization
    ProcessINI();
    std::cout << "[LateralKinematicBicycleLMPC] Initialized." << std::endl;
}

LateralKinematicBicycleLMPC::~LateralKinematicBicycleLMPC() {
}



void LateralKinematicBicycleLMPC::BuildControllerVehicleState(const VehicleState& vehicle_state, tk::Map road_map){
    ProcessINI();
    i_vehicle_state_ = vehicle_state;
    i_road_map_      = road_map;
}

boost::optional<ControlCommand> LateralKinematicBicycleLMPC::CalculateOptimalTireSteering(const ControlTrajectory& ref){
    ControlCommand control_command;
    // if (i_vehicle_state_.vehicle_can.operation_mode != OperationMode::AUTONOMOUS){
    //     // initialize
    //     control_command.control_trajectory.control_point.clear();
    //     control_command.steering_tire_angle = i_vehicle_state_.vehicle_can.steering_tire_angle*RAD2DEG;
    // }
    // else{
        vector<State> operation_point = GenerateOperationPoints(ref); 
        vector<OptVariables> opt_vals = SolveHPIPM(operation_point,ref);   

        control_command.control_trajectory.control_point.clear();
        control_command.control_trajectory.header = ref.header;
        for(int i = 0; i<opt_vals.size() ; i++){
            ControlPoint point = {opt_vals[i].first(0), opt_vals[i].first(1), opt_vals[i].first(2),  0};
            point.steer = opt_vals[i].first(3)*RAD2DEG;
            point.dsteer = opt_vals[i].second(0)*RAD2DEG;

            control_command.control_trajectory.control_point.push_back(point);
        }
        if(opt_vals.size() >= 1){
            control_command.steering_tire_angle = opt_vals[params_.shift_input_step].first(3)*RAD2DEG;
        }
        else{
            control_command.steering_tire_angle = 0.0;
        }
        CheckSolvedStateTwisted(ref, control_command);

    // }

    if(is_solved_ == true){
        return control_command;
    }
    else{
        return boost::none;
    }
}

vector<LateralKinematicBicycleLMPC::State> LateralKinematicBicycleLMPC::GenerateOperationPoints(const ControlTrajectory& ref){
    vector<LateralKinematicBicycleLMPC::State> operation_points;

    if (operation_inputs_.empty()){
        std::cout<<"empty operation input"<<std::endl;
        LateralKinematicBicycleLMPC::Input operate_input;
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
        operation_points.push_back(next_state);
    }
    
    return operation_points;
}

LateralKinematicBicycleLMPC::State LateralKinematicBicycleLMPC::GetReferenceState(const ControlTrajectory& ref, const uint32_t& idx){
    State x_ref;
    x_ref <<    ref.control_point[idx].x,
                ref.control_point[idx].y,
                ref.control_point[idx].yaw,
                0.0;
    return x_ref;
}


LateralKinematicBicycleLMPC::State LateralKinematicBicycleLMPC::GetInitState(bool is_prev_state_empty){
    State x0;
    if(is_prev_state_empty){
        x0 << 0.0, 0.0, 0.0, 0.0;
    }
    else{
        x0 << 0.0, 0.0, 0.0, optimal_states_.front()(3);
    }
    return x0;
};

LateralKinematicBicycleLMPC::State LateralKinematicBicycleLMPC::GetContinuousModelDeriv(const State &x,const Input &u, const double &v)
{
    // MODEL VARIANT
    State f;
    f(0) = v*std::cos(x(2));
    f(1) = v*std::sin(x(2));
    f(2) = v*std::tan(x(3))/params_.vehicle_wheel_base;
    f(3) = u(0);
    return f;
}

LateralKinematicBicycleLMPC::LinModelMatrix LateralKinematicBicycleLMPC::GetDiscreteLinModelMatrix(const State &x, const Input &u, const double &v){

    const double phi    = x(2);
    const double delta  = x(3);

    A_MPC   A_c = A_MPC::Zero();
    B_MPC   B_c = B_MPC::Zero();
    C_MPC   C_c = C_MPC::Zero();

    const State f = GetContinuousModelDeriv(x,u,v);
    A_c(0,2) = -v*sin(phi);
    A_c(1,2) =  v*cos(phi);
    A_c(2,3) = (v*(pow(tan(delta),2.0)+1.0))/params_.vehicle_wheel_base;
    B_c(3,0) = 1.0;
    C_c = f - A_c*x - B_c*u;

    // disctetize the continuous time linear model \dot x = A x + B u + g using ZHO
    Eigen::Matrix<double,dim_state_+dim_input_+1,dim_state_+dim_input_+1> temp = Eigen::Matrix<double,dim_state_+dim_input_+1,dim_state_+dim_input_+1>::Zero();
    // building matrix necessary for expm
    // temp = Ts*[A,B,g;zeros]
    
    temp.block<dim_state_,dim_state_>(0,0)            = A_c;
    temp.block<dim_state_,dim_input_>(0,dim_state_)   = B_c;
    temp.block<dim_state_,1>(0,dim_state_+dim_input_) = C_c;

    temp = temp*step_dt_;
    // std::cout << "temp\n" << temp << std::endl;
    // take the matrix exponential of temp
    const Eigen::Matrix<double,dim_state_+dim_input_+1,dim_state_+dim_input_+1> temp_res = temp.exp();
    // std::cout << "temp_res\n" << temp_res << std::endl;
    // extract dynamics out of big matrix
    // x_{k+1} = Ad x_k + Bd u_k + gd
    //temp_res = [Ad,Bd,gd;zeros]
    const A_MPC A_d = temp_res.block<dim_state_,dim_state_>(0,0);
    const B_MPC B_d = temp_res.block<dim_state_,dim_input_>(0,dim_state_);
    const C_MPC g_d = temp_res.block<dim_state_,1>(0,dim_state_+dim_input_);
    
    // std::cout << "A_d\n" << A_d << "\n\nB_d\n" << B_d << "\n\nC_d\n" << g_d << std::endl;

    return {A_d,B_d,g_d};
}


vector<LateralKinematicBicycleLMPC::OptVariables> LateralKinematicBicycleLMPC::SolveHPIPM(const vector<State>& operation_points, const ControlTrajectory& ref){
    
    std::vector<hpipm::OcpQp> qp(step_num_+2);
    
    Eigen::Matrix<double,dim_state_,dim_state_> Q;
    Eigen::Matrix<double,dim_state_,dim_state_> Qf;
    Eigen::Matrix<double,dim_input_,dim_input_> R;
    Eigen::Matrix<double,dim_state_,dim_input_> S;

    // cost
    // MODEL VARIANT //
    Q.setZero();  Q.diagonal()  << params_.weight_x,params_.weight_y, params_.weight_yaw, params_.weight_steer;
    Qf.setZero(); Qf.diagonal() << params_.weight_x_end, params_.weight_y_end,params_.weight_yaw_end, params_.weight_steer_end;
    S.setZero();
    R.setZero();  R.diagonal()  << params_.weight_dsteer;

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
        qp[step_num_+1].Q = Qf.transpose();
        qp[step_num_+1].q = (- Qf * x_ref).transpose();

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
        qp[i].idxbx = {3};
        qp[i].lbx = (Eigen::VectorXd(1) << -MAX_STEERING_TIRE_ANGLE*DEG2RAD).finished();
        qp[i].ubx = (Eigen::VectorXd(1) <<  MAX_STEERING_TIRE_ANGLE*DEG2RAD).finished();
    }

    //input constraint
    for (int i=0; i<step_num_+1; ++i) {
        qp[i].idxbu = {0};
        qp[i].lbu = (Eigen::VectorXd(1) << -MAX_STEERING_TIRE_ANGLE*DEG2RAD).finished();
        qp[i].ubu = (Eigen::VectorXd(1) <<  MAX_STEERING_TIRE_ANGLE*DEG2RAD).finished();
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

    auto t_solve_start = std::chrono::high_resolution_clock::now();
    auto opt_sol = solver.solve(x0, qp, solution);
    auto t_solve_end = std::chrono::high_resolution_clock::now();
    last_solve_time_ms_ = std::chrono::duration_cast<std::chrono::microseconds>(t_solve_end - t_solve_start).count() * 0.001;

    vector<LateralKinematicBicycleLMPC::OptVariables> opt_vals;
    if(opt_sol != hpipm::HpipmStatus::Success){
        std::cerr << "[LateralKinematicBicycleLMPC] SOLVER FAILED - using previous solution." << std::endl;
        for (int i=0; i<step_num_+1; ++i) {
            LateralKinematicBicycleLMPC::OptVariables opt_var;
            opt_var.first = optimal_states_[i];
            opt_var.second = operation_inputs_[i];

            opt_vals.push_back(opt_var);           
        }
        optimal_states_.clear();
        operation_inputs_.clear();
        return opt_vals;
    }
    else{
        if (opt_sol == hpipm::HpipmStatus::Success) is_solved_ = true;
        optimal_states_.clear();
        operation_inputs_.clear();
        for (int i=1; i<step_num_+2; ++i) {
            LateralKinematicBicycleLMPC::OptVariables opt_var;
            opt_var.first = solution[i].x;
            opt_var.second = solution[i-1].u;

            opt_vals.push_back(opt_var);           
            optimal_states_.push_back(opt_var.first);
            operation_inputs_.push_back(opt_var.second);
        }
    }
    
    return opt_vals;
}

void LateralKinematicBicycleLMPC::CheckSolvedStateTwisted(const ControlTrajectory& ref, const ControlCommand& control_command){
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
        std::cerr << "[LateralKinematicBicycleLMPC] Solution diverged (twisted), resetting solver." << std::endl;
        is_solved_ = false;
        optimal_states_.clear();
        operation_inputs_.clear();
    }

    static double prev_timestamp = i_vehicle_state_.header.stamp;
    double curr_timestamp = i_vehicle_state_.header.stamp;

    if(curr_timestamp < prev_timestamp){
        std::cerr << "[LateralKinematicBicycleLMPC] Timestamp went backward, resetting solver." << std::endl;
        is_solved_ = false;
        optimal_states_.clear();
        operation_inputs_.clear();
    }

    prev_timestamp = curr_timestamp;

}

void LateralKinematicBicycleLMPC::ProcessINI() {
    if (util_ini_parser_.IsFileUpdated()){
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "step_dt",               params_.step_dt             );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "step_num",              params_.step_num            );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "shift_input_step",      params_.shift_input_step    );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "weight_x",              params_.weight_x            );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "weight_y",              params_.weight_y            );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "weight_yaw",            params_.weight_yaw          );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "weight_steer",          params_.weight_steer        );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "weight_x_end",          params_.weight_x_end        );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "weight_y_end",          params_.weight_y_end        );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "weight_yaw_end",        params_.weight_yaw_end      );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "weight_steer_end",      params_.weight_steer_end    );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "weight_dsteer",         params_.weight_dsteer       );

        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "avg_distance_error_max",params_.avg_distance_error_max  );
        util_ini_parser_.ParseConfig("Lateral Kinematic Bicycle LMPC", "avg_yaw_error_max",     params_.avg_yaw_error_max       );
        util_ini_parser_.ParseConfig("Vehicle", "wheel_base", params_.vehicle_wheel_base);

        step_num_ = params_.step_num;
        step_dt_ = params_.step_dt;

        cout << "Update Lateral Kinematic Bicycle LMPC ini" << endl;

    }
}