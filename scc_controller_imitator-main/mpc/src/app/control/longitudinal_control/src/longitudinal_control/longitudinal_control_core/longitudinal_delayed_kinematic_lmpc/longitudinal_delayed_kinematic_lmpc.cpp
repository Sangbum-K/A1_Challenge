#include <longitudinal_control/longitudinal_control_core/longitudinal_delayed_kinematic_lmpc/longitudinal_delayed_kinematic_lmpc.hpp>
#include <algorithm>   // [scc_imitator fork] std::min for jerk cap

LongitudinalDelayedKinematicLMPC::LongitudinalDelayedKinematicLMPC() :LongitudinalControllerInterface() {   
    std::string dir(getenv("PWD"));
    std::string ini_dir("/config/control.ini");  
    util_ini_parser_.Init((dir+ini_dir).c_str());

    // Parameters initialization
    ProcessINI();
    std::cout << "init LongitudinalDelayedKinematicLMPC" << std::endl;
}

LongitudinalDelayedKinematicLMPC::~LongitudinalDelayedKinematicLMPC() {
}



void LongitudinalDelayedKinematicLMPC::BuildControllerVehicleState(const VehicleState& vehicle_state, tk::Map road_map){
    std::cout << "LongitudinalDelayedKinematicLMPC : " << (int)live_cnt_++ << std::endl;

    ProcessINI();
    i_vehicle_state_ = vehicle_state;
    i_road_map_      = road_map;
}

ControlCommand LongitudinalDelayedKinematicLMPC::CalculateOptimalLongitudinalCommand(const ControlTrajectory& ref){
    ControlCommand control_command;
    if (i_vehicle_state_.vehicle_can.operation_mode != OperationMode::AUTONOMOUS){
        // initialize
        control_command.control_trajectory.control_point.clear();
        control_command.trq = 0.0;
        control_command.ax  = 0.0;
    }
    else{
        if ((int)ref.control_point.size() < step_num_ + 2) {
            std::cerr << "[LongitudinalDelayedKinematicLMPC] ref too short: "
                << ref.control_point.size() << " < " << step_num_ + 2
                << ". Increase min_lookahead_distance in planning.ini." << std::endl;
            control_command.trq = 0.0;
            control_command.ax  = 0.0;
            return control_command;
        }
        vector<State> operation_point = GenerateOperationPoints(ref);
        vector<OptVariables> opt_vals = SolveHPIPM(operation_point,ref);

        control_command.control_trajectory.control_point.clear();
        control_command.control_trajectory.header = ref.header;
        for(int i = 0; i<opt_vals.size() ; i++){
            ControlPoint point = ref.control_point[i];
            point.vx = opt_vals[i].first(0);
            point.ax = opt_vals[i].first(1);
            point.trq = opt_vals[i].first(1)*params_.vehicle_mass*params_.vehicle_wheel_radius;
            control_command.control_trajectory.control_point.push_back(point);
        }
        if(opt_vals.size() >= 1){
            control_command.ax = opt_vals[params_.shift_input_step].first(8);
            control_command.trq = opt_vals[params_.shift_input_step].first(8)*params_.vehicle_mass*params_.vehicle_wheel_radius;
        }
        else{
            control_command.ax = 0.0;
            control_command.trq = 0.0;
        }

    }
    return control_command;
}

vector<LongitudinalDelayedKinematicLMPC::State> LongitudinalDelayedKinematicLMPC::GenerateOperationPoints(const ControlTrajectory& ref){
    vector<LongitudinalDelayedKinematicLMPC::State> operation_points;

    if (operation_inputs_.empty()){
        std::cout<<"empty operation input"<<std::endl;
        LongitudinalDelayedKinematicLMPC::Input operate_input;
        for(uint8_t i=0; i<step_num_+1; i++){
            operate_input.setZero();     
            operation_inputs_.push_back(operate_input);
        } 
    }    

    operation_points.push_back(GetInitState(optimal_states_.empty()));
    for(uint8_t i=0; i<step_num_+1; i++) 
    {
        // MODEL VARIANT
        double slope = -i_vehicle_state_.pitch;
        State curr_state = operation_points.back();
        Input input = operation_inputs_[i];
        State state_derive = GetContinuousModelDeriv(curr_state , input, slope);
        State next_state = curr_state + state_derive * step_dt_;

        next_state(2) = curr_state(3);
        next_state(3) = curr_state(4);
        next_state(4) = curr_state(5);
        next_state(5) = curr_state(6);
        next_state(6) = curr_state(7);
        next_state(7) = curr_state(8);

        operation_points.push_back(next_state);

    }
    
    return operation_points;
}

LongitudinalDelayedKinematicLMPC::State LongitudinalDelayedKinematicLMPC::GetReferenceState(const ControlTrajectory& ref, const uint32_t& idx){
    State x_ref;
    x_ref <<    ref.control_point[idx].vx,
                ref.control_point[idx].ax,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0;
    return x_ref;
}


LongitudinalDelayedKinematicLMPC::State LongitudinalDelayedKinematicLMPC::GetInitState(bool is_prev_state_empty){
    State x0;
    if(is_prev_state_empty){
        x0 << i_vehicle_state_.vx,
                i_vehicle_state_.ax,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0;
    }
    else{
        x0 << i_vehicle_state_.vx, 
                i_vehicle_state_.ax,
                optimal_states_.front()(2),
                optimal_states_.front()(3),
                optimal_states_.front()(4),
                optimal_states_.front()(5),
                optimal_states_.front()(6),
                optimal_states_.front()(7),
                optimal_states_.front()(8);
    }
    return x0;
};

LongitudinalDelayedKinematicLMPC::State LongitudinalDelayedKinematicLMPC::GetContinuousModelDeriv(const State &x,const Input &u, const double &slope)
{
    // MODEL VARIANT
    // v
    // a
    State f;
    f(0) = x(1);
    f(1) = (-x(1) + x(2))/params_.time_constant;
    f(2) = 0;
    f(3) = 0;
    f(4) = 0;
    f(5) = 0;
    f(6) = 0;
    f(7) = 0;
    f(8) = u(0);
    
    return f;
}

LongitudinalDelayedKinematicLMPC::LinModelMatrix LongitudinalDelayedKinematicLMPC::GetDiscreteLinModelMatrix(const State &x, const Input &u, const double &slope){

    A_MPC   A_d = A_MPC::Zero();
    B_MPC   B_d = B_MPC::Zero();
    C_MPC   C_d = C_MPC::Zero();


    A_d << 1.0 , step_dt_                          ,                   0.0,                0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
           0.0 , 1.0-step_dt_/params_.time_constant,       step_dt_/params_.time_constant, 0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
           0.0 , 0.0                               ,                   0.0,                1.0,    0.0,    0.0,    0.0,    0.0,    0.0,
           0.0 , 0.0                               ,                   0.0,                0.0,    1.0,    0.0,    0.0,    0.0,    0.0,
           0.0 , 0.0                               ,                   0.0,                0.0,    0.0,    1.0,    0.0,    0.0,    0.0,
           0.0 , 0.0                               ,                   0.0,                0.0,    0.0,    0.0,    1.0,    0.0,    0.0,
           0.0 , 0.0                               ,                   0.0,                0.0,    0.0,    0.0,    0.0,    1.0,    0.0,
           0.0 , 0.0                               ,                   0.0,                0.0,    0.0,    0.0,    0.0,    0.0,    1.0,
           0.0 , 0.0                               ,                   0.0,                0.0,    0.0,    0.0,    0.0,    0.0,    1.0;
    B_d <<  0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            step_dt_;

    return {A_d,B_d,C_d};
}


vector<LongitudinalDelayedKinematicLMPC::OptVariables> LongitudinalDelayedKinematicLMPC::SolveHPIPM(const vector<State>& operation_points, const ControlTrajectory& ref){
    
    std::vector<hpipm::OcpQp> qp(step_num_+2);
    
    Eigen::Matrix<double,dim_state_,dim_state_> Q;
    Eigen::Matrix<double,dim_state_,dim_state_> Qf;
    Eigen::Matrix<double,dim_input_,dim_input_> R;
    Eigen::Matrix<double,dim_state_,dim_input_> S;

    // cost
    // MODEL VARIANT //
    Q.setZero();  Q.diagonal()  << params_.weight_v,params_.weight_a , 0 ,0 ,0 ,0, 0, 0, 0;
    Qf.setZero(); Qf.diagonal() << params_.weight_v_end, params_.weight_a_end, 0 ,0 ,0 ,0, 0, 0, 0;
    S.setZero();
    R.setZero();  R.diagonal()  << params_.weight_jerk;

    Eigen::VectorXd x_ref(dim_state_);
    const Eigen::VectorXd r = Eigen::VectorXd::Zero(dim_input_);
    for (int i=0; i<step_num_+1; ++i) {
        x_ref = GetReferenceState(ref,i); 

        qp[i].Q = Q.transpose();
        qp[i].R = R.transpose();
        qp[i].S = S.transpose();
        qp[i].q = (- Q * x_ref).transpose();
        qp[i].r = r;
        Q *= params_.weight_damping_ratio;

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
        double slope = -i_vehicle_state_.pitch;
        const auto [a_d, b_d, c_d] = GetDiscreteLinModelMatrix(operation_points.at(i),operation_inputs_.at(i), slope);
        qp[i].A = a_d;
        qp[i].B = b_d;
        qp[i].b = c_d;
    }

    // state constraint — ax (state index 1) 의 절대값 제한
    // [scc_controller_imitator fork v3] state ax bound 활성화 + jerk cap.
    // ROS 는 warm-chain 이라 누적 안 됨. synth_batch 는 cold single-solve →
    // 이전 ±171 m/s² 출력 → 5 m/s² 한정.
    constexpr double kMaxAxAbs = 20.0;       // m/s² (user-spec ±20, 차량 물리 brake 한계)
    constexpr double kMaxJerkAbs = 100.0;    // m/s³ (user-spec ±100). config max_jerk(=1000) 와 min()
    for (int i=1; i<=step_num_+1; ++i) {
        qp[i].idxbx = {1};
        qp[i].lbx = (Eigen::VectorXd(1) << -kMaxAxAbs).finished();
        qp[i].ubx = (Eigen::VectorXd(1) <<  kMaxAxAbs).finished();
    }

    //input constraint (jerk)
    const double j_lim = std::min(static_cast<double>(params_.max_jerk), kMaxJerkAbs);
    for (int i=0; i<step_num_+1; ++i) {
        qp[i].idxbu = {0};
        qp[i].lbu = (Eigen::VectorXd(1) << -j_lim).finished();
        qp[i].ubu = (Eigen::VectorXd(1) <<  j_lim).finished();
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
   
    vector<LongitudinalDelayedKinematicLMPC::OptVariables> opt_vals;
    if(opt_sol != hpipm::HpipmStatus::Success){
        std::cout << "!!!!!!!!!!!!!!!! CAN'T SOLVE !!!!!!!!!!!!!!!!"<< std::endl;
        for (int i=0; i<step_num_+1; ++i) {
            LongitudinalDelayedKinematicLMPC::OptVariables opt_var;
            opt_var.first  = (i < (int)optimal_states_.size())    ? optimal_states_[i]    : State::Zero();
            opt_var.second = (i < (int)operation_inputs_.size())  ? operation_inputs_[i]  : Input::Zero();

            opt_vals.push_back(opt_var);
        }
        optimal_states_.clear();
        operation_inputs_.clear();
        return opt_vals;
    }
    else{
        optimal_states_.clear();
        operation_inputs_.clear();
        for (int i=1; i<step_num_+2; ++i) {
            LongitudinalDelayedKinematicLMPC::OptVariables opt_var;
            opt_var.first = solution[i].x;
            opt_var.second = solution[i-1].u;

            opt_vals.push_back(opt_var);           
            optimal_states_.push_back(opt_var.first);
            operation_inputs_.push_back(opt_var.second);
        }
    }
    
    return opt_vals;
}

void LongitudinalDelayedKinematicLMPC::ProcessINI() {
    if (util_ini_parser_.IsFileUpdated()){
        util_ini_parser_.ParseConfig("Longitudinal Delayed Kinematic LMPC", "step_dt",                params_.step_dt             );
        util_ini_parser_.ParseConfig("Longitudinal Delayed Kinematic LMPC", "step_num",               params_.step_num            );
        util_ini_parser_.ParseConfig("Longitudinal Delayed Kinematic LMPC", "shift_input_step",       params_.shift_input_step    );
        util_ini_parser_.ParseConfig("Longitudinal Delayed Kinematic LMPC", "time_constant",          params_.time_constant       );
        util_ini_parser_.ParseConfig("Longitudinal Delayed Kinematic LMPC", "max_jerk",                 params_.max_jerk              );
        util_ini_parser_.ParseConfig("Longitudinal Delayed Kinematic LMPC", "weight_damping_ratio",   params_.weight_damping_ratio);
        
        util_ini_parser_.ParseConfig("Longitudinal Delayed Kinematic LMPC", "weight_v",               params_.weight_v            );
        util_ini_parser_.ParseConfig("Longitudinal Delayed Kinematic LMPC", "weight_a",               params_.weight_a            );
        util_ini_parser_.ParseConfig("Longitudinal Delayed Kinematic LMPC", "weight_v_end",           params_.weight_v_end        );
        util_ini_parser_.ParseConfig("Longitudinal Delayed Kinematic LMPC", "weight_a_end",           params_.weight_a_end        );
        util_ini_parser_.ParseConfig("Longitudinal Delayed Kinematic LMPC", "weight_jerk",            params_.weight_jerk           );
        util_ini_parser_.ParseConfig("Vehicle", "vehicle_mass",         params_.vehicle_mass        );
        util_ini_parser_.ParseConfig("Vehicle", "vehicle_wheel_radius", params_.vehicle_wheel_radius);

        step_num_ = params_.step_num;
        step_dt_ = params_.step_dt;

        cout << "Update Longitudinal Delayed Kinematic LMPC ini" << endl;
    }
}