#include <longitudinal_control/longitudinal_control_core/longitudinal_dynamic_lmpc/longitudinal_dynamic_lmpc.hpp>

LongitudinalDynamicLMPC::LongitudinalDynamicLMPC() :LongitudinalControllerInterface() {   
    std::string dir(getenv("PWD"));
    std::string ini_dir("/config/control.ini");  
    util_ini_parser_.Init((dir+ini_dir).c_str());

    // Parameters initialization
    ProcessINI();
    std::cout << "init LongitudinalDynamicLMPC" << std::endl;
}

LongitudinalDynamicLMPC::~LongitudinalDynamicLMPC() {
}



void LongitudinalDynamicLMPC::BuildControllerVehicleState(const VehicleState& vehicle_state, tk::Map road_map){
    std::cout << "LongitudinalDynamicLMPC : " << (int)live_cnt_++ << std::endl;

    ProcessINI();
    i_vehicle_state_ = vehicle_state;
    i_road_map_      = road_map;
}

ControlCommand LongitudinalDynamicLMPC::CalculateOptimalLongitudinalCommand(const ControlTrajectory& ref){
    ControlCommand control_command;
    if (i_vehicle_state_.vehicle_can.operation_mode != OperationMode::AUTONOMOUS){
        // initialize
        control_command.control_trajectory.control_point.clear();
        control_command.trq = 0.0;
        control_command.ax  = 0.0;
    }
    else{
        vector<State> operation_point = GenerateOperationPoints(ref); 
        vector<OptVariables> opt_vals = SolveHPIPM(operation_point,ref);   

        control_command.control_trajectory.control_point.clear();
        control_command.control_trajectory.header.frame_id = target_frame_;
        for(int i = 0; i<opt_vals.size() ; i++){
            ControlPoint point;
            point.x = ref.control_point[i].x;
            point.y = ref.control_point[i].y;
            point.vx = opt_vals[i].first(0);
            point.ax = opt_vals[i].first(1)/params_.vehicle_mass;
            point.trq = opt_vals[i].first(1)*params_.vehicle_wheel_radius;
            control_command.control_trajectory.control_point.push_back(point);
        }
        if(opt_vals.size() >= 1){
            control_command.ax = opt_vals[params_.shift_input_step].first(1)/params_.vehicle_mass;
            control_command.trq = opt_vals[params_.shift_input_step].first(1)*params_.vehicle_wheel_radius;
        }
        else{
            control_command.ax = 0.0;
            control_command.trq = 0.0;
        }

    }
    return control_command;
}

vector<LongitudinalDynamicLMPC::State> LongitudinalDynamicLMPC::GenerateOperationPoints(const ControlTrajectory& ref){
    vector<LongitudinalDynamicLMPC::State> operation_points;

    if (operation_inputs_.empty()){
        std::cout<<"empty operation input"<<std::endl;
        LongitudinalDynamicLMPC::Input operate_input;
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
        operation_points.push_back(next_state);

    }
    
    return operation_points;
}

LongitudinalDynamicLMPC::State LongitudinalDynamicLMPC::GetReferenceState(const ControlTrajectory& ref, const uint32_t& idx){
    State x_ref;
    x_ref <<    ref.control_point[idx].vx,
                0.0;
    return x_ref;
}


LongitudinalDynamicLMPC::State LongitudinalDynamicLMPC::GetInitState(bool is_prev_state_empty){
    State x0;
    if(is_prev_state_empty){
        x0 << i_vehicle_state_.vx, 0.0;
    }
    else{
        x0 << i_vehicle_state_.vx, optimal_states_.front()(1);
    }
    return x0;
};

LongitudinalDynamicLMPC::State LongitudinalDynamicLMPC::GetContinuousModelDeriv(const State &x,const Input &u, const double &slope)
{
    // MODEL VARIANT
    // v
    // F
    State f;
    f(0) = x(1)/params_.vehicle_mass 
            - params_.gravity*sin(slope) 
            - params_.rolling_resistance*params_.gravity*cos(slope)
            - 0.5*params_.air_drag_coeff*params_.air_density*params_.vehicle_front_area*x(0)*x(0)/params_.vehicle_mass ;
    f(1) = u(0);
    return f;
}

LongitudinalDynamicLMPC::LinModelMatrix LongitudinalDynamicLMPC::GetDiscreteLinModelMatrix(const State &x, const Input &u, const double &slope){
    A_MPC   A_d = A_MPC::Zero();
    B_MPC   B_d = B_MPC::Zero();
    C_MPC   g_d = C_MPC::Zero();
    double drag_force_per_vel = -0.5*params_.air_drag_coeff*params_.air_density*params_.vehicle_front_area*x(0); // Actual drag is proportional to velocity^2
    double drag_force_air = -drag_force_per_vel*x(0);
    double drag_force_gravity = params_.gravity*params_.vehicle_mass*sin(slope);
    double drag_force_rolling = -params_.rolling_resistance*params_.gravity*params_.vehicle_mass*cos(slope);

    A_d << 1.0-step_dt_*drag_force_per_vel/params_.vehicle_mass, step_dt_/params_.vehicle_mass,
           0.0                                         , 1.0;

    B_d <<  0.0,
            step_dt_;

    g_d <<  (step_dt_*(drag_force_gravity+drag_force_air+drag_force_rolling))/params_.vehicle_mass,
            0.0;


    // std::cout << "A_d\n" << A_d << "\n\nB_d\n" << B_d << "\n\nC_d\n" << g_d << std::endl;

    return {A_d,B_d,g_d};
}

vector<LongitudinalDynamicLMPC::OptVariables> LongitudinalDynamicLMPC::SolveHPIPM(const vector<State>& operation_points, const ControlTrajectory& ref){
    
    std::vector<hpipm::OcpQp> qp(step_num_+2);
    
    Eigen::Matrix<double,dim_state_,dim_state_> Q;
    Eigen::Matrix<double,dim_state_,dim_state_> Qf;
    Eigen::Matrix<double,dim_input_,dim_input_> R;
    Eigen::Matrix<double,dim_state_,dim_input_> S;

    // cost
    // MODEL VARIANT //
    Q.setZero();  Q.diagonal()  << params_.weight_v,params_.weight_F;
    Qf.setZero(); Qf.diagonal() << params_.weight_v_end, params_.weight_F_end;
    S.setZero();
    R.setZero();  R.diagonal()  << params_.weight_dF;

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

    // // state constraint
    // for (int i=1; i<=step_num_+1; ++i) {
    //     // v, F
    //     qp[i].idxbx = {1};
    //     qp[i].lbx = (Eigen::VectorXd(1) << T_LL/params_.vehicle_wheel_radius).finished();
    //     qp[i].ubx = (Eigen::VectorXd(1) << T_UL/params_.vehicle_wheel_radius).finished();
    // }

    //input constraint
    for (int i=0; i<step_num_+1; ++i) {
        qp[i].idxbu = {0};
        qp[i].lbu = (Eigen::VectorXd(1) << -params_.max_dF).finished();
        qp[i].ubu = (Eigen::VectorXd(1) << params_.max_dF).finished();

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

    vector<LongitudinalDynamicLMPC::OptVariables> opt_vals;
    if(opt_sol != hpipm::HpipmStatus::Success){
        std::cout << "!!!!!!!!!!!!!!!! CAN'T SOLVE !!!!!!!!!!!!!!!!"<< std::endl;
        for (int i=0; i<step_num_+1; ++i) {
            LongitudinalDynamicLMPC::OptVariables opt_var;
            opt_var.first = optimal_states_[i];
            opt_var.second = operation_inputs_[i];

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
            LongitudinalDynamicLMPC::OptVariables opt_var;
            opt_var.first = solution[i].x;
            opt_var.second = solution[i-1].u;

            opt_vals.push_back(opt_var);           
            optimal_states_.push_back(opt_var.first);
            operation_inputs_.push_back(opt_var.second);
        }
    }
    
    return opt_vals;
}

void LongitudinalDynamicLMPC::ProcessINI() {
    if (util_ini_parser_.IsFileUpdated()){
        util_ini_parser_.ParseConfig("Longitudinal Dynamic LMPC", "step_dt",                params_.step_dt             );
        util_ini_parser_.ParseConfig("Longitudinal Dynamic LMPC", "step_num",               params_.step_num            );
        util_ini_parser_.ParseConfig("Longitudinal Dynamic LMPC", "shift_input_step",       params_.shift_input_step    );
        util_ini_parser_.ParseConfig("Longitudinal Dynamic LMPC", "max_dF",                 params_.max_dF              );
        util_ini_parser_.ParseConfig("Longitudinal Dynamic LMPC", "weight_damping_ratio",   params_.weight_damping_ratio);
        
        util_ini_parser_.ParseConfig("Longitudinal Dynamic LMPC", "weight_v",               params_.weight_v            );
        util_ini_parser_.ParseConfig("Longitudinal Dynamic LMPC", "weight_F",               params_.weight_F            );
        util_ini_parser_.ParseConfig("Longitudinal Dynamic LMPC", "weight_v_end",           params_.weight_v_end        );
        util_ini_parser_.ParseConfig("Longitudinal Dynamic LMPC", "weight_F_end",           params_.weight_F_end        );
        util_ini_parser_.ParseConfig("Longitudinal Dynamic LMPC", "weight_dF",              params_.weight_dF           );
        util_ini_parser_.ParseConfig("Vehicle", "vehicle_mass",         params_.vehicle_mass        );
        util_ini_parser_.ParseConfig("Vehicle", "gravity",              params_.gravity             );
        util_ini_parser_.ParseConfig("Vehicle", "rolling_resistance",   params_.rolling_resistance  );
        util_ini_parser_.ParseConfig("Vehicle", "air_density",          params_.air_density         );
        util_ini_parser_.ParseConfig("Vehicle", "vehicle_front_area",   params_.vehicle_front_area  );
        util_ini_parser_.ParseConfig("Vehicle", "air_drag_coeff",       params_.air_drag_coeff      );
        util_ini_parser_.ParseConfig("Vehicle", "vehicle_wheel_radius", params_.vehicle_wheel_radius);

        step_num_ = params_.step_num;
        step_dt_ = params_.step_dt;

        cout << "Update Longitudinal Dynamic LMPC ini" << endl;

    }
}