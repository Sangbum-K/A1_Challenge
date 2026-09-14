from math import cos, sqrt
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import numpy as np
from casadi import *
import casadi as cs
from casadi import fabs

# constants
lf = 1.4    # 전축에서 무게중심까지의 거리 [m]
lr = 0.95    # 후축에서 무게중심까지의 거리 [m]
L = lf + lr # 차량의 휠베이스 [m]


IDX_X_X = 0
IDX_X_Y = 1
IDX_X_YAW = 2
IDX_X_SPEED = 3
IDX_X_DELTA = 4
IDX_X_DELTA1 = 5
IDX_X_DELTA2 = 6
IDX_X_DELTA3 = 7
IDX_X_DELTA4 = 8
IDX_X_DELTA5 = 9
IDX_X_DELTA6 = 10
IDX_X_ACC = 11
IDX_X_A1 = 12
IDX_X_A2 = 13
IDX_X_A3 = 14
IDX_X_A4 = 15
IDX_X_A5 = 16
IDX_X_A6 = 17
IDX_X_S = 18
IDX_U_DD = 0
IDX_U_JERK = 1

def ode_model(ego_frame_rear) -> AcadosModel:
    model_name = 'integrate_kinematic_bicycle_nmpc'
    # set up states: [x, y, yaw, speed, delta, delta1, delta2, delta3, ddelta, a, a1, a2, a3]
    x       = SX.sym('x')
    y       = SX.sym('y')
    yaw     = SX.sym('yaw')
    speed   = SX.sym('speed')
    delta   = SX.sym('delta')
    delta1  = SX.sym('delta1')
    delta2  = SX.sym('delta2')
    delta3  = SX.sym('delta3')
    delta4  = SX.sym('delta4')
    delta5  = SX.sym('delta5')
    delta6  = SX.sym('delta6')
    a       = SX.sym('a')
    a1      = SX.sym('a1')
    a2      = SX.sym('a2')
    a3      = SX.sym('a3')
    a4      = SX.sym('a4')
    a5      = SX.sym('a5')
    a6      = SX.sym('a6')
    s       = SX.sym('s')
    state = vertcat(x, y, yaw, speed, delta, delta1, delta2, delta3, delta4, delta5, delta6, a, a1, a2, a3, a4, a5, a6, s)

    # set up control inputs: [ddelta, jerk]
    input_ddelta = SX.sym('input_ddelta')
    input_jerk = SX.sym('input_jerk')
    input = vertcat(input_ddelta, input_jerk)

    # 적분 시각
    dt = SX.sym('dt')

    # 시상수
    lag_tau = SX.sym('lag_tau')

    # 진행방향과 차이
    x_ref = SX.sym('x_ref')
    y_ref = SX.sym('y_ref')
    yaw_ref = SX.sym('yaw_ref')
    kappa = SX.sym('kappa')
    dx = x - x_ref
    dy = y - y_ref
    mu = acos(cos(yaw - yaw_ref))
    n = dx * sin(-yaw_ref) + dy * cos(-yaw_ref)

    # states derivatives (continuous time dynamics)
    # beta = atan2(lr*tan(delta),L)
    if ego_frame_rear == 1:
        beta = 0.0
    else:
        beta = atan2(lr*tan(delta),L)
    
    # xdot: [x_dot, y_dot, yaw_dot, speed_dot, delta_dot, a_dot]
    # states derivatives (continuous time dynamics)
    x_dot    = speed * cos(yaw + beta)
    y_dot    = speed * sin(yaw + beta)
    yaw_dot  = speed * tan(delta)/L
    speed_dot = a
    delta_dot = (delta1 - delta)/lag_tau
    delta1_dot = (delta2 - delta1)/dt
    delta2_dot = (delta3 - delta2)/dt
    delta3_dot = (delta4 - delta3)/dt
    delta4_dot = (delta5 - delta4)/dt
    delta5_dot = (delta6 - delta5)/dt
    delta6_dot = input_ddelta
    a_dot     = (a1 - a)/lag_tau
    a1_dot    = (a2 - a1)/dt
    a2_dot    = (a3 - a2)/dt
    a3_dot    = (a4 - a3)/dt
    a4_dot    = (a5 - a4)/dt
    a5_dot    = (a6 - a5)/dt
    a6_dot    = input_jerk
    s_dot = speed / cs.if_else(1 - n * kappa < 1e-6, 1e-6, 1 - n * kappa) # kappa_p와 n 필요
    # s_dot = speed * cos(mu)
    # s_dot = speed
    
    xdot = vertcat(
        x_dot,
        y_dot,
        yaw_dot,
        speed_dot,
        delta_dot,
        delta1_dot,
        delta2_dot,
        delta3_dot,
        delta4_dot,
        delta5_dot,
        delta6_dot,
        a_dot,
        a1_dot,
        a2_dot,
        a3_dot,
        a4_dot,
        a5_dot,
        a6_dot,
        s_dot
        )

    model = AcadosModel()
    model.f_expl_expr = xdot
    model.x = state
    model.u = input
    model.name = model_name
    model.n_states = state.size()[0]
    model.n_inputs = input.size()[0]
    model.lag_tau = lag_tau
    model.beta = beta
    model.dt = dt
    model.x_ref = x_ref
    model.y_ref = y_ref
    model.yaw_ref = yaw_ref
    model.kappa = kappa
    model.mu = mu
    model.n = n
    return model

def pseudo_huber_loss(x, delta=1.0):
    # delta: smoothing parameter
    return delta * delta * (cs.sqrt(1 + (x/delta)**2) - 1)

def acados_settings(build=True, num_step=40, final_time=4, ego_frame_rear=1, shooting_nodes=None):
    N = num_step
    Tf = final_time
    
    ocp = AcadosOcp()
    # set model
    model = ode_model(ego_frame_rear)

    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    
    # parameters
    x_ref = model.x_ref
    y_ref = model.y_ref
    yaw_ref = model.yaw_ref
    speed_ref = SX.sym('speed_ref')
    kappa = model.kappa
    max_speed = SX.sym('max_speed')
    ax_max = SX.sym('ax_max')
    ax_min = SX.sym('ax_min')
    ay_max = SX.sym('ay_max')
    super_ellipse_exponent = SX.sym('super_ellipse_exponent')

    s_w = SX.sym('s_w')
    d_w = SX.sym('d_w')
    d_dot_w = SX.sym('d_dot_w')
    mu_w = SX.sym('mu_w')
    speed_w = SX.sym('speed_w')
    delta_w = SX.sym('delta_w')
    accel_w = SX.sym('accel_w')
    ddelta_w = SX.sym('ddelta_w')
    jerk_w = SX.sym('jerk_w')
    obj_w = SX.sym('obj_w')
    dddelta_w = SX.sym('dddelta_w')

    obj0 = SX.sym("obj0",6) # [s_ofront, n_ofront, s_ocenter, n_ocenter, s_orear, n_orear]
    thw = SX.sym("thw")

    dt = model.dt
    lag_tau = model.lag_tau
    mu = model.mu
    xdot = model.f_expl_expr
    p = vertcat(x_ref,y_ref,yaw_ref,speed_ref,
                ax_max,ax_min,ay_max,super_ellipse_exponent,
                kappa,
                s_w,d_w,d_dot_w,mu_w,speed_w,delta_w,accel_w,ddelta_w,jerk_w,obj_w, dddelta_w, #16
                obj0, thw, # 6, 1
                dt,
                lag_tau # 1
                )
    
    ocp.model.p = p

    # Dynamically determine the number of parameters
    n_params = ocp.model.p.size()[0]
    ocp.parameter_values = np.zeros(n_params)

    # path cost
    x = ocp.model.x # [s, d, mu, v, delta, delta1, delta2, delta3, a, a1, a2, a3]
    u = ocp.model.u # [ddelta,jerk]

    ## Cost function 
    # State cost
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    going_cost = cs.sqrt(cs.fabs(speed_ref - x[IDX_X_SPEED]))
    normalized_yaw_cost = mu

    n = model.n
    n_cost = cs.sqrt(pseudo_huber_loss(n, 1.0))
    d_dot_cost = x[IDX_X_SPEED] * sin(mu)
    speed_cost = x[IDX_X_SPEED]
        
    delta_cost = x[IDX_X_DELTA]
    accel_cost = x[IDX_X_ACC]
    ddelta_1 = (x[IDX_X_DELTA1] - x[IDX_X_DELTA])/dt
    ddelta_2 = (x[IDX_X_DELTA2] - x[IDX_X_DELTA1])/dt
    dddelta_cost = (ddelta_1 - ddelta_2)/dt
    cost_x_expr = vertcat(going_cost, n_cost, speed_cost, normalized_yaw_cost, delta_cost, accel_cost, d_dot_cost, dddelta_cost) #
    # cost_x_expr = vertcat(n_cost, speed_cost, normalized_yaw_cost, delta_cost, accel_cost, d_dot_cost) #
    Q_mat = 2*np.diag([s_w, d_w, speed_w, mu_w, delta_w, accel_w, d_dot_w, dddelta_w]) #, delta_w])
    # Q_mat = 2*np.diag([d_w, speed_w, mu_w, delta_w, accel_w, d_dot_w]) #, delta_w])
    R_mat = 2*np.diag([ddelta_w, jerk_w])

    ep_going_cost = cs.sqrt((x_ref - x[IDX_X_X])**2 + (y_ref - x[IDX_X_Y])**2)
    
    # Cost function apply
    ocp.model.cost_expr_ext_cost = 0.5 * (cost_x_expr.T @ Q_mat @ cost_x_expr) \
                                    + 0.5 * (u.T @ R_mat @ u)
    ocp.model.cost_expr_ext_cost_e = 0.5 * (cost_x_expr.T @ Q_mat @ cost_x_expr)# + 0.5 * (ep_going_cost.T @ s_w @ ep_going_cost)


    ## set constraints
    # State constraints
    delta_max = 0.61 # ~ 35 deg
    ocp.constraints.lbx = np.array([-1.0,-delta_max])
    ocp.constraints.ubx = np.array([14.72,+delta_max])
    ocp.constraints.idxbx = np.array([IDX_X_SPEED, IDX_X_DELTA]) # [v, delta]
    # nbx = ocp.constraints.idxbx.shape[0]
    # ocp.constraints.Jsbx  = np.eye(nbx) # If enable, it will be soft constraint
    
    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx_e = ocp.constraints.idxbx
    # nbx = ocp.constraints.idxbx.shape[0]
    # ocp.constraints.Jsbx_e  = np.eye(nbx) # If enable, it will be soft constraint

    # Control input constraints
    ddelta_max = 0.6
    jerk_max = 5
    ocp.constraints.lbu = np.array([-ddelta_max, -jerk_max])
    ocp.constraints.ubu = np.array([+ddelta_max, +jerk_max])
    ocp.constraints.idxbu = np.array([IDX_U_DD, IDX_U_JERK]) # [ddelta, jerk]
    # nbu = ocp.constraints.idxbu.shape[0]
    # ocp.constraints.Jsbu  = np.eye(nbu) # If enable, it will be soft constraint

    ### Nonlinear constraints
    ## Lane boundary constraint
    bound = n_cost

    ## lateral accel and jerk constraint
    v = x[IDX_X_SPEED]
    tan_delta = tan(x[IDX_X_DELTA])
    # a_lat = v**2 * tan_delta / L # kinematic model
    a_lat = v**2 * kappa # Road based model
    j_lat = (2*v*x[IDX_X_ACC]*tan_delta + v**2*(1 + tan_delta**2)*u[IDX_U_DD])/L

    # Reference speed for lateral accel constraint
    v_cost = x[IDX_X_SPEED]

    ## object constraint. SCC condition and Constant velocity with time headway
    # if you want to follow front vehicle only 
    s_front = x[IDX_X_S] + L # *cos(x[IDX_X_YAW]) # Simplified # Ego vehicle front bumper position (actualy it is front axel position)

    # safety distance
    safe_dist = 5.0

    # Distance from front bumper to object[front, center, rear]
    shortest_obj_s = cs.fmin(obj0[0], cs.fmin(obj0[2], obj0[4]))
    # If shortest_obj_s is negative, set it to 1e7. It means that the object is behind the ego vehicle.
    shortest_obj_s = cs.if_else(shortest_obj_s < 0, 1e7, shortest_obj_s)

    # Diffrent of distance between object and ego vehicle
    need_to_go_for_thw = shortest_obj_s - s_front - safe_dist

    # Carculate accelration for SCC
    distance_margin = need_to_go_for_thw-v*thw
    
    ## Friction circle constraint
    # Set friction circle constraint with super-ellipse form
    ax = x[IDX_X_ACC]  # Longitudinal acceleration
    ay = cs.fabs(a_lat)  # Lateral acceleration
    
    # Maximum acceleration limit
    ax_max_acc = fabs(ax_max)   # Acceleration limit (m/s^2)
    ax_max_dec = fabs(ax_min)   # Deceleration limit (m/s^2)
    # ax_max_dec = -15   # Deceleration limit (m/s^2)
    ay_max = fabs(ay_max)       # Maximum lateral acceleration (m/s^2)
    
    # 초타원 지수
    n = fabs(super_ellipse_exponent) # Super-ellipse exponent: 0.6 is astroid, 1.0 is diamond, 2.0 is circle, 4.0 is square

    # ax의 절대값을 최대 가속도로 정규화 (가속/감속에 따라 다르게 적용)
    normalized_ax = if_else(ax >= 0,
                          fabs(ax/ax_max_acc),  # Acceleration
                          fabs(ax/ax_max_dec))  # Deceleration
    normalized_ay = fabs(ay/ay_max)
    
    # 초타원 제약조건
    friction_circle_constraint_square = cs.fabs(normalized_ax**n + normalized_ay**n) # < 1
    # friction_circle_constraint = 0.1 # < 1

    ### Nonlinear constraints apply
    # Set constraints
    ocp.model.con_h_expr = vertcat(
                                   bound, 
                                   v_cost, # a_lat, 
                                   j_lat,
                                   distance_margin, # SCC condition
                                   friction_circle_constraint_square # Friction circle constraint
                                   )
    # Upper bound of constraints
    ocp.constraints.lh    = np.array([
                                    -1.0,
                                    0.0, #-10.5, 
                                    -10,
                                    0.0,
                                    -1.0 # unbounded friction_circle_constraint is always positive
                                    ]) 
    # Lower bound of constraints
    ocp.constraints.uh    = np.array([
                                    1.0,
                                    1e9, #10.5, 
                                    10,
                                    1e9,
                                    1.0 # Circle constraint
                                      ]) 
    # Set nonlinear constraint as soft constraint
    nh = ocp.model.con_h_expr.shape[0]
    ocp.constraints.Jsh = np.eye(nh)

    ## End point constraints
    ocp.model.con_h_expr_e = vertcat(
                                   bound,
                                   v_cost, # a_lat,
                                   distance_margin,
                                   friction_circle_constraint_square
                                   )
    # Upper bound of constraints
    ocp.constraints.lh_e = np.array([
                                    -1.0,
                                    0.0,
                                    0.0,
                                    -1.0
                                    ])
    # Lower bound of constraints
    ocp.constraints.uh_e = np.array([
                                    1.0,
                                    1e9,
                                    1e9, # distance_margin is always positive
                                    1.0
                                      ])
    # Set nonlinear constraint as soft constraint
    nh_e = ocp.model.con_h_expr_e.shape[0]
    ocp.constraints.Jsh_e = np.eye(nh_e)

    ## Nonlinear constraint penalty
    z_1 = np.array([
                    1e3, # bound
                    1e3, # v_cost
                    1e3, # j_lat
                    1e3, # distance_margin
                    1e6, # friction_circle_constraint_square
                    ])
    z_2 = np.array([
                    1e5, # lane boundary constraint
                    1e5, # v_cost
                    1e5, # j_lat
                    1e6, # distance_margin
                    1e1, # friction_circle_constraint_square (already squared)
                    ])
    ocp.cost.Zl = z_2
    # Quadratic penalty to when the constraint is violated in upper bound in the stage cost
    ocp.cost.Zu = z_2
    # Linear penalty to when the constraint is violated in lower bound in the stage cost
    ocp.cost.zl = z_1
    # Linear penalty to when the constraint is violated in upper bound in the stage cost    
    ocp.cost.zu = z_1

    ## End point constraints
    z_1_e = z_2[[0, 1, 2, 4]]
    z_2_e = z_2[[0, 1, 2, 4]]
    ocp.cost.Zl_e = z_2_e
    ocp.cost.Zu_e = z_2_e
    ocp.cost.zl_e = z_1_e
    ocp.cost.zu_e = z_1_e

    ## Initial condition
    ocp.constraints.x0 = np.array(np.zeros(ocp.model.x.shape[0]))  # Initialize with zeros based on the size of ocp.model.x

    ### Solver options
    so = ocp.solver_options

    so.N_horizon = N # Number of shooting intervals == number of inputs
    so.tf = Tf
    if shooting_nodes is None:
        # For uniform time discretization
        pass
    else:
        # For non-uniform time discretization 
        # so.shooting_nodes = None # https://docs.acados.org/python_interface/index.    html#acados_template.acados_ocp_options.AcadosOcpOptions.shooting_nodes
        so.shooting_nodes = shooting_nodes

    # Basic options for all solvers
    so.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    so.hessian_approx = 'GAUSS_NEWTON'
    so.integrator_type = 'ERK'
    so.sim_method_num_stages = 1 # Runge-Kutta int. stages: (1) RK1, (2) RK2, (4) RK4
    so.sim_method_num_steps = 1
    
    ## Solver type
    # NOTE: scc_controller_imitator 사본. v0237 trajectory 분포에서 SQP full 이 NaN 만들어
    # ROS node 와 동일하게 SQP_RTI 로 운영. (Stage B 합성에서 ROS node 동작 재현 보장.)
    solver_type = 'SQP_RTI'  # Choose from: 'SQP_RTI', 'SQP'
    # Solver specific options
    if solver_type == 'SQP_RTI':
        so.nlp_solver_type = 'SQP_RTI'
        so.qp_solver_iter_max = 30
        so.hpipm_mode = "SPEED"
        so.qp_tol = 1e0  # QP 솔버의 수렴 허용 오차 설정
        so.warm_start_first_qp = True  # 첫 번째 QP에서 웜 스타트 사용
        so.levenberg_marquardt = 1e3  # 수치적 안정성 향상
        so.regularize_method = 'CONVEXIFY' # 'CONVEXIFY', 'NO_REGULARIZE', 'MIRROR', 'PROJECT', 'PROJECT_REDUC_HESS'
        so.globalization_method = 'FIXED_STEP'

    elif solver_type == 'SQP':
        so.nlp_solver_type = 'SQP'
        so.nlp_solver_max_iter = 100   # was 3 — full convergence (batch synthesis, not real-time)
        so.qp_solver_iter_max = 100    # was 30 — inner QP convergence headroom
        so.hpipm_mode = "BALANCE"      # was SPEED — accuracy over speed for batch
        so.qp_tol = 1e-4               # was 3e-1 — tighter QP convergence
        so.warm_start_first_qp = True
        so.levenberg_marquardt = 1e3   # 수치적 안정성 유지
        so.regularize_method = 'MIRROR'
        so.globalization_method = 'FIXED_STEP'
        so.fixed_step_size = 1.0       # was 1e-3 — full Newton steps for fast convergence
        so.tol = 1e-4                  # was 3e-1 — tight NLP tolerance
        so.print_level = 0             # set to 2 for per-iter KKT debug
        so.early_stop_tol = 1e-3       # was 1e-1 — only early-stop when actually converged
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    
    # Create solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json', generate=build, build=build, verbose=True)

    return acados_solver, nx, nu

