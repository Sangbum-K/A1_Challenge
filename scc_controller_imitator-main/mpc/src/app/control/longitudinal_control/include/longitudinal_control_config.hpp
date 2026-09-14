#ifndef __LONGITUDINAL_CONTROL_CONFIG_HPP__
#define __LONGITUDINAL_CONTROL_CONFIG_HPP__

#include <stdint.h>
#include <string>

typedef enum {
    DELAYED_DYNAMIC_LMPC= 0,
    DYNAMIC_LMPC,
    PID,
    AX_FEEDFORWARD,
    DELAYED_KINEMATIC_LMPC
} LongitudinalControlMethod;

typedef struct {
    int     longitudinal_control_method;
    double  reference_time_offset;
    int     use_user_speed;
    double  user_speed;
    double  zero_speed;
    int     use_speed_compensator;
    double  look_ahead_time;
    double  ref_speed_lpf;
    double  vel_kp;
    double  vel_ki;
    double  vel_kd;
    double  max_i_error;

    int     use_cte_deceleration;
    double  deceleration_min_cte;
    double  deceleration_min_accel;
    double  deceleration_max_cte;
    double  deceleration_max_accel;

    std::string pedal_map_file_name;

    // Emergency braking thresholds
    double  emergency_brake_torque = -2500.0;  // [Nm]
    double  emergency_decel        = -3.0;     // [m/s²]

    // Vehicle physics — loaded from [Vehicle] in control.ini
    double  vehicle_mass          = 630.0;   // [kg]
    double  gravity               = 9.806;   // [m/s²]
    double  rolling_resistance    = 0.015;   // [-]
    double  air_density           = 1.205;   // [kg/m³]
    double  vehicle_front_area    = 1.2;     // [m²]
    double  air_drag_coeff        = 0.7;     // [-]
    double  vehicle_wheel_radius  = 0.280;   // [m]
} LongitudinalControlParams;

typedef struct {    
    double  step_dt;
    int     step_num;
    int     shift_input_step;   
    double  time_constant;
    double  max_dF; 
    double  weight_damping_ratio;
    double  weight_v;
    double  weight_F;
    double  weight_v_end;
    double  weight_F_end;
    double  weight_dF;
} LongitudinalDelayedDynamicLMPCParams;

typedef struct {
    double  step_dt;
    int     step_num;
    int     shift_input_step;
    double  time_constant;
    double  max_jerk;
    double  weight_damping_ratio;
    double  weight_v;
    double  weight_a;
    double  weight_v_end;
    double  weight_a_end;
    double  weight_jerk;

    // Vehicle physics — loaded from [Vehicle] in control.ini
    double  vehicle_mass          = 630.0;   // [kg]
    double  vehicle_wheel_radius  = 0.280;   // [m]
} LongitudinalDelayedKinematicLMPCParams;

typedef struct {
    double  step_dt;
    int     step_num;
    int     shift_input_step;
    double  max_dF;
    double  weight_damping_ratio;
    double  weight_v;
    double  weight_F;
    double  weight_v_end;
    double  weight_F_end;
    double  weight_dF;

    // Vehicle physics — loaded from [Vehicle] in control.ini
    double  vehicle_mass          = 630.0;   // [kg]
    double  gravity               = 9.806;   // [m/s²]
    double  rolling_resistance    = 0.015;   // [-]
    double  air_density           = 1.205;   // [kg/m³]
    double  vehicle_front_area    = 1.2;     // [m²]
    double  air_drag_coeff        = 0.7;     // [-]
    double  vehicle_wheel_radius  = 0.280;   // [m]
} LongitudinalDynamicLMPCParams;

typedef struct {
    double  look_ahead_time;
    double  step_dt;
    int     step_num;
    double  vel_kp;
    double  vel_ki;
    double  vel_kd;
    double  max_i_error;
    double  ref_speed_lpf;

    // Vehicle physics — loaded from [Vehicle] in control.ini
    double  vehicle_mass         = 630.0;   // [kg]
    double  gravity              = 9.806;   // [m/s²]
    double  rolling_resistance   = 0.015;   // [-]
    double  air_density          = 1.205;   // [kg/m³]
    double  vehicle_front_area   = 1.2;     // [m²]
    double  air_drag_coeff       = 0.7;     // [-]
    double  vehicle_wheel_radius = 0.280;   // [m]
} PIDParams;

typedef struct {    
    double  look_ahead_time;
    double  step_dt;
    int     step_num;
    double  ref_speed_lpf;
} AxFeedForwardParams;
#endif // __LONGITUDINAL_CONTROL_CONFIG_HPP__