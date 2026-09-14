#ifndef __LATERAL_CONTROL_CONFIG_HPP__
#define __LATERAL_CONTROL_CONFIG_HPP__
#pragma once

// STD Header
#include <stdint.h>
#include <string>


typedef enum {
    DELAYED_KINEMATIC_BICYCLE_LMPC = 0,
    KINEMATIC_BICYCLE_LMPC,    
    KINEMATIC_BICYCLE_NMPC,
    DYNAMIC_ERROR_LMPC,
    PURE_PURSUIT
} LateralControlMethod;


typedef struct {
    int     lateral_control_method;
    double  reference_time_offset;
    int     use_reference_speed;
    double  time_delay_for_compensators;

    int     use_understeer_gradient;
    double  understeer_gradient_lpf;
    double  understeer_gradient_rate_limit;
    double  understeer_gradient_kv;

    int     use_vehicle_twisting_compensator;
    double  vehicle_twisting_constant;
    double  max_vehicle_twisting_compensation;

    double  vehicle_wheel_base = 2.35;  // [m] — loaded from [Vehicle] in control.ini
} LateralControlParams;



typedef struct {
    double  step_dt;
    int     step_num;
    int     shift_input_step;
    double  time_constant;
    double  weight_x;
    double  weight_y;
    double  weight_yaw;
    double  weight_steer;
    double  weight_dsteer;
    double  weight_dsteer_n;
    double  weight_dsteer_n_thresh;
    double  weight_dsteer_lpf;

    double  avg_distance_error_max;
    double  avg_yaw_error_max;

    double  vehicle_wheel_base = 2.35;  // [m] — loaded from [Vehicle] in control.ini
} LateralDelayedKinematicBicycleLMPCParams;


typedef struct {    
    double  step_dt;
    int     shift_input_step;    
    double  weight_x;
    double  weight_y;
    double  weight_yaw;
    double  weight_steer;
    double  weight_x_end;
    double  weight_y_end;
    double  weight_yaw_end;
    double  weight_steer_end;
    double  weight_dsteer;
    double  avg_distance_error_max;
    double  avg_yaw_error_max;
} LateralKinematicBicycleNMPCParams;



typedef struct {
    double  step_dt;
    int     step_num;
    int     shift_input_step;
    double  weight_x;
    double  weight_y;
    double  weight_yaw;
    double  weight_steer;
    double  weight_x_end;
    double  weight_y_end;
    double  weight_yaw_end;
    double  weight_steer_end;
    double  weight_dsteer;
    double  avg_distance_error_max;
    double  avg_yaw_error_max;

    double  vehicle_wheel_base = 2.35;  // [m] — loaded from [Vehicle] in control.ini
} LateralKinematicBicycleLMPCParams;



typedef struct {    
    double  step_dt;
    int     step_num;
    int     shift_input_step;    
    double  time_constant;
    double  weight_n;
    double  weight_dn;
    double  weight_mu;
    double  weight_dmu;
    double  weight_steer;
    double  weight_n_end;
    double  weight_dn_end;
    double  weight_mu_end;
    double  weight_dmu_end;
    double  weight_steer_end;
    double  weight_dsteer;
} LateralDynamicErrorLMPCParams;


typedef struct {
    double  vehicle_wheel_base = 2.35;  // [m] — loaded from [Vehicle] in control.ini
    double  min_look_ahead;
    double  look_ahead_gain;
    double  step_dt;
    int     step_num;
} PurePursuitParams;


typedef struct {    
    double  gain_k;    
} StanleyParams;



#endif // __LATERAL_CONTROL_CONFIG_HPP__