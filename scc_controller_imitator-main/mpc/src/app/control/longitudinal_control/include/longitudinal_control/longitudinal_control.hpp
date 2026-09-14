#ifndef __LONGITUDINAL_CONTROL_HPP__
#define __LONGITUDINAL_CONTROL_HPP__
#pragma once

// STD Header
#include <iostream>
#include <vector>
#include <list>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <mutex>
#include <utility>
#include <memory>

// Interface Header
#include "interface_control.hpp"
#include "interface_objects.hpp"
#include "interface_trajectories.hpp"
#include "interface_vehicle_state.hpp"

// Util Function Header
#include <ini_parser.h>
#include <pedal_map.hpp>
#include "function_control.hpp"
#include "function_objects.hpp"
#include "function_trajectories.hpp"
#include "function_vehicle_state.hpp"

// Library Header
#include "spline.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

// Parameter Header
#include <longitudinal_control_config.hpp>

// Algorithm header
#include <longitudinal_control/longitudinal_control_core/longitudinal_control_interface.hpp>
#include <longitudinal_control/longitudinal_control_core/longitudinal_pid/longitudinal_pid.hpp>
#include <longitudinal_control/longitudinal_control_core/longitudinal_dynamic_lmpc/longitudinal_dynamic_lmpc.hpp>
#include <longitudinal_control/longitudinal_control_core/longitudinal_delayed_kinematic_lmpc/longitudinal_delayed_kinematic_lmpc.hpp>

using namespace std;
using namespace interface;
using namespace longitudinal_controller;

class LongitudinalControl {
    public:
        explicit LongitudinalControl();
        virtual ~LongitudinalControl();

		bool Init();
		void ProcessINI();

    public:
        std::tuple<ControlCommand, ControlTrajectory, PathTrackingInfos> Run(
			const interface::VehicleState& vehicle_state,
			const interface::PredictObjects& predict_objects,
			const interface::Trajectory& trajectory);

		// [scc_imitator fork] inner LMPC 의 optimal_states_ 비움 (cold-start)
		void Reset() { if (ptr_longitudinal_controller_) ptr_longitudinal_controller_->ResetInternalState(); }
    
    private:
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
	    // Functions
		void CreateLongitudinalController();
		ControlTrajectory GenerateTargetSpeedProfile(const Trajectory& trajectory, const VehicleState& vehicle_state, tk::Map road_map,
										const std::string& frame_out = "ego_frame");
		void CalculateSpeedTrackingErrors(const ControlTrajectory& profile, const VehicleState& vehicle_state);
		float SpeedCompensator(const ControlTrajectory& profile, const VehicleState& vehicle_state, const double& dt);
		void FaultCheck(ControlCommand& control_command);
		Trajectory GenerateDTTargetTrajectory(const VehicleState& vehicle_state, const Trajectory& trajectory, tk::Map road_map);
		inline double ComputeLongitudinalForce(double ax, double vx, double pitch) const {
			return params_.vehicle_mass * ax
				 + params_.vehicle_mass * params_.gravity * std::sin(-pitch)
				 + params_.vehicle_mass * params_.rolling_resistance * params_.gravity * std::cos(-pitch)
				 + 0.5 * params_.air_density * params_.vehicle_front_area * params_.air_drag_coeff * vx * vx;
		}
		void PedalValuesFilter(PedalMap::PedalValues& pedal_values);

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Variables     

		std::shared_ptr<LongitudinalControllerInterface> ptr_longitudinal_controller_;
		PathTrackingInfos speed_tracking_infos_;
		double sum_speed_error_ = 0.0;
		double prev_speed_error_ = 0.0;
		int prev_control_method_ = -1;
		double cte_error_ = 0.0;
		double prev_brake_pedal_ = -1.0;
		double speed_compensator_ref_speed_ = 0.0;

		// Inputs
		std::shared_ptr<const VehicleState> i_vehicle_state_;
		std::shared_ptr<const Trajectory> 	i_reference_trajectory_;

		// Outputs
	
		// Pedal Map
		PedalMap pedal_map_;
	
		// Environments
        IniParser util_ini_parser_;	

		// Parameter
		LongitudinalControlParams params_;

        // Global Variables
        // ADD GLOBAL VARIABLES HERE

};

#endif // __LONGITUDINAL_CONTROL_HPP__