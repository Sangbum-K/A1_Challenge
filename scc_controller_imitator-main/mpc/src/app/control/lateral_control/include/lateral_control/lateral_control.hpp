#ifndef __LATERAL_CONTROL_HPP__
#define __LATERAL_CONTROL_HPP__
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
#include <chrono>

// Interface Header
#include "interface_constants.hpp"
#include "interface_control.hpp"
#include "interface_trajectories.hpp"
#include "interface_vehicle_state.hpp"

// Util Function Header
#include "ini_parser.h"
#include "function_control.hpp"
#include "function_print.hpp"
#include "function_trajectories.hpp"
#include "function_vehicle_state.hpp"

// Library Header
#include <spline.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

// Parameter Header
#include <lateral_control_config.hpp>

// Algorithm Header
#include <lateral_control_core/lateral_control_interface.hpp>
#include <lateral_control_core/lateral_purepursuit/lateral_purepursuit.hpp>
#include <lateral_control_core/lateral_kinematic_bicycle_lmpc/lateral_kinematic_bicycle_lmpc.hpp>
#include <lateral_control_core/lateral_delayed_kinematic_bicycle_lmpc/lateral_delayed_kinematic_bicycle_lmpc.hpp>
#include <lateral_control_fallback/lateral_control_fallback.hpp>

using namespace std;
using namespace interface;
using namespace lateral_controller;

class LateralControl{	
	public:
        explicit LateralControl();
        virtual ~LateralControl();

        bool Init();
		void ProcessINI();		

		std::tuple<ControlCommand, ControlTrajectory, PathTrackingInfos> Run (
			const VehicleState& vehicle_state,
			const Trajectory& trajectory);

		// [scc_imitator fork] inner LMPC 의 optimal_states_ 비움 (cold-start)
		void Reset() { if (ptr_lateral_controller_) ptr_lateral_controller_->ResetInternalState(); }
	   
	private:
	    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
	    // Functions
		void CreateLateralController();
		ControlTrajectory GenerateTargetTrajectory(const Trajectory& trajectory, tk::Map road_map,
										const std::string& frame_out = "ego_frame");
				
		double CompensateUndersteerGradient(const ControlTrajectory& ref);
		Trajectory GenerateDTTargetTrajectory(const VehicleState& vehicle_state, const Trajectory& trajectory, tk::Map road_map);

		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
		// Variables
		std::shared_ptr<LateralControllerInterface> ptr_lateral_controller_; 
		std::unique_ptr<LateralControlFallback>     ptr_lateral_fallback_; 

		PathTrackingInfos path_tracking_info_;
		int prev_control_method_ = -1;

		// Inputs
		std::shared_ptr<const VehicleState> i_vehicle_state_;
		std::shared_ptr<const Trajectory> 	i_reference_trajectory_;
		
        // Environments
        IniParser util_ini_parser_;	

		// Parameter
		LateralControlParams params_;
};

#endif  // __DYNAMIC_MPC_HPP__
