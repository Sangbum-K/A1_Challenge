#ifndef __LATERAL_PUREPURSUIT_HPP__
#define __LATERAL_PUREPURSUIT_HPP__
#pragma once

// STD header
#include <iostream>
#include <vector>
#include <list>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <mutex>
#include <utility>

// Utility header
#include <ini_parser.h>
#include <spline.h>

// Interface Header
#include "interface_constants.hpp"
#include "interface_trajectories.hpp"
#include "interface_vehicle_state.hpp"
#include "interface_control.hpp"

#include <lateral_control/lateral_control_core/lateral_control_interface.hpp>

// Parameter Header
#include <lateral_control_config.hpp>

using namespace std;
using namespace tk;
using namespace lateral_controller;

class LateralPurePursuit: public LateralControllerInterface{	
	public:
        explicit LateralPurePursuit();
        virtual ~LateralPurePursuit();

    public:

        void ProcessINI();		
		void BuildControllerVehicleState(const interface::VehicleState& vehicle_state, tk::Map road_map) override;
	   	boost::optional<interface::ControlCommand> CalculateOptimalTireSteering(const interface::ControlTrajectory& ref) override;

	private:
	    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
	    // Functions


		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
		// Variables
		uint8_t live_cnt_=0;

		// Inputs
		interface::VehicleState i_vehicle_state_;
		interface::ControlTrajectory i_reference_trajectory_;
		tk::Map i_road_map_;
		// Outputs
		
        // Environments
        IniParser util_ini_parser_;

        // Configuration parameters
		PurePursuitParams params_;



    public:
		inline std::string GetTargetFrame(){
			return "ego_frame";
		};
		inline int GetControlTrajectoryStepNum(){
			return params_.step_num;
		};
		inline double GetControlTrajectoryStepTime(){
			return params_.step_dt;
		};
};

#endif  // __LATERAL_PUREPURSUIT_HPP__
