#ifndef __LATERAL_KINEMATIC_BICYCLE_LMPC_HPP__
#define __LATERAL_KINEMATIC_BICYCLE_LMPC_HPP__
#pragma once

// STD header
#include <iostream>
#include <vector>
#include <tuple>
#include <list>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <mutex>
#include <utility>
#include <memory>
// Utility header
#include <ini_parser.h>
#include <spline.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

// Interface Header
#include "interface_constants.hpp"
#include "interface_trajectories.hpp"
#include "interface_vehicle_state.hpp"
#include "interface_control.hpp"

#include <lateral_control/lateral_control_core/lateral_control_interface.hpp>

// Solver header
#include "hpipm-cpp.hpp"

// Parameter Header
#include <lateral_control_config.hpp>

using namespace std;
using namespace interface;
using namespace lateral_controller;
// using namespace lmpc_model;
class LateralKinematicBicycleLMPC: public LateralControllerInterface{	
	public:
        explicit LateralKinematicBicycleLMPC();
        virtual ~LateralKinematicBicycleLMPC();

    public:

        void ProcessINI();		
		void BuildControllerVehicleState(const VehicleState& vehicle_state, tk::Map road_map) override;
	   	boost::optional<ControlCommand> CalculateOptimalTireSteering(const ControlTrajectory& ref) override;

	private:
		static constexpr int	dim_state_ = 4;
		static constexpr int 	dim_input_ = 1;
		const string target_frame_ = "ego_frame"; 

		typedef Eigen::Matrix<double,dim_state_,1> State;
		typedef Eigen::Matrix<double,dim_input_,1> Input;
		typedef pair<State,Input> OptVariables;

		typedef Eigen::Matrix<double,dim_state_,dim_state_> A_MPC;
		typedef Eigen::Matrix<double,dim_state_,dim_input_> B_MPC;
		typedef Eigen::Matrix<double,dim_state_,1>  		C_MPC;
		typedef tuple<A_MPC,B_MPC,C_MPC> LinModelMatrix;

	    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
	    // Functions
		vector<State> GenerateOperationPoints(const ControlTrajectory& ref);
		vector<OptVariables> SolveHPIPM(const vector<State>& operation_point, const ControlTrajectory& ref);
        void CheckSolvedStateTwisted(const ControlTrajectory &ref, const ControlCommand &control_command);
        State GetContinuousModelDeriv(const State &x, const Input &u, const double &v);
        State GetReferenceState(const ControlTrajectory& ref, const uint32_t& idx);
		State GetInitState(bool is_prev_state_empty);
		LinModelMatrix GetDiscreteLinModelMatrix(const State &x, const Input &u, const double &v);
		double GetLastSolveTimeMs() const override { return last_solve_time_ms_; }

		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
		// Variables
		bool is_solved_ = false;

		int 	step_num_ = 0;
		double 	step_dt_ = 0;

		double last_solve_time_ms_{0.0};

		
		vector<Input> operation_inputs_;
		vector<State> optimal_states_;

		// Inputs
		VehicleState i_vehicle_state_;
		ControlTrajectory i_reference_trajectory_;
		tk::Map i_road_map_;
		// Outputs
		
        // Environments
        IniParser util_ini_parser_;

        // Configuration parameters
		LateralKinematicBicycleLMPCParams params_;

	

    public:
		inline std::string GetTargetFrame(){
			return target_frame_;
		};
		inline int GetControlTrajectoryStepNum(){
			return params_.step_num;
		};
		inline double GetControlTrajectoryStepTime(){
			return params_.step_dt;
		};
};

#endif  // __LATERAL_KINEMATIC_BICYCLE_LMPC_HPP__
