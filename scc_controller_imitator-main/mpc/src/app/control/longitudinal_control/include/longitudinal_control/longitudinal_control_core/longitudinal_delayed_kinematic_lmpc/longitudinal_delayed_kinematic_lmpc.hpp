#ifndef __LONGITUDINAL_DELAYED_KINEMATIC_LMPC_HPP__
#define __LONGITUDINAL_DELAYED_KINEMATIC_LMPC_HPP__
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

#include <longitudinal_control/longitudinal_control_core/longitudinal_control_interface.hpp>

// Solver header
#include "hpipm-cpp.hpp"

// Parameter Header
#include <longitudinal_control_config.hpp>

using namespace std;
using namespace interface;
using namespace longitudinal_controller;
// using namespace lmpc_model;
class LongitudinalDelayedKinematicLMPC: public LongitudinalControllerInterface{	
	public:
        explicit LongitudinalDelayedKinematicLMPC();
        virtual ~LongitudinalDelayedKinematicLMPC();

    public:

        void ProcessINI();
		void BuildControllerVehicleState(const VehicleState& vehicle_state, tk::Map road_map) override;
	   	ControlCommand CalculateOptimalLongitudinalCommand(const ControlTrajectory& ref) override;

		// [scc_imitator fork] cold-start 강제: optimal_states_ + operation_inputs_ 비움
		void ResetInternalState() override {
			optimal_states_.clear();
			operation_inputs_.clear();
		}

	private:
		static constexpr int	dim_state_ = 9;
		static constexpr int 	dim_input_ = 1;
		// const string target_frame_ = "CG"; 
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
		State GetContinuousModelDeriv(const State &x,const Input &u, const double &slope);
		State GetReferenceState(const ControlTrajectory& ref, const uint32_t& idx);
		State GetInitState(bool is_prev_state_empty);
		LinModelMatrix GetDiscreteLinModelMatrix(const State &x, const Input &u, const double &slope);

		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
		// Variables
		uint8_t live_cnt_ = 0;
		 
		int 	step_num_ = 0;
		double 	step_dt_ = 0;
		
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
		LongitudinalDelayedKinematicLMPCParams params_;

	

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

#endif  // __LONGITUDINAL_PUREPURSUIT_HPP__
