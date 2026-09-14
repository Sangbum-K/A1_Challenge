#ifndef _LATERAL_CONTROL_FALLBACK_HPP_
#define _LATERAL_CONTROL_FALLBACK_HPP_

// STD header
#include <iostream>
#include <vector>
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
#include <boost/optional.hpp>

// Interface Header
#include "interface_constants.hpp"
#include "interface_trajectories.hpp"
#include "interface_vehicle_state.hpp"
#include "interface_control.hpp"

// Parameter Header
#include <lateral_control_config.hpp>

using namespace std;

typedef struct {
    double x;
    double y;
    double distance;
    double s;
    double n;
} LookAheadPoint;

class LateralControlFallback
{
    public:
        explicit LateralControlFallback();
        virtual ~LateralControlFallback();
    
        bool Init();
        void ProcessINI();	
        interface::ControlCommand PurePursuitSteeringAngle(const interface::VehicleState& vehicle_state, const interface::Trajectory& trajectory);

    private:
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Functions
        interface::Trajectory ClipTrajectoryFromClosestPoint(const interface::VehicleState& vehicle_state, const interface::Trajectory& trajectory);
        boost::optional<LookAheadPoint> GetLookAheadPoint(const interface::VehicleState& vehicle_state, const interface::Trajectory& trajectory);
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Variables
        // interface changed (no s n dist in trajectorypoint)

        // Environments
        IniParser util_ini_parser_;	

        
        // Configuration parameters
		PurePursuitParams params_;
};
#endif  // _LATERAL_CONTROL_FALLBACK_HPP_
