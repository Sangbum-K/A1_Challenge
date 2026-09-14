#ifndef _LONGITUDINAL_CONTROLLER_INTERFACE_HPP_
#define _LONGITUDINAL_CONTROLLER_INTERFACE_HPP_

#include <string>
#include <spline.h>

// Interface Header
#include "interface_constants.hpp"
#include "interface_trajectories.hpp"
#include "interface_vehicle_state.hpp"
#include "interface_control.hpp"


namespace longitudinal_controller
{

class LongitudinalControllerInterface
{
    protected:

    public:
        LongitudinalControllerInterface();
        virtual ~LongitudinalControllerInterface() = default;
        // [scc_imitator fork] cold-start 강제 (default no-op for non-LMPC controllers)
        virtual void ResetInternalState() {}

        virtual void BuildControllerVehicleState(const interface::VehicleState& vehicle_state, tk::Map road_map) = 0;
        virtual interface::ControlCommand CalculateOptimalLongitudinalCommand(const interface::ControlTrajectory& ref) = 0;
      
    public:
        virtual std::string GetTargetFrame() = 0;
        virtual int GetControlTrajectoryStepNum() = 0;
        virtual double GetControlTrajectoryStepTime() = 0;
};
}  // longitudinal_controller
#endif  // _LONGITUDINAL_CONTROLLER_INTERFACE_HPP_
