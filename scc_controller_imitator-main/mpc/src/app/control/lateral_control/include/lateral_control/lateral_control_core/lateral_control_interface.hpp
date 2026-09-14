#ifndef _LATERAL_CONTROLLER_INTERFACE_HPP_
#define _LATERAL_CONTROLLER_INTERFACE_HPP_

#include <string>
#include <spline.h>
#include <boost/optional.hpp>

// Interface Header
#include "interface_constants.hpp"
#include "interface_trajectories.hpp"
#include "interface_vehicle_state.hpp"
#include "interface_control.hpp"


// using namespace tk;

namespace lateral_controller
{

class LateralControllerInterface
{
protected:

public:
  LateralControllerInterface();
  virtual ~LateralControllerInterface() = default;
  // [scc_imitator fork] cold-start 강제 (default no-op for non-LMPC controllers)
  virtual void ResetInternalState() {}

  virtual void BuildControllerVehicleState(const interface::VehicleState& vehicle_state, tk::Map road_map) = 0;
  virtual double GetLastSolveTimeMs() const { return 0.0; }
  virtual boost::optional<interface::ControlCommand> CalculateOptimalTireSteering(const interface::ControlTrajectory& ref) = 0;
  
public:
  virtual std::string GetTargetFrame() = 0;
  virtual int GetControlTrajectoryStepNum() = 0;
  virtual double GetControlTrajectoryStepTime() = 0;
};
}  // lateral_controller
#endif  // _LATERAL_CONTROLLER_INTERFACE_HPP_
