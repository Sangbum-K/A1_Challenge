
cmake_minimum_required(VERSION 2.8.3)
project(autohyu_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs  
  geometry_msgs
  sensor_msgs
  visualization_msgs
  message_generation
)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
    FILES

    ## BSW
    FrameFD.msg

    ## APP
    # Localization
    WGS84.msg
    Reference.msg
    VehicleCAN.msg
    VehicleState.msg
    
    # Perception   
    ObjectDimension.msg
    Object2DState.msg
    Object3DState.msg
    DetectObject2D.msg
    DetectObjects2D.msg
    DetectObject3D.msg
    DetectObjects3D.msg
    TrackObject.msg
    TrackObjects.msg
    TrafficLight.msg
    PredictObjectMultimodal.msg
    PredictObject.msg
    PredictObjects.msg

    # Lanelet
    MapBin.msg
    Lanelet.msg
    LaneletLine.msg
    LaneletLanePoint.msg
    LaneletLane.msg
    LaneletRelation.msg
    LaneletPath.msg
    LaneletRoute.msg
    LaneletTrafficLight.msg
    LaneletTrafficSign.msg
    LaneletRightOfWay.msg

    # Planning
    GoalPoint.msg
    GoalPoints.msg
    PlanningSpace.msg
    PlanningSpaceEgo.msg
    PlanningSpaceObject.msg
    PlanningSpacePoint.msg
    PlanningSpaceState.msg
    PlanningSpaceVectorMap.msg
    PlanningSpaceTrafficLight.msg

    SpatialTrajectoryPoint.msg
    SpatialTrajectory.msg
    TemporalTrajectoryPoint.msg
    TemporalTrajectory.msg

    TrajectoryBoundaryPoint.msg
    TrajectoryBoundary.msg
    BehaviorTrajectoryPoint.msg
    BehaviorTrajectory.msg
    BehaviorTrajectories.msg
    
    TrajectoryPoint.msg
    Trajectory.msg
    Trajectories.msg

    PlanningInputOutput.msg
    DAgger.msg
    
    # Event Points
    EventPoint.msg
    EventPoints.msg
    
    # Control    
    VehicleCmd.msg    
    ControlInfo.msg
    ControlInputOutput.msg
)

# Generate services in the 'srv' folder
add_service_files(
  FILES
  MapBinSrv.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs
   nav_msgs
   geometry_msgs
   sensor_msgs
   visualization_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS 
  roscpp 
  rospy
  std_msgs 
  nav_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  message_runtime
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)