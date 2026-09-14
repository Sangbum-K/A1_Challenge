cmake_minimum_required(VERSION 2.8.3)
project(morai_bridge)

## Compile as C++17, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  tf
  std_msgs
  autohyu_msgs
  morai_msgs
  sensor_msgs
  morai_cpp_networks

  cv_bridge
  image_transport

  lanelet2_core
  lanelet2_io
  lanelet2_projection
  lanelet2_routing
  lanelet2_traffic_rules
)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

###################################
## catkin specific configuration ##
###################################
catkin_package(
)

###########
## Build ##
###########

## ini_parser is a sibling top-level dir in moraibridge/ (moraibridge/ini_parser).
## The BSW support headers (task_manager / ros_bridge / util_function / interface)
## and the spline lib are vendored into this package under bsw_support/.
set(INI_PARSER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ini_parser)
set(BSW_SUPPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bsw_support)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}

  # ini_parser (sibling top-level dir)
  ${INI_PARSER_DIR}

  # BSW support headers (vendored)
  ${BSW_SUPPORT_DIR}/task_manager
  ${BSW_SUPPORT_DIR}/interface
  ${BSW_SUPPORT_DIR}/ros_bridge
  ${BSW_SUPPORT_DIR}/util_function

  # Library
  ${BSW_SUPPORT_DIR}/spline

  # MORAI_CPP_NETWORK_MODULE
  ${morai_cpp_networks_INCLUDE_DIRS}
)

## Declare C++ executables (qualifier-essential: 3 nodes only)
add_executable(morai_vehicle_driver
  ${INI_PARSER_DIR}/ini_parser.cpp
  src/vehicle_driver/vehicle_driver_node.cpp
)

add_executable(morai_vehicle_control
  ${INI_PARSER_DIR}/ini_parser.cpp
  src/vehicle_control/vehicle_control_node.cpp
)

add_executable(morai_camera_driver
  ${INI_PARSER_DIR}/ini_parser.cpp
  src/camera_driver/camera_driver_node.cpp
)

## Add cmake target dependencies of the executables
add_dependencies(morai_vehicle_driver
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  autohyu_msgs_generate_messages_cpp
  morai_msgs_generate_messages_cpp
)

add_dependencies(morai_vehicle_control
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  autohyu_msgs_generate_messages_cpp
  morai_msgs_generate_messages_cpp
)

add_dependencies(morai_camera_driver
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  autohyu_msgs_generate_messages_cpp
  morai_msgs_generate_messages_cpp
)

## Specify libraries to link a library or executable target against
target_link_libraries(morai_vehicle_driver
    ${catkin_LIBRARIES}
    ${morai_cpp_networks_LIBRARIES}
)

target_link_libraries(morai_vehicle_control
    ${catkin_LIBRARIES}
    ${morai_cpp_networks_LIBRARIES}
)

target_link_libraries(morai_camera_driver
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${morai_cpp_networks_LIBRARIES}
)

# Install Python scripts
install(PROGRAMS
  script/control_mode_button.py
  script/accel_to_pedal.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
