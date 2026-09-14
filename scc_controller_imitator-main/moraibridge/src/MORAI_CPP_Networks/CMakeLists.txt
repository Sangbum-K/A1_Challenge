cmake_minimum_required(VERSION 2.8.3)
project(morai_cpp_networks)

## Compile as C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
)

# Platform specific settings
if(WIN32)
    # Windows socket libraries
    set(PLATFORM_LIBS ws2_32 wsock32)
else()
    set(PLATFORM_LIBS pthread)
endif()

# Find OpenCV package
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

## Declare catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES 
    ${PROJECT_NAME}_network
    ${PROJECT_NAME}_sensor
    ${PROJECT_NAME}_actuator
    ${PROJECT_NAME}_simulator
  CATKIN_DEPENDS roscpp
  DEPENDS OpenCV Eigen3
)

# Include directories
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

# Network library sources
add_library(${PROJECT_NAME}_network
    src/network/udp_receiver.cpp
    src/network/udp_sender.cpp
)

target_link_libraries(${PROJECT_NAME}_network
    ${PLATFORM_LIBS}
    ${catkin_LIBRARIES}
)

# Sensor library sources
add_library(${PROJECT_NAME}_sensor
    src/sensors/camera.cpp
    src/sensors/gps.cpp
    src/sensors/imu.cpp
    src/sensors/object_info.cpp
    src/sensors/traffic_light.cpp
    src/sensors/vehicle_state.cpp
    src/sensors/vehicle_noisy_state.cpp

    src/sensors/camera2.cpp
    src/sensors/gt_sensor2.cpp
    src/sensors/vehicle_state2.cpp
)

target_link_libraries(${PROJECT_NAME}_sensor
    ${PLATFORM_LIBS}
    ${PROJECT_NAME}_network
    ${OpenCV_LIBS}
    ${EIGEN3_LIBRARIES}
    ${catkin_LIBRARIES}
)

# Actuator library sources
add_library(${PROJECT_NAME}_actuator
    src/actuator/ego_ctl.cpp
    src/actuator/turn_signal_lamp_control.cpp
    src/actuator/ego_ctl2.cpp
)

target_link_libraries(${PROJECT_NAME}_actuator
    ${PLATFORM_LIBS}
    ${PROJECT_NAME}_network
    ${catkin_LIBRARIES}
)

# Simulator library sources
add_library(${PROJECT_NAME}_simulator
    src/simulator/multi_ego_setting.cpp
    src/simulator/transform_control2.cpp
)

target_link_libraries(${PROJECT_NAME}_simulator
    ${PLATFORM_LIBS}
    ${PROJECT_NAME}_network
    ${catkin_LIBRARIES}
)
