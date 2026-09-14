cmake_minimum_required(VERSION 2.8.3)
project(longitudinal_control)

## Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    tf
    std_msgs
    nav_msgs
    geometry_msgs
    visualization_msgs
    autohyu_msgs
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)

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
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}

    # BSW
    /home/ailab-12/git/scc_controller_imitator/mpc/src/bsw/system/ini_parser
    /home/ailab-12/git/scc_controller_imitator/mpc/src/bsw/system/task_manager
    /home/ailab-12/git/scc_controller_imitator/mpc/src/bsw/system/interface
    /home/ailab-12/git/scc_controller_imitator/mpc/src/bsw/system/ros_bridge 
    /home/ailab-12/git/scc_controller_imitator/mpc/src/bsw/system/util_function
    
    # Library
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/spline

    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/blasfeo/lib/include
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/hpipm/lib/include
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/include

    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/pedal_map
)

## Library 빌드 (node 진입점 제외) — 외부 패키지 가 link 가능
add_library(${PROJECT_NAME}_lib
    # BSW
    /home/ailab-12/git/scc_controller_imitator/mpc/src/bsw/system/ini_parser/ini_parser.cpp

    # Library
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/spline/spline.cpp

    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/src/detail/d_ocp_qp_dim_wrapper.cpp
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/src/detail/d_ocp_qp_wrapper.cpp
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/src/detail/d_ocp_qp_sol_wrapper.cpp
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/src/detail/d_ocp_qp_ipm_arg_wrapper.cpp
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/src/detail/d_ocp_qp_ipm_ws_wrapper.cpp
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/src/ocp_qp_dim.cpp
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/src/ocp_qp_ipm_solver_settings.cpp
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/src/ocp_qp_ipm_solver_statistics.cpp
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/src/ocp_qp_ipm_solver.cpp

    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/pedal_map/pedal_map.cpp

    ## controller basic
    src/longitudinal_control/longitudinal_control.cpp
    src/longitudinal_control/longitudinal_control_core/longitudinal_control_interface.cpp

    ## controller lib
    src/longitudinal_control/longitudinal_control_core/longitudinal_pid/longitudinal_pid.cpp
    src/longitudinal_control/longitudinal_control_core/longitudinal_dynamic_lmpc/longitudinal_dynamic_lmpc.cpp
    src/longitudinal_control/longitudinal_control_core/longitudinal_delayed_kinematic_lmpc/longitudinal_delayed_kinematic_lmpc.cpp
)
add_dependencies(${PROJECT_NAME}_lib
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
    autohyu_msgs_generate_messages_cpp
)
target_link_libraries(${PROJECT_NAME}_lib
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/hpipm/lib/lib/libhpipm.a
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/blasfeo/lib/lib/libblasfeo.a
    ${catkin_LIBRARIES}
)

## ROS node 진입점 — lib link
add_executable(${PROJECT_NAME}
    src/longitudinal_control_node.cpp
)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME}
    ${PROJECT_NAME}_lib
)
