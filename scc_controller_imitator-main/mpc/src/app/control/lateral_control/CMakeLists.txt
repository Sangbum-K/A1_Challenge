cmake_minimum_required(VERSION 2.8.3)
project(lateral_control)

## Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
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
execute_process(
  COMMAND bash install_hpipm.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  RESULT_VARIABLE result
)

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

    ## core/fallback controllers
    ${PROJECT_SOURCE_DIR}/include/lateral_control
)

# Collect all .cpp files in the src directory and its subdirectories
file(GLOB_RECURSE HPIPM_WRAPPER_SOURCES "/home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/src/*.cpp")
file(GLOB_RECURSE LATERAL_CONTROLLER "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Library 빌드용: ROS node 진입점 제외 (lateral_control_node.cpp).
# 우리 외부 패키지 (scc_batch_synth 등) 가 link 하기 위해 분리.
set(LATERAL_CONTROLLER_LIB_SOURCES ${LATERAL_CONTROLLER})
list(FILTER LATERAL_CONTROLLER_LIB_SOURCES EXCLUDE REGEX ".*lateral_control_node\\.cpp$")

## Build as library (no ROS node entry) — 외부 패키지가 link 가능
add_library(${PROJECT_NAME}_lib
    /home/ailab-12/git/scc_controller_imitator/mpc/src/bsw/system/ini_parser/ini_parser.cpp
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/spline/spline.cpp
    ${HPIPM_WRAPPER_SOURCES}
    ${LATERAL_CONTROLLER_LIB_SOURCES}
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

## Declare a C++ executable — node entry. lib link.
add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/src/lateral_control_node.cpp
)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME}
    ${PROJECT_NAME}_lib
)