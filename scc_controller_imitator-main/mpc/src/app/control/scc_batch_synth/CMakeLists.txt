cmake_minimum_required(VERSION 2.8.3)
project(scc_batch_synth)

add_compile_options(-std=c++17 -O2 -Wall)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rosbag
    autohyu_msgs
    scc_replanner
    lateral_control
    longitudinal_control
)

find_package(Eigen3 REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}_lib
    CATKIN_DEPENDS scc_replanner lateral_control longitudinal_control
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}

    /home/ailab-12/git/scc_controller_imitator/mpc/src/bsw/system/interface
    /home/ailab-12/git/scc_controller_imitator/mpc/src/bsw/system/util_function
    /home/ailab-12/git/scc_controller_imitator/mpc/src/bsw/system/ini_parser
    /home/ailab-12/git/scc_controller_imitator/mpc/src/bsw/system/ros_bridge
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/spline
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/blasfeo/lib/include
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/hpipm/lib/include
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/hpipm/include
    /home/ailab-12/git/scc_controller_imitator/mpc/src/lib/pedal_map

    # Controller headers
    /home/ailab-12/git/scc_controller_imitator/mpc/src/app/control/lateral_control/include
    /home/ailab-12/git/scc_controller_imitator/mpc/src/app/control/lateral_control/include/lateral_control
    /home/ailab-12/git/scc_controller_imitator/mpc/src/app/control/longitudinal_control/include
    /home/ailab-12/git/scc_controller_imitator/mpc/src/app/control/longitudinal_control/include/longitudinal_control
)

add_library(${PROJECT_NAME}_lib
    src/ego_parser.cpp
    src/object_parser.cpp
    src/frame_loader.cpp
)
add_dependencies(${PROJECT_NAME}_lib ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_lib ${catkin_LIBRARIES})

# Test executable: load one frame, print summary
add_executable(frame_loader_test test/frame_loader_test.cpp)
add_dependencies(frame_loader_test ${PROJECT_NAME}_lib)
target_link_libraries(frame_loader_test
    ${PROJECT_NAME}_lib
    ${catkin_LIBRARIES}
)

# Batch GT synthesis main executable — FrameLoader + SccReplanner + lateral/longitudinal
add_executable(synth_batch src/synth_batch.cpp)
add_dependencies(synth_batch ${PROJECT_NAME}_lib)
target_link_libraries(synth_batch
    ${PROJECT_NAME}_lib
    ${catkin_LIBRARIES}
)

# Bag replay test: 실 ROS msg 그대로 controller 에 먹여서 fallback 원인 진단
add_executable(bag_replay_test test/bag_replay_test.cpp)
add_dependencies(bag_replay_test ${PROJECT_NAME}_lib)
target_link_libraries(bag_replay_test
    ${PROJECT_NAME}_lib
    ${catkin_LIBRARIES}
)
