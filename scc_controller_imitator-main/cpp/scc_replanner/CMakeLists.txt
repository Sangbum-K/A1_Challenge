cmake_minimum_required(VERSION 2.8.3)
project(scc_replanner)

add_compile_options(-std=c++17 -O2 -Wall)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    autohyu_msgs
)

find_package(Eigen3 REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}

    # BSW (A1_Rule)
    ${CMAKE_SOURCE_DIR}/bsw/system/interface
    ${CMAKE_SOURCE_DIR}/bsw/system/util_function
    ${CMAKE_SOURCE_DIR}/bsw/system/ros_bridge
)

add_library(${PROJECT_NAME}
    src/scc_speed_replanner.cpp
)
add_dependencies(${PROJECT_NAME}
    ${catkin_EXPORTED_TARGETS}
    autohyu_msgs_generate_messages_cpp
)
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
)

# unit-style test executable
add_executable(scc_replanner_test test/scc_replanner_test.cpp)
add_dependencies(scc_replanner_test ${PROJECT_NAME})
target_link_libraries(scc_replanner_test
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
)
