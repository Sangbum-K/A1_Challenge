// bag_replay_test.cpp
// MPC.bag 의 실 /app/pla/trajectory + /app/loc/vehicle_state 메시지를 그대로 controller 에
// 먹여서 정상 동작 여부 확인. fallback 발생 시 우리 frame_loader 출력과 비교 가능.

#include <cstdio>
#include <cstring>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <autohyu_msgs/Trajectory.h>
#include <autohyu_msgs/VehicleState.h>
#include <autohyu_msgs/PredictObjects.h>

#include "ros_bridge_trajectories.hpp"
#include "ros_bridge_vehicle_state.hpp"
#include "ros_bridge_objects.hpp"
#include "lateral_control/lateral_control.hpp"
#include "longitudinal_control/longitudinal_control.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "사용: %s <MPC.bag>\n", argv[0]);
        return 1;
    }

    rosbag::Bag bag(argv[1], rosbag::bagmode::Read);
    rosbag::View view(bag);

    autohyu_msgs::Trajectory::ConstPtr first_traj;
    autohyu_msgs::VehicleState::ConstPtr last_vs;
    autohyu_msgs::PredictObjects::ConstPtr last_objs;

    int n_seen_traj = 0;
    int n_seen_vs = 0;
    int n_iter = 0;
    for (const auto& m : view) {
        n_iter++;
        const std::string topic = m.getTopic();
        if (topic == "/app/loc/vehicle_state" || topic == "/app/pla/trajectory") {
            std::fprintf(stderr, "  iter %d t=%.3f topic=%s\n", n_iter,
                          m.getTime().toSec(), topic.c_str());
        }
        if (topic == "/app/loc/vehicle_state") {
            last_vs = m.instantiate<autohyu_msgs::VehicleState>();
            n_seen_vs++;
        } else if (topic == "/app/perc/predict_objects") {
            last_objs = m.instantiate<autohyu_msgs::PredictObjects>();
        } else if (topic == "/app/pla/trajectory") {
            n_seen_traj++;
        }
        // 두 메시지 모두 1회 이상 확인되면 가장 최근 traj + vs 쌍 선택해서 종료
        if (last_vs && n_seen_traj > 0) {
            // 다음 trajectory msg 가 오면 그것을 선택 (vs 후속)
            if (topic == "/app/pla/trajectory") {
                first_traj = m.instantiate<autohyu_msgs::Trajectory>();
                std::fprintf(stderr, "[match] iter=%d  datatype=%s  md5=%s  instantiate=%s\n",
                              n_iter, m.getDataType().c_str(), m.getMD5Sum().c_str(),
                              first_traj ? "OK" : "NULL");
                break;
            }
        }
    }
    std::fprintf(stderr, "[scan] seen vs=%d  traj=%d\n", n_seen_vs, n_seen_traj);
    bag.close();

    if (!first_traj || !last_vs) {
        std::fprintf(stderr, "msg 부족 — first_traj=%p last_vs=%p\n",
                      (void*)first_traj.get(), (void*)last_vs.get());
        return 2;
    }
    std::printf("=== bag msg loaded ===\n");
    std::printf("  vehicle_state: x=%.2f y=%.2f yaw=%.3f rad  vx=%.2f\n",
                 last_vs->x, last_vs->y, last_vs->yaw, last_vs->vx);
    std::printf("  trajectory: %zu points\n", first_traj->point.size());
    if (!first_traj->point.empty()) {
        const auto& p0 = first_traj->point[0];
        const auto& pe = first_traj->point.back();
        std::printf("    p0: x=%.2f y=%.2f yaw=%.3f speed=%.2f time=%.3f\n",
                     p0.x, p0.y, p0.yaw, p0.speed, p0.time);
        std::printf("    pN: x=%.2f y=%.2f yaw=%.3f speed=%.2f time=%.3f\n",
                     pe.x, pe.y, pe.yaw, pe.speed, pe.time);
    }

    interface::VehicleState vs   = ros_bridge::GetVehicleState(*last_vs);
    interface::Trajectory   traj = ros_bridge::GetTrajectory(*first_traj);
    interface::PredictObjects objs;
    if (last_objs) objs = ros_bridge::GetPredictObjects(*last_objs);

    std::printf("\n=== ConvertedC interface ===\n");
    std::printf("  vs interface: x=%.2f y=%.2f yaw=%.3f vx=%.2f\n", vs.x, vs.y, vs.yaw, vs.vx);
    std::printf("  traj interface: %zu points\n", traj.point.size());

    // Controllers
    LateralControl lat;
    LongitudinalControl lng;
    if (!lat.Init() || !lng.Init()) {
        std::fprintf(stderr, "Init 실패. PWD 가 A1_Rule root 인지 확인.\n");
        return 3;
    }
    std::printf("  controllers ready.\n\n");

    auto [lat_cmd, lat_traj, lat_info] = lat.Run(vs, traj);
    auto [lng_cmd, lng_traj, lng_info] = lng.Run(vs, objs, traj);

    std::printf("=== Lateral output ===\n");
    std::printf("  steer_tire_angle = %.4f rad (= %.2f deg)\n",
                 lat_cmd.steering_tire_angle,
                 lat_cmd.steering_tire_angle * 180.0 / 3.14159265);
    std::printf("  control_trajectory N = %zu\n", lat_cmd.control_trajectory.control_point.size());

    std::printf("\n=== Longitudinal output ===\n");
    std::printf("  ax = %.4f m/s²,  trq = %.2f Nm\n", lng_cmd.ax, lng_cmd.trq);
    std::printf("  accel_pedal = %.3f, brake_pedal = %.3f\n",
                 lng_cmd.accel_pedal, lng_cmd.brake_pedal);
    std::printf("  control_trajectory N = %zu\n", lng_cmd.control_trajectory.control_point.size());
    if (!lng_cmd.control_trajectory.control_point.empty()) {
        const auto& cp = lng_cmd.control_trajectory.control_point[0];
        std::printf("  cp[0]: vx=%.2f ax=%.2f steer=%.4f time=%.3f\n",
                     cp.vx, cp.ax, cp.steer, cp.time);
    }
    return 0;
}
