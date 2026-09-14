#include "lateral_control/lateral_control_fallback/lateral_control_fallback.hpp"

LateralControlFallback::LateralControlFallback()
{

}

LateralControlFallback::~LateralControlFallback(){

}


bool LateralControlFallback::Init(){
    // Ini initialization
    const char* pwd_env = getenv("PWD");
    if (pwd_env == nullptr) {
        std::cerr << "[LateralControlFallback] PWD environment variable is not set. Cannot load config." << std::endl;
        return false;
    }
    std::string dir(pwd_env);
    std::string ini_dir("/config/control.ini");
    util_ini_parser_.Init((dir+ini_dir).c_str());

    // Parameters initialization
    ProcessINI();
    
    return true;
}


interface::ControlCommand LateralControlFallback::PurePursuitSteeringAngle(const interface::VehicleState& vehicle_state, const interface::Trajectory& trajectory){
    ProcessINI();
    interface::ControlCommand cmd;
    cmd.control_trajectory.header.frame_id = "world";
    
    interface::Trajectory cliped_trajectory = ClipTrajectoryFromClosestPoint(vehicle_state, trajectory);
    boost::optional<LookAheadPoint> look_ahead_point = GetLookAheadPoint(vehicle_state, cliped_trajectory);
    

    if(look_ahead_point){
        cmd.steering_tire_angle = atan2(2.0 * params_.vehicle_wheel_base * look_ahead_point->n, pow(look_ahead_point->distance,2))*interface::RAD2DEG;
        cmd.control_trajectory.control_point.push_back({look_ahead_point->x, look_ahead_point->y, 0.0, 0.0, 0.0});
        cmd.control_trajectory.control_point.push_back({look_ahead_point->x, look_ahead_point->y, 0.5, 0.5, 0.5});
    }
    else{
        cmd.steering_tire_angle = vehicle_state.vehicle_can.steering_tire_angle*interface::RAD2DEG;
        cmd.control_trajectory.control_point.push_back({vehicle_state.x, vehicle_state.y, 0.0, 0.0, 0.0});
        cmd.control_trajectory.control_point.push_back({vehicle_state.x, vehicle_state.y, 0.5, 0.5, 0.5});
        return cmd;
    }

    return cmd;
}


interface::Trajectory LateralControlFallback::ClipTrajectoryFromClosestPoint(const interface::VehicleState& vehicle_state, const interface::Trajectory& trajectory)
{
    double min_distance = HUGE_VAL;
    int    closest_idx  = 0;

    for (int i = 0; i < trajectory.point.size(); i++)
    {
        // Get Euclidian Distance from vehicle pose(Ego Frame. Rear) to each trajectory point
        double temp_distance =
            sqrt(pow(vehicle_state.x - trajectory.point[i].x, 2) +
                 pow(vehicle_state.y - trajectory.point[i].y, 2));

        // Find Lookahead Point that have distance over lookahead distance and min
        if (temp_distance < min_distance)
        {
            closest_idx = i;
            min_distance = temp_distance;
        }
    }

    interface::Trajectory clipped_trajectory;
    double sum_distance = 0.0;

    for (int i = closest_idx; i < trajectory.point.size()-1; i++)
    {
        interface::TrajectoryPoint tp;
        double temp_distance = 
            sqrt(pow(trajectory.point[i+1].x - trajectory.point[i].x, 2) +
                 pow(trajectory.point[i+1].y - trajectory.point[i].y, 2));
        
        tp.distance = sum_distance;
        tp.x = trajectory.point.at(i).x;
        tp.y = trajectory.point.at(i).y;
        
        clipped_trajectory.point.push_back(tp);
        
        sum_distance += temp_distance;
    }

    return clipped_trajectory;
}

boost::optional<LookAheadPoint> LateralControlFallback::GetLookAheadPoint(const interface::VehicleState& vehicle_state, const interface::Trajectory& trajectory){
    LookAheadPoint look_ahead_point;
    
    double look_ahead_distance = params_.min_look_ahead +  params_.look_ahead_gain*vehicle_state.vx;

    // Select process for short path that shorter than lookahead distance
    if(!trajectory.point.empty()){
        if(trajectory.point.back().distance > look_ahead_distance){

            // Trajectory is longer than lookahead distance
            for(int i=0; i<trajectory.point.size(); i++){
                // Find Lookahead Point that have distance over lookahead distance and min
                if(trajectory.point.at(i).distance >= look_ahead_distance){
                    look_ahead_point.x = trajectory.point[i].x;
                    look_ahead_point.y = trajectory.point[i].y;
                    // look_ahead_point.distance = trajectory.point[i].distance; 
                    look_ahead_point.distance = trajectory.point[i].distance; 
                    break;
                }
            }

        }
        else{
            // Trajectory is shorter than lookahead distance
            if(!trajectory.point.empty()){
                std::cout << "traj is short than ld" << endl;
                look_ahead_point.x = trajectory.point.back().x;
                look_ahead_point.y = trajectory.point.back().y;
                look_ahead_point.distance = trajectory.point.back().distance;
            }
            else{
                
            }
        }
    }
    else{
        std::cerr << "[LateralControlFallback] No Trajectory Point - cannot compute steering." << std::endl;
        return boost::none;
    }
    

    // Find look ahead point's lateral deviation
    interface::TrajectoryPoint ego_point;
    Eigen::Affine3d point_global_tf, point_local_tf, ego_frame_tf;
    point_global_tf = Eigen::Affine3d::Identity();
    point_local_tf = Eigen::Affine3d::Identity();
    ego_frame_tf = Eigen::Affine3d::Identity();

    point_global_tf.translate(Eigen::Vector3d(look_ahead_point.x, look_ahead_point.y, 0.0));
    point_global_tf.rotate( Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) * 
                            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()));

    ego_frame_tf.translation()[0] = vehicle_state.x;
    ego_frame_tf.translation()[1] = vehicle_state.y;
    ego_frame_tf.translation()[2] = 0.0;

    ego_frame_tf = Eigen::Translation3d(ego_frame_tf.translation()) *
                    (Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) *
                        Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                        Eigen::AngleAxisd(vehicle_state.yaw, Eigen::Vector3d::UnitZ()));

    point_local_tf = ego_frame_tf.inverse() * point_global_tf;
    Eigen::Vector3d ego_point_rpy = point_local_tf.rotation().eulerAngles(0,1,2);

    // deviation from local frame
    look_ahead_point.s  = point_local_tf.translation().x();
    look_ahead_point.n  = point_local_tf.translation().y();

    return look_ahead_point;
}

void LateralControlFallback::ProcessINI(){
	if (util_ini_parser_.IsFileUpdated()){
        util_ini_parser_.ParseConfig("Vehicle",      "wheel_base",       params_.vehicle_wheel_base);
        util_ini_parser_.ParseConfig("Pure pursuit", "min_look_ahead",   params_.min_look_ahead);
        util_ini_parser_.ParseConfig("Pure pursuit", "look_ahead_gain",  params_.look_ahead_gain);
        std::cout <<"min_look_ahead :" <<params_.min_look_ahead << ", look_ahead_gain :" << params_.look_ahead_gain << std::endl;
    }
}