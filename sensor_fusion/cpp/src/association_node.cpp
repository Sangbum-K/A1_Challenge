#include "a1_fusion/association.hpp"
#include "a1_fusion/camera_evidence.hpp"
#include "a1_fusion/exposure_control.hpp"
#include "a1_fusion/gnss_ego.hpp"
#include "a1_fusion/map_filter_ros.hpp"
#include "a1_fusion/raceline.hpp"
#include "a1_fusion/scan_rate.hpp"
#include "a1_fusion/tracker.hpp"
#include "a1_fusion/ui_recorder.hpp"
#include "a1_fusion_cpp/msg/a2_features.hpp"
#include "a1_fusion_cpp/msg/association.hpp"
#include "a1_fusion_cpp/msg/brake_estimate.hpp"
#include "a1_fusion_cpp/msg/ego_motion.hpp"
#include <patchwork/patchworkpp.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/opencv.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
using Cloud=sensor_msgs::msg::PointCloud2;
using Image=sensor_msgs::msg::Image;
using A2=a1_fusion_cpp::msg::A2Features;
using Association=a1_fusion_cpp::msg::Association;
using EgoMsg=a1_fusion_cpp::msg::EgoMotion;
using BrakeMsg=a1_fusion_cpp::msg::BrakeEstimate;
using Ins=gsof_msgs::msg::NavigationSolution49;
using a1::Ego;
using namespace a1::inspection;
namespace {
const char* visibility_name(Visibility v){
  switch(v){
    case Visibility::Visible:return "visible";
    case Visibility::Occluded:return "occluded";
    case Visibility::Outside:return "outside";
    default:return "silent";
  }
}
// Everything the render thread needs, copied once per scan.
struct DrawTrack {
  Box box{Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),0,{}};
  Projection image,raw_image;
  cv::Scalar color;
  std::string label,tag;
  bool highlighted{},matched{};
};
struct RenderSnapshot {
  Association result;
  Image::ConstSharedPtr image;
  std::vector<BinRegion> regions;
  std::array<double,40> probabilities{};
  std::vector<Eigen::Vector2f> ground,nonground;
  std::vector<DrawTrack> tracks;
  std::string map_status;
  int map_ground_removed{},map_wall_removed{};
  double received{};
};
}
class AssociationNode:public rclcpp::Node {
 public:
  AssociationNode():Node("a1_association"){
    cv::setNumThreads(1);
    map_filter_=std::make_unique<MapFilterRos>(*this);
    declare_parameter("show_debug_clusters",false);
    declare_parameter("bev_forward_range_m",100.);
    declare_parameter("processing_range_m",120.);
    declare_parameter("max_targets",2);
    TrackerConfig tracking;
    tracking.coast_s=declare_parameter("track_coast_s",.5);
    tracking.memory_s=declare_parameter("track_memory_s",1.5);
    tracking.max_position_variance=declare_parameter("track_max_position_variance_m2",1.);
    tracking.weak_min_range=declare_parameter("weak_min_range_m",40.);
    tracking.max_weak_tracks=declare_parameter("max_weak_tracks",30);
    tracking.guided=declare_parameter("track_guided_gathering",true);
    if(!std::isfinite(tracking.coast_s)||!std::isfinite(tracking.memory_s)||!std::isfinite(tracking.max_position_variance)||tracking.coast_s<=0||tracking.memory_s<tracking.coast_s||tracking.max_position_variance<=0)throw std::invalid_argument("invalid tracker lifetime/variance");
    tracker_=Tracker(tracking);
    weak_min_range_=tracking.weak_min_range;
    SelectorConfig selection;
    selection.l_conf=declare_parameter("evidence_confirm",3.);
    selection.l_drop=declare_parameter("evidence_release",1.);
    selection.far_entry_range=declare_parameter("far_entry_range_m",50.);
    selection.far_keep_range=declare_parameter("far_keep_range_m",35.);
    selection.target_min=declare_parameter("target_min_range_m",3.);
    selection.target_max=declare_parameter("target_max_range_m",50.);
    selection.camera_silent_s=declare_parameter("camera_silent_max_s",2.);
    selection.switch_hold_s=declare_parameter("target_switch_hold_s",.3);
    selector_=EvidenceSelector(selection);
    camera_silent_s_=selection.camera_silent_s;
    // Scans, not seconds, drive track confirmation and camera evidence; see scan_rate.hpp.
    scan_rate_=ScanRate(declare_parameter("scan_rate_warn_hz",7.),declare_parameter("scan_rate_min_hz",4.));
    far_.near_limit=selection.far_keep_range;
    far_.promote=declare_parameter("far_score_promote",3.);
    far_.release=declare_parameter("far_score_release",1.5);
    evidence_.tau=declare_parameter("evidence_tau_s",4.);
    evidence_.p0=declare_parameter("evidence_neutral_probability",.25);
    evidence_.alpha=declare_parameter("evidence_alpha",.15);
    evidence_.pos=declare_parameter("evidence_step_max",1.5);
    evidence_.neg=declare_parameter("evidence_step_min",-.2);
    evidence_.match_score=declare_parameter("evidence_match_probability",.4);
    bin_dilation_=declare_parameter("camera_bin_dilation",1);
    if(selection.target_min<0||selection.target_max<=selection.target_min||get_parameter("max_targets").as_int()<1||get_parameter("max_targets").as_int()>2||selection.l_drop>selection.l_conf||far_.release>far_.promote||bin_dilation_<0)throw std::runtime_error("invalid target selection configuration");
    if(selection.target_max>val("processing_range_m"))throw std::runtime_error("target range exceeds processing range");
    declare_parameter("lidar_roi_center_deg",0.);
    declare_parameter("camera_position_m",std::vector<double>{0.,0.,0.});
    declare_parameter("camera_rpy_rad",std::vector<double>{0.,0.,0.});
    declare_parameter("camera_fx",640.);declare_parameter("camera_fy",640.);
    declare_parameter("camera_cx",640.);declare_parameter("camera_cy",360.);
    const auto distortion=declare_parameter("camera_distortion",std::vector<double>{0.,0.,0.,0.});
    if(distortion.size()!=4)throw std::invalid_argument("camera_distortion needs k1 k2 p1 p2");
    distortion_={distortion[0],distortion[1],distortion[2],distortion[3]};
    const auto mount=declare_parameter("base_from_lidar",std::vector<double>{0.,0.,0.,0.,0.,M_PI/2});
    if(mount.size()!=6)throw std::invalid_argument("base_from_lidar needs xyz roll pitch yaw");
    lidar_lever_=Eigen::Vector2d(mount[0],mount[1]);lidar_yaw_=mount[5];
    declare_parameter("calibration_verified",false);declare_parameter("clock_verified",false);
    declare_parameter("lidar_time_offset_s",0.);declare_parameter("pair_slop_s",.08);
    declare_parameter("presence_hold_threshold",.35);declare_parameter("presence_edge_threshold",.25);
    declare_parameter("fragment_merge_gap_m",.45);declare_parameter("presence_threshold",.5);
    declare_parameter("bin_search_min",0);declare_parameter("bin_search_max",39);
    declare_parameter("corridor_half_width_m",12.);
    declare_parameter("roi_half_width_m",12.);declare_parameter("roi_half_tangent",.58);declare_parameter("roi_grade_margin",.1);
    declare_parameter("cluster_tolerance_m",.45);
    declare_parameter("cluster_range_start_m",20.);declare_parameter("cluster_range_slope",.01);
    declare_parameter("cluster_max_tolerance_m",.85);declare_parameter("cluster_height_gap_m",1.2);declare_parameter("voxel_size_m",.16);
    declare_parameter("cluster_min_points",5);declare_parameter("far_cluster_min_points",3);
    declare_parameter("show_window",false);declare_parameter("min_height_above_ground_m",.15);
    show_window_=get_parameter("show_window").as_bool();
    show_debug_=get_parameter("show_debug_clusters").as_bool();
    auto recording=declare_parameter<std::string>("recording_dir","");
    if(!recording.empty())recorder_=std::make_unique<UiRecorder>(recording);
    preview_=declare_parameter<std::string>("preview_path","");
    const auto raceline_path=declare_parameter<std::string>("raceline_path","");
    const bool raceline_closed=declare_parameter("raceline_closed_loop",true);
    const auto raceline_pose=declare_parameter("map_from_raceline",std::vector<double>{0.,0.,0.});
    if(raceline_pose.size()!=3)throw std::invalid_argument("map_from_raceline needs x y yaw");
    map_from_raceline_={raceline_pose[0],raceline_pose[1],raceline_pose[2]};
    raceline_verified_=declare_parameter("raceline_verified",false);
    if(!raceline_path.empty())raceline_=Raceline::load(raceline_path,raceline_closed);
    auto csv=declare_parameter<std::string>("csv_path","");
    if(!csv.empty())open_csv(csv);
    if(!preview_.empty())std::filesystem::create_directories(std::filesystem::path(preview_).parent_path());
    patchwork::Params p;p.sensor_height=declare_parameter("sensor_height_m",1.2);p.min_range=1.;
    if(!std::isfinite(val("processing_range_m"))||val("processing_range_m")>150)throw std::runtime_error("invalid processing range");
    p.max_range=val("processing_range_m");
    p.verbose=false;p.enable_RNR=false; // XYZ only: do not invent intensity for reflection-noise removal.
    ground_=std::make_unique<patchwork::PatchWorkpp>(p);
    image_pub_=create_publisher<Image>("/lead_fusion/association/image",rclcpp::SensorDataQoS());
    assoc_pub_=create_publisher<Association>("/lead_fusion/association",rclcpp::SensorDataQoS());
    front_pub_=create_publisher<Cloud>("/lead_fusion/front_points",rclcpp::SensorDataQoS());
    map_removed_pub_=create_publisher<Cloud>("/lead_fusion/map_removed",rclcpp::SensorDataQoS());
    ground_pub_=create_publisher<Cloud>("/lead_fusion/ground",rclcpp::SensorDataQoS());
    nonground_pub_=create_publisher<Cloud>("/lead_fusion/nonground",rclcpp::SensorDataQoS());
    clusters_pub_=create_publisher<Cloud>("/lead_fusion/clusters",rclcpp::SensorDataQoS());
    ego_sub_=create_subscription<EgoMsg>("/lead_fusion/ego_motion",10,[this](EgoMsg::ConstSharedPtr m){std::lock_guard<std::mutex> lock(ui_mutex_);ego_ui_=m;});
    brake_sub_=create_subscription<BrakeMsg>("/lead_fusion/estimate",10,[this](BrakeMsg::ConstSharedPtr m){std::lock_guard<std::mutex> lock(ui_mutex_);brake_ui_=m;});
    gnss_status_offset_=declare_parameter("gnss_status_stamp_offset_s",0.);
    gnss_status_sub_=create_subscription<Ins>(declare_parameter<std::string>("gnss_status_topic","/gsof_client/gsof/ins_solution_49"),rclcpp::SensorDataQoS(),[this](Ins::ConstSharedPtr m){gnss_status_=m;});
    gnss_odom_sub_=create_subscription<nav_msgs::msg::Odometry>(declare_parameter<std::string>("gnss_odom_topic","/a1_gnss/odom"),rclcpp::SensorDataQoS(),[this](nav_msgs::msg::Odometry::ConstSharedPtr m){gnss_odom(*m);});
    image_sub_=create_subscription<Image>(declare_parameter<std::string>("image_topic","/camera_driver_gv/vis/image_raw"),rclcpp::SensorDataQoS().keep_last(4),[this](Image::ConstSharedPtr m){images_.push_back(m);while(images_.size()>30)images_.pop_front();});
    a2_sub_=create_subscription<A2>("/lead_fusion/a2",rclcpp::SensorDataQoS().keep_last(4),[this](A2::ConstSharedPtr m){a2s_.push_back(m);while(a2s_.size()>40)a2s_.pop_front();});
    const auto lidar_topic=declare_parameter<std::string>("lidar_topic","/lidar_points");
    cloud_sub_=create_subscription<Cloud>(lidar_topic,rclcpp::SensorDataQoS().keep_last(2),[this,lidar_topic](Cloud::ConstSharedPtr m){
      if(count_publishers(lidar_topic)!=1){
        pending_.clear();reset_targets();
        RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),2000,"LiDAR input requires exactly one publisher; refusing mixed sources");
        return;
      }
      pending_.emplace_back(std::chrono::steady_clock::now(),m);while(pending_.size()>4)pending_.pop_front();
    });
    // Wait for the camera inference of the matching frame; don't pair against the older
    // last completed inference merely because the cloud callback arrived first.
    timer_=create_wall_timer(std::chrono::milliseconds(10),[this]{
      const auto ready=[](const auto& entry){return std::chrono::duration<double>(std::chrono::steady_clock::now()-entry.first).count()>=.10;};
      // Falling behind must cost old scans, not latency: P5 rejects data older than max_age_s.
      while(pending_.size()>1&&ready(pending_[1])){pending_.pop_front();++dropped_scans_;}
      if(pending_.empty()||!ready(pending_.front()))return;
      auto m=pending_.front().second;pending_.pop_front();
      try{process(*m);}catch(const std::exception& e){reset_targets();RCLCPP_ERROR_THROTTLE(get_logger(),*get_clock(),2000,"inspection: %s",e.what());}
    });
    exposure_=std::make_unique<ExposureControl>(*this);
    ui_top_=cv::Mat::zeros(655,1600,CV_8UC3);
    render_thread_=std::thread([this]{render_loop();});
    RCLCPP_INFO(get_logger(),"Patchwork++ / clusters / CTRV tracker / camera evidence preview. NO brake output.");
  }
  ~AssociationNode() override {stop_=true;if(render_thread_.joinable())render_thread_.join();}
 private:
  double val(const char* n){return get_parameter(n).as_double();}
  static double stamp(const std_msgs::msg::Header& h){return rclcpp::Time(h.stamp).seconds();}
  static cv::Scalar color(int64_t id){return cv::Scalar(60+(id*53)%195,60+(id*97)%195,60+(id*151)%195);}
  template<class M> typename M::ConstSharedPtr nearest(const std::deque<typename M::ConstSharedPtr>& q,double t){typename M::ConstSharedPtr best;double gap=1e100;for(auto& m:q){double d=std::abs(stamp(m->header)-t);if(d<gap){gap=d;best=m;}}return best;}
  void reset_targets(){tracker_.reset();selector_.reset();primary_id_=-1;}
  void open_csv(const std::string& csv){
    std::filesystem::create_directories(std::filesystem::path(csv).parent_path());
    csv_.open(csv);if(!csv_)throw std::runtime_error("cannot open association CSV");
    csv_<<"stamp,status,unique,verified,track,bin_left,bin_right,ground,nonground,tracks,image_dt,a2_dt,processing_ms,candidate_ids,lidar_candidate_ids,ambiguous,primary_id,standby_id,primary_status,track_geometry,range_stage_counts,camera_stamp,camera_regions,camera_probabilities,primary_camera_age_s,map_filter_status,map_ground_removed,map_wall_removed,ego_status,raceline_status,dropped_scans,scan_dt_s\n";
    auto detail=std::filesystem::path(csv);detail.replace_extension("tracks.csv");track_csv_.open(detail);if(!track_csv_)throw std::runtime_error("cannot open track diagnostics");
    track_csv_<<"stamp,id,measured,hits,age_s,geometry,reason,raw_x,raw_y,raw_z,raw_dx,raw_dy,raw_dz,points,x,y,z,range_m,vx,vy,var_x,var_y,projectable,u0,v0,u1,v1,camera_score,camera_state,camera_matched,selected,footprint_valid,footprint_length_m,footprint_width_m,footprint_yaw_rad,evidence,world_speed,heading,yaw_rate,weak,guided,rear_visible,near_surface_x,static_ratio,map_static,map_covered,far_score,far_vehicle,path_s,path_n\n";
  }
  void gnss_odom(const nav_msgs::msg::Odometry& m){
    const double t=rclcpp::Time(m.header.stamp).seconds();
    if(!gnss_status_||!a1::navigation_aligned(*gnss_status_)||std::abs(t-(stamp(gnss_status_->header)+gnss_status_offset_))>val("pair_slop_s"))return;
    Ego e;if(!parse_gnss_odom(m,e).empty())return;
    egos_.push_integrated(e,.6);
  }
  Pose2 lidar_pose(const Ego& e) const {
    const double c=std::cos(e.yaw),s=std::sin(e.yaw);
    return {e.position.x()+c*lidar_lever_.x()-s*lidar_lever_.y(),e.position.y()+s*lidar_lever_.x()+c*lidar_lever_.y(),e.yaw};
  }
  std::optional<Pose2> pose_at(double t) const {
    if(!ego_link_.compensated())return Pose2{};
    const auto e=egos_.at(t,.5);
    if(!e||!e->valid)return std::nullopt;
    return lidar_pose(*e);
  }
  void cloud_pub(const Eigen::MatrixX3f& points,const std_msgs::msg::Header& h,rclcpp::Publisher<Cloud>::SharedPtr pub){
    if(pub->get_subscription_count()==0)return;
    pcl::PointCloud<pcl::PointXYZ> pc;pc.reserve(points.rows());for(int i=0;i<points.rows();++i)pc.push_back({points(i,0),points(i,1),points(i,2)});Cloud m;pcl::toROSMsg(pc,m);m.header=h;pub->publish(m);
  }
  void process(const Cloud& m){
    const auto start=std::chrono::steady_clock::now();
    if(m.is_bigendian||m.row_step<uint64_t(m.width)*m.point_step||m.data.size()<uint64_t(m.row_step)*m.height)throw std::runtime_error("invalid cloud layout");
    for(const char* name:{"x","y","z"}){bool ok=false;for(auto& f:m.fields)if(f.name==name&&f.datatype==7&&f.count==1&&f.offset+4<=m.point_step)ok=true;if(!ok)throw std::runtime_error("invalid XYZ fields");}
    const double range_limit=val("processing_range_m"),time_offset=val("lidar_time_offset_s");
    const Eigen::Matrix3d lidar_to_body=body_from_lidar(lidar_yaw_);
    pcl::PointCloud<pcl::PointXYZ> raw;pcl::fromROSMsg(m,raw);
    const double hstamp=stamp(m.header);double t=hstamp;bool point_time_ok=false;
    ScanTiming timing;
    for(auto& f:m.fields)if(f.name=="timestamp"&&f.datatype==8&&f.count==1&&f.offset+8<=m.point_step){
      double begin=1e100,end=-1e100;bool ok=true;
      for(uint32_t row=0;row<m.height&&ok;++row)for(uint32_t col=0;col<m.width;++col){
        double v;std::memcpy(&v,m.data.data()+size_t(row)*m.row_step+size_t(col)*m.point_step+f.offset,8);
        if(!std::isfinite(v)||std::abs(v-hstamp)>.3){ok=false;break;}
        begin=std::min(begin,v);end=std::max(end,v);
        const auto& p=raw[size_t(row)*m.width+col];
        const Eigen::Vector3d q=lidar_to_body*Eigen::Vector3d(p.x,p.y,0.);
        timing.add(std::atan2(q.y(),q.x()),v+time_offset);
      }
      if(ok&&end>=begin&&end-begin<=.15){t=end;point_time_ok=true;}
    }
    if(!point_time_ok)timing=ScanTiming{};
    t+=time_offset;
    timing.finish(t);
    if(t<=last_stamp_){reset_targets();images_.clear();a2s_.clear();egos_.clear();last_a2_stamp_=-1;}
    last_stamp_=t;
    scan_rate_.add(t);
    const auto rate_state=scan_rate_.state();
    if(rate_state==ScanRate::State::Slow||rate_state==ScanRate::State::TooSlow)
      RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),5000,"LiDAR scans arrive at %.1f Hz; tracking and camera evidence are tuned for 10 Hz (warning below %.0f Hz, no target below %.0f Hz)",scan_rate_.hz(),scan_rate_.warn_hz(),scan_rate_.min_hz());
    if(ego_link_.update(t,egos_.at(t,.5).has_value()))reset_targets();
    tracker_.set_process_scale(ego_link_.compensated()?1.:4.);
    const auto ego_at_scan=pose_at(t);
    const Pose2 scan_pose=ego_at_scan?*ego_at_scan:Pose2{};
    auto a=nearest<A2>(a2s_,t);
    auto im=nearest<Image>(images_,a?stamp(a->header):t);
    if(im&&(im->width!=1280||im->height!=720))throw std::runtime_error("expected 1280x720 camera ROI");
    Association result;result.header.stamp=rclcpp::Time(int64_t(t*1e9));result.header.frame_id="lidar_vehicle";
    result.image_delta_s=im?stamp(im->header)-t:std::numeric_limits<double>::quiet_NaN();
    result.a2_delta_s=a?stamp(a->header)-t:std::numeric_limits<double>::quiet_NaN();
    if(im)result.image_stamp=im->header.stamp;
    if(a)result.a2_stamp=a->header.stamp;
    result.calibration_verified=get_parameter("calibration_verified").as_bool();
    result.clock_verified=get_parameter("clock_verified").as_bool()&&point_time_ok;
    result.primary_track_id=result.standby_track_id=-1;
    result.motion_compensated=ego_link_.compensated();
    result.ego_status=ego_link_.compensated()?"compensated":"unavailable";
    result.candidate_track_id=-1;result.bin_left=result.bin_right=-1;
    result.dropped_scans=uint32_t(dropped_scans_);
    Eigen::MatrixXf input(raw.size(),3);int count=0;
    for(auto& p:raw){Eigen::Vector3d q(p.x,p.y,p.z);if(!q.allFinite()||q.norm()<1||q.norm()>range_limit)continue;q=lidar_to_body*q;input.row(count++)=q.cast<float>();}input.conservativeResize(count,3);
    const double roi_angle=val("lidar_roi_center_deg")*M_PI/180;
    Eigen::MatrixX3f front(count,3);int nf=0;
    for(int i=0;i<count;++i)if(in_front_half(input.row(i).cast<double>().transpose(),roi_angle))front.row(nf++)=input.row(i);
    front.conservativeResize(nf,3);cloud_pub(front,result.header,front_pub_);
    if(count<50)throw std::runtime_error("too few finite LiDAR points");
    ground_->estimateGround(input);auto gp=ground_->getGround(),ng=ground_->getNonground();
    result.ground_points=gp.rows();result.nonground_points=ng.rows();
    cloud_pub(gp,result.header,ground_pub_);cloud_pub(ng,result.header,nonground_pub_);
    const double roi_half_width=val("roi_half_width_m"),roi_tangent=val("roi_half_tangent"),roi_grade=val("roi_grade_margin");
    auto roi=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for(int i=0;i<ng.rows();++i){auto q=ng.row(i);if(in_front_half(q.cast<double>().transpose(),roi_angle)&&q.x()>1&&q.x()<=range_limit&&in_processing_roi(q.x(),q.y(),q.z(),roi_half_width,roi_tangent,roi_grade))roi->push_back({q.x(),q.y(),q.z()});}
    auto vox=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();pcl::VoxelGrid<pcl::PointXYZ> vg;vg.setInputCloud(roi);float leaf=val("voxel_size_m");if(leaf<=0)throw std::runtime_error("voxel size must be positive");vg.setLeafSize(leaf,leaf,leaf);vg.filter(*vox);
    const std::string map_filter_status=map_filter_->begin(t,m.header.frame_id,lidar_to_body);
    result.map_status=map_filter_status=="active"?(map_filter_->has_structure()?"ok":"no_structure"):map_filter_status;
    const bool map_ok=result.map_status=="ok";
    // Per-azimuth pose: a point fired earlier in the scan was seen from an earlier ego position.
    std::array<std::optional<Pose2>,360> cell_pose;
    auto pose_for=[&](double azimuth)->Pose2{
      const int cell=((int(std::floor(azimuth*180/M_PI))%360)+360)%360;
      if(!cell_pose[cell]){const auto p=pose_at(timing.at(azimuth));cell_pose[cell]=p?*p:scan_pose;}
      return *cell_pose[cell];
    };
    auto to_map=[&](const Eigen::Vector3d& q){
      const Eigen::Vector2d at_scan=scan_pose.to_body(pose_for(std::atan2(q.y(),q.x())).to_world(q.head<2>()));
      return Eigen::Vector3d(map_filter_->map_from_body()*Eigen::Vector3d(at_scan.x(),at_scan.y(),q.z()));
    };
    int map_ground_removed=0,map_wall_removed=0;
    auto map_kept=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::PointCloud<pcl::PointXYZI> map_removed;
    const bool publish_removed=map_removed_pub_->get_subscription_count()>0;
    for(const auto& q:*vox){
      const Eigen::Vector3d body(q.x,q.y,q.z);
      const int label=map_filter_->active()?map_filter_->match(to_map(body),body.head<2>().norm()):0;
      if(!label){map_kept->push_back(q);continue;}
      if(label==1)++map_ground_removed;else ++map_wall_removed;
      if(publish_removed){pcl::PointXYZI removed;removed.x=q.x;removed.y=q.y;removed.z=q.z;removed.intensity=label;map_removed.push_back(removed);}
    }
    vox=map_kept;
    if(publish_removed){Cloud removed;pcl::toROSMsg(map_removed,removed);removed.header=result.header;map_removed_pub_->publish(removed);}
    // Patchwork++ leaves sparse/unclassified ground patches in nonground. Prevent a
    // thin residual floor from joining separate objects into one Euclidean cluster.
    // This adapter guard uses only locally supported ground, not a global flat-road plane.
    const double min_height=val("min_height_above_ground_m");
    std::map<std::pair<int,int>,std::pair<double,int>> ground_cells;
    for(int i=0;i<gp.rows();++i){auto& cell=ground_cells[{int(std::floor(gp(i,0))),int(std::floor(gp(i,1)))}];cell.first+=gp(i,2);++cell.second;}
    auto elevated=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for(auto& q:*vox){std::vector<double> heights;int ix=std::floor(q.x),iy=std::floor(q.y);
      for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy){auto it=ground_cells.find({ix+dx,iy+dy});if(it!=ground_cells.end()&&it->second.second>=3)heights.push_back(it->second.first/it->second.second);}
      if(heights.size()>=3){std::sort(heights.begin(),heights.end());if(q.z-heights[heights.size()/2]<min_height)continue;}
      elevated->push_back(q);
    }
    // Counts per 10 m shell: input / nonground / voxel / elevated. CSV diagnostics only.
    std::array<std::array<int,4>,12> stage_counts{};
    auto count_stage=[&](double x,double y,int stage){int bin=int(std::hypot(x,y)/10.);if(bin>=0&&bin<12)++stage_counts[bin][stage];};
    for(int i=0;i<input.rows();++i)if(in_front_half(input.row(i).cast<double>().transpose(),roi_angle))count_stage(input(i,0),input(i,1),0);
    for(const auto& q:*roi)count_stage(q.x,q.y,1);
    for(const auto& q:*vox)count_stage(q.x,q.y,2);
    for(const auto& q:*elevated)count_stage(q.x,q.y,3);
    vox=elevated;
    Scan scan;scan.stamp=t;scan.ego=scan_pose;scan.map_ok=map_ok;
    scan.body_points.reserve(vox->size());scan.world_points.reserve(vox->size());scan.point_time.reserve(vox->size());
    std::vector<Eigen::Vector3d> map_points;map_points.reserve(vox->size());
    for(const auto& q:*vox){
      const Eigen::Vector3d body(q.x,q.y,q.z);
      const double azimuth=std::atan2(body.y(),body.x());
      const Eigen::Vector2d world=pose_for(azimuth).to_world(body.head<2>());
      scan.body_points.push_back(body);scan.world_points.emplace_back(world.x(),world.y(),body.z());
      scan.point_time.push_back(timing.at(azimuth));
      map_points.push_back(map_ok?to_map(body):body);
    }
    if(map_ok){
      scan.is_structure=[&](int i){return map_filter_->is_structure(map_points[i],scan.body_points[i].head<2>().norm());};
      scan.is_covered=[&](int i){return map_filter_->covered(map_points[i]);};
    }
    const std::string raceline_status=update_raceline(t,scan_pose);
    result.raceline_status=raceline_status;
    const bool raceline_active=raceline_status=="active";
    scan.heading_prior=[&,yaw=scan_pose.yaw](const Eigen::Vector2d& world){
      if(!raceline_active)return yaw;
      // Racing-line tangent, carried from the map frame into the continuous frame.
      const auto map_from_body=map_filter_->map_from_body();
      const Eigen::Vector2d body=scan_pose.to_body(world);
      const Eigen::Vector3d point=map_from_body*Eigen::Vector3d(body.x(),body.y(),0.);
      int hint=ego_path_hint_;
      const auto frenet=raceline_->project(map_from_raceline_.to_body(point.head<2>()),hint,801);
      const Eigen::Vector3d forward=map_from_body.linear()*Eigen::Vector3d::UnitX();
      return std::remainder(frenet.yaw+map_from_raceline_.yaw-std::atan2(forward.y(),forward.x())+yaw,2*M_PI);
    };
    std::vector<std::vector<int>> groups;std::vector<Box> boxes;
    if(!vox->empty()){
      ClusterConfig cc;cc.near_radius=val("cluster_tolerance_m");cc.start_range=val("cluster_range_start_m");cc.slope=val("cluster_range_slope");cc.max_radius=val("cluster_max_tolerance_m");cc.max_height_gap=val("cluster_height_gap_m");
      const int near_points=get_parameter("cluster_min_points").as_int(),far_points=get_parameter("far_cluster_min_points").as_int();
      cc.min_points=std::min(near_points,far_points);
      for(auto& indices:adaptive_clusters(scan.body_points,cc)){
        Box b;b.lo=Eigen::Vector3d::Constant(1e9);b.hi=-b.lo;b.points=indices.size();
        for(int j:indices){const auto& v=scan.body_points[j];b.lo=b.lo.cwiseMin(v);b.hi=b.hi.cwiseMax(v);}
        const auto dim=b.hi-b.lo;const double extent=std::max(dim.x(),dim.y());
        // A far vehicle may be a single LiDAR ring: flat and with few points.
        const bool far=b.center().head<2>().norm()>=weak_min_range_;
        if(int(indices.size())<(far?far_points:near_points)||extent<.3||extent>10||(!far&&dim.z()<.15))continue;
        boxes.push_back(b);groups.push_back(std::move(indices));
      }
    }
    // Bounded fragment attachment preserves actual measured extents.
    for(auto& ids:merge_fragments(boxes,val("fragment_merge_gap_m"))){
      std::vector<int> joined;
      for(int k:ids)joined.insert(joined.end(),groups[k].begin(),groups[k].end());
      Detection d=make_detection(scan,joined,weak_min_range_);
      if(d.quality!=Quality::Reject)scan.detections.push_back(std::move(d));
    }
    publish_clusters(scan,result);
    auto& tracks=tracker_.update(scan);
    update_presence_display(a,im,result);
    const bool ambiguous=accumulate_camera_evidence(t);
    result.association_ambiguous=ambiguous;
    const bool camera_silent=last_camera_stamp_<0||t-last_camera_stamp_>camera_silent_s_;
    // Evidence survives a short camera gap, but a match needs a current frame.
    const bool camera_current=last_camera_stamp_>=0&&t-last_camera_stamp_<=.3;
    const double corridor=val("corridor_half_width_m");
    std::vector<TargetObservation> observations;
    std::map<int64_t,std::string> reasons;
    for(auto& tr:tracks){
      const Eigen::Vector2d body=scan_pose.to_body(tr.anchor());
      update_path(tr,scan_pose);
      FarInputs inputs;inputs.map_ok=map_ok;inputs.ego_compensated=ego_link_.compensated();inputs.raceline_active=raceline_active;
      inputs.speed_sigma=std::sqrt(std::max(0.,tr.P(2,2)));inputs.range=body.norm();inputs.path_n=tr.path_n;
      update_far_state(tr,inputs,far_);
      const std::string geometry=vehicle_geometry_status(tr);
      const bool is_primary=tr.id==primary_id_;
      TargetObservation o;o.id=tr.id;o.range=body.norm();
      o.distance=raceline_active&&std::isfinite(tr.path_s)?tr.path_s:o.range;
      o.usable=tracker_.usable(tr,t);o.measured=tr.measured;o.evidence=tr.evidence;
      o.vehicle_shape=camera_surface_geometry(tr.raw_box,is_primary)||geometry=="full_geometry"||geometry=="retained_partial"||geometry=="far_partial";
      o.map_static=tr.map_static;o.far_vehicle=tr.far_vehicle;o.visibility=camera_silent?Visibility::Silent:tr.visibility;
      o.camera_matched=camera_current&&tr.visibility==Visibility::Visible&&tr.camera_score>=evidence_.match_score;
      o.in_lane=raceline_active?tr.path_s>0&&same_lane(tr.path_n,ego_path_n_,is_primary):std::abs(body.y())<=corridor;
      observations.push_back(o);
      reasons[tr.id]=!vehicle_extent(tr.raw_box)?"extent_exceeded":!tr.confirmed?"track_unconfirmed":
        !o.usable?"track_expired_or_uncertain":tr.map_static?"map_static":!o.in_lane?"outside_lane":
        o.range>val("target_max_range_m")?"outside_target_range":o.range>val("far_entry_range_m")?"far_not_vehicle":
        !o.vehicle_shape?"insufficient_geometry":"camera_evidence_low";
    }
    auto selected=selector_.update(t,last_camera_stamp_,observations);
    if(rate_state==ScanRate::State::TooSlow){selected.primary=selected.standby=-1;selected.status="lidar_rate_low";}
    primary_id_=selected.primary;
    if(selected.primary>=0)reasons[selected.primary]="selected";
    if(selected.standby>=0)reasons[selected.standby]="standby";
    result.primary_track_id=selected.primary;
    result.standby_track_id=get_parameter("max_targets").as_int()>1?selected.standby:-1;
    result.primary_status=selected.status;
    result.primary_camera_age_s=selected.camera_age;
    std::vector<int64_t> candidates;
    auto matched=[&](int64_t id){for(const auto& o:observations)if(o.id==id)return o.camera_matched;return false;};
    for(auto id:{result.primary_track_id,result.standby_track_id})if(id>=0){
      result.lidar_candidate_track_ids.push_back(id);
      if(matched(id))candidates.push_back(id);
    }
    result.candidate_track_ids=candidates;
    auto snapshot=std::make_shared<RenderSnapshot>();
    const auto [optical,camera]=camera_pose();
    for(size_t index=0;index<tracks.size();++index){
      const auto& tr=tracks[index];
      const Box box=track_body_box(tr,scan_pose);
      const auto p=project_box(box,optical,camera);
      const auto raw_projection=project_box(tr.raw_box,optical,camera);
      const auto span=bins_for(project_box(track_rear_box(tr,scan_pose),optical,camera));
      const Eigen::Vector2d ego_velocity=ego_velocity_at(t);
      const Eigen::Vector2d relative=relative_velocity(tr,scan_pose,ego_velocity);
      const Eigen::Vector3d center=box.center(),size=box.hi-box.lo;
      a1_fusion_cpp::msg::ClusterTrack out;out.id=tr.id;out.measured=tr.measured;out.prediction_valid=tracker_.usable(tr,t);out.observation_age_s=std::max(0.,t-tr.last_seen);
      out.motion_reacquiring=!tr.rear_visible;
      const double c=std::cos(scan_pose.yaw),s=std::sin(scan_pose.yaw);
      const double xx=tr.P(0,0),xy=tr.P(0,1),yy=tr.P(1,1);
      const double body_xx=c*c*xx+2*c*s*xy+s*s*yy,body_yy=s*s*xx-2*c*s*xy+c*c*yy,body_xy=(c*c-s*s)*xy+c*s*(yy-xx);
      out.position_covariance_xy={body_xx,body_xy,body_xy,body_yy};
      out.confirmed=tr.confirmed;out.points=tr.measured?tr.raw_box.points:0;
      for(int k=0;k<3;++k){out.raw_center_m[k]=tr.raw_box.center()[k];out.raw_size_m[k]=(tr.raw_box.hi-tr.raw_box.lo)[k];out.center_m[k]=center[k];out.size_m[k]=size[k];}
      out.relative_velocity_mps={relative.x(),relative.y()};out.near_surface_x_m=near_surface_x(tr,scan_pose);
      out.projectable=p.valid;out.image_box_xyxy={p.x0,p.y0,p.x1,p.y1};
      out.range_m=scan_pose.to_body(tr.anchor()).norm();
      const bool candidate=matched(tr.id);
      out.camera_matched=candidate;out.matched_bin_left=span.valid?span.left:-1;out.matched_bin_right=span.valid?span.right:-1;
      out.geometry_status=vehicle_geometry_status(tr);out.selection_reason=reasons.at(tr.id);
      out.weak_measurement=tr.weak_update;out.camera_evidence=tr.evidence;
      out.camera_state=visibility_name(camera_silent?Visibility::Silent:tr.visibility);
      out.map_static_ratio=tr.static_ratio;out.map_static=tr.map_static;out.far_score=tr.far_score;out.far_vehicle=tr.far_vehicle;
      out.path_s_m=tr.path_s;out.path_n_m=tr.path_n;
      out.lidar_candidate=std::find(result.lidar_candidate_track_ids.begin(),result.lidar_candidate_track_ids.end(),tr.id)!=result.lidar_candidate_track_ids.end();
      const bool visible=out.lidar_candidate;
      write_track_row(tr,out,t,p,candidate,visible);
      result.tracks.push_back(out);
      if(visible||show_debug_){
        DrawTrack draw;draw.box=box;draw.image=p;draw.raw_image=raw_projection;draw.matched=candidate;draw.highlighted=visible;
        draw.color=!tr.measured&&visible?cv::Scalar(255,180,40):candidate?cv::Scalar(40,40,255):visible?cv::Scalar(255,180,40):cv::Scalar(100,100,100);
        draw.tag="T"+std::to_string(tr.id);
        draw.label=draw.tag+" r="+cv::format("%.1f",out.range_m)+"m L="+cv::format("%.1f",tr.evidence)+(tr.id==primary_id_?" PRIMARY":"")+(!tr.measured?" PREDICTED":candidate?" CAMERA+LIDAR":visible?" HELD":" UNSELECTED");
        snapshot->tracks.push_back(std::move(draw));
      }
    }
    result.candidate_unique=candidates.size()==1;if(result.candidate_unique)result.candidate_track_id=candidates[0];
    result.verified_match=result.candidate_unique&&!result.association_ambiguous&&result.calibration_verified&&result.clock_verified;
    result.status=rate_state==ScanRate::State::TooSlow?"lidar_rate_low":!im?"no_image":!a?"no_a2":!a->history_ready?"a2_history_unready":selected.primary<0?"no_target":selected.status=="camera_held"?"camera_held":!result.calibration_verified?"candidate_extrinsics_unverified":!result.clock_verified?"candidate_clock_unverified":"matched_geometry_only";
    result.processing_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    assoc_pub_->publish(result);
    write_association_row(result,tracks,stage_counts,map_filter_status,map_ground_removed,map_wall_removed);
    if(render_needed()){
      snapshot->result=result;snapshot->image=im;snapshot->regions=regions_;snapshot->probabilities=probabilities_;
      snapshot->map_status=result.map_status;snapshot->map_ground_removed=map_ground_removed;snapshot->map_wall_removed=map_wall_removed;
      snapshot->received=now().seconds();
      snapshot->ground.reserve(gp.rows()/3+1);snapshot->nonground.reserve(ng.rows());
      for(int i=0;i<gp.rows();i+=3)if(in_front_half(gp.row(i).cast<double>().transpose(),roi_angle))snapshot->ground.emplace_back(gp(i,0),gp(i,1));
      for(int i=0;i<ng.rows();++i)if(in_front_half(ng.row(i).cast<double>().transpose(),roi_angle))snapshot->nonground.emplace_back(ng(i,0),ng(i,1));
      std::lock_guard<std::mutex> lock(ui_mutex_);latest_=std::move(snapshot);
    }
  }
  std::pair<Eigen::Matrix3d,Eigen::Vector3d> camera_pose(){
    auto pos=get_parameter("camera_position_m").as_double_array(),angles=get_parameter("camera_rpy_rad").as_double_array();
    if(pos.size()!=3||angles.size()!=3)throw std::runtime_error("camera mount requires xyz/rpy triples");
    Eigen::Matrix3d mount=(Eigen::AngleAxisd(angles[2],Eigen::Vector3d::UnitZ())*Eigen::AngleAxisd(angles[1],Eigen::Vector3d::UnitY())*Eigen::AngleAxisd(angles[0],Eigen::Vector3d::UnitX())).toRotationMatrix();
    return {Eigen::Matrix3d(nominal_optical_from_body()*mount.transpose()),Eigen::Vector3d(pos[0],pos[1],pos[2])};
  }
  Projection project_box(const Box& box,const Eigen::Matrix3d& optical,const Eigen::Vector3d& camera){
    return project(box,optical,camera,val("camera_fx"),val("camera_fy"),val("camera_cx"),val("camera_cy"),1280,720,distortion_);
  }
  Eigen::Vector2d ego_velocity_at(double t) const {
    if(!ego_link_.compensated())return Eigen::Vector2d::Zero();
    const auto e=egos_.at(t,.5);
    return e?Eigen::Vector2d(e->velocity.x(),e->velocity.y()):Eigen::Vector2d::Zero();
  }
  std::string update_raceline(double t,const Pose2&){
    if(!raceline_)return "disabled";
    if(!map_filter_->active())return "no_localization";
    const Eigen::Vector3d origin=map_filter_->map_from_body()*Eigen::Vector3d::Zero();
    const auto frenet=raceline_->project(map_from_raceline_.to_body(origin.head<2>()),ego_path_hint_);
    ego_path_s_=frenet.s;ego_path_n_=frenet.n;
    alignment_.add(t,std::abs(frenet.n));
    if(alignment_.misaligned())return "misaligned";
    return raceline_verified_?"active":"unverified";
  }
  void update_path(Track& tr,const Pose2& ego){
    tr.path_s=tr.path_n=std::numeric_limits<double>::quiet_NaN();
    if(!raceline_||!map_filter_->active())return;
    const Eigen::Vector2d body=ego.to_body(tr.anchor());
    const Eigen::Vector3d point=map_filter_->map_from_body()*Eigen::Vector3d(body.x(),body.y(),0.);
    const auto frenet=raceline_->project(map_from_raceline_.to_body(point.head<2>()),tr.path_hint);
    tr.path_s=raceline_->ahead(ego_path_s_,frenet.s);tr.path_n=frenet.n;
  }
  void publish_clusters(const Scan& scan,const Association& result){
    if(clusters_pub_->get_subscription_count()==0)return;
    pcl::PointCloud<pcl::PointXYZRGB> colored;
    for(const auto& d:scan.detections){
      const bool strong=d.quality==Quality::Strong;
      for(int i:d.points){const auto& q=scan.body_points[i];pcl::PointXYZRGB v;v.x=q.x();v.y=q.y();v.z=q.z();v.r=strong?40:255;v.g=strong?255:180;v.b=40;colored.push_back(v);}
    }
    Cloud message;pcl::toROSMsg(colored,message);message.header=result.header;clusters_pub_->publish(message);
  }
  // Display only: the smoothed regions drawn on screen. Target decisions use the evidence.
  void update_presence_display(const A2::ConstSharedPtr& a,const Image::ConstSharedPtr& im,Association& result){
    const bool paired=im&&a&&std::abs(stamp(im->header)-stamp(a->header))<.005&&std::abs(result.image_delta_s)<=val("pair_slop_s")&&std::abs(result.a2_delta_s)<=val("pair_slop_s");
    if(!paired||!a->history_ready){regions_.clear();presence_stamp_=-1;probabilities_.fill(0);return;}
    const double at=stamp(a->header);
    if(at!=presence_stamp_){
      const bool reset=presence_stamp_<0||at<=presence_stamp_||at-presence_stamp_>.25;
      const double alpha=reset?1.:1-std::exp(-(at-presence_stamp_)/.15);
      if(reset)regions_.clear();
      for(int j=0;j<40;++j){const double raw=a->raw[j*7];const double q=std::isfinite(raw)?1/(1+std::exp(-std::clamp(raw,-80.,80.))):0.;
        probabilities_[j]=(1-alpha)*probabilities_[j]+alpha*q;}
      presence_stamp_=at;
    }
    const int lower=get_parameter("bin_search_min").as_int(),upper=get_parameter("bin_search_max").as_int();
    if(lower<0||upper>39||lower>upper)throw std::runtime_error("invalid bin search range");
    regions_=camera_regions(probabilities_,regions_,val("presence_threshold"),val("presence_hold_threshold"),val("presence_edge_threshold"),lower,upper);
    if(regions_.size()==1){result.bin_left=regions_[0].left;result.bin_right=regions_[0].right;}
  }
  // Each new A2 frame scores every track at that frame's own time and ego pose.
  bool accumulate_camera_evidence(double t){
    bool ambiguous=false;
    auto& tracks=tracker_.tracks();
    const auto [optical,camera]=camera_pose();
    for(const auto& frame:a2s_){
      const double at=stamp(frame->header);
      if(at<=last_a2_stamp_||at>t)continue;
      last_a2_stamp_=at;
      if(!frame->history_ready)continue;
      const auto ego=pose_at(at);
      if(!ego)continue;
      last_camera_stamp_=at;
      std::array<double,40> probabilities{};
      for(int j=0;j<40;++j){const double raw=frame->raw[j*7];probabilities[j]=std::isfinite(raw)?1/(1+std::exp(-std::clamp(raw,-80.,80.))):0.;}
      std::vector<TrackView> views;views.reserve(tracks.size());
      for(auto& tr:tracks){
        Track at_frame=tr;tracker_.peek(tr,at,at_frame.x,at_frame.P);
        TrackView v;v.track=&tr;v.range=ego->to_body(at_frame.anchor()).norm();
        v.projection=project_box(track_rear_box(at_frame,*ego),optical,camera);
        if(v.projection.valid&&(v.projection.y1<235||v.projection.y0>485))v.projection.valid=false;
        v.span=bins_for(v.projection);
        v.can_own=tr.confirmed&&vehicle_extent(tr.raw_box)&&!tr.map_static;
        views.push_back(v);
      }
      ambiguous=assign_ownership(views,probabilities,bin_dilation_)||ambiguous;
      for(const auto& v:views)observe(*v.track,at,v.score,v.visibility,evidence_);
    }
    return ambiguous;
  }
  void write_track_row(const Track& tr,const a1_fusion_cpp::msg::ClusterTrack& out,double t,const Projection& p,bool candidate,bool visible){
    if(!track_csv_)return;
    const Eigen::Vector3d rc=tr.raw_box.center(),rd=tr.raw_box.hi-tr.raw_box.lo;
    track_csv_<<std::setprecision(15)<<t<<','<<tr.id<<','<<tr.measured<<','<<tr.hits<<','<<out.observation_age_s<<','<<out.geometry_status<<','<<out.selection_reason;
    for(int k=0;k<3;++k)track_csv_<<','<<rc[k];
    for(int k=0;k<3;++k)track_csv_<<','<<rd[k];
    track_csv_<<','<<tr.raw_box.points;
    for(int k=0;k<3;++k)track_csv_<<','<<out.center_m[k];
    track_csv_<<','<<out.range_m<<','<<out.relative_velocity_mps[0]<<','<<out.relative_velocity_mps[1]<<','<<out.position_covariance_xy[0]<<','<<out.position_covariance_xy[3]<<','<<p.valid<<','<<p.x0<<','<<p.y0<<','<<p.x1<<','<<p.y1<<','<<tr.camera_score<<','<<out.camera_state<<','<<candidate<<','<<visible<<','<<tr.raw_box.footprint.valid<<','<<tr.raw_box.footprint.length<<','<<tr.raw_box.footprint.width<<','<<tr.raw_box.footprint.yaw;
    track_csv_<<','<<tr.evidence<<','<<tr.velocity()<<','<<tr.heading()<<','<<tr.yaw_rate()<<','<<tr.weak_update<<','<<tr.guided<<','<<tr.rear_visible<<','<<out.near_surface_x_m<<','<<tr.static_ratio<<','<<tr.map_static<<','<<tr.map_covered<<','<<tr.far_score<<','<<tr.far_vehicle<<','<<tr.path_s<<','<<tr.path_n<<'\n';
  }
  void write_association_row(const Association& result,const std::vector<Track>& tracks,const std::array<std::array<int,4>,12>& stage_counts,const std::string& map_filter_status,int map_ground_removed,int map_wall_removed){
    if(!csv_)return;
    const double t=rclcpp::Time(result.header.stamp).seconds();
    csv_<<std::setprecision(15)<<t<<','<<result.status<<','<<result.candidate_unique<<','<<result.verified_match<<','<<result.candidate_track_id<<','<<result.bin_left<<','<<result.bin_right<<','<<result.ground_points<<','<<result.nonground_points<<','<<result.tracks.size()<<','<<result.image_delta_s<<','<<result.a2_delta_s<<','<<result.processing_ms<<',';
    for(size_t i=0;i<result.candidate_track_ids.size();++i){if(i)csv_<<';';csv_<<result.candidate_track_ids[i];}csv_<<',';
    for(size_t i=0;i<result.lidar_candidate_track_ids.size();++i){if(i)csv_<<';';csv_<<result.lidar_candidate_track_ids[i];}
    csv_<<','<<result.association_ambiguous<<','<<result.primary_track_id<<','<<result.standby_track_id<<','<<result.primary_status<<',';
    for(const auto& tr:tracks)csv_<<tr.id<<':'<<vehicle_geometry_status(tr)<<':'<<tr.raw_box.points<<':'<<(tr.raw_box.hi-tr.raw_box.lo).transpose()<<';';
    csv_<<',';for(size_t i=0;i<stage_counts.size();++i){csv_<<i*10<<'-'<<(i+1)*10<<':';for(int n:stage_counts[i])csv_<<n<<'/';csv_<<';';}
    csv_<<','<<last_camera_stamp_<<',';for(const auto& region:regions_)csv_<<region.left<<':'<<region.right<<':'<<region.peak<<';';
    csv_<<',';for(double q:probabilities_)csv_<<q<<';';
    csv_<<','<<result.primary_camera_age_s<<','<<map_filter_status<<','<<map_ground_removed<<','<<map_wall_removed<<','<<result.ego_status<<','<<result.raceline_status<<','<<result.dropped_scans<<','<<scan_rate_.last_interval()<<'\n';csv_.flush();
    if(track_csv_)track_csv_.flush();
  }
  bool render_needed() const {return show_window_||recorder_||!preview_.empty()||image_pub_->get_subscription_count()>0;}
  // All HighGUI calls stay on this one thread: window toolkits are not thread-agnostic,
  // and drawing must never delay perception.
  void render_loop(){
    if(show_window_){
      cv::namedWindow(kWindow,cv::WINDOW_AUTOSIZE);
      cv::setMouseCallback(kWindow,[](int event,int x,int y,int,void* context){static_cast<ExposureControl*>(context)->mouse(event,x,y);},exposure_.get());
    }
    std::shared_ptr<const RenderSnapshot> drawn;double last_cloud=0;
    while(!stop_&&rclcpp::ok()){
      const auto begin=std::chrono::steady_clock::now();
      std::shared_ptr<const RenderSnapshot> latest;
      {std::lock_guard<std::mutex> lock(ui_mutex_);latest=latest_;}
      try{
        if(latest&&latest!=drawn){compose_view(*latest);ui_result_=latest->result;last_cloud=latest->received;drawn=latest;}
        if(render_needed())draw_dashboard(now().seconds()-last_cloud>.3);
      }catch(const std::exception& e){RCLCPP_ERROR_THROTTLE(get_logger(),*get_clock(),2000,"render: %s",e.what());}
      std::this_thread::sleep_until(begin+std::chrono::milliseconds(100));
    }
  }
  void compose_view(const RenderSnapshot& s){
    const auto& result=s.result;
    cv::Mat view=s.image?cv_bridge::toCvCopy(s.image,"bgr8")->image:cv::Mat::zeros(720,1280,CV_8UC3);
    if(view.cols!=1280||view.rows!=720)throw std::runtime_error("expected 1280x720 camera ROI");
    for(const auto& region:s.regions)cv::rectangle(view,cv::Rect(region.left*32,235,(region.right-region.left+1)*32,250),cv::Scalar(0,230,230),2);
    for(int j=0;j<40;++j){int bar=int(50*s.probabilities[j]);if(bar>0)cv::rectangle(view,cv::Rect(j*32,710-bar,30,bar),cv::Scalar(40,200,40),-1);}
    cv::Mat bev(720,480,CV_8UC3,cv::Scalar(20,20,20));
    const double range=val("bev_forward_range_m");if(range<20||range>150)throw std::runtime_error("BEV range must be 20..150 m");
    const double scale=600/range,roi_angle=val("lidar_roi_center_deg")*M_PI/180;
    auto pixel=[scale](double x,double y){return cv::Point(int(240-y*scale),int(680-x*scale));};
    const auto [optical,camera]=camera_pose();
    for(const auto& q:s.ground){auto uv=pixel(q.x(),q.y());if(uv.x>=0&&uv.x<480&&uv.y>=0&&uv.y<720)bev.at<cv::Vec3b>(uv)=cv::Vec3b(45,65,45);}
    for(int meters=10;meters<=int(range);meters+=10){auto pt=pixel(meters,0);if(pt.y<145)continue;cv::line(bev,{0,pt.y},{479,pt.y},{40,40,40},1);cv::putText(bev,std::to_string(meters)+"m",{5,pt.y-3},cv::FONT_HERSHEY_SIMPLEX,.4,{150,150,150},1);}
    cv::line(bev,{240,0},{240,719},{70,70,70});cv::circle(bev,pixel(0,0),6,{255,255,255},-1);
    // Camera and cloud geometry share the exact same transform as image projection.
    // The 180-degree LiDAR half-plane is wider than the camera image FOV.
    // Crop heading is independent of the camera extrinsic transform.
    const double arc_radius=std::min(25.,215/scale);
    auto sector_point=[&](double r,double angle){return pixel(r*std::cos(angle),r*std::sin(angle));};
    cv::line(bev,sector_point(arc_radius,roi_angle-M_PI/2),pixel(0,0),{180,180,180},1);
    cv::line(bev,pixel(0,0),sector_point(arc_radius,roi_angle+M_PI/2),{180,180,180},1);
    for(int deg=-90;deg<90;++deg){double a0=roi_angle+deg*M_PI/180,a1=roi_angle+(deg+1)*M_PI/180;
      cv::line(bev,sector_point(arc_radius,a0),sector_point(arc_radius,a1),{90,90,90},1);}
    cv::arrowedLine(bev,pixel(0,0),sector_point(arc_radius,roi_angle),{230,230,230},2);
    // Every point, drawn bright: the view must show the real density, not a subsample.
    for(const auto& q:s.nonground){auto uv=pixel(q.x(),q.y());if(uv.x>=0&&uv.x<480&&uv.y>=145&&uv.y<720)bev.at<cv::Vec3b>(uv)=cv::Vec3b(210,210,210);}
    const double cx=val("camera_cx"),fx=val("camera_fx");
    for(double u:{0.,cx,1279.}){
      Eigen::Vector3d ray=optical.transpose()*Eigen::Vector3d((u-cx)/fx,0,1);
      if(ray.head<2>().norm()>1e-6){ray*=18/ray.head<2>().norm();
        cv::line(bev,pixel(camera.x(),camera.y()),pixel(camera.x()+ray.x(),camera.y()+ray.y()),{0,220,220},u==cx?2:1);}
    }
    cv::circle(bev,pixel(camera.x(),camera.y()),4,{0,220,220},-1);
    for(const auto& tr:s.tracks){
      if(tr.matched&&tr.raw_image.valid)cv::rectangle(view,{int(tr.raw_image.x0),int(tr.raw_image.y0)},{int(tr.raw_image.x1),int(tr.raw_image.y1)},{130,130,130},1);
      if(tr.image.valid){cv::rectangle(view,{int(tr.image.x0),int(tr.image.y0)},{int(tr.image.x1),int(tr.image.y1)},tr.color,tr.matched?3:2);cv::putText(view,tr.label,{int(tr.image.x0),std::max(85,int(tr.image.y0)-5)},cv::FONT_HERSHEY_SIMPLEX,.45,tr.color,1);}
      cv::rectangle(bev,pixel(tr.box.hi.x(),tr.box.hi.y()),pixel(tr.box.lo.x(),tr.box.lo.y()),tr.color,tr.matched?3:1);
      cv::putText(bev,tr.tag,pixel(tr.box.center().x(),tr.box.center().y()),cv::FONT_HERSHEY_SIMPLEX,.45,tr.color,1);
    }
    cv::rectangle(view,{0,0},{1279,100},{0,0,0},-1);
    cv::putText(view,"PRIMARY: "+result.primary_status+" | camera: "+result.status,{12,25},cv::FONT_HERSHEY_SIMPLEX,.7,{0,220,255},2);
    cv::putText(view,"PREVIEW ONLY | ego: "+result.ego_status+" | raceline: "+result.raceline_status+" | camera metrics UNUSED",{12,52},cv::FONT_HERSHEY_SIMPLEX,.55,{255,255,255},1);
    std::string flags=std::string(result.calibration_verified?"EXTRINSICS VERIFIED":"EXTRINSICS UNVERIFIED")+" | "+(result.clock_verified?"CLOCK VERIFIED":"CLOCK UNVERIFIED")+cv::format(" | image dt %.0f ms | A2 dt %.0f ms | %.0f ms",result.image_delta_s*1000,result.a2_delta_s*1000,result.processing_ms);
    cv::putText(view,flags,{12,80},cv::FONT_HERSHEY_SIMPLEX,.5,{0,180,255},1);
    cv::putText(bev,"CAMERA FORWARD = UP",{10,25},cv::FONT_HERSHEY_SIMPLEX,.6,{255,255,255},1);
    cv::putText(bev,cv::format("WHITE: LiDAR ROI 180deg / right %.0fdeg",-val("lidar_roi_center_deg")),{10,50},cv::FONT_HERSHEY_SIMPLEX,.45,{230,230,230},1);
    cv::putText(bev,"YELLOW: camera axis / FOV (unverified)",{10,75},cv::FONT_HERSHEY_SIMPLEX,.45,{0,220,220},1);
    cv::putText(bev,"LEFT = LiDAR +X | RIGHT = LiDAR -X",{10,100},cv::FONT_HERSHEY_SIMPLEX,.43,{230,230,230},1);
    cv::putText(bev,show_debug_?"DEBUG clusters are NOT vehicle detections":"Red: matched / cyan: held / yellow: camera",{10,125},cv::FONT_HERSHEY_SIMPLEX,.43,{230,230,230},1);
    const bool map_on=s.map_status=="ok";
    cv::putText(bev,(map_on?std::string("MAP: "):std::string("MAP OFF: "))+s.map_status+" G:"+std::to_string(s.map_ground_removed)+" W:"+std::to_string(s.map_wall_removed),{10,150},cv::FONT_HERSHEY_SIMPLEX,.42,map_on?cv::Scalar(230,230,230):cv::Scalar(70,70,255),1);
    cv::Mat top;cv::hconcat(view,bev,top);cv::resize(top,ui_top_,{1600,655});
  }
  void draw_dashboard(bool stale){
    const double time=now().seconds();
    EgoMsg::ConstSharedPtr ego_ui;BrakeMsg::ConstSharedPtr brake_ui;
    {std::lock_guard<std::mutex> lock(ui_mutex_);ego_ui=ego_ui_;brake_ui=brake_ui_;}
    cv::Mat panel(860,1600,CV_8UC3,cv::Scalar(16,20,27));ui_top_.copyTo(panel(cv::Rect(0,0,1600,655)));
    auto text=[&](const std::string& v,int x,int y,double size,cv::Scalar c){cv::putText(panel,v,{x,y},cv::FONT_HERSHEY_SIMPLEX,size,c,1,cv::LINE_AA);};
    auto fresh=[&](const std_msgs::msg::Header& h){double age=time-stamp(h);return age>=0&&age<=.25;};
    const bool ego_ok=ego_ui&&fresh(ego_ui->header)&&ego_ui->valid&&std::isfinite(ego_ui->speed_mps);
    const bool ego_logged=ego_ui&&fresh(ego_ui->header)&&!ego_ok&&std::isfinite(ego_ui->speed_unverified_mps);
    const bool accel_ok=ego_ok&&std::isfinite(ego_ui->longitudinal_acceleration_mps2);
    const bool brake_fresh=brake_ui&&fresh(brake_ui->header);
    const bool brake_matches=!stale&&brake_fresh&&brake_ui->source=="lidar_primary"&&brake_ui->track_id==ui_result_.primary_track_id&&!brake_ui->what_if;
    const bool brake_ok=brake_matches&&brake_ui->valid&&time<=rclcpp::Time(brake_ui->valid_until).seconds()&&std::isfinite(brake_ui->request_mps2);
    const bool brake_logged=brake_matches&&!brake_ok&&std::isfinite(brake_ui->request_unverified_mps2);
    cv::Scalar white(235,240,245),muted(155,165,180),red(70,70,255),cyan(240,210,80);
    for(int i=0;i<4;++i)cv::rectangle(panel,{i*400+8,665},{i*400+392,824},{30,36,46},-1);
    text("EGO SPEED / NAVIGATION",24,694,.55,muted);
    text(ego_ok?cv::format("%.2f m/s",ego_ui->speed_mps):ego_logged?cv::format("%.2f m/s",ego_ui->speed_unverified_mps):"-- m/s",24,737,.95,ego_logged?muted:white);
    text(ego_ok?cv::format("%.1f km/h",ego_ui->speed_mps*3.6):ego_logged?"UNVERIFIED":"WAITING FOR VALID EGO",24,770,.5,cyan);
    text(ego_ui?ego_ui->source:"No ego source",24,800,.44,muted);
    text("LONGITUDINAL ACCELERATION",424,694,.5,muted);
    text(accel_ok?cv::format("%+.2f m/s^2",ego_ui->longitudinal_acceleration_mps2):"-- m/s^2",424,737,.85,white);
    text("Body forward / unavailable if unknown",424,770,.46,cyan);
    std::string ego_reason=!ego_ui?"NO EGO TELEMETRY":!fresh(ego_ui->header)?"EGO INPUT STALE":ego_ui->reason;
    text(ego_reason.substr(0,39),424,800,.40,muted);
    text(cv::format("CAMERA TARGET / %.0f-%.0fm",val("target_min_range_m"),val("target_max_range_m")),824,694,.5,muted);
    int row=0;
    if(!stale)for(auto id:ui_result_.lidar_candidate_track_ids)for(const auto& tr:ui_result_.tracks)if(tr.id==id&&row<2){
      text(cv::format("%s T%ld %.1fm Vx%+.1f%s",id==ui_result_.primary_track_id?"P":"S",long(id),tr.range_m,tr.relative_velocity_mps[0],ui_result_.primary_status=="measured"&&tr.measured?"":" ~"),824,729+row*38,.54,tr.camera_matched?red:cv::Scalar(0,165,255));++row;
    }
    if(row==0)text("NO TARGET",824,737,.44,white);
    text(row==0?"Waiting for camera evidence":ui_result_.primary_status=="measured"?"Nearest rear surface / relative Vx":"HELD/PREDICTED: NOT VERIFIED",824,800,.43,muted);
    text("P5 BRAKE REQUEST / SHADOW",1224,694,.49,muted);
    text(brake_ok?cv::format("%+.2f m/s^2",brake_ui->request_mps2):brake_logged?cv::format("%+.2f m/s^2",brake_ui->request_unverified_mps2):"-- m/s^2",1224,737,.85,brake_ok?red:brake_logged?muted:white);
    text(brake_ok?cv::format("Raw P5 %+.2f  gate %.2f",brake_ui->raw_p5_mps2,brake_ui->gate):brake_logged?"UNVERIFIED: "+brake_ui->reason.substr(0,23):!brake_fresh?"P5 INPUT MISSING / STALE":brake_ui->what_if?"WHAT-IF EXCLUDED":brake_ui->reason.substr(0,35),1224,770,.42,muted);
    text("Selected primary only",1224,800,.37,muted);
    text("SHADOW ONLY - NO CAN OUTPUT",16,847,.48,red);
    text(cv::format("LiDAR view: %.0fm | red: matched | cyan: held | yellow: camera",val("bev_forward_range_m")),460,847,.45,white);
    if(stale){cv::rectangle(panel,{0,0},{1599,66},{20,20,110},-1);text("LIDAR / ASSOCIATION INPUT STALE - DISPLAY NOT LIVE",20,43,.8,white);}
    exposure_->draw(panel);
    if(recorder_){
      text(recorder_->failed()?"RECORDING ERROR":cv::format("REC MP4 %05d",recorder_->segment()),1370,847,.43,red);
      recorder_->push(panel,time);
    }
    auto header=ui_result_.header;if(stale)header.stamp=now();
    if(image_pub_->get_subscription_count()>0)image_pub_->publish(*cv_bridge::CvImage(header,"bgr8",panel).toImageMsg());
    if(!preview_.empty()&&time-last_preview_>1){cv::imwrite(preview_,panel);last_preview_=time;}
    if(show_window_){cv::imshow(kWindow,panel);exposure_->key(cv::waitKeyEx(1));}
  }

  static constexpr const char* kWindow="A1 camera-LiDAR association (unverified preview)";
  std::unique_ptr<MapFilterRos> map_filter_;rclcpp::Publisher<Cloud>::SharedPtr map_removed_pub_;
  std::unique_ptr<ExposureControl> exposure_;
  std::unique_ptr<UiRecorder> recorder_;
  std::thread render_thread_;std::atomic<bool> stop_{false};std::mutex ui_mutex_;
  std::shared_ptr<const RenderSnapshot> latest_;
  cv::Mat ui_top_;Association ui_result_;
  EgoMsg::ConstSharedPtr ego_ui_;BrakeMsg::ConstSharedPtr brake_ui_;
  rclcpp::Subscription<EgoMsg>::SharedPtr ego_sub_;rclcpp::Subscription<BrakeMsg>::SharedPtr brake_sub_;
  std::array<double,40> probabilities_{};double presence_stamp_{-1};std::vector<BinRegion> regions_;int64_t primary_id_{-1};
  Tracker tracker_;EvidenceSelector selector_;EvidenceConfig evidence_;FarConfig far_;Distortion distortion_;
  a1::EgoHistory egos_;EgoLink ego_link_;Eigen::Vector2d lidar_lever_=Eigen::Vector2d::Zero();double lidar_yaw_{M_PI/2};
  Ins::ConstSharedPtr gnss_status_;double gnss_status_offset_{};
  rclcpp::Subscription<Ins>::SharedPtr gnss_status_sub_;rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gnss_odom_sub_;
  std::optional<Raceline> raceline_;Pose2 map_from_raceline_;AlignmentMonitor alignment_;bool raceline_verified_{};
  int ego_path_hint_{-1},bin_dilation_{1};double ego_path_s_{},ego_path_n_{};
  std::unique_ptr<patchwork::PatchWorkpp> ground_;double weak_min_range_{40.},camera_silent_s_{2.};
  double last_stamp_{-1},last_preview_{},last_a2_stamp_{-1},last_camera_stamp_{-1};
  size_t dropped_scans_{};ScanRate scan_rate_;
  std::string preview_;std::ofstream csv_,track_csv_;bool show_window_{},show_debug_{};
  std::deque<Image::ConstSharedPtr> images_;std::deque<A2::ConstSharedPtr> a2s_;
  std::deque<std::pair<std::chrono::steady_clock::time_point,Cloud::ConstSharedPtr>> pending_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<Image>::SharedPtr image_sub_;rclcpp::Subscription<A2>::SharedPtr a2_sub_;rclcpp::Subscription<Cloud>::SharedPtr cloud_sub_;
  rclcpp::Publisher<Image>::SharedPtr image_pub_;rclcpp::Publisher<Association>::SharedPtr assoc_pub_;rclcpp::Publisher<Cloud>::SharedPtr ground_pub_,nonground_pub_,clusters_pub_,front_pub_;
};
int main(int argc,char** argv){rclcpp::init(argc,argv);try{rclcpp::spin(std::make_shared<AssociationNode>());}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}rclcpp::shutdown();}
