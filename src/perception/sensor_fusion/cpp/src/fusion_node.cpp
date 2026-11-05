#include "a1_fusion_cpp/msg/ego_motion.hpp"
#include "a1_fusion_cpp/msg/association.hpp"
#include "a1_fusion/core.hpp"
#include "a1_fusion_cpp/msg/a2_features.hpp"
#include "a1_fusion_cpp/msg/brake_estimate.hpp"
#include <rclcpp/rclcpp.hpp>
#include <gsof_msgs/msg/navigation_solution49.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <cstring>
#include <fstream>

using Features=a1_fusion_cpp::msg::A2Features;
using Output=a1_fusion_cpp::msg::BrakeEstimate;
using Ins=gsof_msgs::msg::NavigationSolution49;
using Cloud=sensor_msgs::msg::PointCloud2;
class FusionNode: public rclcpp::Node {
 public:
  FusionNode():Node("a1_fusion") {
    mode_=declare_parameter<std::string>("mode","fusion");
    if(mode_!="fusion"&&mode_!="camera"&&mode_!="association") throw std::invalid_argument("mode must be camera, fusion or association");
    cfg_.clock_valid=declare_parameter("clock_verified",false);
    cfg_.calibration_valid=declare_parameter("extrinsics_verified",false);
    cfg_.camera_metric_valid=declare_parameter("camera_metric_validated",false);
    cfg_.bumper_x=declare_parameter("bumper_x_m",0.0);
    cfg_.plan_speed=declare_parameter("plan_speed_mps",-1.0);
    cfg_.max_age=declare_parameter("max_age_s",.25);
    cfg_.pair_slop=declare_parameter("pair_slop_s",.08);
    cfg_.camera_fx=declare_parameter("camera_fx",640.0);
    cfg_.camera_cx=declare_parameter("camera_cx",640.0);
    cfg_.camera_width=declare_parameter("camera_width",1280.0);
    cfg_.corridor_half_width=declare_parameter("corridor_half_width_m",1.8);
    cfg_.object_threshold=declare_parameter("object_threshold",.5);
    lidar_from_=transform("base_from_lidar");
    cfg_.camera_from_base=transform("camera_from_base");
    ins_yaw_=declare_parameter("ins_to_vehicle_yaw_rad",0.0);
    ins_verified_=declare_parameter("ins_mount_verified",false);
    auto lever=declare_parameter<std::vector<double>>("base_to_ins_m",{0,0,0});
    if(lever.size()!=3) throw std::invalid_argument("base_to_ins_m needs xyz");
    lever_=a1::Vec(lever[0],lever[1],lever[2]);
    ins_offset_=declare_parameter("ins_stamp_offset_s",0.0);
    time_field_=declare_parameter<std::string>("point_time_field","timestamp");
    cloud_frame_=declare_parameter<std::string>("lidar_frame","hesai_lidar");
    camera_frame_=declare_parameter<std::string>("camera_frame","hikrobot_camera_vis");
    const auto log=declare_parameter<std::string>("csv_path","");
    if(!log.empty()) {
      csv_.open(log);if(!csv_) throw std::runtime_error("cannot open CSV: "+log);
      csv_<<"stamp,source,ego_source,valid,reason,track,bin,x,y,range,gap,vrel,ego,lead,plan,gate,p5,request,what_if\n";
      csv_.precision(17);
    }
    if(cfg_.max_age<=0||cfg_.pair_slop<=0||cfg_.camera_fx<=0||cfg_.camera_width<=0||cfg_.object_threshold<=0||cfg_.object_threshold>=1)
      throw std::invalid_argument("invalid thresholds or intrinsics");
    engine_=std::make_unique<a1::Engine>(cfg_);
    primary_coast_s_=declare_parameter("primary_coast_s",.3);
    primary_variance_=declare_parameter("primary_max_position_variance_m2",1.);
    primary_motion_verified_=declare_parameter("primary_motion_verified",false);
    if(!std::isfinite(primary_coast_s_)||primary_coast_s_<=0||!std::isfinite(primary_variance_)||primary_variance_<=0)throw std::invalid_argument("invalid primary freshness/variance");
    ego_source_=declare_parameter<std::string>("ego_source","imu");
    if(ego_source_!="imu"&&ego_source_!="ins")throw std::invalid_argument("ego_source must be imu or ins");
    a1::ImuConfig ic;ic.stationary_start_confirmed=declare_parameter("imu_stationary_start_confirmed",false);
    ic.mount_verified=declare_parameter("imu_mount_verified",false);
    ic.calibration_s=declare_parameter("imu_calibration_s",2.0);
    ic.max_integration_s=declare_parameter("imu_max_integration_s",10.0);
    ic.base_from_imu=transform("base_from_imu").linear();
    if(ic.calibration_s<.5||ic.max_integration_s<=0||!std::isfinite(ic.max_integration_s))throw std::invalid_argument("invalid IMU time limits");
    imu_=std::make_unique<a1::ImuSpeed>(ic);
    imu_frame_=declare_parameter<std::string>("imu_frame","ins");
    ego_pub_=create_publisher<a1_fusion_cpp::msg::EgoMotion>("/lead_fusion/ego_motion",10);
    pub_=create_publisher<Output>("/lead_fusion/estimate",10);
    const auto ins_topic=declare_parameter<std::string>("ins_topic","/gsof_client/gsof/ins_solution_49");
    const auto imu_topic=declare_parameter<std::string>("imu_topic","/gnss/imu");
    const auto lidar_topic=declare_parameter<std::string>("lidar_topic","/lidar_points");
    if(ego_source_=="ins")ins_sub_=create_subscription<Ins>(ins_topic,rclcpp::SensorDataQoS(),[this](Ins::ConstSharedPtr m){ins(*m);});
    if(ego_source_=="imu")imu_sub_=create_subscription<sensor_msgs::msg::Imu>(imu_topic,rclcpp::SensorDataQoS(),[this](sensor_msgs::msg::Imu::ConstSharedPtr m){
      if(ego_source_!="imu")return;
      const double t=rclcpp::Time(m->header.stamp).seconds();
      if(m->header.frame_id!=imu_frame_||m->linear_acceleration_covariance[0]<0||m->angular_velocity_covariance[0]<0){egos_.clear();engine_->reset();a1::Ego bad;bad.stamp=t;publish_ego(bad,"imu_frame_or_fields_invalid");invalid("imu_frame_or_fields_invalid",t);return;}
      auto e=imu_->update(t,a1::Vec(m->linear_acceleration.x,m->linear_acceleration.y,m->linear_acceleration.z),a1::Vec(m->angular_velocity.x,m->angular_velocity.y,m->angular_velocity.z));
      publish_ego(e,imu_->reason());
      egos_.push_back(e);while(egos_.size()>400)egos_.pop_front();
      if(!e.valid){engine_->reset();invalid(imu_->reason(),t);}
    });
    if(mode_=="association")association_sub_=create_subscription<a1_fusion_cpp::msg::Association>("/lead_fusion/association",rclcpp::SensorDataQoS().keep_last(2),[this](a1_fusion_cpp::msg::Association::ConstSharedPtr m){primary(*m);});
    if(mode_!="association")a2_sub_=create_subscription<Features>("/lead_fusion/a2",rclcpp::SensorDataQoS().keep_last(2),[this](Features::ConstSharedPtr m){features(*m);});
    if(mode_=="fusion")lidar_sub_=create_subscription<Cloud>(lidar_topic,rclcpp::SensorDataQoS().keep_last(1),[this](Cloud::ConstSharedPtr m){cloud(*m);});
    timer_=create_wall_timer(std::chrono::milliseconds(100),[this]{
      if(now().seconds()-last_publish_>cfg_.max_age) invalid("input_timeout",now().seconds());
    });
    RCLCPP_INFO(get_logger(),"C++ %s shadow P5; ego_source=%s clock=%d extrinsics=%d",mode_.c_str(),ego_source_.c_str(),cfg_.clock_valid,cfg_.calibration_valid);
  }
 private:
  Eigen::Isometry3d transform(const std::string& key) {
    auto v=declare_parameter<std::vector<double>>(key,{0,0,0,0,0,0});
    if(v.size()!=6||!std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x);}))throw std::invalid_argument(key+" needs finite xyz roll pitch yaw");
    Eigen::Isometry3d t=Eigen::Isometry3d::Identity();t.translation()=a1::Vec(v[0],v[1],v[2]);
    t.linear()=(Eigen::AngleAxisd(v[5],a1::Vec::UnitZ())*Eigen::AngleAxisd(v[4],a1::Vec::UnitY())*Eigen::AngleAxisd(v[3],a1::Vec::UnitX())).toRotationMatrix();return t;
  }
  void ins(const Ins& m) {
    if(ego_source_!="ins")return;
    constexpr double rad=3.14159265358979323846/180;
    a1::Ego e;e.stamp=rclcpp::Time(m.header.stamp).seconds()+ins_offset_;
    e.yaw=(90-m.heading)*rad-ins_yaw_;e.yaw_rate=-m.ang_rate_down*rad;
    e.velocity=a1::Vec(m.velocity.east,m.velocity.north,-m.velocity.down);
    const a1::Vec lever_world=Eigen::AngleAxisd(e.yaw,a1::Vec::UnitZ())*lever_;
    e.velocity-=a1::Vec(0,0,e.yaw_rate).cross(lever_world);
    // This baseline uses planar motion: tilted INS/roads require a full 6DoF backend.
    e.valid=ins_verified_&&m.status.imu_alignment==4&&m.status.gnss>=1&&m.status.gnss<=6&&
      e.velocity.allFinite()&&std::isfinite(e.yaw)&&std::isfinite(e.yaw_rate)&&
      std::isfinite(m.roll)&&std::isfinite(m.pitch)&&std::abs(m.roll)<10&&std::abs(m.pitch)<10;
    if(!egos_.empty()) {
      const auto& prev=egos_.back();const double dt=e.stamp-prev.stamp;
      if(dt<=0||dt>.25||!e.valid||!prev.valid){egos_.clear();engine_->reset();}
      else e.position=prev.position+.5*(prev.velocity+e.velocity)*dt;
    }
    egos_.push_back(e);while(egos_.size()>400) egos_.pop_front();
  }
  a1::Ego ego_at(double t) const {
    for(auto it=egos_.rbegin();it!=egos_.rend();++it) if(it->stamp<=t) {
      auto e=*it;double dt=t-e.stamp;
      if(dt>cfg_.pair_slop){e.valid=false;return e;}
      // Propagate causally from an actual navigation observation, bounded by slop.
      e.position+=e.velocity*dt;e.yaw+=e.yaw_rate*dt;return e;
    }
    return {};
  }
  a1::A2 a2_at(double t) const {
    for(auto it=a2s_.rbegin();it!=a2s_.rend();++it) if(it->stamp<=t)return *it;
    return {};
  }
  void features(const Features& m) {
    if(mode_=="association")return;
    if(m.header.frame_id!=camera_frame_){invalid("camera_frame_mismatch",now().seconds());return;}
    a1::A2 a;a.stamp=rclcpp::Time(m.header.stamp).seconds();a.raw=m.raw;a.ready=m.history_ready;
    if(!a2s_.empty()&&a.stamp<=a2s_.back().stamp){a2s_.clear();engine_->reset();invalid("a2_nonmonotonic",a.stamp);return;}
    a2s_.push_back(a);while(a2s_.size()>64)a2s_.pop_front();
    if(mode_=="camera")publish(engine_->camera(a,ego_at(a.stamp),now().seconds()));
  }
  void primary(const a1_fusion_cpp::msg::Association& m){
    const double t=rclcpp::Time(m.header.stamp).seconds();
    if(m.header.frame_id!="lidar_vehicle"){invalid("primary_frame_mismatch",t);return;}
    if(t<=last_primary_){invalid("primary_nonmonotonic",t);return;}last_primary_=t;
    if(!m.clock_verified||!cfg_.clock_valid){invalid("clock_unverified",t);return;}
    if(!m.calibration_verified||!cfg_.calibration_valid){invalid("extrinsics_unverified",t);return;}
    const auto e=ego_at(t);
    for(const auto& tr:m.tracks)if(tr.id==m.primary_track_id&&tr.id>=0){
      if(m.primary_status!="measured"||!tr.camera_matched||!tr.measured||m.association_ambiguous||
         !std::isfinite(m.primary_camera_age_s)||m.primary_camera_age_s<0||m.primary_camera_age_s>primary_coast_s_){
        invalid("primary_camera_unconfirmed",t);return;
      }
      a1::PrimaryTarget target;target.id=tr.id;target.stamp=t;
      const double age=tr.observation_age_s;
      target.valid_until=t+primary_coast_s_-age;
      target.center=a1::Vec(tr.center_m[0],tr.center_m[1],tr.center_m[2]);
      target.gap=tr.center_m[0]-.5*tr.size_m[0]-cfg_.bumper_x;
      // Body-coordinate derivative -> relative longitudinal velocity (planar yaw correction).
      target.relative_speed=tr.apparent_velocity_mps[0]-e.yaw_rate*tr.center_m[1];
      target.predicted=!tr.measured;
      target.valid=tr.confirmed&&tr.prediction_valid&&std::isfinite(age)&&age>=0&&age<=primary_coast_s_&&
        std::isfinite(tr.size_m[0])&&tr.size_m[0]>=0&&
        std::isfinite(tr.position_covariance_xy[0])&&std::isfinite(tr.position_covariance_xy[3])&&
        tr.position_covariance_xy[0]>=0&&tr.position_covariance_xy[3]>=0&&
        tr.position_covariance_xy[0]<=primary_variance_&&tr.position_covariance_xy[3]<=primary_variance_;
      auto result=engine_->primary(target,e,now().seconds());
      if(result.valid&&!primary_motion_verified_){result.valid=false;result.request=a1::nan;result.reason="primary_motion_unverified";}
      publish(result);return;
    }
    invalid("no_valid_primary",t);
  }
  void cloud(const Cloud& m) {
    if(mode_!="fusion")return;
    const double header=rclcpp::Time(m.header.stamp).seconds();
    if(m.header.frame_id!=cloud_frame_){invalid("lidar_frame_mismatch",header);return;}
    if(!cfg_.calibration_valid){invalid("extrinsics_unverified",header);return;}
    // Inspect fields and strides; never assume the historical bag's byte offsets.
    const sensor_msgs::msg::PointField* tf=nullptr;
    for(const auto& f:m.fields) if(f.name==time_field_)tf=&f;
    if(!tf||tf->datatype!=sensor_msgs::msg::PointField::FLOAT64||tf->count!=1||tf->offset+8>m.point_step||m.is_bigendian||
       m.row_step<static_cast<uint64_t>(m.width)*m.point_step||m.data.size()<static_cast<uint64_t>(m.row_step)*m.height){invalid("point_time_schema_invalid",header);return;}
    pcl::PointCloud<pcl::PointXYZ> raw;
    for(const char* field:{"x","y","z"}) {
      bool ok=false;for(const auto& f:m.fields)if(f.name==field&&f.datatype==sensor_msgs::msg::PointField::FLOAT32&&f.count==1&&f.offset+4<=m.point_step)ok=true;
      if(!ok){invalid("xyz_schema_invalid",header);return;}
    }
    pcl::fromROSMsg(m,raw);
    std::vector<double> times;times.reserve(raw.size());double begin=1e100,end=-1e100;
    for(uint32_t row=0;row<m.height;++row)for(uint32_t col=0;col<m.width;++col){
      double t;std::memcpy(&t,&m.data[static_cast<size_t>(row)*m.row_step+static_cast<size_t>(col)*m.point_step+tf->offset],8);
      if(!std::isfinite(t)||std::abs(t-header)>.3){invalid("point_clock_invalid",header);return;}
      times.push_back(t);begin=std::min(begin,t);end=std::max(end,t);
    }
    if(times.empty()||end-begin>.15||end<=last_cloud_){invalid("scan_time_invalid",header);engine_->reset();last_cloud_=end;return;}
    last_cloud_=end;
    const auto e=ego_at(end);
    if(!e.valid||!cfg_.clock_valid){invalid("ins_or_clock_unverified",end);engine_->reset();return;}
    if(now().seconds()<end||now().seconds()-end>cfg_.max_age){invalid("stale_or_future_cloud",end);return;}
    auto points=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto seeds=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for(size_t i=0;i<raw.size();++i){
      a1::Vec p(raw[i].x,raw[i].y,raw[i].z);if(!p.allFinite())continue;
      p=lidar_from_*p;
      // Constant navigation twist during this scan, transformed to scan-end base.
      a1::Ego at=e;const double dt=times[i]-end;at.yaw+=e.yaw_rate*dt;at.position+=e.velocity*dt;
      p=a1::to_body(a1::to_world(p,at),e);
      if(p.x()<1||p.x()>80||std::abs(p.y())>10||p.z()<-3||p.z()>4)continue;
      points->push_back(pcl::PointXYZ(p.x(),p.y(),p.z()));
      if(p.x()<25&&std::abs(p.y())<7&&p.z()<-.5)seeds->push_back(points->back());
    }
    if(seeds->size()<30){invalid("ground_insufficient",end);return;}
    pcl::SACSegmentation<pcl::PointXYZ> seg;seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);seg.setAxis(Eigen::Vector3f::UnitZ());seg.setEpsAngle(.28);
    seg.setMethodType(pcl::SAC_RANSAC);seg.setDistanceThreshold(.12);seg.setMaxIterations(60);seg.setInputCloud(seeds);
    pcl::PointIndices inliers;pcl::ModelCoefficients plane;seg.segment(inliers,plane);
    if(plane.values.size()!=4||inliers.indices.size()<30){invalid("ground_fit_failed",end);return;}
    if(plane.values[2]<0)for(auto& x:plane.values)x=-x;
    auto above=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for(const auto& p:*points){double h=plane.values[0]*p.x+plane.values[1]*p.y+plane.values[2]*p.z+plane.values[3];if(h>.22&&h<2.8)above->push_back(p);}
    auto vox=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();pcl::VoxelGrid<pcl::PointXYZ> vg;vg.setInputCloud(above);vg.setLeafSize(.16,.16,.16);vg.filter(*vox);
    std::vector<a1::Detection> detections;
    if(!vox->empty()) {
      auto tree=pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();tree->setInputCloud(vox);
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;ec.setClusterTolerance(.65);ec.setMinClusterSize(3);ec.setMaxClusterSize(20000);ec.setSearchMethod(tree);ec.setInputCloud(vox);
      std::vector<pcl::PointIndices> clusters;ec.extract(clusters);
      for(const auto& group:clusters){
        a1::Vec lo=a1::Vec::Constant(1e9),hi=-lo;
        for(int idx:group.indices){const auto& p=(*vox)[idx];a1::Vec v(p.x,p.y,p.z);lo=lo.cwiseMin(v);hi=hi.cwiseMax(v);}
        const auto dim=hi-lo;if(std::max(dim.x(),dim.y())>8||std::max(dim.x(),dim.y())<.3)continue;
        detections.push_back({(lo+hi)*.5,lo,hi});
      }
    }
    publish(engine_->fusion(a2_at(end),e,end,detections,now().seconds()));
  }
  void publish_ego(const a1::Ego& e,const std::string& reason){
    a1_fusion_cpp::msg::EgoMotion m;m.header.stamp=rclcpp::Time(int64_t(e.stamp*1e9));m.header.frame_id="base_link";
    m.valid=e.valid;m.source="imu_integrated_experimental";m.reason=reason;
    m.speed_mps=e.valid?e.forward():a1::nan;
    m.longitudinal_acceleration_mps2=e.valid?std::cos(e.yaw)*e.acceleration.x()+std::sin(e.yaw)*e.acceleration.y():a1::nan;
    ego_pub_->publish(m);
  }
  void invalid(const std::string& reason,double stamp){a1::Result r;r.reason=reason;r.stamp=stamp;r.valid_until=stamp;r.source=mode_;publish(r);}
  void publish(const a1::Result& r) {
    Output m;m.header.stamp=rclcpp::Time(static_cast<int64_t>(r.stamp*1e9));m.header.frame_id="base_link";
    m.valid_until=rclcpp::Time(static_cast<int64_t>(r.valid_until*1e9));m.valid=r.valid;m.what_if=r.what_if;
    m.source=r.source;m.ego_source=r.ego_source.empty()?ego_source_:r.ego_source;m.reason=r.reason;m.gap_definition=r.gap_definition;m.track_id=r.track_id;m.bin=r.bin;
    m.x_m=r.x;m.y_m=r.y;m.range_m=r.range;m.gap_m=r.gap;m.relative_speed_mps=r.relative_speed;
    m.ego_speed_mps=r.ego_speed;m.lead_speed_mps=r.lead_speed;m.plan_speed_mps=r.plan_speed;
    m.gate=r.gate;m.raw_p5_mps2=r.raw_p5;m.request_mps2=r.request;pub_->publish(m);last_publish_=now().seconds();
    if(csv_)csv_<<r.stamp<<','<<r.source<<','<<m.ego_source<<','<<r.valid<<','<<r.reason<<','<<r.track_id<<','<<r.bin<<','<<r.x<<','<<r.y<<','<<r.range<<','<<r.gap<<','<<r.relative_speed<<','<<r.ego_speed<<','<<r.lead_speed<<','<<r.plan_speed<<','<<r.gate<<','<<r.raw_p5<<','<<r.request<<','<<r.what_if<<'\n';
  }
  rclcpp::Subscription<a1_fusion_cpp::msg::Association>::SharedPtr association_sub_;
  double primary_coast_s_{.3},primary_variance_{1.},last_primary_{-1};bool primary_motion_verified_{};
  a1::Config cfg_;std::unique_ptr<a1::Engine> engine_;std::string mode_,time_field_,cloud_frame_,camera_frame_;
  std::string ego_source_,imu_frame_;std::unique_ptr<a1::ImuSpeed> imu_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  Eigen::Isometry3d lidar_from_;a1::Vec lever_;bool ins_verified_{};double ins_yaw_{},ins_offset_{},last_cloud_{-1},last_publish_{};
  std::deque<a1::Ego> egos_;std::deque<a1::A2> a2s_;std::ofstream csv_;
  rclcpp::Subscription<Ins>::SharedPtr ins_sub_;rclcpp::Subscription<Features>::SharedPtr a2_sub_;rclcpp::Subscription<Cloud>::SharedPtr lidar_sub_;
  rclcpp::Publisher<a1_fusion_cpp::msg::EgoMotion>::SharedPtr ego_pub_;
  rclcpp::Publisher<Output>::SharedPtr pub_;rclcpp::TimerBase::SharedPtr timer_;
};
int main(int argc,char** argv){rclcpp::init(argc,argv);try{rclcpp::spin(std::make_shared<FusionNode>());}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}rclcpp::shutdown();}
