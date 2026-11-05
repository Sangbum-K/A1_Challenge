#include "a1_fusion/ui_recorder.hpp"
#include "a1_fusion/exposure_control.hpp"
#include "a1_fusion_cpp/msg/ego_motion.hpp"
#include "a1_fusion_cpp/msg/brake_estimate.hpp"
#include "a1_fusion/association.hpp"
#include "a1_fusion/map_filter_ros.hpp"
#include "a1_fusion_cpp/msg/a2_features.hpp"
#include "a1_fusion_cpp/msg/association.hpp"
#include <patchwork/patchworkpp.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <opencv2/opencv.hpp>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <map>
#include <array>
using Cloud=sensor_msgs::msg::PointCloud2;
using Image=sensor_msgs::msg::Image;
using A2=a1_fusion_cpp::msg::A2Features;
using Association=a1_fusion_cpp::msg::Association;
using namespace a1::inspection;
class AssociationNode:public rclcpp::Node {
 public:
  AssociationNode():Node("a1_association"){
    cv::setNumThreads(1);
    map_filter_=std::make_unique<MapFilterRos>(*this);
    declare_parameter("lidar_yaw_rad",M_PI/2);
    declare_parameter("show_debug_clusters",false);
    declare_parameter("bev_forward_range_m",100.);
    declare_parameter("processing_range_m",100.);
    declare_parameter("target_min_range_m",10.);declare_parameter("target_max_range_m",50.);
    declare_parameter("max_targets",2);
    TrackerConfig tracking;tracking.coast_s=declare_parameter("track_coast_s",.3);
    tracking.memory_s=declare_parameter("track_memory_s",.8);
    tracking.max_position_variance=declare_parameter("track_max_position_variance_m2",1.);
    if(!std::isfinite(tracking.coast_s)||!std::isfinite(tracking.memory_s)||!std::isfinite(tracking.max_position_variance)||tracking.coast_s<=0||tracking.memory_s<tracking.coast_s||tracking.max_position_variance<=0)throw std::invalid_argument("invalid tracker lifetime/variance");
    tracker_=Tracker(tracking);
    const double camera_hold=declare_parameter("camera_target_hold_s",.3);
    if(!std::isfinite(camera_hold)||camera_hold<=0||camera_hold>tracking.coast_s)throw std::invalid_argument("camera hold must be positive and <= track coast");
    const int confirmations=declare_parameter("camera_match_confirmations",3);
    if(confirmations<1)throw std::invalid_argument("camera_match_confirmations must be >= 1");
    selector_=CameraTargetSelector(camera_hold,confirmations);
    if(val("target_min_range_m")<0||!std::isfinite(val("target_min_range_m"))||!std::isfinite(val("target_max_range_m"))||val("target_max_range_m")<=val("target_min_range_m")||val("target_max_range_m")>val("processing_range_m")||get_parameter("max_targets").as_int()<1||get_parameter("max_targets").as_int()>2)throw std::runtime_error("invalid target range / max_targets (1..2)");
    declare_parameter("lidar_roi_center_deg",0.);
    declare_parameter("camera_position_m",std::vector<double>{0.,0.,0.});
    declare_parameter("camera_rpy_rad",std::vector<double>{0.,0.,0.});
    declare_parameter("camera_fx",640.);declare_parameter("camera_fy",640.);
    declare_parameter("camera_cx",640.);declare_parameter("camera_cy",360.);
    declare_parameter("calibration_verified",false);declare_parameter("clock_verified",false);
    declare_parameter("lidar_time_offset_s",0.);declare_parameter("pair_slop_s",.08);
    declare_parameter("presence_hold_threshold",.35);declare_parameter("presence_edge_threshold",.25);
    declare_parameter("fragment_merge_gap_m",.45);declare_parameter("presence_threshold",.5);declare_parameter("horizontal_iou_threshold",.15);
    declare_parameter("bin_search_min",0);declare_parameter("bin_search_max",39);
    declare_parameter("corridor_half_width_m",1.8);
    declare_parameter("cluster_tolerance_m",.45);
    declare_parameter("cluster_range_start_m",20.);declare_parameter("cluster_range_slope",.01);
    declare_parameter("cluster_max_tolerance_m",.85);declare_parameter("cluster_height_gap_m",1.2);declare_parameter("voxel_size_m",.16);
    declare_parameter("cluster_min_points",5);declare_parameter("show_window",false);
    declare_parameter("min_height_above_ground_m",.15);
    auto recording=declare_parameter<std::string>("recording_dir","");
    if(!recording.empty())recorder_=std::make_unique<UiRecorder>(recording);
    preview_=declare_parameter<std::string>("preview_path","");
    auto csv=declare_parameter<std::string>("csv_path","");
    if(!csv.empty()){std::filesystem::create_directories(std::filesystem::path(csv).parent_path());csv_.open(csv);if(!csv_)throw std::runtime_error("cannot open association CSV");csv_<<"stamp,status,unique,verified,track,bin_left,bin_right,ground,nonground,tracks,image_dt,a2_dt,processing_ms,candidate_ids,lidar_candidate_ids,ambiguous,primary_id,standby_id,primary_status,track_geometry,range_stage_counts,camera_stamp,camera_regions,camera_probabilities,primary_camera_age_s,map_filter_status,map_ground_removed,map_wall_removed\n";
      auto detail=std::filesystem::path(csv);detail.replace_extension("tracks.csv");track_csv_.open(detail);if(!track_csv_)throw std::runtime_error("cannot open track diagnostics");
      track_csv_<<"stamp,id,measured,hits,age_s,geometry,reason,raw_x,raw_y,raw_z,raw_dx,raw_dy,raw_dz,points,x,y,z,range_m,vx,vy,var_x,var_y,projectable,u0,v0,u1,v1,best_iou,best_region,camera_matched,selected,footprint_valid,footprint_length_m,footprint_width_m,footprint_yaw_rad\n";}
    if(!preview_.empty())std::filesystem::create_directories(std::filesystem::path(preview_).parent_path());
    patchwork::Params p;p.sensor_height=declare_parameter("sensor_height_m",1.8);p.min_range=1.;
    if(!std::isfinite(val("processing_range_m"))||val("processing_range_m")<val("target_max_range_m")||val("processing_range_m")>150)throw std::runtime_error("invalid processing range");
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
    ego_sub_=create_subscription<a1_fusion_cpp::msg::EgoMotion>("/lead_fusion/ego_motion",10,[this](a1_fusion_cpp::msg::EgoMotion::ConstSharedPtr m){ego_ui_=m;});
    brake_sub_=create_subscription<a1_fusion_cpp::msg::BrakeEstimate>("/lead_fusion/estimate",10,[this](a1_fusion_cpp::msg::BrakeEstimate::ConstSharedPtr m){brake_ui_=m;});
    image_sub_=create_subscription<Image>(declare_parameter<std::string>("image_topic","/camera_driver_gv/vis/image_raw"),rclcpp::SensorDataQoS().keep_last(4),[this](Image::ConstSharedPtr m){images_.push_back(m);while(images_.size()>30)images_.pop_front();});
    a2_sub_=create_subscription<A2>("/lead_fusion/a2",rclcpp::SensorDataQoS().keep_last(4),[this](A2::ConstSharedPtr m){a2s_.push_back(m);while(a2s_.size()>40)a2s_.pop_front();});
    const auto lidar_topic=declare_parameter<std::string>("lidar_topic","/lidar_points");
    cloud_sub_=create_subscription<Cloud>(lidar_topic,rclcpp::SensorDataQoS().keep_last(2),[this,lidar_topic](Cloud::ConstSharedPtr m){
      if(count_publishers(lidar_topic)!=1){
        pending_.clear();tracker_.reset();selector_.reset();primary_id_=-1;
        RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),2000,"LiDAR input requires exactly one publisher; refusing mixed sources");
        return;
      }
      pending_.emplace_back(std::chrono::steady_clock::now(),m);while(pending_.size()>4)pending_.pop_front();
    });
    // Wait for the camera inference of the matching frame; don't pair against the older
    // last completed inference merely because the cloud callback arrived first.
    timer_=create_wall_timer(std::chrono::milliseconds(10),[this]{
      if(!ui_top_.empty()&&now().seconds()-ui_last_cloud_>.3&&now().seconds()-ui_last_draw_>.1)draw_dashboard(true);
      if(pending_.empty()||std::chrono::duration<double>(std::chrono::steady_clock::now()-pending_.front().first).count()<.10)return;
      auto m=pending_.front().second;pending_.pop_front();
      try{process(*m);}catch(const std::exception& e){tracker_.reset();selector_.reset();primary_id_=-1;RCLCPP_ERROR_THROTTLE(get_logger(),*get_clock(),2000,"inspection: %s",e.what());}
    });
    exposure_=std::make_unique<ExposureControl>(*this);
    if(get_parameter("show_window").as_bool()){
      cv::namedWindow("A1 camera-LiDAR association (unverified preview)",cv::WINDOW_AUTOSIZE);
      cv::setMouseCallback("A1 camera-LiDAR association (unverified preview)",[](int event,int x,int y,int,void* context){static_cast<ExposureControl*>(context)->mouse(event,x,y);},exposure_.get());
    }
    ui_top_=cv::Mat::zeros(655,1600,CV_8UC3);draw_dashboard(true);
    RCLCPP_INFO(get_logger(),"Patchwork++ / PCL clusters / CV-Kalman Hungarian / camera-bin projection preview. NO brake output.");
  }
 private:
  double val(const char* n){return get_parameter(n).as_double();}
  static double stamp(const std_msgs::msg::Header& h){return rclcpp::Time(h.stamp).seconds();}
  static cv::Scalar color(int64_t id){return cv::Scalar(60+(id*53)%195,60+(id*97)%195,60+(id*151)%195);}
  template<class M> typename M::ConstSharedPtr nearest(const std::deque<typename M::ConstSharedPtr>& q,double t){typename M::ConstSharedPtr best;double gap=1e100;for(auto& m:q){double d=std::abs(stamp(m->header)-t);if(d<gap){gap=d;best=m;}}return best;}
  void cloud_pub(const Eigen::MatrixX3f& points,const std_msgs::msg::Header& h,rclcpp::Publisher<Cloud>::SharedPtr pub){
    if(pub->get_subscription_count()==0)return;
    pcl::PointCloud<pcl::PointXYZ> pc;pc.reserve(points.rows());for(int i=0;i<points.rows();++i)pc.push_back({points(i,0),points(i,1),points(i,2)});Cloud m;pcl::toROSMsg(pc,m);m.header=h;pub->publish(m);
  }
  void process(const Cloud& m){
    const auto start=std::chrono::steady_clock::now();
    if(m.is_bigendian||m.row_step<uint64_t(m.width)*m.point_step||m.data.size()<uint64_t(m.row_step)*m.height)throw std::runtime_error("invalid cloud layout");
    for(const char* name:{"x","y","z"}){bool ok=false;for(auto& f:m.fields)if(f.name==name&&f.datatype==7&&f.count==1&&f.offset+4<=m.point_step)ok=true;if(!ok)throw std::runtime_error("invalid XYZ fields");}
    const double hstamp=stamp(m.header);double t=hstamp;bool point_time_ok=false;
    for(auto& f:m.fields)if(f.name=="timestamp"&&f.datatype==8&&f.count==1&&f.offset+8<=m.point_step){
      double begin=1e100,end=-1e100;bool ok=true;
      for(uint32_t row=0;row<m.height;++row)for(uint32_t col=0;col<m.width;++col){double v;std::memcpy(&v,m.data.data()+size_t(row)*m.row_step+size_t(col)*m.point_step+f.offset,8);if(!std::isfinite(v)||std::abs(v-hstamp)>.3){ok=false;break;}begin=std::min(begin,v);end=std::max(end,v);}
      if(ok&&end>=begin&&end-begin<=.15){t=end;point_time_ok=true;}
    }
    t+=val("lidar_time_offset_s");
    if(t<=last_stamp_){tracker_.reset();selector_.reset();primary_id_=-1;images_.clear();a2s_.clear();}last_stamp_=t;
    auto a=nearest<A2>(a2s_,t);
    auto im=nearest<Image>(images_,a?stamp(a->header):t);
    Association result;result.header.stamp=rclcpp::Time(int64_t(t*1e9));result.header.frame_id="lidar_vehicle";
    result.image_delta_s=im?stamp(im->header)-t:std::numeric_limits<double>::quiet_NaN();
    result.a2_delta_s=a?stamp(a->header)-t:std::numeric_limits<double>::quiet_NaN();
    if(im)result.image_stamp=im->header.stamp;
    if(a)result.a2_stamp=a->header.stamp;
    result.calibration_verified=get_parameter("calibration_verified").as_bool();
    result.clock_verified=get_parameter("clock_verified").as_bool()&&point_time_ok;
    result.primary_track_id=result.standby_track_id=-1;
    result.motion_compensated=false;result.candidate_track_id=-1;result.bin_left=result.bin_right=-1;
    pcl::PointCloud<pcl::PointXYZ> raw;pcl::fromROSMsg(m,raw);
    Eigen::MatrixXf input(raw.size(),3);int count=0;
    const Eigen::Matrix3d lidar_to_body=body_from_lidar(val("lidar_yaw_rad"));
    for(auto& p:raw){Eigen::Vector3d q(p.x,p.y,p.z);if(!q.allFinite()||q.norm()<1||q.norm()>val("processing_range_m"))continue;q=lidar_to_body*q;input.row(count++)=q.cast<float>();}input.conservativeResize(count,3);
    const double roi_angle=val("lidar_roi_center_deg")*M_PI/180;
    Eigen::MatrixX3f front(count,3);int nf=0;
    for(int i=0;i<count;++i)if(in_front_half(input.row(i).cast<double>().transpose(),roi_angle))front.row(nf++)=input.row(i);
    front.conservativeResize(nf,3);cloud_pub(front,result.header,front_pub_);
    if(count<50)throw std::runtime_error("too few finite LiDAR points");
    ground_->estimateGround(input);auto gp=ground_->getGround(),ng=ground_->getNonground();
    result.ground_points=gp.rows();result.nonground_points=ng.rows();
    cloud_pub(gp,result.header,ground_pub_);cloud_pub(ng,result.header,nonground_pub_);
    auto roi=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for(int i=0;i<ng.rows();++i){auto q=ng.row(i);if(in_front_half(q.cast<double>().transpose(),roi_angle)&&q.x()>1&&q.x()<=val("processing_range_m")&&std::abs(q.y())<val("corridor_half_width_m")&&q.z()>-3&&q.z()<4)roi->push_back({q.x(),q.y(),q.z()});}
    auto vox=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();pcl::VoxelGrid<pcl::PointXYZ> vg;vg.setInputCloud(roi);float leaf=val("voxel_size_m");if(leaf<=0)throw std::runtime_error("voxel size must be positive");vg.setLeafSize(leaf,leaf,leaf);vg.filter(*vox);
    const std::string map_status=map_filter_->begin(t,m.header.frame_id,lidar_to_body);
    int map_ground_removed=0,map_wall_removed=0;
    auto map_kept=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::PointCloud<pcl::PointXYZI> map_removed;
    for(const auto& q:*vox){int label=map_filter_->match(Eigen::Vector3d(q.x,q.y,q.z));
      if(!label){map_kept->push_back(q);continue;}
      if(label==1)++map_ground_removed;else ++map_wall_removed;
      if(map_removed_pub_->get_subscription_count()>0){pcl::PointXYZI removed;removed.x=q.x;removed.y=q.y;removed.z=q.z;removed.intensity=label;map_removed.push_back(removed);}
    }
    vox=map_kept;
    if(map_removed_pub_->get_subscription_count()>0){Cloud removed;pcl::toROSMsg(map_removed,removed);removed.header=result.header;map_removed_pub_->publish(removed);}
    // Patchwork++ leaves sparse/unclassified ground patches in nonground. Prevent a
    // thin residual floor from joining separate objects into one Euclidean cluster.
    // This adapter guard uses only locally supported ground, not a global flat-road plane.
    std::map<std::pair<int,int>,std::pair<double,int>> ground_cells;
    for(int i=0;i<gp.rows();++i){auto& cell=ground_cells[{int(std::floor(gp(i,0))),int(std::floor(gp(i,1)))}];cell.first+=gp(i,2);++cell.second;}
    auto elevated=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for(auto& q:*vox){std::vector<double> heights;int ix=std::floor(q.x),iy=std::floor(q.y);
      for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy){auto it=ground_cells.find({ix+dx,iy+dy});if(it!=ground_cells.end()&&it->second.second>=3)heights.push_back(it->second.first/it->second.second);}
      if(heights.size()>=3){std::sort(heights.begin(),heights.end());if(q.z-heights[heights.size()/2]<val("min_height_above_ground_m"))continue;}
      elevated->push_back(q);
    }
    // Counts per 10 m shell: input / nonground / voxel / elevated. CSV diagnostics only.
    std::array<std::array<int,4>,10> stage_counts{};
    auto count_stage=[&](double x,double y,int stage){int bin=int(std::hypot(x,y)/10.);if(bin>=0&&bin<10)++stage_counts[bin][stage];};
    for(int i=0;i<input.rows();++i)if(in_front_half(input.row(i).cast<double>().transpose(),roi_angle))count_stage(input(i,0),input(i,1),0);
    for(const auto& q:*roi)count_stage(q.x,q.y,1);
    for(const auto& q:*vox)count_stage(q.x,q.y,2);
    for(const auto& q:*elevated)count_stage(q.x,q.y,3);
    vox=elevated;
    std::vector<Box> boxes;std::vector<pcl::PointIndices> groups;
    if(!vox->empty()){
      std::vector<Eigen::Vector3d> xyz;xyz.reserve(vox->size());for(const auto& q:*vox)xyz.emplace_back(q.x,q.y,q.z);
      ClusterConfig cc;cc.near_radius=val("cluster_tolerance_m");cc.start_range=val("cluster_range_start_m");cc.slope=val("cluster_range_slope");cc.max_radius=val("cluster_max_tolerance_m");cc.max_height_gap=val("cluster_height_gap_m");cc.min_points=get_parameter("cluster_min_points").as_int();
      std::vector<pcl::PointIndices> all;for(auto& indices:adaptive_clusters(xyz,cc)){pcl::PointIndices group;group.indices=std::move(indices);all.push_back(std::move(group));}
      for(auto& group:all){Box b;b.lo=Eigen::Vector3d::Constant(1e9);b.hi=-b.lo;b.points=group.indices.size();for(int j:group.indices){auto& p=(*vox)[j];Eigen::Vector3d v(p.x,p.y,p.z);b.lo=b.lo.cwiseMin(v);b.hi=b.hi.cwiseMax(v);}auto dim=b.hi-b.lo;if(std::max(dim.x(),dim.y())<.3||std::max(dim.x(),dim.y())>10||dim.z()<.15)continue;boxes.push_back(b);groups.push_back(group);}
    }
    // Bounded fragment attachment preserves actual measured extents.
    std::vector<Box> merged;std::vector<pcl::PointIndices> merged_groups;
    for(auto& ids:merge_fragments(boxes,val("fragment_merge_gap_m"))){
      Box b=boxes[ids[0]];pcl::PointIndices g=groups[ids[0]];
      for(size_t k=1;k<ids.size();++k){auto& q=boxes[ids[k]];b.lo=b.lo.cwiseMin(q.lo);b.hi=b.hi.cwiseMax(q.hi);b.points+=q.points;g.indices.insert(g.indices.end(),groups[ids[k]].indices.begin(),groups[ids[k]].indices.end());}
      // Refit after fragment attachment so metadata describes the entire union.
      // Fragment merging itself keeps its conservative axis-aligned extent gate.
      std::vector<Eigen::Vector3d> measured;measured.reserve(g.indices.size());
      for(int j:g.indices){const auto& q=(*vox)[j];measured.emplace_back(q.x,q.y,q.z);}
      b.footprint=measured_footprint(measured);
      merged.push_back(b);merged_groups.push_back(g);
    }
    boxes=std::move(merged);groups=std::move(merged_groups);
    const auto& tracks=tracker_.update(t,boxes);
    cv::Mat view=im?cv_bridge::toCvCopy(im,"bgr8")->image:cv::Mat::zeros(720,1280,CV_8UC3);
    if(view.cols!=1280||view.rows!=720)throw std::runtime_error("expected 1280x720 camera ROI");
    auto pos=get_parameter("camera_position_m").as_double_array(),angles=get_parameter("camera_rpy_rad").as_double_array();
    if(pos.size()!=3||angles.size()!=3)throw std::runtime_error("camera mount requires xyz/rpy triples");
    Eigen::Vector3d camera(pos[0],pos[1],pos[2]);Eigen::Matrix3d optical=nominal_optical_from_body();
    Eigen::Matrix3d mount=(Eigen::AngleAxisd(angles[2],Eigen::Vector3d::UnitZ())*Eigen::AngleAxisd(angles[1],Eigen::Vector3d::UnitY())*Eigen::AngleAxisd(angles[0],Eigen::Vector3d::UnitX())).toRotationMatrix();optical=optical*mount.transpose();
    const bool paired=im&&a&&std::abs(stamp(im->header)-stamp(a->header))<.005&&std::abs(result.image_delta_s)<=val("pair_slop_s")&&std::abs(result.a2_delta_s)<=val("pair_slop_s");
    bool presence=false;
    if(paired&&a->history_ready){
      double at=stamp(a->header);
      if(at!=presence_stamp_){
        const bool reset=presence_stamp_<0||at<=presence_stamp_||at-presence_stamp_>.25;
        const double alpha=reset?1.:1-std::exp(-(at-presence_stamp_)/.15);
        if(reset)regions_.clear();
        for(int j=0;j<40;++j){double raw=a->raw[j*7];double q=std::isfinite(raw)?1/(1+std::exp(-std::clamp(raw,-80.,80.))):0.;
          probabilities_[j]=(1-alpha)*probabilities_[j]+alpha*q;}
        presence_stamp_=at;
      }
      int lower=get_parameter("bin_search_min").as_int(),upper=get_parameter("bin_search_max").as_int();
      if(lower<0||upper>39||lower>upper)throw std::runtime_error("invalid bin search range");
      regions_=camera_regions(probabilities_,regions_,val("presence_threshold"),val("presence_hold_threshold"),val("presence_edge_threshold"),lower,upper);
      presence=!regions_.empty();
      // Legacy scalar bin fields only describe a genuinely single camera region.
      if(regions_.size()==1){result.bin_left=regions_[0].left;result.bin_right=regions_[0].right;}
      for(const auto& region:regions_)cv::rectangle(view,cv::Rect(region.left*32,235,(region.right-region.left+1)*32,250),cv::Scalar(0,230,230),2);
      for(int j=0;j<40;++j){int bar=int(50*probabilities_[j]);if(bar>0)cv::rectangle(view,cv::Rect(j*32,710-bar,30,bar),cv::Scalar(40,200,40),-1);}
    }else{regions_.clear();presence_stamp_=-1;probabilities_.fill(0);}
    cv::Mat bev(720,480,CV_8UC3,cv::Scalar(20,20,20));const double range=val("bev_forward_range_m");if(range<20||range>150)throw std::runtime_error("BEV range must be 20..150 m");const double scale=600/range;auto pixel=[scale](double x,double y){return cv::Point(int(240-y*scale),int(680-x*scale));};
    for(int i=0;i<gp.rows();i+=3){auto q=gp.row(i);if(!in_front_half(q.cast<double>().transpose(),roi_angle))continue;auto uv=pixel(q.x(),q.y());if(uv.x>=0&&uv.x<480&&uv.y>=0&&uv.y<720)bev.at<cv::Vec3b>(uv)=cv::Vec3b(45,65,45);}
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
    for(int i=0;i<ng.rows();i+=4){auto q=ng.row(i);if(!in_front_half(q.cast<double>().transpose(),roi_angle))continue;
      auto uv=pixel(q.x(),q.y());if(uv.x>=0&&uv.x<480&&uv.y>=145&&uv.y<720)bev.at<cv::Vec3b>(uv)=cv::Vec3b(80,80,80);}
    for(double u:{0.,val("camera_cx"),1279.}){
      Eigen::Vector3d ray=optical.transpose()*Eigen::Vector3d((u-val("camera_cx"))/val("camera_fx"),0,1);
      if(ray.head<2>().norm()>1e-6){ray*=18/ray.head<2>().norm();
        cv::line(bev,pixel(camera.x(),camera.y()),pixel(camera.x()+ray.x(),camera.y()+ray.y()),{0,220,220},u==val("camera_cx")?2:1);}
    }
    cv::circle(bev,pixel(camera.x(),camera.y()),4,{0,220,220},-1);
    const bool debug=get_parameter("show_debug_clusters").as_bool();
    std::vector<int64_t> candidates;
    std::vector<const Track*> eligible;
    
    // Only camera-supported LiDAR targets drive the range card and shadow P5.
    // Keep rejected tracks for diagnosis; do not populate a primary with the nearest background object.
    std::map<int64_t,std::string> reasons;
    std::map<int64_t,Projection> projections;
    std::map<int64_t,double> best_iou;
    std::map<int64_t,int> best_region,matched_regions;
    for(const auto& tr:tracks){
      projections[tr.id]=project(tr.box,optical,camera,val("camera_fx"),val("camera_fy"),val("camera_cx"),val("camera_cy"),1280,720);
      const auto geometry=std::string(vehicle_geometry_status(tr));
      auto& reason=reasons[tr.id];
      if(!vehicle_extent(tr.raw_box))reason="extent_exceeded";
      else if(!camera_surface_geometry(tr.raw_box,tr.id==primary_id_)&&geometry!="full_geometry"&&geometry!="retained_partial"&&geometry!="far_partial")reason="insufficient_geometry";
      else if(tr.hits<3)reason="track_unconfirmed";
      else if(!tracker_.usable(tr,t))reason="track_expired_or_uncertain";
      else if(!in_target_range(tr.box,tr.id==primary_id_?1.:val("target_min_range_m"),val("target_max_range_m")))reason="outside_target_range";
      else if(std::abs(tr.box.center().y())>val("corridor_half_width_m"))reason="outside_corridor";
      else {eligible.push_back(&tr);reason="no_camera_support";}
    }
    std::sort(eligible.begin(),eligible.end(),[](const Track* a,const Track* b){return a->box.center().head<2>().norm()<b->box.center().head<2>().norm();});
    Eigen::MatrixXd overlaps=Eigen::MatrixXd::Zero(eligible.size(),regions_.size());
    for(size_t i=0;i<eligible.size();++i){const auto& tr=*eligible[i];const auto p=projections.at(tr.id);auto& reason=reasons[tr.id];
      if(!paired||!a->history_ready||!presence){reason="no_camera_support";continue;}
      if(!p.valid){reason="projection_outside_image";continue;}
      if(p.y1<235||p.y0>485){reason="projection_outside_camera_roi";continue;}
      if(!tr.measured){reason="predicted_only";continue;}
      reason="no_horizontal_overlap";
      for(size_t j=0;j<regions_.size();++j){overlaps(i,j)=horizontal_iou(p,regions_[j].left*32,(regions_[j].right+1)*32);
        if(overlaps(i,j)>best_iou[tr.id]){best_iou[tr.id]=overlaps(i,j);best_region[tr.id]=j;}}
    }
    auto matches=unique_region_matches(overlaps,val("horizontal_iou_threshold"));
    std::vector<TargetObservation> observations;
    for(size_t i=0;i<eligible.size();++i){const auto& tr=*eligible[i];
      bool ambiguous=matches[i]<0&&overlaps.cols()>0&&overlaps.row(i).maxCoeff()>=val("horizontal_iou_threshold");
      if(ambiguous){result.association_ambiguous=true;reasons[tr.id]="ambiguous_camera_match";}
      if(matches[i]>=0){matched_regions[tr.id]=matches[i];reasons[tr.id]="camera_matched";}
      observations.push_back({tr.id,tr.box.center().head<2>().norm(),true,tr.measured,matches[i]>=0,ambiguous});
    }
    const double camera_stamp=paired&&a->history_ready?stamp(a->header):std::numeric_limits<double>::quiet_NaN();
    auto selected=selector_.update(t,camera_stamp,observations);
    primary_id_=selected.primary;
    result.primary_track_id=selected.primary;
    result.standby_track_id=get_parameter("max_targets").as_int()>1?selected.standby:-1;
    result.primary_status=selected.status;
    result.primary_camera_age_s=selected.primary>=0?selected.camera_age:std::numeric_limits<double>::quiet_NaN();
    for(auto id:{result.primary_track_id,result.standby_track_id})if(id>=0){
      result.lidar_candidate_track_ids.push_back(id);
      if(matched_regions.count(id))candidates.push_back(id);
    }
    // This list contains at most two selected current matches; diagnostic matches
    // on other tracks remain available in ClusterTrack.camera_matched.
    result.candidate_track_ids=candidates;
    pcl::PointCloud<pcl::PointXYZRGB> colored;
    for(auto& tr:tracks){auto p=projections.at(tr.id);
      a1_fusion_cpp::msg::ClusterTrack out;out.id=tr.id;out.measured=tr.measured;out.prediction_valid=tracker_.usable(tr,t);out.observation_age_s=std::max(0.,t-tr.last_seen);
      out.position_covariance_xy={tr.covariance(0,0),tr.covariance(0,1),tr.covariance(1,0),tr.covariance(1,1)};
      out.confirmed=tr.hits>=3;out.points=tr.measured?tr.box.points:0;auto center=tr.box.center();auto size=tr.box.hi-tr.box.lo;
      for(int k=0;k<3;++k){out.raw_center_m[k]=tr.raw_box.center()[k];out.raw_size_m[k]=(tr.raw_box.hi-tr.raw_box.lo)[k];out.center_m[k]=center[k];out.size_m[k]=size[k];}out.apparent_velocity_mps={tr.state[2],tr.state[3]};out.projectable=p.valid;out.image_box_xyxy={p.x0,p.y0,p.x1,p.y1};
      out.range_m=center.head<2>().norm();out.matched_bin_left=out.matched_bin_right=-1;
      const auto match=matched_regions.find(tr.id);const bool candidate=match!=matched_regions.end();
      out.camera_matched=candidate;out.geometry_status=vehicle_geometry_status(tr);out.selection_reason=reasons.at(tr.id);
      out.lidar_candidate=std::find(result.lidar_candidate_track_ids.begin(),result.lidar_candidate_track_ids.end(),tr.id)!=result.lidar_candidate_track_ids.end();
      if(candidate){const auto& region=regions_[match->second];out.matched_bin_left=region.left;out.matched_bin_right=region.right;out.horizontal_iou=horizontal_iou(p,region.left*32,(region.right+1)*32);}
      const bool visible=out.lidar_candidate;
      if(track_csv_){const Eigen::Vector3d rc=tr.raw_box.center(),rd=tr.raw_box.hi-tr.raw_box.lo;
        track_csv_<<std::setprecision(15)<<t<<','<<tr.id<<','<<tr.measured<<','<<tr.hits<<','<<out.observation_age_s<<','<<out.geometry_status<<','<<out.selection_reason;
        for(int k=0;k<3;++k)track_csv_<<','<<rc[k];
        for(int k=0;k<3;++k)track_csv_<<','<<rd[k];
        track_csv_<<','<<tr.raw_box.points;
        for(int k=0;k<3;++k)track_csv_<<','<<center[k];
        track_csv_<<','<<out.range_m<<','<<tr.state[2]<<','<<tr.state[3]<<','<<tr.covariance(0,0)<<','<<tr.covariance(1,1)<<','<<p.valid<<','<<p.x0<<','<<p.y0<<','<<p.x1<<','<<p.y1<<','<<best_iou[tr.id]<<','<<(best_region.count(tr.id)?best_region.at(tr.id):-1)<<','<<candidate<<','<<visible<<','<<tr.raw_box.footprint.valid<<','<<tr.raw_box.footprint.length<<','<<tr.raw_box.footprint.width<<','<<tr.raw_box.footprint.yaw<<'\n';}
      auto col=!tr.measured&&visible?cv::Scalar(255,180,40):candidate?cv::Scalar(40,40,255):visible?cv::Scalar(255,180,40):cv::Scalar(100,100,100);
      std::string label="T"+std::to_string(tr.id)+" r="+cv::format("%.1f",out.range_m)+"m"+(tr.id==primary_id_?" PRIMARY":"")+(!tr.measured?" PREDICTED":candidate?" CAMERA+LIDAR":visible?" HELD - UNVERIFIED":" UNSELECTED");
      auto raw_projection=project(tr.raw_box,optical,camera,val("camera_fx"),val("camera_fy"),val("camera_cx"),val("camera_cy"),1280,720);
      if(candidate&&raw_projection.valid)cv::rectangle(view,{int(raw_projection.x0),int(raw_projection.y0)},{int(raw_projection.x1),int(raw_projection.y1)},{130,130,130},1);
      if(p.valid&&(visible||debug)){cv::rectangle(view,{int(p.x0),int(p.y0)},{int(p.x1),int(p.y1)},col,candidate?3:2);cv::putText(view,label,{int(p.x0),std::max(85,int(p.y0)-5)},cv::FONT_HERSHEY_SIMPLEX,.45,col,1);}
      if(visible||debug){cv::rectangle(bev,pixel(tr.box.hi.x(),tr.box.hi.y()),pixel(tr.box.lo.x(),tr.box.lo.y()),col,candidate?3:1);cv::putText(bev,"T"+std::to_string(tr.id),pixel(center.x(),center.y()),cv::FONT_HERSHEY_SIMPLEX,.45,col,1);}
      result.tracks.push_back(out);
      if(clusters_pub_->get_subscription_count()>0)for(size_t j=0;j<boxes.size();++j)if(tr.measured&&(boxes[j].center()-tr.raw_box.center()).norm()<1e-6){for(int idx:groups[j].indices){auto& q=(*vox)[idx];pcl::PointXYZRGB v;v.x=q.x;v.y=q.y;v.z=q.z;v.b=col[0];v.g=col[1];v.r=col[2];colored.push_back(v);}break;}
    }
    if(clusters_pub_->get_subscription_count()>0){Cloud colored_msg;pcl::toROSMsg(colored,colored_msg);colored_msg.header=result.header;clusters_pub_->publish(colored_msg);}
    result.candidate_unique=candidates.size()==1;if(result.candidate_unique)result.candidate_track_id=candidates[0];
    result.verified_match=result.candidate_unique&&!result.association_ambiguous&&result.calibration_verified&&result.clock_verified;
    result.status=!im?"no_image":!a?"no_a2":!paired?"time_pair_unavailable":!a->history_ready?"a2_history_unready":!presence?"no_camera_presence":result.association_ambiguous?"ambiguous_clusters":candidates.empty()?(matched_regions.empty()?"no_confirmed_cluster":"camera_match_confirming"):!result.calibration_verified?"candidate_extrinsics_unverified":!result.clock_verified?"candidate_clock_unverified":"matched_geometry_only";
    cv::rectangle(view,{0,0},{1279,100},{0,0,0},-1);
    cv::putText(view,"PRIMARY: "+result.primary_status+" | camera: "+result.status,{12,25},cv::FONT_HERSHEY_SIMPLEX,.7,{0,220,255},2);
    cv::putText(view,"PREVIEW ONLY | no ego deskew | apparent velocity | camera metrics UNUSED",{12,52},cv::FONT_HERSHEY_SIMPLEX,.55,{255,255,255},1);
    std::string flags=std::string(result.calibration_verified?"EXTRINSICS VERIFIED":"EXTRINSICS UNVERIFIED")+" | "+(result.clock_verified?"CLOCK VERIFIED":"CLOCK UNVERIFIED")+cv::format(" | image dt %.0f ms | A2 dt %.0f ms",result.image_delta_s*1000,result.a2_delta_s*1000);
    cv::putText(view,flags,{12,80},cv::FONT_HERSHEY_SIMPLEX,.5,{0,180,255},1);
    cv::putText(bev,"CAMERA FORWARD = UP",{10,25},cv::FONT_HERSHEY_SIMPLEX,.6,{255,255,255},1);
    cv::putText(bev,cv::format("WHITE: LiDAR ROI 180deg / right %.0fdeg",-val("lidar_roi_center_deg")),{10,50},cv::FONT_HERSHEY_SIMPLEX,.45,{230,230,230},1);
    cv::putText(bev,"YELLOW: camera axis / FOV (unverified)",{10,75},cv::FONT_HERSHEY_SIMPLEX,.45,{0,220,220},1);
    cv::putText(bev,"LEFT = LiDAR +X | RIGHT = LiDAR -X",{10,100},cv::FONT_HERSHEY_SIMPLEX,.43,{230,230,230},1);
    cv::putText(bev,debug?"DEBUG clusters are NOT vehicle detections":"Red: matched / cyan: held / yellow: camera",{10,125},cv::FONT_HERSHEY_SIMPLEX,.43,{230,230,230},1);
    cv::putText(bev,"MAP: "+map_status+" G:"+std::to_string(map_ground_removed)+" W:"+std::to_string(map_wall_removed),{10,150},cv::FONT_HERSHEY_SIMPLEX,.42,{230,230,230},1);
    cv::Mat top;cv::hconcat(view,bev,top);cv::resize(top,ui_top_,{1600,655});
    ui_result_=result;ui_last_cloud_=now().seconds();draw_dashboard(false);
    result.processing_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();assoc_pub_->publish(result);
    if(csv_){csv_<<std::setprecision(15)<<t<<','<<result.status<<','<<result.candidate_unique<<','<<result.verified_match<<','<<result.candidate_track_id<<','<<result.bin_left<<','<<result.bin_right<<','<<result.ground_points<<','<<result.nonground_points<<','<<result.tracks.size()<<','<<result.image_delta_s<<','<<result.a2_delta_s<<','<<result.processing_ms<<',';
      for(size_t i=0;i<result.candidate_track_ids.size();++i){if(i)csv_<<';';csv_<<result.candidate_track_ids[i];}csv_<<',';
      for(size_t i=0;i<result.lidar_candidate_track_ids.size();++i){if(i)csv_<<';';csv_<<result.lidar_candidate_track_ids[i];}
      csv_<<','<<result.association_ambiguous<<','<<result.primary_track_id<<','<<result.standby_track_id<<','<<result.primary_status<<',';for(const auto& tr:tracks)csv_<<tr.id<<':'<<vehicle_geometry_status(tr)<<':'<<tr.raw_box.points<<':'<<(tr.raw_box.hi-tr.raw_box.lo).transpose()<<';';csv_<<',';for(size_t i=0;i<stage_counts.size();++i){csv_<<i*10<<'-'<<(i+1)*10<<':';for(int n:stage_counts[i])csv_<<n<<'/';csv_<<';';}csv_<<','<<camera_stamp<<',';for(const auto& region:regions_)csv_<<region.left<<':'<<region.right<<':'<<region.peak<<';';csv_<<',';for(double q:probabilities_)csv_<<q<<';';csv_<<','<<result.primary_camera_age_s<<','<<map_status<<','<<map_ground_removed<<','<<map_wall_removed<<'\n';csv_.flush();}
    if(track_csv_)track_csv_.flush();
  }

  void draw_dashboard(bool stale){
    const double time=now().seconds();ui_last_draw_=time;
    cv::Mat panel(860,1600,CV_8UC3,cv::Scalar(16,20,27));ui_top_.copyTo(panel(cv::Rect(0,0,1600,655)));
    auto text=[&](const std::string& v,int x,int y,double size,cv::Scalar c){cv::putText(panel,v,{x,y},cv::FONT_HERSHEY_SIMPLEX,size,c,1,cv::LINE_AA);};
    auto fresh=[&](const std_msgs::msg::Header& h){double age=time-stamp(h);return age>=0&&age<=.25;};
    const bool ego_ok=ego_ui_&&fresh(ego_ui_->header)&&ego_ui_->valid&&std::isfinite(ego_ui_->speed_mps);
    const bool accel_ok=ego_ok&&std::isfinite(ego_ui_->longitudinal_acceleration_mps2);
    const bool brake_fresh=brake_ui_&&fresh(brake_ui_->header);
    const bool brake_ok=!stale&&brake_fresh&&brake_ui_->source=="lidar_primary"&&brake_ui_->track_id==ui_result_.primary_track_id&&brake_ui_->valid&&!brake_ui_->what_if&&time<=rclcpp::Time(brake_ui_->valid_until).seconds()&&std::isfinite(brake_ui_->request_mps2);
    cv::Scalar white(235,240,245),muted(155,165,180),red(70,70,255),cyan(240,210,80);
    for(int i=0;i<4;++i)cv::rectangle(panel,{i*400+8,665},{i*400+392,824},{30,36,46},-1);
    text("EGO SPEED / IMU",24,694,.55,muted);
    text(ego_ok?cv::format("%.2f m/s",ego_ui_->speed_mps):"-- m/s",24,737,.95,white);
    text(ego_ok?cv::format("%.1f km/h",ego_ui_->speed_mps*3.6):"WAITING FOR VALID IMU",24,770,.5,cyan);
    text("Integrated estimate / drift limited",24,800,.44,muted);
    text("LONGITUDINAL ACCELERATION",424,694,.5,muted);
    text(accel_ok?cv::format("%+.2f m/s^2",ego_ui_->longitudinal_acceleration_mps2):"-- m/s^2",424,737,.85,white);
    text("Gravity removed / body forward",424,770,.46,cyan);
    std::string ego_reason=!ego_ui_?"NO IMU TELEMETRY":!fresh(ego_ui_->header)?"IMU INPUT STALE":ego_ui_->reason;
    text(ego_reason.substr(0,39),424,800,.40,muted);
    text(cv::format("CAMERA TARGET / %.0f-%.0fm",val("target_min_range_m"),val("target_max_range_m")),824,694,.5,muted);
    int row=0;
    if(!stale)for(auto id:ui_result_.lidar_candidate_track_ids)for(const auto& tr:ui_result_.tracks)if(tr.id==id&&row<2){
      text(cv::format("%s T%ld %.1fm Vx%+.1f%s",id==ui_result_.primary_track_id?"P":"S",long(id),tr.range_m,tr.apparent_velocity_mps[0],ui_result_.primary_status=="measured"&&tr.measured?"":" ~"),824,729+row*38,.54,tr.camera_matched?red:cv::Scalar(0,165,255));++row;
    }
    if(row==0)text("CAMERA TARGET: NO LIDAR MATCH",824,737,.44,white);
    text(row==0?"Waiting for camera-LiDAR agreement":ui_result_.primary_status=="measured"?"Planar center range / apparent Vx":"HELD/PREDICTED: NOT VERIFIED",824,800,.43,muted);
    text("P5 BRAKE REQUEST / SHADOW",1224,694,.49,muted);
    text(brake_ok?cv::format("%+.2f m/s^2",brake_ui_->request_mps2):"-- m/s^2",1224,737,.85,brake_ok?red:white);
    text(brake_ok?cv::format("Raw P5 %+.2f  gate %.2f",brake_ui_->raw_p5_mps2,brake_ui_->gate):!brake_fresh?"P5 INPUT MISSING / STALE":brake_ui_->what_if?"WHAT-IF EXCLUDED":brake_ui_->reason.substr(0,35),1224,770,.42,muted);
    text("Camera-matched primary only",1224,800,.37,muted);
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
    if(get_parameter("show_window").as_bool()){cv::imshow("A1 camera-LiDAR association (unverified preview)",panel);exposure_->key(cv::waitKeyEx(1));}
  }

  std::unique_ptr<MapFilterRos> map_filter_;rclcpp::Publisher<Cloud>::SharedPtr map_removed_pub_;
  std::unique_ptr<ExposureControl> exposure_;
  std::unique_ptr<UiRecorder> recorder_;
  cv::Mat ui_top_;Association ui_result_;double ui_last_cloud_{},ui_last_draw_{};
  a1_fusion_cpp::msg::EgoMotion::ConstSharedPtr ego_ui_;
  a1_fusion_cpp::msg::BrakeEstimate::ConstSharedPtr brake_ui_;
  rclcpp::Subscription<a1_fusion_cpp::msg::EgoMotion>::SharedPtr ego_sub_;
  rclcpp::Subscription<a1_fusion_cpp::msg::BrakeEstimate>::SharedPtr brake_sub_;
  std::array<double,40> probabilities_{};double presence_stamp_{-1};std::vector<BinRegion> regions_;int64_t primary_id_{-1};
  CameraTargetSelector selector_;std::ofstream track_csv_;
  Tracker tracker_;std::unique_ptr<patchwork::PatchWorkpp> ground_;double last_stamp_{-1},last_preview_{},last_image_{};std::string preview_;std::ofstream csv_;
  std::deque<Image::ConstSharedPtr> images_;std::deque<A2::ConstSharedPtr> a2s_;
  std::deque<std::pair<std::chrono::steady_clock::time_point,Cloud::ConstSharedPtr>> pending_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<Image>::SharedPtr image_sub_;rclcpp::Subscription<A2>::SharedPtr a2_sub_;rclcpp::Subscription<Cloud>::SharedPtr cloud_sub_;
  rclcpp::Publisher<Image>::SharedPtr image_pub_;rclcpp::Publisher<Association>::SharedPtr assoc_pub_;rclcpp::Publisher<Cloud>::SharedPtr ground_pub_,nonground_pub_,clusters_pub_,front_pub_;
};
int main(int argc,char** argv){rclcpp::init(argc,argv);try{rclcpp::spin(std::make_shared<AssociationNode>());}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}rclcpp::shutdown();}
