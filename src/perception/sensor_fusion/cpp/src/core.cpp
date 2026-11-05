#include "a1_fusion/core.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace a1 {
void ImuSpeed::reset() {
  first_=last_=ready_at_=-1;samples_=0;sum_a_.setZero();sum_w_.setZero();sum_sq_.setZero();
  bias_a_.setZero();bias_w_.setZero();previous_accel_.setZero();attitude_.setIdentity();ego_=Ego{};
}
Ego ImuSpeed::update(double stamp,const Vec& acceleration,const Vec& angular_rate) {
  ego_.valid=false;ego_.source="imu_integrated_experimental";ego_.stamp=stamp;
  if(!config_.stationary_start_confirmed){reason_="imu_initial_stationary_unconfirmed";return ego_;}
  if(!config_.mount_verified){reason_="imu_mount_unverified";return ego_;}
  if(!std::isfinite(stamp)||!acceleration.allFinite()||!angular_rate.allFinite()||stamp<=0) {
    reason_="imu_nonfinite";ready_at_=-2;return ego_;
  }
  const Vec a=config_.base_from_imu*acceleration,w=config_.base_from_imu*angular_rate;
  const double dt=last_<0?0:stamp-last_;
  if(last_>=0&&(dt<=0||dt>config_.max_dt)) {
    // A gap needs a new explicit estimator reset/confirmed stationary start, not hidden zero velocity.
    reason_="imu_time_discontinuity";ready_at_=-2;return ego_;
  }
  if(ready_at_==-2){reason_="imu_reinitialization_required";return ego_;}
  if(first_<0)first_=stamp;
  last_=stamp;
  if(ready_at_<0) {
    if(w.norm()>.05||a.norm()<8.5||a.norm()>11.0){reason_="imu_not_stationary";first_=stamp;samples_=0;sum_a_.setZero();sum_w_.setZero();sum_sq_.setZero();return ego_;}
    sum_a_+=a;sum_w_+=w;sum_sq_+=a.cwiseProduct(a);++samples_;reason_="imu_bias_warmup";
    if(stamp-first_<config_.calibration_s||samples_<30)return ego_;
    const Vec mean=sum_a_/samples_;
    if((sum_sq_/samples_-mean.cwiseProduct(mean)).maxCoeff()>.04){reason_="imu_calibration_motion";ready_at_=-2;return ego_;}
    attitude_=Eigen::Quaterniond::FromTwoVectors(mean,Vec::UnitZ()).normalized();
    bias_a_=mean-attitude_.conjugate()*Vec(0,0,9.80665);bias_w_=sum_w_/samples_;
    ready_at_=stamp;ego_.position.setZero();ego_.velocity.setZero();previous_accel_.setZero();ego_.acceleration.setZero();
  } else {
    if(stamp-ready_at_>config_.max_integration_s){reason_="imu_drift_window_expired";return ego_;}
    const Vec omega=w-bias_w_;const double angle=omega.norm()*dt;
    if(angle>0)attitude_=(attitude_*Eigen::Quaterniond(Eigen::AngleAxisd(angle,omega.normalized()))).normalized();
    const Vec world_accel=attitude_*(a-bias_a_)-Vec(0,0,9.80665);
    const Vec previous_velocity=ego_.velocity;
    ego_.velocity+=.5*(previous_accel_+world_accel)*dt;
    ego_.position+=.5*(previous_velocity+ego_.velocity)*dt;
    previous_accel_=world_accel;ego_.acceleration=world_accel;
  }
  const auto rot=attitude_.toRotationMatrix();ego_.yaw=std::atan2(rot(1,0),rot(0,0));
  ego_.yaw_rate=(w-bias_w_).z();
  if(std::abs(rot(2,0))>std::sin(10.0*M_PI/180)||std::abs(rot(2,1))>std::sin(10.0*M_PI/180)) {reason_="imu_planar_limit";return ego_;}
  ego_.valid=ego_.velocity.allFinite()&&ego_.velocity.norm()<60;
  reason_=ego_.valid?"imu_experimental_ready":"imu_kinematics_invalid";return ego_;
}
double probability(double x) { return 1/(1+std::exp(-std::clamp(x,-80.0,80.0))); }
double p5(double plan, double gap, double lead) {
  if (!std::isfinite(plan)||plan<0||!std::isfinite(gap)||!std::isfinite(lead))
    throw std::invalid_argument("nonfinite P5 input or negative planning speed");
  std::array<double,40> t{},u{};
  const double minimum=7+6*std::log1p(plan/5);
  double x=0;
  for(int i=0;i<40;++i) {
    t[i]=i*0.1;
    if(i) x+=plan*0.1;
    u[i]=std::min(plan,std::max(0.0,(gap+std::max(0.0,lead)*t[i]-x-minimum)/2));
  }
  for(int i=38;i>=0;--i) u[i]=std::min(u[i],u[i+1]+20*(t[i+1]-t[i]));
  u[0]=plan;
  for(int i=1;i<40;++i) u[i]=std::max(0.0,std::clamp(u[i],u[i-1]-20*(t[i]-t[i-1]),u[i-1]+5*(t[i]-t[i-1])));
  const double a=(u[10]-u[0])/t[10];
  return std::abs(a)<1e-6 ? 0 : a;
}
Vec to_world(const Vec& p,const Ego& e) { return Eigen::AngleAxisd(e.yaw,Vec::UnitZ())*p+e.position; }
Vec to_body(const Vec& p,const Ego& e) { return Eigen::AngleAxisd(-e.yaw,Vec::UnitZ())*(p-e.position); }
const std::vector<Track>& Tracker::update(double stamp,const std::vector<Detection>& ds,const Ego& e) {
  if(stamp<=last_stamp_) { reset(); return tracks_; }
  last_stamp_=stamp;
  tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[&](const auto& t){return stamp-t.stamp>config_.timeout;}),tracks_.end());
  const size_t old_size=tracks_.size();
  for(auto& t:tracks_) t.measured=false;
  // Mutual nearest-neighbour association, rejecting near-ties instead of switching IDs.
  std::vector<Vec> world;
  for(const auto& d:ds) world.push_back(to_world(d.center,e));
  std::vector<int> choices(ds.size(),-1), reverse(old_size,-1);
  for(size_t i=0;i<ds.size();++i) {
    double best=config_.track_gate, second=config_.track_gate;
    for(size_t j=0;j<old_size;++j) {
      const auto& tr=tracks_[j];
      const double d=(world[i]-tr.position-tr.velocity*(stamp-tr.stamp)).head<2>().norm();
      if(d<best) { second=best;best=d;choices[i]=static_cast<int>(j); }
      else second=std::min(second,d);
    }
    if(second-best<0.3) choices[i]=-1;
  }
  for(size_t j=0;j<old_size;++j) {
    double best=config_.track_gate,second=config_.track_gate;
    for(size_t i=0;i<ds.size();++i) {
      const auto& tr=tracks_[j];
      const double d=(world[i]-tr.position-tr.velocity*(stamp-tr.stamp)).head<2>().norm();
      if(d<best){second=best;best=d;reverse[j]=static_cast<int>(i);}else second=std::min(second,d);
    }
    if(second-best<0.3) reverse[j]=-1;
  }
  for(size_t i=0;i<ds.size();++i) {
    const int j=choices[i];
    if(j>=0 && reverse[j]==static_cast<int>(i)) {
      auto& tr=tracks_[j];
      const double dt=stamp-tr.stamp;
      const Vec v=(world[i]-tr.position)/dt;
      if(v.head<2>().norm()>60) continue;
      tr.velocity=tr.hits==1?v:0.65*tr.velocity+0.35*v;
      tr.position=world[i];tr.stamp=stamp;tr.box=ds[i];tr.measured=true;
      tr.velocity_valid=++tr.hits>=3;
    } else {
      Track tr;tr.id=next_id_++;tr.hits=1;tr.stamp=stamp;
      tr.position=world[i];tr.box=ds[i];tr.measured=true;tracks_.push_back(tr);
    }
  }
  return tracks_;
}
Result Engine::base(const A2& a,const Ego& e,double stamp,double now) {
  Result r;r.stamp=stamp;r.valid_until=stamp+config_.max_age;r.ego_source=e.source;
  if(!config_.clock_valid){r.reason="clock_unverified";return r;}
  if(!std::isfinite(now)||!std::isfinite(stamp)||stamp<=0||now<stamp||now-stamp>config_.max_age){r.reason="stale_or_future_measurement";return r;}
  if(!e.valid||!e.velocity.allFinite()||!e.position.allFinite()||!std::isfinite(e.yaw)||!std::isfinite(e.stamp)||e.stamp>stamp||stamp-e.stamp>config_.pair_slop){r.reason="ins_invalid_or_unsynchronized";return r;}
  if(!a.ready||!std::isfinite(a.stamp)||a.stamp>stamp||stamp-a.stamp>config_.pair_slop){r.reason="a2_unready_or_unsynchronized";return r;}
  int k=16;
  // Fusion deliberately inspects only presence logits; metric channels can be NaN.
  for(int i=0;i<40;++i) if(!std::isfinite(a.raw[i*7])){r.reason="a2_presence_nonfinite";return r;}
  for(int i=17;i<=25;++i) if(a.raw[i*7]>a.raw[k*7]) k=i;
  r.bin=k;
  if(probability(a.raw[k*7])<config_.object_threshold){r.reason="no_camera_lead";return r;}
  if(a.stamp<gate_stamp_) { gate_=0;gate_stamp_=-1; }
  if(a.stamp>gate_stamp_) {
    if(gate_stamp_>0 && a.stamp-gate_stamp_>config_.max_age) gate_=0;
    gate_=0.7*gate_+0.3*probability((a.raw[k*7]-0.3)*5);
    gate_stamp_=a.stamp;
  }
  r.gate=gate_;r.reason="ready";
  return r;
}
void Engine::finish(Result& r,const Ego& e) {
  r.ego_speed=e.forward();r.plan_speed=config_.plan_speed<0?r.ego_speed:config_.plan_speed;
  r.what_if=config_.plan_speed>=0;
  if(!std::isfinite(r.gap)||r.gap<=0||!std::isfinite(r.relative_speed)||r.ego_speed<0||!std::isfinite(r.plan_speed)) {r.reason="invalid_kinematics";return;}
  r.lead_speed=std::max(0.0,r.ego_speed+r.relative_speed);
  r.raw_p5=p5(r.plan_speed,r.gap,r.lead_speed);
  // F2 is a signed brake-only acceleration request, never positive throttle.
  r.request=std::clamp(r.gate*std::min(0.0,r.raw_p5),-3.0,0.0);
  r.valid=true;r.reason="shadow_only";
}
Result Engine::primary(const PrimaryTarget& target,const Ego& e,double now){
  Result r;r.source="lidar_primary";r.ego_source=e.source;r.stamp=target.stamp;
  r.track_id=target.id;r.gap_definition="tracked_lidar_near_surface_x_minus_bumper";
  r.valid_until=std::min(target.stamp+config_.max_age,target.valid_until);
  r.x=target.center.x();r.y=target.center.y();r.range=target.center.head<2>().norm();
  r.gap=target.gap;r.relative_speed=target.relative_speed;
  if(!config_.clock_valid){r.reason="clock_unverified";return r;}
  if(!config_.calibration_valid){r.reason="extrinsics_unverified";return r;}
  if(!target.valid||target.id<0||!target.center.allFinite()||!std::isfinite(target.stamp)||!std::isfinite(target.valid_until)||!std::isfinite(now)||now<target.stamp||now>r.valid_until||target.stamp<=0){r.reason="primary_invalid_or_expired";return r;}
  if(!e.valid||!e.velocity.allFinite()||!std::isfinite(e.yaw)||!std::isfinite(e.stamp)||e.stamp>target.stamp||target.stamp-e.stamp>config_.pair_slop){r.reason="ego_invalid_or_unsynchronized";return r;}
  // LiDAR-led mode: camera visibility is not a continuously applied brake gate.
  // The original raw P5 equation and request clamp are unchanged.
  r.gate=1.;finish(r,e);
  if(r.valid&&target.predicted)r.reason="shadow_predicted_primary";
  return r;
}
Result Engine::camera(const A2& a,const Ego& e,double now) {
  auto r=base(a,e,a.stamp,now);r.source="camera";r.gap_definition="a2_radial_projected_proxy";
  if(r.reason!="ready") return r;
  const double d=40*a.raw[r.bin*7+6], v=20*a.raw[r.bin*7+3];
  if(!std::isfinite(d)||d<=0||!std::isfinite(v)){r.reason="a2_metric_invalid";return r;}
  const double theta=std::atan(((r.bin+0.5)*32-640)/640);
  r.x=d*std::cos(theta);r.y=-d*std::sin(theta);r.range=d;r.gap=r.x-config_.bumper_x;r.relative_speed=v;
  if(!config_.camera_metric_valid){r.reason="camera_metric_unvalidated";return r;}
  finish(r,e);return r;
}
Result Engine::fusion(const A2& a,const Ego& e,double stamp,const std::vector<Detection>& ds,double now) {
  auto r=base(a,e,stamp,now);r.source="fusion";r.gap_definition="lidar_near_surface_x_minus_bumper";
  if(r.reason!="ready"){tracker_.reset();return r;}
  if(!config_.calibration_valid){r.reason="extrinsics_unverified";return r;}
  const auto& tracks=tracker_.update(stamp,ds,e);
  int left=r.bin,right=r.bin;
  while(left>0&&probability(a.raw[(left-1)*7])>=config_.object_threshold) --left;
  while(right<39&&probability(a.raw[(right+1)*7])>=config_.object_threshold) ++right;
  std::vector<const Track*> matched;
  for(const auto& t:tracks) {
    if(!t.measured||!t.velocity_valid) continue;
    const auto& b=t.box;
    if(b.lo.x()<=config_.bumper_x||std::abs(b.center.y())>config_.corridor_half_width) continue;
    const Vec c=config_.camera_from_base*b.center;
    if(c.z()<=0) continue;
    const double u=config_.camera_fx*c.x()/c.z()+config_.camera_cx;
    if(u>=left*config_.camera_width/40 && u<(right+1)*config_.camera_width/40) matched.push_back(&t);
  }
  if(matched.size()!=1){r.reason=matched.empty()?"no_confirmed_lidar_match":"ambiguous_camera_lidar_match";return r;}
  const auto& t=*matched.front();
  r.track_id=t.id;r.x=t.box.center.x();r.y=t.box.center.y();r.range=t.box.center.norm();r.gap=t.box.lo.x()-config_.bumper_x;
  const Vec relative=Eigen::AngleAxisd(-e.yaw,Vec::UnitZ())*(t.velocity-e.velocity);
  r.relative_speed=relative.x();finish(r,e);return r;
}
} // namespace a1
