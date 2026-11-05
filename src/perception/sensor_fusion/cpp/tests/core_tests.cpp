#include "a1_fusion/core.hpp"
#include <iostream>
#include <stdexcept>
#define CHECK(x) do { if(!(x)) throw std::runtime_error(#x); } while(false)
int main() {
  using namespace a1;
  ImuConfig ic;ImuSpeed disabled(ic);CHECK(!disabled.update(1,Vec(0,0,9.80665),Vec::Zero()).valid);
  ic.stationary_start_confirmed=true;ic.mount_verified=true;ic.calibration_s=.5;ic.max_integration_s=2;
  ImuSpeed inertial(ic);Ego estimated;
  for(int i=0;i<=100;++i)estimated=inertial.update(1+i*.01,Vec(0,0,9.80665),Vec::Zero());
  CHECK(estimated.valid);CHECK(estimated.velocity.norm()<1e-9);CHECK(estimated.acceleration.norm()<1e-9);
  for(int i=1;i<=100;++i)estimated=inertial.update(2+i*.01,Vec(1,0,9.80665),Vec::Zero());
  CHECK(estimated.valid);CHECK(std::abs(estimated.forward()-.995)<1e-8);CHECK((estimated.acceleration-Vec(1,0,0)).norm()<1e-9);
  CHECK(!inertial.update(3.2,Vec(0,0,9.80665),Vec::Zero()).valid);
  CHECK(!inertial.update(3.21,Vec(0,0,9.80665),Vec::Zero()).valid);
  CHECK(p5(0,10,0)==0);CHECK(p5(10,10,0)<0);CHECK(p5(10,200,10)==0);
  Config c;c.clock_valid=true;c.calibration_valid=true;c.camera_metric_valid=true;
  c.camera_from_base.linear()<<0,-1,0,0,0,-1,1,0,0;
  A2 a;a.ready=true;a.stamp=1;
  for(int k=0;k<40;++k) a.raw[k*7]=-8;
  a.raw[20*7]=8;a.raw[20*7+6]=0.5;a.raw[20*7+3]=-0.25;
  Ego e;e.valid=true;e.stamp=1;e.velocity=Vec(10,0,0);
  PrimaryTarget target;target.id=42;target.stamp=1;target.valid_until=1.3;target.center=Vec(20,0,0);target.gap=19;target.relative_speed=-5;target.valid=true;
  Engine primary(c);auto initial=primary.primary(target,e,1.01);CHECK(initial.valid&&initial.track_id==42);CHECK(initial.raw_p5==p5(10,19,5));
  target.predicted=true;target.stamp=1.2;target.center.x()=19;target.gap=18;e.stamp=1.2;
  auto predicted=primary.primary(target,e,1.21);CHECK(predicted.valid&&predicted.track_id==42&&predicted.reason=="shadow_predicted_primary");CHECK(predicted.valid_until<=1.3);
  CHECK(!primary.primary(target,e,1.31).valid);target.valid=false;CHECK(!primary.primary(target,e,1.21).valid);
  target.valid=true;target.id=43;target.predicted=false;target.gap=30;auto switched=primary.primary(target,e,1.21);CHECK(switched.raw_p5==p5(10,30,5));
  e.stamp=1;
  Engine camera(c);auto r=camera.camera(a,e,1.01);
  CHECK(r.valid);CHECK(std::abs(r.relative_speed+5)<1e-9);CHECK(r.lead_speed==5);CHECK(r.request<0);
  CHECK(!camera.camera(a,e,2).valid);
  e.valid=false;CHECK(!camera.camera(a,e,1.01).valid);e.valid=true;
  a.raw[20*7+6]=a1::nan;CHECK(!camera.camera(a,e,1.01).valid);
  // Fusion must ignore all A2 metric channels, including NaNs.
  for(int k=0;k<40;++k)for(int ch=1;ch<7;++ch)a.raw[k*7+ch]=a1::nan;
  Engine fusion(c);
  for(int i=0;i<10;++i) {
    const double t=1+i*.1;
    a.stamp=e.stamp=t;e.position=Vec(i,0,0);
    const Vec center(20-i*.5,-.3,0);
    Detection d{center,center-Vec(1,.6,.5),center+Vec(1,.6,.5)};
    r=fusion.fusion(a,e,t,{d},t+.01);
  }
  CHECK(r.valid);CHECK(std::abs(r.relative_speed+5)<1e-8);CHECK(std::abs(r.lead_speed-5)<1e-8);
  CHECK(r.gap_definition=="lidar_near_surface_x_minus_bumper");
  a.stamp=e.stamp=2;r=fusion.fusion(a,e,2,{},2.01);CHECK(!r.valid);
  // Rotation of a stationary world target must not invent world velocity.
  Tracker tracker(c);Vec world(20,0,0);
  for(int i=0;i<6;++i){e.stamp=3+.1*i;e.yaw=.02*i;e.position=Vec::Zero();
    const Vec b=to_body(world,e);Detection d{b,b-Vec(1,1,1),b+Vec(1,1,1)};
    const auto& ts=tracker.update(e.stamp,{d},e);CHECK(ts.size()==1);
    if(i>=2) CHECK(ts.front().velocity.norm()<1e-9);
  }
  CHECK(tracker.update(e.stamp,{},e).empty());
  c.clock_valid=false;Engine unverified(c);CHECK(!unverified.camera(a,e,3).valid);
  std::cout<<"core: P5, camera, INS invalid, stale, NaN isolation, tracking, yaw, duplicate passed\n";
}
