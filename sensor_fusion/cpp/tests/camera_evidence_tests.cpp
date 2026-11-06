#include "a1_fusion/camera_evidence.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#define CHECK(x) do{if(!(x))throw std::runtime_error(std::string(#x)+" line "+std::to_string(__LINE__));}while(false)
using namespace a1::inspection;
namespace {
using Bins=std::array<double,40>;
Projection box_at(double u0,double u1){Projection p;p.valid=true;p.x0=u0;p.x1=u1;p.y0=300;p.y1=400;return p;}
TrackView view(Track& t,double range,const Projection& p,bool can_own=true){
  TrackView v;v.track=&t;v.range=range;v.projection=p;v.span=bins_for(p);v.can_own=can_own;return v;
}
TargetObservation target(int64_t id,double distance,double evidence,bool measured=true){
  TargetObservation o;o.id=id;o.distance=o.range=distance;o.evidence=evidence;o.usable=true;o.measured=measured;
  o.in_lane=true;o.vehicle_shape=true;o.visibility=Visibility::Visible;return o;
}
}
int main(){
  { // Scenario, synthetic only: lead through a corner while A2 sees it about 40 % of the time,
    // with LiDAR dropouts and a rail fragment closer than the lead.
    using V2=Eigen::Vector2d;
    const double omega=.3,radius=20/omega,gap_angle=.5;
    auto ego_at=[&](double t){const double a=omega*t;return Pose2{radius*std::sin(a),radius*(1-std::cos(a)),a};};
    auto lead_at=[&](double t,V2& rear,double& heading){const double a=omega*t+gap_angle;rear=V2(radius*std::sin(a),radius*(1-std::cos(a)));heading=a;};
    const Eigen::Matrix3d optical=nominal_optical_from_body();const Eigen::Vector3d camera(.8,0,-.3);
    const double fx=1100,cx=640,cy=360; // Assumes a calibrated camera: this scenario does not exercise a wrong fx.
    Tracker tracker;EvidenceSelector selector;EvidenceConfig config;
    double last_camera=-1;int scans=0,held=0;int64_t lead_id=-1;int switches=0;
    for(int i=0;i<160;++i){
      const double t=i*.1+.1;const Pose2 ego=ego_at(t);
      V2 rear;double heading;lead_at(t,rear,heading);
      const V2 u(std::cos(heading),std::sin(heading)),v(-u.y(),u.x());
      Scan scan;scan.stamp=t;scan.ego=ego;scan.map_ok=true;
      scan.heading_prior=[yaw=ego.yaw](const V2&){return yaw;};
      std::vector<int> car,rail;
      auto add=[&](const V2& world,double z,std::vector<int>& ids){
        const V2 body=ego.to_body(world);ids.push_back(int(scan.world_points.size()));
        scan.world_points.emplace_back(world.x(),world.y(),z);scan.body_points.emplace_back(body.x(),body.y(),z);scan.point_time.push_back(t);};
      const bool lidar_lost=(i>=70&&i<73)||(i>=120&&i<122);
      if(!lidar_lost)for(double b=-.95;b<=.95;b+=.1)for(double z:{-.6,-.2,.2})add(rear+v*b,z,car);
      // Static rail fragment near the path: the map knows it.
      const V2 rail_origin=ego.to_world(V2(12,6));
      for(double a=0;a<=1.5;a+=.1)for(double z:{-.9,-.5})add(rail_origin+V2(a,0),z,rail);
      const int rail_begin=car.empty()?0:int(car.size());
      scan.is_structure=[rail_begin](int index){return index>=rail_begin;};scan.is_covered=[](int){return true;};
      if(!car.empty()){auto d=make_detection(scan,car,40.);if(d.quality!=Quality::Reject)scan.detections.push_back(d);}
      {auto d=make_detection(scan,rail,40.);if(d.quality!=Quality::Reject)scan.detections.push_back(d);}
      auto& tracks=tracker.update(scan);
      for(int k=0;k<2;++k){
        const double at=t-.05+k*.05;const Pose2 at_ego=ego_at(at);
        V2 rear_at;double heading_at;lead_at(at,rear_at,heading_at);
        const V2 side(-std::sin(heading_at),std::cos(heading_at));
        const V2 left=at_ego.to_body(rear_at+side*.95),right=at_ego.to_body(rear_at-side*.95);
        Box truth{Eigen::Vector3d(std::min(left.x(),right.x()),std::min(left.y(),right.y()),-.6),Eigen::Vector3d(std::max(left.x(),right.x()),std::max(left.y(),right.y()),.2),0,{}};
        const auto seen=bins_for(project(truth,optical,camera,fx,fx,cx,cy,1280,720));
        const bool detected=std::fmod(at,2.5)<1.; // 1 s on, 1.5 s off, about what the video showed.
        Bins p;p.fill(.02);
        if(seen.valid)for(int j=seen.left;j<=seen.right;++j)p[j]=detected?.7:.1;
        std::vector<TrackView> views;
        for(auto& tr:tracks){
          Track copy=tr;tracker.peek(tr,at,copy.x,copy.P);
          TrackView view_of;view_of.track=&tr;view_of.range=at_ego.to_body(copy.anchor()).norm();
          view_of.projection=project(track_rear_box(copy,at_ego),optical,camera,fx,fx,cx,cy,1280,720);
          view_of.span=bins_for(view_of.projection);view_of.can_own=tr.confirmed&&!tr.map_static;
          views.push_back(view_of);
        }
        assign_ownership(views,p);
        for(const auto& view_of:views)observe(*view_of.track,at,view_of.score,view_of.visibility,config);
        last_camera=at;
      }
      std::vector<TargetObservation> observations;
      for(const auto& tr:tracks){
        TargetObservation o;o.id=tr.id;o.range=o.distance=ego.to_body(tr.anchor()).norm();o.usable=tracker.usable(tr,t);o.measured=tr.measured;
        o.evidence=tr.evidence;o.in_lane=true;o.vehicle_shape=camera_surface_geometry(tr.raw_box,tr.id==lead_id)||full_vehicle_geometry(tr.raw_box);
        o.map_static=tr.map_static;o.visibility=tr.visibility;o.camera_matched=tr.visibility==Visibility::Visible&&tr.camera_score>=.4;
        observations.push_back(o);
      }
      const auto selected=selector.update(t,last_camera,observations);
      if(i>=30){
        ++scans;
        const Track* chosen=nullptr;for(const auto& tr:tracks)if(tr.id==selected.primary)chosen=&tr;
        CHECK(selected.primary<0||(chosen&&(chosen->anchor()-rear).norm()<4)); // Never the rail.
        if(selected.primary>=0){++held;if(lead_id>=0&&selected.primary!=lead_id)++switches;lead_id=selected.primary;}
      }
    }
    CHECK(held>=scans*.95);CHECK(switches==0);
    std::cout<<"Scenario: corner, A2 at 40 %, dropouts, static rail: target held "<<held<<"/"<<scans<<": PASS\n";
  }
  { // C1: projected bins and scores.
    CHECK(!bins_for(Projection{}).valid);
    auto span=bins_for(box_at(320,400));CHECK(span.valid&&span.left==10&&span.right==12);
    auto clipped=bins_for(box_at(1200,1279));CHECK(clipped.right==39);
    Bins p{};p[12]=.8;std::array<bool,40> others{};
    CHECK(span_score(p,span,others,0)==.8);
    CHECK(span_score(p,bins_for(box_at(320,350)),others,0)==0);
    CHECK(span_score(p,bins_for(box_at(320,350)),others,2)==.8); // Reading window allows calibration error.
    others[12]=true;CHECK(span_score(p,span,others,1)==0);       // Never read another track's bins.
    std::cout<<"Bin spans and scores: PASS\n";
  }
  { // C3: the nearer vehicle owns its bins; the hidden one keeps its evidence.
    Track near_car,far_car;Bins p{};p[20]=.9;
    std::vector<TrackView> views{view(far_car,60,box_at(640,672)),view(near_car,20,box_at(620,700))};
    CHECK(!assign_ownership(views,p));
    for(const auto& v:views){
      if(v.track==&near_car){CHECK(v.visibility==Visibility::Visible);CHECK(v.score==.9);}
      else CHECK(v.visibility==Visibility::Occluded);
    }
    std::cout<<"Occlusion ownership: PASS\n";
  }
  { // C3: a nearby background fragment that cannot own bins does not steal the car's evidence.
    Track fragment,lead;Bins p{};p[20]=.9;
    std::vector<TrackView> views{view(fragment,12,box_at(640,672),false),view(lead,40,box_at(640,672))};
    CHECK(!assign_ownership(views,p));
    for(const auto& v:views)if(v.track==&lead){CHECK(v.visibility==Visibility::Visible);CHECK(v.score==.9);}
                            else CHECK(v.visibility==Visibility::Occluded);
    std::cout<<"Background fragment cannot take evidence: PASS\n";
  }
  { // C3: same range and same bins cannot be ordered.
    Track a,b;Bins p{};p[20]=.9;
    std::vector<TrackView> views{view(a,30,box_at(630,700)),view(b,31,box_at(630,700))};
    CHECK(assign_ownership(views,p));
    for(const auto& v:views)CHECK(v.visibility==Visibility::Occluded);
    std::cout<<"Unorderable pair is ambiguous: PASS\n";
  }
  { // C2: evidence builds and decays at the designed pace.
    EvidenceConfig config;Track t;t.evidence_stamp=0;double stamp=0;
    int frames=0;
    while(t.evidence<3&&frames<100){stamp+=1/21.;observe(t,stamp,.6,Visibility::Visible,config);++frames;}
    CHECK(frames>=12&&frames<=18); // About 0.7 s.
    t.evidence=8;int weak=0;
    while(t.evidence>1&&weak<200){stamp+=1/21.;observe(t,stamp,.1,Visibility::Visible,config);++weak;}
    CHECK(weak>=80&&weak<=105);    // About 4.4 s: a real loss still releases the target.
    t.evidence=5;
    for(int i=0;i<100;++i){stamp+=1/21.;observe(t,stamp,0,Visibility::Occluded,config);}
    CHECK(t.evidence==5);
    for(int i=0;i<100;++i){stamp+=1/21.;observe(t,stamp,0,Visibility::Outside,config);}
    CHECK(t.evidence==5);          // A camera that cannot see the car proves nothing.
    double held=t.evidence;observe(t,stamp,.9,Visibility::Visible,config);CHECK(t.evidence==held); // Repeated frame.
    observe(t,stamp-1,.9,Visibility::Visible,config);CHECK(t.evidence==held);
    for(int i=0;i<2000;++i){stamp+=1/21.;observe(t,stamp,.99,Visibility::Visible,config);}
    CHECK(t.evidence<=config.l_max);
    std::cout<<"Evidence build-up, decay, hold, duplicate frames, bounds: PASS\n";
  }
  { // Far vehicle score: no single missing cue may decide; validity gates still apply.
    FarConfig config;
    auto base=[]{Track t;t.confirmed=true;t.seen=0xFF;t.map_covered=true;t.x[2]=0;t.observed_width=.6;return t;};
    FarInputs on;on.map_ok=true;on.ego_compensated=true;on.raceline_active=false;on.speed_sigma=.5;on.range=80;
    Track moving=base();moving.x[2]=20;moving.seen=0x155;update_far_state(moving,on,config);CHECK(moving.far_vehicle); // 1 + 3
    Track stopped_seen=base();stopped_seen.seen=0x155;stopped_seen.evidence=3;
    FarInputs line=on;line.raceline_active=true;line.path_n=2;update_far_state(stopped_seen,line,config);CHECK(stopped_seen.far_vehicle); // 1+1+1.5
    Track stopped=base();stopped.seen=0x155;update_far_state(stopped,line,config);CHECK(!stopped.far_vehicle);CHECK(stopped.far_score==2);
    Track off_track=stopped_seen;off_track.far_vehicle=false;FarInputs outside=line;outside.path_n=-20;update_far_state(off_track,outside,config);CHECK(!off_track.far_vehicle);
    Track rail=moving;rail.far_vehicle=false;rail.map_static=true;update_far_state(rail,on,config);CHECK(!rail.far_vehicle);CHECK(std::isnan(rail.far_score));
    Track unmapped=moving;unmapped.far_vehicle=false;FarInputs no_map=on;no_map.map_ok=false;update_far_state(unmapped,no_map,config);CHECK(!unmapped.far_vehicle);
    Track beyond=moving;beyond.far_vehicle=false;beyond.map_covered=false;update_far_state(beyond,on,config);CHECK(!beyond.far_vehicle);
    Track wobbly=moving;wobbly.far_vehicle=false;FarInputs bad=on;bad.speed_sigma=3;update_far_state(wobbly,bad,config);CHECK(!wobbly.far_vehicle); // Motion needs a settled speed.
    Track ghost=moving;ghost.far_vehicle=false;FarInputs no_ego=on;no_ego.ego_compensated=false;update_far_state(ghost,no_ego,config);CHECK(!ghost.far_vehicle);
    Track tentative=moving;tentative.confirmed=false;tentative.far_vehicle=false;update_far_state(tentative,on,config);CHECK(!tentative.far_vehicle);
    Track hysteresis=base();hysteresis.seen=0xFF;hysteresis.far_vehicle=true;hysteresis.observed_width=1.2;
    update_far_state(hysteresis,on,config);CHECK(hysteresis.far_vehicle);CHECK(hysteresis.far_score==2); // 1.5 + .5 stays above the release level.
    hysteresis.seen=0;hysteresis.observed_width=0;update_far_state(hysteresis,on,config);CHECK(!hysteresis.far_vehicle);
    Track close=moving;FarInputs near=on;near.range=30;update_far_state(close,near,config);CHECK(!close.far_vehicle);CHECK(std::isnan(close.far_score));
    std::cout<<"Far vehicle score, hysteresis, validity gates: PASS\n";
  }
  { // T877 regression: near background without camera evidence never becomes primary.
    SelectorConfig config;EvidenceSelector selector(config);
    auto selected=selector.update(1.,1.,{target(877,13.2,0)});CHECK(selected.primary==-1);CHECK(selected.status=="no_target");
    selected=selector.update(1.1,1.1,{target(877,10.4,0),target(879,30,3.5)});CHECK(selected.primary==879&&selected.status=="camera_held");
    std::cout<<"T877 background rejection: PASS\n";
  }
  { // Evidence hold: a weak camera keeps the target, a lost LiDAR is only predicted.
    SelectorConfig config;EvidenceSelector selector(config);
    auto strong=target(5,30,4);strong.camera_matched=true;
    CHECK(selector.update(1.,1.,{strong}).status=="measured");
    auto weak=target(5,30,1.5);auto held=selector.update(1.1,1.1,{weak});CHECK(held.primary==5&&held.status=="camera_held");
    CHECK(std::abs(held.camera_age-.1)<1e-9);
    auto lost=target(5,30,1.5,false);auto predicted=selector.update(1.2,1.2,{lost});CHECK(predicted.primary==5&&predicted.status=="predicted");
    auto gone=target(5,30,.5);CHECK(selector.update(1.3,1.3,{gone}).primary==-1);
    auto unusable=target(6,30,4);CHECK(selector.update(1.4,1.4,{unusable}).primary==6);
    unusable.usable=false;CHECK(selector.update(1.5,1.5,{unusable}).primary==-1);
    std::cout<<"Evidence hold, predicted, release: PASS\n";
  }
  { // Lane, map and range conditions.
    SelectorConfig config;EvidenceSelector selector(config);
    auto outside=target(1,30,4);outside.in_lane=false;CHECK(selector.update(1,1,{outside}).primary==-1);
    auto wall=target(2,30,4);wall.map_static=true;CHECK(selector.update(1.1,1.1,{wall}).primary==-1);
    auto thin=target(3,30,4);thin.vehicle_shape=false;CHECK(selector.update(1.2,1.2,{thin}).primary==-1);
    auto close=target(4,2,4);CHECK(selector.update(1.3,1.3,{close}).primary==-1);
    auto far=target(7,80,4);CHECK(selector.update(1.4,1.4,{far}).primary==-1); // Limit stays at 50 m until raised.
    far.far_vehicle=true;CHECK(selector.update(1.5,1.5,{far}).primary==-1);
    std::cout<<"Target lane, map, shape and range gates: PASS\n";
  }
  { // Raised range: a far vehicle needs no camera, and stays down to 35 m.
    SelectorConfig config;config.target_max=120;EvidenceSelector selector(config);
    auto far=target(9,80,0);far.vehicle_shape=false;far.far_vehicle=true;
    CHECK(selector.update(1,1,{far}).primary==9);
    auto closer=far;closer.distance=closer.range=45;CHECK(selector.update(1.1,1.1,{closer}).primary==9);
    closer.distance=closer.range=30;CHECK(selector.update(1.2,1.2,{closer}).primary==-1);
    std::cout<<"Far target entry and handover: PASS\n";
  }
  { // Cut-in: a closer confirmed target replaces the primary after the hold, a blip does not.
    SelectorConfig config;EvidenceSelector selector(config);
    CHECK(selector.update(1,1,{target(1,40,5)}).primary==1);
    auto first=selector.update(1.1,1.1,{target(1,40,5),target(2,15,4)});CHECK(first.primary==1&&first.standby==2);
    CHECK(selector.update(1.2,1.2,{target(1,40,5)}).primary==1);            // Blip cancels the timer.
    CHECK(selector.update(1.3,1.3,{target(1,40,5),target(2,15,4)}).primary==1);
    CHECK(selector.update(1.5,1.5,{target(1,40,5),target(2,15,4)}).primary==1);
    CHECK(selector.update(1.61,1.61,{target(1,40,5),target(2,15,4)}).primary==2);
    std::cout<<"Cut-in replacement: PASS\n";
  }
  { // Camera silence releases near targets; time reversal resets.
    SelectorConfig config;EvidenceSelector selector(config);
    CHECK(selector.update(1,1,{target(1,30,5)}).primary==1);
    CHECK(selector.update(2.9,1,{target(1,30,5)}).primary==1);
    CHECK(selector.update(3.1,1,{target(1,30,5)}).primary==-1);
    CHECK(selector.update(3.2,1,{target(1,30,5)}).primary==-1);
    CHECK(selector.update(4,4,{target(1,30,5)}).primary==1);
    CHECK(selector.update(3,3,{}).primary==-1);
    CHECK(selector.update(std::nan(""),3,{target(1,30,5)}).primary==-1);
    SelectorConfig raised=config;raised.target_max=120;EvidenceSelector far(raised);
    auto vehicle=target(3,80,0);vehicle.vehicle_shape=false;vehicle.far_vehicle=true;
    CHECK(far.update(1,1,{vehicle}).primary==3);CHECK(far.update(4,1,{vehicle}).primary==3);
    std::cout<<"Camera silence and time reversal: PASS\n";
  }
}
