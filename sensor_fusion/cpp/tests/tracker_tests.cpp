#include "a1_fusion/tracker.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#define CHECK(x) do{if(!(x))throw std::runtime_error(std::string(#x)+" line "+std::to_string(__LINE__));}while(false)
using namespace a1::inspection;
using V2=Eigen::Vector2d;
using V3=Eigen::Vector3d;
namespace {
struct Object { std::vector<V2> xy; std::vector<double> heights{-.6,-.2,.2,.6}; };

// Rear and sensor-facing side surfaces of a rectangular vehicle, world frame.
std::vector<V2> car_surface(const V2& rear,double heading,const V2& sensor,double length=4.5,double width=1.9,double step=.12){
  const V2 u(std::cos(heading),std::sin(heading)),v(-u.y(),u.x());
  std::vector<V2> out;
  const bool rear_seen=(sensor-rear).dot(u)<0;
  if(rear_seen)for(double b=-width/2;b<=width/2+1e-9;b+=step)out.push_back(rear+v*b);
  const double side=(sensor-rear).dot(v)>0?width/2:-width/2;
  if(std::abs((sensor-rear).dot(v))>width/2)for(double a=0;a<=length+1e-9;a+=step)out.push_back(rear+u*a+v*side);
  return out;
}

Scan make_scan(double t,const Pose2& ego,const std::vector<Object>& objects,bool map_ok=true){
  Scan s;s.stamp=t;s.ego=ego;s.map_ok=map_ok;
  s.heading_prior=[yaw=ego.yaw](const V2&){return yaw;};
  for(const auto& o:objects){
    std::vector<int> ids;
    for(const auto& p:o.xy)for(double z:o.heights){
      const V2 b=ego.to_body(p);
      ids.push_back(int(s.world_points.size()));
      s.world_points.emplace_back(p.x(),p.y(),z);s.body_points.emplace_back(b.x(),b.y(),z);s.point_time.push_back(t);
    }
    if(ids.empty())continue;
    Detection d=make_detection(s,ids,40.);
    if(d.quality!=Quality::Reject)s.detections.push_back(d);
  }
  return s;
}
const Track* only_confirmed(const std::vector<Track>& tracks){
  const Track* out=nullptr;
  for(const auto& t:tracks)if(t.confirmed){CHECK(out==nullptr);out=&t;}
  return out;
}
struct Circle { double radius,omega,phase;
  V2 at(double t)const{const double a=phase+omega*t;return V2(radius*std::sin(a),radius*(1-std::cos(a)));}
  double heading(double t)const{return phase+omega*t;} };
}

int main(){
  const double pi=std::acos(-1.);
  { // Pose conversions are inverse of each other.
    Pose2 p{10,-3,.7};V2 q(4,5);CHECK((p.to_body(p.to_world(q))-q).norm()<1e-12);
    Pose2 yaw90{0,0,pi/2};CHECK((yaw90.to_world(V2(1,0))-V2(0,1)).norm()<1e-12);
  }
  { // Per-azimuth scan time, with a safe fallback when timestamps are bunched.
    ScanTiming timing;
    for(int a=0;a<360;a+=1)timing.add(a*pi/180,100.+a/3600.);
    timing.finish(100.1);CHECK(timing.valid());
    CHECK(std::abs(timing.at(90*pi/180)-100.025)<1e-3);
    CHECK(std::abs(timing.at(-90*pi/180)-100.075)<1e-3); // Same cell as +270 deg.
    ScanTiming bunched;for(int a=0;a<360;++a)bunched.add(a*pi/180,100.);
    bunched.finish(100.1);CHECK(!bunched.valid());CHECK(bunched.at(1.)==100.1);
    ScanTiming sparse;sparse.add(0,100.);sparse.add(pi,100.05);sparse.finish(100.1);
    CHECK(sparse.valid());CHECK(std::abs(sparse.at(pi/2)-100.05)<.051); // Empty cells borrow a neighbour.
    std::cout<<"Pose2 and ScanTiming: PASS\n";
  }
  { // T5: CTRV prediction and Jacobian.
    Track::State x;x<<3,4,20,.7,.25;
    const auto F=ctrv_jacobian(x,.1);
    for(int j=0;j<5;++j){
      Track::State plus=x,minus=x;const double h=1e-6;plus[j]+=h;minus[j]-=h;
      const Track::State d=(ctrv_predict(plus,.1)-ctrv_predict(minus,.1))/(2*h);
      for(int i=0;i<5;++i)CHECK(std::abs(F(i,j)-d[i])<1e-6);
    }
    Track::State straight=x,nearly=x;straight[4]=0;nearly[4]=1e-3+1e-9;
    CHECK((ctrv_predict(straight,.1).head<2>()-ctrv_predict(nearly,.1).head<2>()).norm()<1e-3);
    const auto Fs=ctrv_jacobian(straight,.1);
    for(int j=0;j<5;++j){
      // The yaw-rate column is the limit of the curved model, so difference across the branch.
      const double h=j==4?2e-3:1e-7;
      Track::State plus=straight,minus=straight;plus[j]+=h;minus[j]-=h;
      const Track::State d=(ctrv_predict(plus,.1)-ctrv_predict(minus,.1))/(2*h);
      for(int i=0;i<4;++i)CHECK(std::abs(Fs(i,j)-d[i])<1e-3);
    }
    std::cout<<"CTRV model and Jacobian against numeric derivative: PASS\n";
  }
  { // T2: rear anchor.
    const V2 sensor(0,0);
    auto straight=rear_anchor(car_surface(V2(30,0),0,sensor),0,sensor,1.9);
    CHECK(straight.rear_visible);CHECK((straight.anchor-V2(30,0)).norm()<.12);CHECK(std::abs(straight.width-1.9)<.15);
    for(double degrees:{10.,20.,30.}){
      const double h=degrees*pi/180;
      auto fit=rear_anchor(car_surface(V2(30,0),h,sensor),h,sensor,1.9);
      CHECK(fit.rear_visible);CHECK((fit.anchor-V2(30,0)).norm()<.3);
    }
    // Same rear point, seen with the side surface: the box centre would move, the anchor must not.
    auto turned=rear_anchor(car_surface(V2(30,3),.5,sensor),.5,sensor,1.9);
    CHECK((turned.anchor-V2(30,3)).norm()<.3);
    // Only a narrow part of the rear is visible: lateral uncertainty grows.
    std::vector<V2> narrow;for(double b=.3;b<=.95;b+=.1)narrow.push_back(V2(30,b));
    auto partial=rear_anchor(narrow,0,sensor,1.9);CHECK(partial.sigma_lateral>=.5);CHECK(partial.rear_visible);
    // Passing alongside: the rear is out of sight, so along-track information is discarded.
    std::vector<V2> side;for(double a=0;a<=4.5;a+=.1)side.push_back(V2(a-2,3));
    auto alongside=rear_anchor(side,0,sensor,1.9);CHECK(!alongside.rear_visible);CHECK(alongside.sigma_along>=2.);
    std::cout<<"Rear anchor: rotation, partial width, side-only: PASS\n";
  }
  { // T1: a stationary object seen from a turning ego has no world velocity and keeps its ID.
    Tracker tracker;Circle ego_path{20/.3,.3,0};int64_t id=-1;
    const V2 rear(45,8);
    for(int i=0;i<40;++i){
      const double t=1+i*.1;Pose2 ego{ego_path.at(i*.1).x(),ego_path.at(i*.1).y(),ego_path.heading(i*.1)};
      const V2 sensor(ego.x,ego.y);
      Object post{car_surface(rear,0,sensor,1.,1.)};
      const auto& tracks=tracker.update(make_scan(t,ego,{post}));
      const Track* tr=only_confirmed(tracks);
      if(i>=3){CHECK(tr);if(id<0)id=tr->id;CHECK(tr->id==id);}
      if(i>=15)CHECK(tr->velocity()<1.5);
    }
    std::cout<<"Stationary object from a turning ego: world velocity ~0, stable ID: PASS\n";
  }
  { // T1+T5: lead car on the same circle, three lost scans, same ID, true speed and yaw rate.
    Tracker tracker;Circle ego_path{20/.3,.3,0},lead{20/.3,.3,.45};int64_t id=-1;
    for(int i=0;i<60;++i){
      const double t=1+i*.1;Pose2 ego{ego_path.at(i*.1).x(),ego_path.at(i*.1).y(),ego_path.heading(i*.1)};
      const V2 sensor(ego.x,ego.y);std::vector<Object> objects;
      if(i<40||i>42)objects.push_back({car_surface(lead.at(i*.1),lead.heading(i*.1),sensor)});
      const auto& tracks=tracker.update(make_scan(t,ego,objects));
      const Track* tr=only_confirmed(tracks);
      if(i>=4){CHECK(tr);if(id<0)id=tr->id;CHECK(tr->id==id);}
      if(i>=20){CHECK(std::abs(tr->velocity()-20)<1.5);CHECK((tr->anchor()-lead.at(i*.1)).norm()<1.2);}
      if(i>=30&&i<40)CHECK(std::abs(tr->yaw_rate()-.3)<.15);
    }
    std::cout<<"Lead car through a corner with dropout: same ID, speed, yaw rate: PASS\n";
  }
  { // T5: two-point start links stationary and fast objects within the first scans.
    for(double speed:{0.,30.}){
      Tracker tracker;Pose2 ego{0,0,0};const V2 sensor(0,0);int64_t id=-1;
      for(int i=0;i<4;++i){
        const double t=1+i*.1;const V2 rear(30+speed*i*.1,0);
        const auto& tracks=tracker.update(make_scan(t,ego,{{car_surface(rear,0,sensor)}}));
        CHECK(tracks.size()==1);if(id<0)id=tracks[0].id;CHECK(tracks[0].id==id);
      }
      CHECK(tracker.tracks()[0].confirmed);CHECK(std::abs(tracker.tracks()[0].velocity()-speed)<1.5);
    }
    std::cout<<"Two-point initialisation for 0 and 30 m/s: PASS\n";
  }
  { // Lifecycle: confirmation, usable window, expiry, time reversal.
    TrackerConfig config;Tracker tracker(config);Pose2 ego{0,0,0};const V2 sensor(0,0);
    Object car{car_surface(V2(30,0),0,sensor)};
    CHECK(!tracker.update(make_scan(1.,ego,{car})).at(0).confirmed);
    CHECK(!tracker.update(make_scan(1.1,ego,{car})).at(0).confirmed);
    CHECK(tracker.update(make_scan(1.2,ego,{car})).at(0).confirmed);
    for(int i=3;i<10;++i)tracker.update(make_scan(1+i*.1,ego,{car}));
    const auto coasting=tracker.update(make_scan(2.2,ego,{}));CHECK(coasting.size()==1&&!coasting[0].measured);
    CHECK(tracker.usable(coasting[0],2.2));
    const auto late=tracker.update(make_scan(2.6,ego,{}));CHECK(late.size()==1&&!tracker.usable(late[0],2.6));
    CHECK(tracker.update(make_scan(3.6,ego,{})).empty());
    tracker.update(make_scan(4.,ego,{car}));CHECK(tracker.update(make_scan(3.,ego,{car})).size()==1);
    std::cout<<"Track lifecycle: confirm, usable, expire, time reversal: PASS\n";
  }
  { // T3: fragments become one measurement, no duplicate IDs.
    Tracker tracker;Pose2 ego{0,0,0};const V2 sensor(0,0);
    auto car=car_surface(V2(30,0),0,sensor);
    for(int i=0;i<6;++i)tracker.update(make_scan(1+i*.1,ego,{{car}}));
    const auto id=tracker.tracks()[0].id;
    std::vector<V2> left,right;for(const auto& p:car)(p.y()>0?left:right).push_back(p);
    const auto& split=tracker.update(make_scan(1.6,ego,{{left},{right}}));
    CHECK(split.size()==1);CHECK(split[0].id==id&&split[0].measured);CHECK(split[0].guided);
    CHECK(std::abs(split[0].width-1.9)<.3);
    std::cout<<"Track-guided gathering joins fragments: PASS\n";
  }
  { // T3: a cluster that swallowed the car and a long rail still gives the car measurement.
    Tracker tracker;Pose2 ego{0,0,0};const V2 sensor(0,0);
    auto car=car_surface(V2(30,0),0,sensor);
    for(int i=0;i<6;++i)tracker.update(make_scan(1+i*.1,ego,{{car}}));
    Object merged{car};for(double x=20;x<=60;x+=.2)merged.xy.push_back(V2(x,4.2));
    const auto& tracks=tracker.update(make_scan(1.6,ego,{merged}));
    CHECK(tracks.size()==1&&tracks[0].measured);CHECK((tracks[0].anchor()-V2(30,0)).norm()<.5);
    std::cout<<"Over-merged cluster: car part is gathered: PASS\n";
  }
  { // T3: gathering is cancelled for oversized and mapped-structure points.
    Tracker tracker;Pose2 ego{0,0,0};const V2 sensor(0,0);
    auto car=car_surface(V2(30,0),0,sensor);
    for(int i=0;i<6;++i)tracker.update(make_scan(1+i*.1,ego,{{car}}));
    Object wall;for(double y=-5;y<=5;y+=.1)wall.xy.push_back(V2(30.2,y));
    auto s=make_scan(1.6,ego,{wall});CHECK(s.detections.empty());
    CHECK(!tracker.update(s)[0].measured); // A wall crossing the box must not update the track.
    Tracker railed;for(int i=0;i<6;++i)railed.update(make_scan(1+i*.1,ego,{{car}}));
    auto mapped=make_scan(1.6,ego,{{car}});mapped.is_structure=[](int){return true;};
    mapped.detections.clear();
    CHECK(!railed.update(mapped)[0].measured); // Points known as map structure are not gathered.
    std::cout<<"Gathering rejects walls and map structure: PASS\n";
  }
  for(bool guided:{true,false}){ // T4: a one-ring return continues a confirmed track.
    TrackerConfig config;config.guided=guided;
    Tracker tracker(config);Pose2 ego{0,0,0};const V2 sensor(0,0);
    auto car=car_surface(V2(70,0),0,sensor);
    for(int i=0;i<6;++i)tracker.update(make_scan(1+i*.1,ego,{{car}}));
    auto id=tracker.tracks()[0].id;
    std::vector<V2> ring;for(double b=-.9;b<=.9;b+=.35)ring.push_back(V2(70,b));
    Object one{ring,{.2}};auto s=make_scan(1.6,ego,{one});
    CHECK(s.detections.size()==1&&s.detections[0].quality==Quality::Weak);
    auto& tracks=tracker.update(s);CHECK(tracks.size()==1&&tracks[0].id==id&&tracks[0].measured&&tracks[0].weak_update);
    std::cout<<"Weak one-ring measurement continues a track: PASS\n";
  }
  { // T4: weak births need a healthy map, stay tentative, and are capped.
    Pose2 ego{0,0,0};std::vector<V2> ring;for(double b=-.9;b<=.9;b+=.35)ring.push_back(V2(90,b));
    Object one{ring,{.2}};
    Tracker no_map;CHECK(no_map.update(make_scan(1.,ego,{one},false)).empty());
    Tracker tracker;
    for(int i=0;i<2;++i)CHECK(!tracker.update(make_scan(1+i*.1,ego,{one})).at(0).confirmed);
    CHECK(!tracker.update(make_scan(1.2,ego,{one})).at(0).confirmed); // 3 of 10 is not enough.
    CHECK(tracker.update(make_scan(1.3,ego,{one})).at(0).confirmed);
    TrackerConfig small;small.max_weak_tracks=3;Tracker capped(small);std::vector<Object> many;
    for(int k=0;k<8;++k){std::vector<V2> r;for(double b=-.9;b<=.9;b+=.35)r.push_back(V2(80+k*6,b));many.push_back({r,{.2}});}
    CHECK(capped.update(make_scan(1.,ego,many)).size()==3);
    std::cout<<"Weak births: map gate, 4 of 10 confirmation, cap: PASS\n";
  }
  { // Near-range weak clusters are rejected; thin near surfaces are not vehicles.
    Box rail{{10,2,-1},{10.4,2.86,-.75},18};CHECK(classify(rail,10.,40.)==Quality::Reject);
    Box far_rail=rail;CHECK(classify(far_rail,60.,40.)==Quality::Weak);
    Box car{{30,-.9,-.5},{34,.9,.8},60};CHECK(classify(car,30.,40.)==Quality::Strong);
    Box wall{{8,-4,-1},{14,4,3},100};CHECK(classify(wall,60.,40.)==Quality::Reject);
    Box few=far_rail;few.points=2;CHECK(classify(few,60.,40.)==Quality::Reject);
    std::cout<<"Detection quality classes: PASS\n";
  }
  { // Near-tie measurements update the track instead of dropping it; no duplicate ID.
    TrackerConfig config;config.guided=false; // Gathering would merge these before assignment.
    Tracker tracker(config);Pose2 ego{0,0,0};const V2 sensor(0,0);
    auto car=car_surface(V2(30,0),0,sensor);
    for(int i=0;i<8;++i)tracker.update(make_scan(1+i*.1,ego,{{car}}));
    auto a=car_surface(V2(29.9,0),0,sensor),b=car_surface(V2(30.1,0),0,sensor);
    const auto& tracks=tracker.update(make_scan(1.8,ego,{{a},{b}}));CHECK(tracks.size()==1&&tracks[0].measured);
    std::cout<<"Near-tie measurements keep the track: PASS\n";
  }
  { // Two neighbouring cars keep separate IDs.
    Tracker tracker;Pose2 ego{0,0,0};const V2 sensor(0,0);int64_t left=-1,right=-1;
    for(int i=0;i<30;++i){
      const double t=1+i*.1,x=30+i*.5;
      const auto& tracks=tracker.update(make_scan(t,ego,{{car_surface(V2(x,1.6),0,sensor)},{car_surface(V2(x,-1.6),0,sensor)}}));
      CHECK(tracks.size()==2);
      if(i==5){left=tracks[0].anchor().y()>0?tracks[0].id:tracks[1].id;right=tracks[0].anchor().y()>0?tracks[1].id:tracks[0].id;}
      if(i>5)for(const auto& tr:tracks)CHECK(tr.id==(tr.anchor().y()>0?left:right));
    }
    std::cout<<"Adjacent vehicles keep separate IDs: PASS\n";
  }
  { // Map structure ratio marks static tracks after repeated measurements.
    Tracker tracker;Pose2 ego{0,0,0};const V2 sensor(0,0);Object rail{car_surface(V2(30,5),0,sensor,3.,.4)};
    for(int i=0;i<6;++i){auto s=make_scan(1+i*.1,ego,{rail});s.is_structure=[](int){return true;};s.is_covered=[](int){return true;};tracker.update(s);}
    CHECK(tracker.tracks().at(0).map_static);CHECK(tracker.tracks()[0].map_covered);
    Tracker moving;for(int i=0;i<6;++i){auto s=make_scan(1+i*.1,ego,{rail});s.is_structure=[](int){return false;};s.is_covered=[](int){return false;};moving.update(s);}
    CHECK(!moving.tracks().at(0).map_static);CHECK(!moving.tracks()[0].map_covered);
    Tracker unknown;for(int i=0;i<6;++i)unknown.update(make_scan(1+i*.1,ego,{rail},false));
    CHECK(!unknown.tracks().at(0).map_static);CHECK(std::isnan(unknown.tracks()[0].static_ratio));
    std::cout<<"Map structure ratio and coverage: PASS\n";
  }
  { // Ego link: short gaps never restart the frame, a real loss does, recovery must be stable.
    EgoLink link;
    CHECK(!link.update(1.,true));CHECK(!link.compensated());
    CHECK(!link.update(1.5,true));CHECK(!link.compensated());
    CHECK(link.update(2.1,true));CHECK(link.compensated());       // 1 s continuous: frame changes.
    CHECK(!link.update(2.2,true));
    CHECK(link.update(3.,false));CHECK(!link.compensated());      // Unavailable means past the 0.5 s bridge.
    CHECK(!link.update(3.1,true));CHECK(!link.update(3.5,false)); // A flicker restarts the wait.
    CHECK(!link.update(3.6,true));CHECK(!link.update(4.,true));CHECK(link.update(4.7,true));
    std::cout<<"Ego link transitions: PASS\n";
  }
  { // Geometry audit strings keep their meaning for recorded dropouts.
    Track partial;partial.hits=8;partial.raw_box={{23.8,1.7,-.2},{24.242,3.529,.220},29};
    partial.vehicle_evidence=true;CHECK(std::string(vehicle_geometry_status(partial))=="retained_partial");
    partial.vehicle_evidence=false;CHECK(std::string(vehicle_geometry_status(partial))=="far_partial");
    partial.hits=3;CHECK(std::string(vehicle_geometry_status(partial))=="insufficient_geometry");
    partial.weak_update=true;CHECK(std::string(vehicle_geometry_status(partial))=="weak_surface");
  }
  { // Output geometry: nearest rear corner in body axes, predicted body box.
    Track tr;tr.x<<30,2,20,0,0;tr.width=2;tr.length=4;tr.height=1.5;tr.z=.1;
    Pose2 ego{0,0,0};CHECK(std::abs(near_surface_x(tr,ego)-30)<1e-9);
    Track yawed=tr;yawed.x[3]=pi/2;CHECK(std::abs(near_surface_x(yawed,ego)-29)<1e-9);
    auto box=track_body_box(tr,ego);CHECK(std::abs(box.lo.x()-30)<1e-9);CHECK(std::abs(box.hi.x()-34)<1e-9);
    CHECK(std::abs(box.lo.y()-1)<1e-9);CHECK(std::abs(box.hi.y()-3)<1e-9);CHECK(std::abs(box.hi.z()-.85)<1e-9);
    auto rear=track_rear_box(tr,ego);CHECK(std::abs(rear.hi.x()-30)<1e-9);CHECK(std::abs(rear.lo.y()-1)<1e-9);
    const V2 velocity=relative_velocity(tr,ego,V2(15,0));CHECK(std::abs(velocity.x()-5)<1e-9);
    std::cout<<"Output geometry and relative velocity: PASS\n";
  }
}
