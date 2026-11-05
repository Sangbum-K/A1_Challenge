#include "a1_fusion/association.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
int main(){using namespace a1::inspection;
 // A turn must not reject a car solely because its world-axis width grows.
 auto rotated_box=[](double length,double width,double angle){
  std::vector<Eigen::Vector3d> pts;
  Box b;b.lo=Eigen::Vector3d::Constant(1e9);b.hi=-b.lo;
  for(double x:{-length/2,length/2})for(double y:{-width/2,width/2})for(double z:{0.,1.5}){
   Eigen::Vector3d q(30+x*std::cos(angle)-y*std::sin(angle),y*std::cos(angle)+x*std::sin(angle),z);
   pts.push_back(q);b.lo=b.lo.cwiseMin(q);b.hi=b.hi.cwiseMax(q);
  }
  b.points=pts.size();b.footprint=measured_footprint(pts);return b;
 };
 for(int degrees=0;degrees<=180;degrees+=5){
  auto car=rotated_box(4.5,1.8,degrees*std::acos(-1.)/180.);
  CHECK(car.footprint.valid);CHECK(std::abs(car.footprint.length-4.5)<1e-8);
  CHECK(std::abs(car.footprint.width-1.8)<1e-8);CHECK(full_vehicle_geometry(car));
  CHECK(!vehicle_extent(rotated_box(6.,.4,degrees*std::acos(-1.)/180.)));
  CHECK(!vehicle_extent(rotated_box(4.5,3.,degrees*std::acos(-1.)/180.)));
 }
 auto turning=rotated_box(4.5,1.8,.8);CHECK(turning.hi.y()-turning.lo.y()>2.6);
 turning.footprint.valid=false;CHECK(!vehicle_extent(turning)); // Conservative legacy fallback.
 CHECK(!measured_footprint({}).valid);
 CHECK(!measured_footprint({{1,2,3},{1,2,3}}).valid);
 CHECK(!measured_footprint({{1,2,3},{NAN,2,3}}).valid);
 auto line=measured_footprint({{0,0,0},{1,1,0},{2,2,0},{1,1,1}});
 CHECK(line.valid&&std::abs(line.length-std::sqrt(8.))<1e-8&&line.width<1e-8);
 std::cout<<"Measured footprint: rotation invariance, oversized rejection, invalid/collinear input: PASS\n";
 // Recorded failure shape: rear face has little depth; camera support is required.
 Box rear{{30,-.85,-.3},{30.17,.85,.27},13};
 CHECK(!full_vehicle_geometry(rear));CHECK(camera_surface_geometry(rear));
 Box clipped=rear;clipped.hi.z()=clipped.lo.z()+.2;
 CHECK(!camera_surface_geometry(clipped));CHECK(camera_surface_geometry(clipped,true));
 clipped.hi.z()=clipped.lo.z()+.1;CHECK(!camera_surface_geometry(clipped,true));
 Box low_rail{{10,2,-1},{10.4,2.86,-.75},18};CHECK(!camera_surface_geometry(low_rail));
 CameraTargetSelector confirmation;
 CHECK(confirmation.update(.1,.1,{{1,20,true,true,true,false}}).primary==-1);
 CHECK(confirmation.update(.2,.1,{{1,20,true,true,true,false}}).primary==-1); // Repeated image is not a vote.
 CHECK(confirmation.update(.3,.3,{{1,20,true,true,true,false}}).primary==-1);
 CHECK(confirmation.update(.4,.4,{{1,20,true,true,true,false}}).primary==1);
 CameraTargetSelector camera_primary(.3,1);
 auto selected=camera_primary.update(1.,1.,{{877,13.2,true,true,false,false}});
 CHECK(selected.primary==-1); // T877-like near background may never acquire a primary.
 selected=camera_primary.update(1.1,1.1,{{877,10.4,true,true,false,false},{879,30,true,true,true,false}});
 CHECK(selected.primary==879&&selected.status=="measured");
 selected=camera_primary.update(1.2,1.2,{{877,10,true,true,true,false},{879,29,true,true,true,false}});
 CHECK(selected.primary==879&&selected.standby==877); // Stable target, no nearest-distance hopping.
 selected=camera_primary.update(1.3,1.3,{{877,9,true,true,false,false},{879,28,true,false,false,false}});
 CHECK(selected.primary==879&&selected.status=="predicted");
 selected=camera_primary.update(1.4,1.4,{{879,27,true,true,true,false}});
 CHECK(selected.primary==879&&selected.status=="measured");
 selected=camera_primary.update(1.5,1.5,{{879,26,true,true,false,true}});
 CHECK(selected.primary==879&&selected.status=="held_ambiguous");
 selected=camera_primary.update(1.71,1.71,{{877,8,true,true,false,false},{879,25,true,true,false,false}});
 CHECK(selected.primary==-1); // Camera loss cannot be refreshed by ongoing LiDAR.
 selected=camera_primary.update(2.,2.,{{42,20,true,true,true,false}});CHECK(selected.primary==42);
 selected=camera_primary.update(2.2,2.,{{42,20,true,true,true,false}});CHECK(selected.primary==42);
 selected=camera_primary.update(2.31,2.,{{42,20,true,true,true,false}});CHECK(selected.primary==-1);
 camera_primary.update(3.,3.,{{42,20,true,true,true,false}});
 selected=camera_primary.update(2.9,2.9,{{42,20,true,true,false,false}});CHECK(selected.primary==-1);
 CHECK(camera_primary.update(4.,4.,{{42,20,false,true,true,false}}).primary==-1);
 CHECK(camera_primary.update(NAN,4.,{}).primary==-1);
 std::cout<<"Camera-led target: background rejection, sticky ID, bounded coasting, ambiguity, repeated-camera expiry: PASS\n";
 // Regression: observed rear surface at the recorded 24.13 m dropout.
 Track partial;partial.hits=8;partial.raw_box={{23.8,1.7,-.2},{24.242,3.529,.220},29};
 partial.vehicle_evidence=true;CHECK(std::string(vehicle_geometry_status(partial))=="retained_partial");
 partial.vehicle_evidence=false;CHECK(std::string(vehicle_geometry_status(partial))=="far_partial");
 partial.hits=3;CHECK(std::string(vehicle_geometry_status(partial))=="insufficient_geometry");
 partial.vehicle_evidence=true;partial.raw_box.hi.z()=-.1;CHECK(std::string(vehicle_geometry_status(partial))=="insufficient_geometry");
 // Two separated rear surfaces, sparse vertical rings: fixed 3D .45 m splits them.
 for(double range:{24.,50.,99.}){
  std::vector<Eigen::Vector3d> cloud;
  for(double y:{-2.,2.})for(int col=0;col<5;++col)for(double z:{0.,.7})cloud.emplace_back(range,y+col*.3,z);
  auto clusters=adaptive_clusters(cloud);CHECK(clusters.size()==2);CHECK(clusters[0].size()==10&&clusters[1].size()==10);
  std::reverse(cloud.begin(),cloud.end());clusters=adaptive_clusters(cloud);CHECK(clusters.size()==2);
 }
 // No radial-band boundary, no association of ground and overhead with same XY.
 std::vector<Eigen::Vector3d> boundary;
 for(int i=0;i<8;++i)boundary.emplace_back(19.7+i*.1,0.,0.);
 CHECK(adaptive_clusters(boundary).size()==1);
 auto upper=boundary;for(auto& q:upper)q.z()=2.;boundary.insert(boundary.end(),upper.begin(),upper.end());CHECK(adaptive_clusters(boundary).size()==2);
 bool rejected=false;try{ClusterConfig c;c.slope=-1;adaptive_clusters(boundary,c);}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
 Tracker shrinking;int64_t stable=-1;
 for(int frame=0;frame<50;++frame){double x=20+frame*.1;double depth=std::max(.3,1.-frame*.025);double height=std::max(.25,.8-frame*.02);
  Box observation{{x-depth/2,-.9,-.2},{x+depth/2,.9,-.2+height},20};
  auto ts=shrinking.update(100+frame*.1,{observation});CHECK(ts.size()==1);
  if(stable<0)stable=ts[0].id;
  CHECK(ts[0].id==stable);
  if(frame>=3){CHECK(shrinking.usable(ts[0],100+frame*.1));CHECK(std::string(vehicle_geometry_status(ts[0]))!="insufficient_geometry");}
 }
 auto shrink_coast=shrinking.update(105.,{});CHECK(shrink_coast.size()==1&&shrink_coast[0].id==stable&&shrinking.usable(shrink_coast[0],105.));
 CHECK(shrinking.update(106.,{}).empty());
 std::cout<<"Far partial surfaces, adjacent vehicles, radial boundary, height separation: PASS\n";
 std::array<double,40> ps{};ps[5]=.8;ps[6]=.4;ps[30]=.7;
 auto regions=camera_regions(ps,{},.5,.35,.25);CHECK(regions.size()==2);CHECK(regions[0].left==5&&regions[0].right==6);
 ps[5]=.4;ps[30]=.4;ps[20]=.4;auto held=camera_regions(ps,regions,.5,.35,.25);CHECK(held.size()==2); // New weak region must not inherit another region's hysteresis.
 ps[5]=NAN;ps[6]=0;CHECK(camera_regions(ps,{},.5,.35,.25).empty());
 Eigen::MatrixXd edges(2,2);edges<<.8,0,0,.7;auto matched=unique_region_matches(edges,.15);CHECK(matched[0]==0&&matched[1]==1);
 edges<<.8,0,.7,0;matched=unique_region_matches(edges,.15);CHECK(matched[0]==-1&&matched[1]==-1);
 edges<<.8,.7,0,0;matched=unique_region_matches(edges,.15);CHECK(matched[0]==-1);
 CHECK(unique_region_matches(Eigen::MatrixXd::Zero(2,0),.15)==std::vector<int>({-1,-1}));
 for(double d:{10.,50.}){Box q{{d-1,-1,-1},{d+1,1,1},10};CHECK(in_target_range(q,10,50));}
 for(double d:{9.99,50.01}){Box q{{d-1,-1,-1},{d+1,1,1},10};CHECK(!in_target_range(q,10,50));}
 std::cout<<"Two camera regions, independent hysteresis, ambiguous edges, 10/50m boundaries: PASS\n";
 CHECK(in_front_half(Eigen::Vector3d(1,-2,0),-M_PI/4));
 CHECK(!in_front_half(Eigen::Vector3d(1,2,0),-M_PI/4));
 CHECK(in_front_half(Eigen::Vector3d(-1,-2,0),-M_PI/4));
 auto rotation=body_from_lidar(M_PI/2);
 CHECK((rotation*Eigen::Vector3d(0,-10,0)-Eigen::Vector3d(10,0,0)).norm()<1e-9);
 CHECK(in_front_half(rotation*Eigen::Vector3d(0,-10,0)));
 CHECK(!in_front_half(rotation*Eigen::Vector3d(0,10,0)));
 auto opt=nominal_optical_from_body();Eigen::Vector3d camera(.8,0,-.3);
 Eigen::Vector3d center=opt*(rotation*Eigen::Vector3d(0,-10,-.3)-camera);
 CHECK(center.x()==0 || std::abs(center.x())<1e-9);CHECK(std::abs(center.y())<1e-9);CHECK(center.z()>0);
 CHECK((opt*(rotation*Eigen::Vector3d(2,-10,-.3)-camera)).x()<0);
 CHECK((opt*(rotation*Eigen::Vector3d(-2,-10,-.3)-camera)).x()>0);
 std::cout<<"LiDAR -Y front / +X camera-left / -X camera-right / optical center: PASS\n";
 Eigen::MatrixXd c(2,3);c<<.1,.2,1,.11,.9,1;auto a=hungarian(c);CHECK(a[0]==1&&a[1]==0);
 Tracker tracker;int64_t id=-1;
 for(int i=0;i<50;++i){double x=20-i*.1;Box b{{x-1,-1,-1},{x+1,1,1},100};const auto& ts=tracker.update(1+i*.1,{b});CHECK(ts.size()==1);if(i==0)id=ts[0].id;CHECK(ts[0].id==id);if(i==49)CHECK(std::abs(ts[0].state[2]+1)<.02);}
 CHECK(tracker.update(6.8,{}).empty());
 Eigen::Matrix3d optical;optical<<0,-1,0,0,0,-1,1,0,0;
 Box b{{9,-1,-1},{11,1,1},100};auto p=project(b,optical,Eigen::Vector3d::Zero(),640,640,640,360,1280,720);
 CHECK(p.valid);CHECK(p.x0<640&&p.x1>640);CHECK(horizontal_iou(p,576,704)>.8);
 Box behind{{-11,-1,-1},{-9,1,1},100};CHECK(!project(behind,optical,Eigen::Vector3d::Zero(),640,640,640,360,1280,720).valid);

 Box fragment1{{9,-.9,-1},{10,-.1,.5},50},fragment2{{9,.15,-1},{10,.9,.5},50};
 CHECK(merge_fragments({fragment1,fragment2}).size()==1);
 Box car1{{9,-1,-1},{13,1,.5},100},car2{{9,1.2,-1},{13,3.2,.5},100};
 CHECK(merge_fragments({car1,car2}).size()==2);
 CHECK(vehicle_extent(car1));Box wall{{8,-4,-1},{14,4,3},100};CHECK(!vehicle_extent(wall));
 Tracker jitter;double raw_error=0,filtered_error=0,raw_size_error=0,filtered_size_error=0;
 for(int i=0;i<100;++i){
  double noise=i%2?.2:-.2,x=20-i*.1+noise,half=1+noise;
  Box obs{{x-half,-1,-1},{x+half,1,1},100};auto ts=jitter.update(10+i*.1,{obs});
  CHECK(ts.size()==1);CHECK((ts[0].raw_box.center()-obs.center()).norm()<1e-9);
  if(i>20){raw_error+=noise*noise;filtered_error+=std::pow(ts[0].box.center().x()-(20-i*.1),2);
   raw_size_error+=std::pow(2*noise,2);filtered_size_error+=std::pow((ts[0].box.hi-ts[0].box.lo).x()-2,2);}
 }
 CHECK(filtered_error<raw_error*.5);CHECK(filtered_size_error<raw_size_error*.5);
 std::cout<<"Fragment merge, adjacent-car separation, extent gate, moving-track jitter reduction: PASS\n";
 // Learn a moving target, coast for two scans, then reconnect without resetting velocity.
 Tracker continuity;int64_t stable_id=-1;double expected_x=0;
 for(int i=0;i<30;++i){expected_x=30-.2*i;Box obs{{expected_x-1,-1,-1},{expected_x+1,1,1},100};auto ts=continuity.update(20+i*.1,{obs});stable_id=ts[0].id;}
 const auto before=continuity.update(23.,{});CHECK(before.size()==1);CHECK(!before[0].measured);CHECK(continuity.usable(before[0],23.));
 CHECK(std::abs(before[0].box.center().x()-24.)<.03);CHECK(std::abs(before[0].state[2]+2)<.03);
 const auto coast=continuity.update(23.1,{});CHECK(continuity.usable(coast[0],23.1));
 Box returned{{22.6,-1,-1},{24.6,1,1},100};const auto back=continuity.update(23.2,{returned});CHECK(back.size()==1&&back[0].id==stable_id);CHECK(std::abs(back[0].state[2]+2)<.05);
 const auto hidden=continuity.update(23.6,{});CHECK(hidden.size()==1&&!continuity.usable(hidden[0],23.6));
 CHECK(continuity.update(24.1,{}).empty());
 Tracker ambiguous;Box same{{19,-1,-1},{21,1,1},100};
 for(int i=0;i<10;++i)ambiguous.update(30+i*.1,{same});
 Box left=same,right=same;left.lo.x()-=.1;left.hi.x()-=.1;right.lo.x()+=.1;right.hi.x()+=.1;
 auto uncertain=ambiguous.update(31.,{left,right});CHECK(uncertain.size()==1&&!uncertain[0].measured);
 auto restored=ambiguous.update(31.1,{same});CHECK(restored.size()==1&&restored[0].id==uncertain[0].id);
 Track near,far;near.id=1;far.id=2;near.box={{9,-1,-1},{11,1,1},10};far.box={{19,-1,-1},{21,1,1},10};near.measured=far.measured=true;
 CHECK(select_primary({&far,&near},-1)==1);near.measured=false;CHECK(select_primary({&near,&far},1)==1);CHECK(select_primary({&far},1)==2);CHECK(select_primary({&near},-1)==-1);
 std::cout<<"Moving target coasting/reconnection, expiry, ambiguous duplicate suppression, primary hold/standby: PASS\n";
 std::cout<<"Hungarian global assignment, CV velocity/ID, expiry, forward/behind-camera projection: PASS\n";
}
