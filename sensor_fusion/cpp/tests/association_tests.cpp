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
 std::cout<<"Adjacent vehicles, radial boundary, height separation: PASS\n";
 std::array<double,40> ps{};ps[5]=.8;ps[6]=.4;ps[30]=.7;
 auto regions=camera_regions(ps,{},.5,.35,.25);CHECK(regions.size()==2);CHECK(regions[0].left==5&&regions[0].right==6);
 ps[5]=.4;ps[30]=.4;ps[20]=.4;auto held=camera_regions(ps,regions,.5,.35,.25);CHECK(held.size()==2); // New weak region must not inherit another region's hysteresis.
 ps[5]=NAN;ps[6]=0;CHECK(camera_regions(ps,{},.5,.35,.25).empty());
 for(double d:{10.,50.}){Box q{{d-1,-1,-1},{d+1,1,1},10};CHECK(in_target_range(q,10,50));}
 for(double d:{9.99,50.01}){Box q{{d-1,-1,-1},{d+1,1,1},10};CHECK(!in_target_range(q,10,50));}
 std::cout<<"Two camera regions, independent hysteresis, 10/50m boundaries: PASS\n";
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
 Eigen::Matrix3d optical;optical<<0,-1,0,0,0,-1,1,0,0;
 Box b{{9,-1,-1},{11,1,1},100};auto p=project(b,optical,Eigen::Vector3d::Zero(),640,640,640,360,1280,720);
 CHECK(p.valid);CHECK(p.x0<640&&p.x1>640);
 Box behind{{-11,-1,-1},{-9,1,1},100};CHECK(!project(behind,optical,Eigen::Vector3d::Zero(),640,640,640,360,1280,720).valid);

 Box fragment1{{9,-.9,-1},{10,-.1,.5},50},fragment2{{9,.15,-1},{10,.9,.5},50};
 CHECK(merge_fragments({fragment1,fragment2}).size()==1);
 Box car1{{9,-1,-1},{13,1,.5},100},car2{{9,1.2,-1},{13,3.2,.5},100};
 CHECK(merge_fragments({car1,car2}).size()==2);
 CHECK(vehicle_extent(car1));Box wall{{8,-4,-1},{14,4,3},100};CHECK(!vehicle_extent(wall));
 std::cout<<"Fragment merge, adjacent-car separation, extent gate: PASS\n";
 // Lens distortion moves a corner point outward for positive k1; zero stays a pinhole.
 Box corner{{9,-5,-1},{11,-4,1},10};
 auto pinhole=project(corner,optical,Eigen::Vector3d::Zero(),640,640,640,360,1280,720);
 auto bent=project(corner,optical,Eigen::Vector3d::Zero(),640,640,640,360,1280,720,Distortion{.1,0,0,0});
 CHECK(pinhole.valid&&bent.valid);CHECK(bent.x1>pinhole.x1);
 auto same=project(corner,optical,Eigen::Vector3d::Zero(),640,640,640,360,1280,720,Distortion{});CHECK(same.x1==pinhole.x1);
 // Far corner and hill vehicles stay in the region of interest; near background does not.
 CHECK(in_processing_roi(100,40,0,12,.58,.1));CHECK(in_processing_roi(100,0,12,12,.58,.1));
 CHECK(!in_processing_roi(10,14,0,12,.58,.1));CHECK(!in_processing_roi(10,0,6,12,.58,.1));CHECK(!in_processing_roi(10,0,-5,12,.58,.1));
 std::cout<<"Lens distortion and forward region of interest: PASS\n";
}
