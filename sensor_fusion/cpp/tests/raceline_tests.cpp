#include "a1_fusion/map_structure.hpp"
#include "a1_fusion/raceline.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#define CHECK(x) do{if(!(x))throw std::runtime_error(std::string(#x)+" line "+std::to_string(__LINE__));}while(false)
using namespace a1::inspection;
int main(int argc,char** argv){
  const std::string path=(argc>1?std::string(argv[1])+"/":std::string())+"raceline_test.csv";
  const double pi=std::acos(-1.),radius=100.;
  {
    std::ofstream f(path);f<<"s_m,x_m,y_m,yaw_rad,kappa_radpm,v_mps,ax_mps2\n";
    for(int i=0;i<628;++i){const double a=i*.01;
      f<<radius*a<<','<<radius*std::cos(a)<<','<<radius*std::sin(a)<<','<<a+pi/2<<",0.01,20,0\n";}
  }
  auto loop=Raceline::load(path,true);
  CHECK(std::abs(loop.length()-2*pi*radius)<1.);
  int hint=-1;
  auto on=loop.project({100,0},hint);CHECK(std::abs(on.s)<1e-6);CHECK(std::abs(on.n)<1e-6);CHECK(std::abs(on.yaw-pi/2)<.02);CHECK(hint>=0);
  int inside=-1;auto left=loop.project({95*std::cos(1.),95*std::sin(1.)},inside);CHECK(std::abs(left.n-5)<.05);CHECK(std::abs(left.s-100)<.5);
  int outside=-1;CHECK(std::abs(loop.project({105*std::cos(1.),105*std::sin(1.)},outside).n+5)<.05);
  // The hint stays near the last position.
  int walk=-1;for(int i=0;i<300;++i){const double a=i*.01;auto q=loop.project({radius*std::cos(a),radius*std::sin(a)},walk);CHECK(std::abs(q.s-radius*a)<1.);}
  // Distance ahead wraps the start line and reports targets behind as negative.
  CHECK(std::abs(loop.ahead(620,10)-(loop.length()-620+10))<1e-9);
  CHECK(std::abs(loop.ahead(10,620)+(loop.length()-620+10))<1e-9);
  CHECK(std::abs(loop.ahead(10,50)-40)<1e-9);
  auto open=Raceline::load(path,false);CHECK(open.ahead(620,10)==-610);CHECK(open.length()<loop.length());
  for(const std::string bad:{"s_m,x_m\n1,2\n2,3\n","s_m,x_m,y_m,yaw_rad\n0,0,0,0\n","s_m,x_m,y_m,yaw_rad\n0,0,0,0\n1,nan,0,0\n","s_m,x_m,y_m,yaw_rad\n1,0,0,0\n0,1,0,0\n"}){
    {std::ofstream f(path);f<<bad;}
    bool rejected=false;try{Raceline::load(path,true);}catch(const std::runtime_error&){rejected=true;}CHECK(rejected);
  }
  bool missing=false;try{Raceline::load(path+".missing",true);}catch(const std::runtime_error&){missing=true;}CHECK(missing);
  std::remove(path.c_str());
  // Lane test uses the ego's own offset: the ego is not always on the racing line.
  CHECK(same_lane(3,3.5,false));CHECK(!same_lane(0,3,false));CHECK(same_lane(0,2.4,true));CHECK(!same_lane(0,2.4,false));
  AlignmentMonitor monitor;
  for(int i=0;i<50;++i)monitor.add(1+i*.1,1.);CHECK(!monitor.misaligned());
  for(int i=0;i<100;++i)monitor.add(6+i*.1,20.);CHECK(monitor.misaligned());
  for(int i=0;i<100;++i)monitor.add(16+i*.1,1.);CHECK(!monitor.misaligned());
  AlignmentMonitor early;early.add(1,20.);CHECK(!early.misaligned()); // Needs a filled window.
  std::cout<<"Raceline: CSV, Frenet, hint, closed loop, invalid input, same lane, alignment: PASS\n";

  // Structure label: repeated static returns stay, a passing car does not.
  std::vector<StructurePoint> points;
  for(int k=0;k<5;++k)for(double x=0;x<6;x+=.2){points.push_back({float(x),0.f,0.f,k});
    for(double z:{.4,.8})points.push_back({float(x),.2f,float(z),k});} // Floor plus an 0.8 m rail.
  for(int k=0;k<5;++k)for(double z:{.4,.8,1.2})points.push_back({float(k*2.f+1.f),3.f,float(z),k}); // Moving car.
  for(int k=0;k<5;++k)points.push_back({float(.2),.2f,6.f,k});                                      // Overhead.
  points.push_back({100.f,100.f,0.f,0});
  auto kept=structure_voxels(points,3);
  CHECK(!kept.empty());
  bool rail=false,car=false,floor=false,overhead=false;
  for(const auto& q:kept){rail=rail||(std::abs(q.y()-.2)<.15&&q.z()>.3&&q.z()<.9);car=car||std::abs(q.y()-3)<.15;floor=floor||q.z()<.2;overhead=overhead||q.z()>4;}
  CHECK(rail);CHECK(!car);CHECK(!floor);CHECK(!overhead);
  CHECK(structure_voxels({},3).empty());
  bool invalid=false;try{structure_voxels(points,0);}catch(const std::invalid_argument&){invalid=true;}CHECK(invalid);
  std::cout<<"Map structure voxels: rail kept, moving car, floor and overhead dropped: PASS\n";
}
