#include "a1_fusion/raceline.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
namespace a1::inspection {
namespace {
std::vector<std::string> split(const std::string& line){
  std::vector<std::string> out;std::stringstream stream(line);std::string item;
  while(std::getline(stream,item,','))out.push_back(item);
  return out;
}
int column(const std::vector<std::string>& header,const char* name){
  const auto it=std::find(header.begin(),header.end(),name);
  if(it==header.end())throw std::runtime_error(std::string("raceline column missing: ")+name);
  return int(it-header.begin());
}
}
Raceline Raceline::load(const std::string& path,bool closed_loop){
  std::ifstream file(path);
  if(!file)throw std::runtime_error("cannot open raceline: "+path);
  std::string line;
  if(!std::getline(file,line))throw std::runtime_error("empty raceline: "+path);
  const auto header=split(line);
  const int s=column(header,"s_m"),x=column(header,"x_m"),y=column(header,"y_m");
  const int needed=std::max({s,x,y});
  Raceline out;out.closed_=closed_loop;
  while(std::getline(file,line)){
    if(line.empty())continue;
    const auto row=split(line);
    if(int(row.size())<=needed)throw std::runtime_error("short raceline row");
    Point p{};
    try{p={std::stod(row[s]),std::stod(row[x]),std::stod(row[y])};}
    catch(const std::exception&){throw std::runtime_error("invalid raceline number");}
    if(!std::isfinite(p.s)||!std::isfinite(p.x)||!std::isfinite(p.y))throw std::runtime_error("non-finite raceline row");
    if(!out.points_.empty()&&p.s<=out.points_.back().s)throw std::runtime_error("raceline s must increase");
    out.points_.push_back(p);
  }
  if(out.points_.size()<2)throw std::runtime_error("raceline needs at least two points");
  const auto& first=out.points_.front();const auto& last=out.points_.back();
  out.length_=last.s-first.s+(closed_loop?std::hypot(first.x-last.x,first.y-last.y):0.);
  return out;
}
Raceline::Frenet Raceline::project(const Eigen::Vector2d& q,int& hint,int window) const {
  const int count=int(points_.size()),segments=closed_?count:count-1;
  const bool global=hint<0||hint>=count;
  const int span=global?segments:std::min(segments,window);
  const int start=global?0:hint-span/2;
  double best=1e300;Frenet out;int best_index=0;
  for(int offset=0;offset<span;++offset){
    int k=start+offset;
    if(closed_)k=((k%count)+count)%count;
    else if(k<0||k>=segments)continue;
    const Point& a=points_[k];const Point& b=points_[(k+1)%count];
    const double dx=b.x-a.x,dy=b.y-a.y,length=std::hypot(dx,dy);
    if(length<1e-9)continue;
    const double tx=dx/length,ty=dy/length,along=std::clamp((q.x()-a.x)*tx+(q.y()-a.y)*ty,0.,length);
    const double ex=q.x()-a.x-along*tx,ey=q.y()-a.y-along*ty,distance=ex*ex+ey*ey;
    if(distance<best){
      best=distance;best_index=k;
      out.s=a.s-points_.front().s+along;out.n=tx*(q.y()-a.y)-ty*(q.x()-a.x);out.yaw=std::atan2(ty,tx);
    }
  }
  hint=best_index;
  return out;
}
double Raceline::ahead(double from,double to) const {
  const double difference=to-from;
  return closed_?std::remainder(difference,length_):difference;
}
bool same_lane(double target_n,double ego_n,bool retained){
  return std::abs(target_n-ego_n)<=(retained?2.5:2.0); // Original replanner lane_half_width = 2.0.
}
void AlignmentMonitor::add(double stamp,double ego_abs_n){
  samples_.emplace_back(stamp,ego_abs_n);
  while(stamp-samples_.front().first>5.)samples_.pop_front();
}
bool AlignmentMonitor::misaligned() const {
  if(samples_.size()<2||samples_.back().first-samples_.front().first<4.5)return false;
  std::vector<double> values;values.reserve(samples_.size());
  for(const auto& s:samples_)values.push_back(s.second);
  std::nth_element(values.begin(),values.begin()+values.size()/2,values.end());
  return values[values.size()/2]>5.;
}
}
