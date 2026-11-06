#include "a1_fusion/association.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
namespace a1::inspection {
Eigen::Matrix3d body_from_lidar(double yaw){
 return Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitZ()).toRotationMatrix();
}
Eigen::Matrix3d nominal_optical_from_body(){
 Eigen::Matrix3d r;r<<0,-1,0,0,0,-1,1,0,0;return r;
}
bool in_front_half(const Eigen::Vector3d& q,double center){return q.allFinite()&&std::isfinite(center)&&(std::cos(center)*q.x()+std::sin(center)*q.y())>=-1e-9;}


bool in_target_range(const Box& b,double minimum,double maximum){
 const auto c=b.center();const double range=c.head<2>().norm();
 return c.allFinite()&&c.x()>0&&std::isfinite(minimum)&&std::isfinite(maximum)&&minimum>=0&&maximum>=minimum&&range>=minimum&&range<=maximum;
}
std::vector<BinRegion> camera_regions(const std::array<double,40>& ps,
 const std::vector<BinRegion>& previous,double enter,double hold,double edge,int lower,int upper){
 if(lower<0||upper>39||lower>upper||!std::isfinite(enter)||!std::isfinite(hold)||!std::isfinite(edge)||edge<0||edge>hold||hold>enter||enter>1)throw std::invalid_argument("invalid camera region thresholds/range");
 std::vector<BinRegion> out;
 for(int j=lower;j<=upper;){
  if(!std::isfinite(ps[j])||ps[j]<edge){++j;continue;}
  BinRegion b{j,j,ps[j]};
  while(j+1<=upper&&std::isfinite(ps[j+1])&&ps[j+1]>=edge){++j;b.right=j;b.peak=std::max(b.peak,ps[j]);}
  bool retained=false;for(const auto& old:previous)if(b.left<=old.right&&b.right>=old.left)retained=true;
  if(b.peak>=(retained?hold:enter))out.push_back(b);
  ++j;
 }
 return out;
}
Footprint measured_footprint(const std::vector<Eigen::Vector3d>& points){
 // Monotone-chain convex hull, then evaluate every hull-edge orientation.
 // Use real returns, never the corners of the axis-aligned bounding box.
 std::vector<Eigen::Vector2d> xy;xy.reserve(points.size());
 for(const auto& q:points){if(!q.allFinite())return {};xy.push_back(q.head<2>());}
 std::sort(xy.begin(),xy.end(),[](const auto& a,const auto& b){return a.x()==b.x()?a.y()<b.y():a.x()<b.x();});
 xy.erase(std::unique(xy.begin(),xy.end(),[](const auto& a,const auto& b){return a.x()==b.x()&&a.y()==b.y();}),xy.end());
 if(xy.size()<2)return {};
 auto cross=[](const auto& a,const auto& b,const auto& c){Eigen::Vector2d u=b-a,v=c-a;return u.x()*v.y()-u.y()*v.x();};
 std::vector<Eigen::Vector2d> hull;
 for(const auto& q:xy){while(hull.size()>=2&&cross(hull[hull.size()-2],hull.back(),q)<=0)hull.pop_back();hull.push_back(q);}
 const size_t lower=hull.size();
 for(size_t k=xy.size()-1;k>0;--k){const auto& q=xy[k-1];while(hull.size()>lower&&cross(hull[hull.size()-2],hull.back(),q)<=0)hull.pop_back();hull.push_back(q);}
 hull.pop_back();
 Footprint best;double area=std::numeric_limits<double>::infinity();
 for(size_t i=0;i<hull.size();++i){
  Eigen::Vector2d u=(hull[(i+1)%hull.size()]-hull[i]).normalized(),v(-u.y(),u.x());
  double x0=0,x1=0,y0=0,y1=0;
  for(const auto& q:hull){Eigen::Vector2d d=q-hull[i];double x=d.dot(u),y=d.dot(v);x0=std::min(x0,x);x1=std::max(x1,x);y0=std::min(y0,y);y1=std::max(y1,y);}
  double a=x1-x0,b=y1-y0;
  if(a*b<area){area=a*b;double yaw=std::atan2(u.y(),u.x());if(b>a)yaw+=std::acos(-1.)*.5;
   best={std::max(a,b),std::min(a,b),std::remainder(yaw,std::acos(-1.)),true};}
 }
 return best;
}

bool vehicle_extent(const Box& b){
 const Eigen::Vector3d d=b.hi-b.lo;
 if(!d.allFinite()||d.minCoeff()<0||d.z()>2.3)return false;
 if(!b.footprint.valid)return d.x()<=5.2&&d.y()<=2.6;
 const auto& f=b.footprint;
 return std::isfinite(f.length)&&std::isfinite(f.width)&&std::isfinite(f.yaw)&&
   f.width>=0&&f.length>=f.width&&f.length<=5.2&&f.width<=2.6;
}
bool full_vehicle_geometry(const Box& b){
 const auto d=b.hi-b.lo;return vehicle_extent(b)&&b.points>=5&&d.x()>=.6&&d.y()>=.5&&d.z()>=.5;
}
bool partial_vehicle_geometry(const Box& b){
 const auto d=b.hi-b.lo;return vehicle_extent(b)&&b.points>=5&&d.head<2>().maxCoeff()>=.8&&d.z()>=.15;
}
bool camera_surface_geometry(const Box& b,bool retained_target){
 const auto d=b.hi-b.lo;
 // A visible rear/side surface may have almost zero depth. Camera support is
 // mandatory in the caller; this does not classify LiDAR-only guardrails as cars.
 return vehicle_extent(b)&&b.points>=5&&
   ((d.head<2>().maxCoeff()>=.6&&d.z()>=.3)||(retained_target&&partial_vehicle_geometry(b)));
}
std::vector<std::vector<int>> adaptive_clusters(const std::vector<Eigen::Vector3d>& points,const ClusterConfig& c){
 if(!std::isfinite(c.near_radius)||!std::isfinite(c.start_range)||!std::isfinite(c.slope)||!std::isfinite(c.max_radius)||!std::isfinite(c.max_height_gap)||c.near_radius<=0||c.start_range<0||c.slope<0||c.max_radius<c.near_radius||c.max_height_gap<=0||c.min_points<1)throw std::invalid_argument("invalid adaptive cluster configuration");
 using Cell=std::pair<int,int>;std::map<Cell,std::vector<int>> cells;
 std::vector<double> radii(points.size());std::vector<bool> seen(points.size());
 auto cell=[&](const Eigen::Vector3d& q){return Cell{int(std::floor(q.x()/c.max_radius)),int(std::floor(q.y()/c.max_radius))};};
 for(size_t i=0;i<points.size();++i){if(!points[i].allFinite()){seen[i]=true;continue;}cells[cell(points[i])].push_back(i);radii[i]=std::min(c.max_radius,c.near_radius+c.slope*std::max(0.,points[i].head<2>().norm()-c.start_range));}
 std::vector<std::vector<int>> out;
 for(size_t seed=0;seed<points.size();++seed){if(seen[seed])continue;std::vector<int> group{int(seed)};seen[seed]=true;
  for(size_t k=0;k<group.size();++k){int i=group[k];auto ij=cell(points[i]);
   for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy){auto it=cells.find({ij.first+dx,ij.second+dy});if(it==cells.end())continue;
    for(int j:it->second){if(seen[j])continue;const auto delta=points[i]-points[j];double r=std::min(radii[i],radii[j]);
     if(std::abs(delta.z())<=c.max_height_gap&&delta.head<2>().squaredNorm()<=r*r){seen[j]=true;group.push_back(j);}
    }
   }
  }
  if(group.size()>=size_t(c.min_points)&&group.size()<=30000)out.push_back(std::move(group));
 }
 return out;
}
std::vector<std::vector<int>> merge_fragments(const std::vector<Box>& boxes,double gap){
 std::vector<Box> bs=boxes;std::vector<std::vector<int>> groups;
 for(size_t i=0;i<bs.size();++i)groups.push_back({int(i)});
 // Only attach fragments, never combine two substantial vehicle-sized observations.
 for(;;){double best=gap;int ai=-1,bi=-1;
  for(size_t i=0;i<bs.size();++i)if(!groups[i].empty())
   for(size_t j=i+1;j<bs.size();++j)if(!groups[j].empty()){
    Eigen::Vector3d di=bs[i].hi-bs[i].lo,dj=bs[j].hi-bs[j].lo;
    if(di.x()*di.y()>=1.5&&dj.x()*dj.y()>=1.5)continue;
    Box joint{bs[i].lo.cwiseMin(bs[j].lo),bs[i].hi.cwiseMax(bs[j].hi),bs[i].points+bs[j].points};
    if(!vehicle_extent(joint))continue;
    if(std::min(bs[i].hi.z(),bs[j].hi.z())-std::max(bs[i].lo.z(),bs[j].lo.z())<.2)continue;
    Eigen::Vector3d separation=(bs[i].lo-bs[j].hi).cwiseMax(bs[j].lo-bs[i].hi).cwiseMax(0.);
    double distance=separation.head<2>().norm();
    if(distance<best){best=distance;ai=i;bi=j;}
   }
  if(ai<0)break;
  bs[ai].lo=bs[ai].lo.cwiseMin(bs[bi].lo);bs[ai].hi=bs[ai].hi.cwiseMax(bs[bi].hi);bs[ai].points+=bs[bi].points;
  groups[ai].insert(groups[ai].end(),groups[bi].begin(),groups[bi].end());groups[bi].clear();
 }
 groups.erase(std::remove_if(groups.begin(),groups.end(),[](auto& g){return g.empty();}),groups.end());
 return groups;
}

std::vector<int> hungarian(const Eigen::MatrixXd& a) {
  const int n=a.rows(),m=a.cols(); if(n>m)throw std::invalid_argument("assignment needs rows <= columns");
  std::vector<double> u(n+1),v(m+1);std::vector<int> p(m+1),way(m+1),out(n,-1);
  for(int i=1;i<=n;++i){p[0]=i;int j0=0;std::vector<double> minv(m+1,1e100);std::vector<bool> used(m+1);
    do {used[j0]=true;int i0=p[j0],j1=0;double delta=1e100;
      for(int j=1;j<=m;++j)if(!used[j]){double cur=a(i0-1,j-1)-u[i0]-v[j];if(cur<minv[j]){minv[j]=cur;way[j]=j0;}if(minv[j]<delta){delta=minv[j];j1=j;}}
      for(int j=0;j<=m;++j){if(used[j]){u[p[j]]+=delta;v[j]-=delta;}else minv[j]-=delta;}
      j0=j1;
    }while(p[j0]!=0);
    do{int j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0);
  }
  for(int j=1;j<=m;++j){if(p[j])out[p[j]-1]=j-1;}
  return out;
}
Projection project(const Box& b,const Eigen::Matrix3d& R,const Eigen::Vector3d& camera,
 double fx,double fy,double cx,double cy,int width,int height,const Distortion& lens){
 Projection p;p.x0=p.y0=1e100;p.x1=p.y1=-1e100;
 for(int k=0;k<8;++k){Eigen::Vector3d q(k&1?b.hi.x():b.lo.x(),k&2?b.hi.y():b.lo.y(),k&4?b.hi.z():b.lo.z());q=R*(q-camera);
  if(!q.allFinite()||q.z()<=.1)return {};
  const double x=q.x()/q.z(),y=q.y()/q.z(),r2=x*x+y*y,radial=1+lens.k1*r2+lens.k2*r2*r2;
  const double xd=x*radial+2*lens.p1*x*y+lens.p2*(r2+2*x*x),yd=y*radial+lens.p1*(r2+2*y*y)+2*lens.p2*x*y;
  double u=fx*xd+cx,v=fy*yd+cy;p.x0=std::min(p.x0,u);p.x1=std::max(p.x1,u);p.y0=std::min(p.y0,v);p.y1=std::max(p.y1,v);
 }
 if(p.x1<0||p.x0>=width||p.y1<0||p.y0>=height)return {};
 p.x0=std::clamp(p.x0,0.0,double(width-1));p.x1=std::clamp(p.x1,0.0,double(width-1));
 p.y0=std::clamp(p.y0,0.0,double(height-1));p.y1=std::clamp(p.y1,0.0,double(height-1));p.valid=true;return p;
}
bool in_processing_roi(double x,double y,double z,double half_width,double half_angle_tangent,double grade_margin){
 const double margin=grade_margin*std::max(0.,x);
 return std::abs(y)<std::max(half_width,half_angle_tangent*x)&&z>-3-margin&&z<4+margin;
}
}
