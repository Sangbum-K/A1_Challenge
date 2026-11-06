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
std::vector<int> unique_region_matches(const Eigen::MatrixXd& overlaps,double threshold){
 if(!std::isfinite(threshold)||threshold<=0||threshold>1)throw std::invalid_argument("invalid IoU threshold");
 std::vector<int> out(overlaps.rows(),-1);
 for(int i=0;i<overlaps.rows();++i){int region=-1,n=0;
  for(int j=0;j<overlaps.cols();++j)if(std::isfinite(overlaps(i,j))&&overlaps(i,j)>=threshold){region=j;++n;}
  if(n!=1)continue;
  int competitors=0;for(int k=0;k<overlaps.rows();++k)if(std::isfinite(overlaps(k,region))&&overlaps(k,region)>=threshold)++competitors;
  if(competitors==1)out[i]=region;
 }
 return out;
}

bool vehicle_extent(const Box& b){
 const Eigen::Vector3d d=b.hi-b.lo;
 return d.allFinite()&&d.minCoeff()>=0&&d.x()<=5.2&&d.y()<=2.6&&d.z()<=2.3;
}
bool full_vehicle_geometry(const Box& b){
 const auto d=b.hi-b.lo;return vehicle_extent(b)&&b.points>=5&&d.x()>=.6&&d.y()>=.5&&d.z()>=.5;
}
bool partial_vehicle_geometry(const Box& b){
 const auto d=b.hi-b.lo;return vehicle_extent(b)&&b.points>=5&&d.head<2>().maxCoeff()>=.8&&d.z()>=.15;
}
const char* vehicle_geometry_status(const Track& t){
 if(!vehicle_extent(t.raw_box))return "extent_exceeded";
 if(full_vehicle_geometry(t.raw_box))return "full_geometry";
 if(t.vehicle_evidence&&partial_vehicle_geometry(t.raw_box))return "retained_partial";
 const auto d=t.raw_box.hi-t.raw_box.lo;
 if(t.hits>=6&&t.raw_box.center().head<2>().norm()>=20&&partial_vehicle_geometry(t.raw_box)&&d.z()>=.3)return "far_partial";
 return "insufficient_geometry";
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
bool Tracker::usable(const Track& t,double stamp)const{
 const double age=stamp-t.last_seen;
 return t.hits>=3&&std::isfinite(stamp)&&age>=-1e-9&&age<=config_.coast_s+1e-9&&t.state.allFinite()&&t.covariance.allFinite()&&t.covariance(0,0)<=config_.max_position_variance&&t.covariance(1,1)<=config_.max_position_variance;
}
int64_t select_primary(const std::vector<const Track*>& eligible,int64_t current){
 for(const auto* t:eligible)if(t->id==current&&!t->measured)return current;
 const Track* nearest=nullptr;
 for(const auto* t:eligible)if(t->measured&&(!nearest||t->box.center().head<2>().squaredNorm()<nearest->box.center().head<2>().squaredNorm()||(t->box.center().head<2>().squaredNorm()==nearest->box.center().head<2>().squaredNorm()&&t->id<nearest->id)))nearest=t;
 return nearest?nearest->id:-1;
}
const std::vector<Track>& Tracker::update(double stamp,const std::vector<Box>& ds){
  if(!std::isfinite(stamp)||stamp<=last_|| (last_>0&&stamp-last_>config_.memory_s)){reset();}
  if(!std::isfinite(stamp))return tracks_;
  const double dt=last_<0?.1:stamp-last_;last_=stamp;
  tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[&](auto& t){return stamp-t.last_seen>config_.memory_s;}),tracks_.end());
  Eigen::Matrix4d F=Eigen::Matrix4d::Identity();F(0,2)=F(1,3)=dt;
  Eigen::Matrix<double,4,2> G;G<<dt*dt/2,0,0,dt*dt/2,dt,0,0,dt;
  Eigen::Matrix<double,2,4> H=Eigen::Matrix<double,2,4>::Zero();H(0,0)=H(1,1)=1;
  const Eigen::Matrix2d R=Eigen::Matrix2d::Identity()*.09;
  const int nt=tracks_.size(),nd=ds.size();Eigen::MatrixXd costs=Eigen::MatrixXd::Constant(nt,nd+nt,1.0);
  for(int i=0;i<nt;++i){auto& t=tracks_[i];t.state=F*t.state;t.covariance=F*t.covariance*F.transpose()+4*G*G.transpose();t.measured=false;
    // The predicted box and state must share the same timestamp, even without a measurement.
    const Eigen::Vector3d extent=t.box.hi-t.box.lo;
    const Eigen::Vector3d center(t.state[0],t.state[1],t.box.center().z());
    t.box.lo=center-extent*.5;t.box.hi=center+extent*.5;
    const Eigen::Matrix2d S=H*t.covariance*H.transpose()+R;
    for(int j=0;j<nd;++j){Eigen::Vector2d d=ds[j].center().head<2>()-t.state.head<2>();
      double mahal=d.dot(S.ldlt().solve(d));double size_error=((ds[j].hi-ds[j].lo).head<2>()-(t.box.hi-t.box.lo).head<2>()).norm();
      costs(i,j)=(mahal<=9.21&&d.norm()<=3&&size_error<=3)? .7*mahal/9.21+.3*std::min(1.0,size_error/3):1e6;
    }
  }
  auto assignments=hungarian(costs);std::vector<bool> consumed(nd),blocked_birth(nd);
  // A plausible observation of a confirmed live track is not immediately a new object.
  // In particular, rejected ambiguous assignments must not spawn duplicate IDs.
  for(int j=0;j<nd;++j)for(int i=0;i<nt;++i)if(tracks_[i].hits>=3&&costs(i,j)<.95)blocked_birth[j]=true;
  for(int i=0;i<nt;++i){auto& t=tracks_[i];int j=assignments[i];
    if(j<0||j>=nd||costs(i,j)>=.95){++t.misses;continue;}
    // Near-identical competing clusters are ambiguous: don't force an ID switch.
    double other=1e6;for(int k=0;k<nd;++k)if(k!=j)other=std::min(other,costs(i,k));
    if(other-costs(i,j)<.03){++t.misses;continue;}
    // Changed visible surfaces make the centroid less reliable, independently per axis.
    const Eigen::Vector2d shape_delta=(ds[j].hi-ds[j].lo).head<2>()-(t.raw_box.hi-t.raw_box.lo).head<2>();
    Eigen::Matrix2d measurement_R=R;
    for(int axis=0;axis<2;++axis)measurement_R(axis,axis)+=std::min(1.,.25*shape_delta[axis]*shape_delta[axis]);
    Eigen::Matrix2d S=H*t.covariance*H.transpose()+measurement_R;Eigen::Matrix<double,4,2> K=t.covariance*H.transpose()*S.inverse();
    t.state+=K*(ds[j].center().head<2>()-H*t.state);
    const Eigen::Matrix4d A=Eigen::Matrix4d::Identity()-K*H;
    t.covariance=A*t.covariance*A.transpose()+K*measurement_R*K.transpose();
    t.raw_box=ds[j];
    t.vehicle_evidence=t.vehicle_evidence||full_vehicle_geometry(ds[j]);
    const double alpha=1-std::exp(-dt/.25);
    Eigen::Vector3d extent=(1-alpha)*(t.box.hi-t.box.lo)+alpha*(ds[j].hi-ds[j].lo);
    Eigen::Vector3d center(t.state[0],t.state[1],(1-alpha)*t.box.center().z()+alpha*ds[j].center().z());
    t.box={center-extent*.5,center+extent*.5,ds[j].points};t.last_seen=stamp;t.measured=true;t.misses=0;++t.hits;consumed[j]=true;
  }
  for(int j=0;j<nd;++j)if(!consumed[j]&&!blocked_birth[j]){Track t;t.id=next_++;t.box=t.raw_box=ds[j];
    t.vehicle_evidence=t.vehicle_evidence||full_vehicle_geometry(ds[j]);t.state.head<2>()=ds[j].center().head<2>();t.covariance.diagonal()<<.09,.09,25,25;t.hits=1;t.vehicle_evidence=full_vehicle_geometry(ds[j]);t.measured=true;t.last_seen=stamp;tracks_.push_back(t);}
  return tracks_;
}
Projection project(const Box& b,const Eigen::Matrix3d& R,const Eigen::Vector3d& camera,
 double fx,double fy,double cx,double cy,int width,int height){
 Projection p;p.x0=p.y0=1e100;p.x1=p.y1=-1e100;
 for(int k=0;k<8;++k){Eigen::Vector3d q(k&1?b.hi.x():b.lo.x(),k&2?b.hi.y():b.lo.y(),k&4?b.hi.z():b.lo.z());q=R*(q-camera);
  if(!q.allFinite()||q.z()<=.1)return {};
  double u=fx*q.x()/q.z()+cx,v=fy*q.y()/q.z()+cy;p.x0=std::min(p.x0,u);p.x1=std::max(p.x1,u);p.y0=std::min(p.y0,v);p.y1=std::max(p.y1,v);
 }
 if(p.x1<0||p.x0>=width||p.y1<0||p.y0>=height)return {};
 p.x0=std::clamp(p.x0,0.0,double(width-1));p.x1=std::clamp(p.x1,0.0,double(width-1));
 p.y0=std::clamp(p.y0,0.0,double(height-1));p.y1=std::clamp(p.y1,0.0,double(height-1));p.valid=true;return p;
}
double horizontal_iou(const Projection& p,double left,double right){if(!p.valid||right<=left)return 0;double i=std::max(0.0,std::min(p.x1,right)-std::max(p.x0,left));return i/std::max(1e-9,std::max(p.x1,right)-std::min(p.x0,left));}
}
