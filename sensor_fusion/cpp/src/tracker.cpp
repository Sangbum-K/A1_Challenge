#include "a1_fusion/tracker.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace a1::inspection {
namespace {
using V2=Eigen::Vector2d;
const double kPi=std::acos(-1.);
double wrap(double angle){return std::remainder(angle,2*kPi);}
std::vector<V2> xy_of(const Scan& s,const std::vector<int>& ids){
  std::vector<V2> out;out.reserve(ids.size());
  for(int i:ids)out.push_back(s.world_points[i].head<2>());
  return out;
}
int count_bits(unsigned value){int n=0;for(;value;value&=value-1)++n;return n;}
Eigen::Matrix2d rotated_noise(double heading,double along,double lateral){
  Eigen::Matrix2d rotation;rotation<<std::cos(heading),-std::sin(heading),std::sin(heading),std::cos(heading);
  Eigen::Matrix2d noise=Eigen::Matrix2d::Zero();noise(0,0)=along*along;noise(1,1)=lateral*lateral;
  return rotation*noise*rotation.transpose();
}
// Rear corners, then front corners, in the continuous frame.
std::array<V2,4> corners(const Track& t){
  const V2 u(std::cos(t.heading()),std::sin(t.heading())),v(-u.y(),u.x());
  const V2 rear=t.anchor(),front=rear+u*t.length;
  return {rear+v*(t.width/2),rear-v*(t.width/2),front+v*(t.width/2),front-v*(t.width/2)};
}
Box body_box(const Track& t,const Pose2& ego,int corner_count){
  const auto all=corners(t);
  Box b{Eigen::Vector3d::Constant(1e100),Eigen::Vector3d::Constant(-1e100),0,{}};
  for(int i=0;i<corner_count;++i){const V2 q=ego.to_body(all[i]);
    b.lo.x()=std::min(b.lo.x(),q.x());b.lo.y()=std::min(b.lo.y(),q.y());
    b.hi.x()=std::max(b.hi.x(),q.x());b.hi.y()=std::max(b.hi.y(),q.y());}
  b.lo.z()=t.z-t.height/2;b.hi.z()=t.z+t.height/2;
  return b;
}
}
V2 Pose2::to_world(const V2& q) const {
  const double c=std::cos(yaw),s=std::sin(yaw);
  return V2(c*q.x()-s*q.y()+x,s*q.x()+c*q.y()+y);
}
V2 Pose2::to_body(const V2& q) const {
  const double c=std::cos(yaw),s=std::sin(yaw),dx=q.x()-x,dy=q.y()-y;
  return V2(c*dx+s*dy,-s*dx+c*dy);
}
bool EgoLink::update(double stamp,bool available){
  if(!available){
    available_since_=-1;
    const bool changed=compensated_;compensated_=false;return changed;
  }
  if(compensated_)return false;
  if(available_since_<0)available_since_=stamp;
  if(stamp-available_since_<1.)return false;
  compensated_=true;return true;
}
int ScanTiming::cell(double azimuth){
  const double degrees=azimuth*180/kPi;
  return ((int(std::floor(degrees))%360)+360)%360;
}
void ScanTiming::add(double azimuth,double time){
  if(!std::isfinite(azimuth)||!std::isfinite(time))return;
  const int c=cell(azimuth);sum_[c]+=time;++count_[c];
  first_=std::min(first_,time);last_=std::max(last_,time);
}
void ScanTiming::finish(double fallback){
  fallback_=fallback;valid_=last_-first_>=.005;
  if(!valid_)return;
  for(int c=0;c<360;++c){
    int gap=0;
    while(gap<180&&!count_[(c+gap)%360]&&!count_[(c-gap+360)%360])++gap;
    const int source=count_[(c+gap)%360]?(c+gap)%360:(c-gap+360)%360;
    time_[c]=sum_[source]/count_[source];
  }
}
double ScanTiming::at(double azimuth) const {return valid_?time_[cell(azimuth)]:fallback_;}
Quality classify(const Box& b,double range,double weak_min_range){
  if(full_vehicle_geometry(b)||camera_surface_geometry(b))return Quality::Strong;
  const auto d=b.hi-b.lo;
  const bool plausible=b.points>=3&&d.head<2>().maxCoeff()>=.3&&vehicle_extent(b);
  return plausible&&range>=weak_min_range?Quality::Weak:Quality::Reject;
}
AnchorFit rear_anchor(const std::vector<V2>& xy,double heading,const V2& sensor,double width_prior){
  AnchorFit fit;
  if(xy.empty())return fit;
  const V2 u(std::cos(heading),std::sin(heading)),v(-u.y(),u.x());
  std::vector<double> along;along.reserve(xy.size());
  V2 centroid=V2::Zero();
  for(const auto& p:xy){along.push_back(p.dot(u));centroid+=p;}
  centroid/=double(xy.size());
  std::sort(along.begin(),along.end());
  const double rear=xy.size()<20?along.front():along[along.size()/20];
  double lo=1e100,hi=-1e100;
  for(const auto& p:xy)if(p.dot(u)<rear+.5){const double b=p.dot(v);lo=std::min(lo,b);hi=std::max(hi,b);}
  fit.width=hi-lo;fit.length=along.back()-rear;
  const double range=(centroid-sensor).norm();
  double middle=(lo+hi)/2;
  const bool wide=fit.width>=1.2;
  if(!wide){
    const double side=sensor.dot(v);
    if(side<lo)middle=lo+width_prior/2;else if(side>hi)middle=hi-width_prior/2;
  }
  fit.anchor=u*rear+v*middle;
  fit.rear_visible=range<1e-6||(centroid-sensor).dot(u)/range>=.5;
  fit.sigma_along=fit.rear_visible?.25:3.;
  fit.sigma_lateral=wide?std::max(.15,.003*range):.6;
  return fit;
}
Track::State ctrv_predict(const Track::State& s,double dt){
  Track::State out=s;
  const double v=s[2],h=s[3],w=s[4];
  if(std::abs(w)>=1e-3){
    out[0]+=v/w*(std::sin(h+w*dt)-std::sin(h));
    out[1]+=v/w*(std::cos(h)-std::cos(h+w*dt));
  }else{
    out[0]+=v*std::cos(h)*dt;out[1]+=v*std::sin(h)*dt;
  }
  out[3]=h+w*dt;
  return out;
}
Track::Cov ctrv_jacobian(const Track::State& s,double dt){
  Track::Cov F=Track::Cov::Identity();
  const double v=s[2],h=s[3],w=s[4],s0=std::sin(h),c0=std::cos(h);
  if(std::abs(w)>=1e-3){
    const double s1=std::sin(h+w*dt),c1=std::cos(h+w*dt);
    F(0,2)=(s1-s0)/w;F(0,3)=v*(c1-c0)/w;F(0,4)=v*(dt*c1/w-(s1-s0)/(w*w));
    F(1,2)=(c0-c1)/w;F(1,3)=v*(s1-s0)/w;F(1,4)=v*(dt*s1/w-(c0-c1)/(w*w));
  }else{
    F(0,2)=c0*dt;F(0,3)=-v*s0*dt;F(0,4)=-.5*v*s0*dt*dt;
    F(1,2)=s0*dt;F(1,3)=v*c0*dt;F(1,4)=.5*v*c0*dt*dt;
  }
  F(3,4)=dt;
  return F;
}
Detection make_detection(const Scan& s,const std::vector<int>& ids,double weak_min_range){
  Detection d;d.points=ids;
  if(ids.empty())return d;
  Box b{Eigen::Vector3d::Constant(1e100),Eigen::Vector3d::Constant(-1e100),int(ids.size()),{}};
  std::vector<Eigen::Vector3d> points;points.reserve(ids.size());
  double time=0;
  for(int i:ids){const auto& q=s.body_points[i];points.push_back(q);b.lo=b.lo.cwiseMin(q);b.hi=b.hi.cwiseMax(q);time+=s.point_time[i];}
  b.footprint=measured_footprint(points);
  d.body_box=b;d.time=time/double(ids.size());
  d.quality=classify(b,b.center().head<2>().norm(),weak_min_range);
  return d;
}
void Tracker::propagate(Track::State& x,Track::Cov& P,double dt) const {
  if(dt==0)return;
  const Track::Cov F=ctrv_jacobian(x,dt);
  const double c=std::cos(x[3]),s=std::sin(x[3]),a=std::abs(dt);
  Eigen::Matrix<double,5,2> G=Eigen::Matrix<double,5,2>::Zero();
  G(0,0)=a*a/2*c;G(1,0)=a*a/2*s;G(2,0)=a;G(3,1)=a*a/2;G(4,1)=a;
  Eigen::Matrix2d q=Eigen::Matrix2d::Zero();
  q(0,0)=config_.sigma_a*config_.sigma_a*process_scale_;q(1,1)=config_.sigma_alpha*config_.sigma_alpha*process_scale_;
  x=ctrv_predict(x,dt);
  P=F*P*F.transpose()+G*q*G.transpose();
}
void Tracker::predict_to(Track& t,double when) const {propagate(t.x,t.P,when-t.t);t.t=when;}
void Tracker::peek(const Track& t,double when,Track::State& x,Track::Cov& P) const {x=t.x;P=t.P;propagate(x,P,when-t.t);}
bool Tracker::usable(const Track& t,double stamp) const {
  const double age=stamp-t.last_seen;
  return t.confirmed&&std::isfinite(stamp)&&age>=-1e-9&&age<=config_.coast_s+1e-9&&t.x.allFinite()&&t.P.allFinite()&&
    t.P(0,0)<=config_.max_position_variance&&t.P(1,1)<=config_.max_position_variance;
}
double Tracker::heading_for(const Track& t,const Scan& s) const {
  if(t.velocity()>=3.||!s.heading_prior)return t.heading();
  return s.heading_prior(t.anchor());
}
void Tracker::observe_map(Track& t,const Detection& d,const Scan& s) const {
  if(!s.map_ok||!s.is_structure){t.static_ratio=std::numeric_limits<double>::quiet_NaN();t.static_updates=0;t.map_static=false;t.map_covered=false;return;}
  int structure=0,covered=0;
  for(int i:d.points){structure+=s.is_structure(i);if(s.is_covered)covered+=s.is_covered(i);}
  const double ratio=double(structure)/double(d.points.size());
  t.static_ratio=std::isfinite(t.static_ratio)?.7*t.static_ratio+.3*ratio:ratio;
  ++t.static_updates;
  t.map_static=t.static_updates>=3&&t.static_ratio>=.6;
  t.map_covered=2*covered>=int(d.points.size());
}
void Tracker::apply(Track& t,const Detection& d,const Scan& s,const AnchorFit& fit,double noise_scale,bool guided){
  if(t.hits==1){
    // Two-point start: one return cannot give velocity, and a wrong guess would
    // leave the first gate in a world-fixed frame.
    const double dt=std::max(1e-3,s.stamp-t.last_seen);
    const V2 moved=fit.anchor-t.anchor();
    t.x[0]=fit.anchor.x();t.x[1]=fit.anchor.y();t.x[2]=moved.norm()/dt;t.x[4]=0;
    if(moved.norm()>=.5)t.x[3]=std::atan2(moved.y(),moved.x());
    t.P=Track::Cov::Zero();t.P(0,0)=t.P(1,1)=.25;t.P(2,2)=25;t.P(3,3)=.5;t.P(4,4)=.25;
    t.t=d.time;
  }else{
    predict_to(t,d.time);
    Eigen::Matrix<double,2,5> H=Eigen::Matrix<double,2,5>::Zero();H(0,0)=H(1,1)=1;
    const double h=heading_for(t,s);
    const Eigen::Matrix2d R=rotated_noise(h,fit.sigma_along,fit.sigma_lateral)*noise_scale;
    const Eigen::Matrix2d S=H*t.P*H.transpose()+R;
    const Eigen::Matrix<double,5,2> K=t.P*H.transpose()*S.inverse();
    const V2 innovation=fit.anchor-t.anchor();
    t.x+=K*innovation;
    const Track::Cov A=Track::Cov::Identity()-K*H;
    t.P=A*t.P*A.transpose()+K*R*K.transpose();
    t.x[2]=std::max(0.,t.x[2]);t.x[3]=wrap(t.x[3]);t.x[4]=std::clamp(t.x[4],-1.,1.);
  }
  if(fit.width>=1.2)t.width=std::clamp(.7*t.width+.3*fit.width,1.2,2.6);
  if(fit.length>t.length)t.length=std::min(5.2,.7*t.length+.3*fit.length);
  const double extent=d.body_box.hi.z()-d.body_box.lo.z();
  t.z=.7*t.z+.3*d.body_box.center().z();
  if(extent>=.8)t.height=.7*t.height+.3*extent;
  t.raw_box=d.body_box;t.rear_visible=fit.rear_visible;t.guided=guided;t.observed_width=fit.width;
  t.weak_update=d.quality==Quality::Weak;
  t.vehicle_evidence=t.vehicle_evidence||full_vehicle_geometry(d.body_box);
  t.last_seen=s.stamp;t.measured=true;t.misses=0;++t.hits;
  observe_map(t,d,s);
}
void Tracker::gather(Scan& s){
  const int count=int(s.world_points.size());
  std::vector<int> owner(count,-1);std::vector<double> best(count,1e100);
  std::vector<int> eligible;
  for(int i=0;i<int(tracks_.size());++i)if(tracks_[i].confirmed&&s.stamp-tracks_[i].last_seen<=config_.coast_s)eligible.push_back(i);
  struct Region{V2 origin,u,v;double rear,front,half_width;};
  std::vector<Region> regions(tracks_.size());
  for(int i:eligible){
    const Track& t=tracks_[i];
    Track::State x;Track::Cov P;peek(t,s.stamp,x,P);
    const double h=heading_for(t,s);
    Region r;r.origin=V2(x[0],x[1]);r.u=V2(std::cos(h),std::sin(h));r.v=V2(-r.u.y(),r.u.x());
    Eigen::Matrix2d xy;xy<<P(0,0),P(0,1),P(1,0),P(1,1);
    const double sa=std::sqrt(std::max(0.,r.u.dot(xy*r.u))),sl=std::sqrt(std::max(0.,r.v.dot(xy*r.v)));
    r.rear=-.8-2*sa;r.front=t.length+.8+2*sa;r.half_width=t.width/2+.6+2*sl;
    regions[i]=r;
    for(int p=0;p<count;++p){
      const V2 d=s.world_points[p].head<2>()-r.origin;
      const double a=d.dot(r.u),b=d.dot(r.v);
      if(a<r.rear||a>r.front||std::abs(b)>r.half_width)continue;
      const double centre=(r.rear+r.front)/2,span=(r.front-r.rear)/2;
      const double score=std::pow((a-centre)/span,2)+std::pow(b/r.half_width,2);
      if(score<best[p]){best[p]=score;owner[p]=i;}
    }
  }
  std::vector<bool> taken(count,false);
  for(int i:eligible){
    std::vector<int> ids;
    for(int p=0;p<count;++p)if(owner[p]==i)ids.push_back(p);
    if(ids.size()<3)continue;
    Track& t=tracks_[i];const Region& r=regions[i];
    double lo=1e100,hi=-1e100,back=1e100,ahead=-1e100;
    for(int p:ids){const V2 d=s.world_points[p].head<2>()-r.origin;const double a=d.dot(r.u),b=d.dot(r.v);lo=std::min(lo,b);hi=std::max(hi,b);back=std::min(back,a);ahead=std::max(ahead,a);}
    // A wall crossing the box looks thin to a minimum-area fit; check the track's own axes.
    if(hi-lo>2.6||ahead-back>5.6)continue;
    Detection d=make_detection(s,ids,config_.weak_min_range);
    if(!vehicle_extent(d.body_box))continue;
    if(s.is_structure){int hits=0;for(int p:ids)hits+=s.is_structure(p);if(hits>=.6*double(ids.size()))continue;}
    const double h=heading_for(t,s);
    const AnchorFit fit=rear_anchor(xy_of(s,ids),h,V2(s.ego.x,s.ego.y),t.width);
    Track::State x;Track::Cov P;peek(t,d.time,x,P);
    Eigen::Matrix2d S;S<<P(0,0),P(0,1),P(1,0),P(1,1);S+=rotated_noise(h,fit.sigma_along,fit.sigma_lateral);
    const V2 innovation=fit.anchor-V2(x[0],x[1]);
    if(innovation.dot(S.inverse()*innovation)>9.21)continue;
    for(int p:ids)taken[p]=true;
    apply(t,d,s,fit,1.,true);
  }
  std::vector<Detection> remaining;
  for(auto& d:s.detections){
    std::vector<int> left;
    for(int p:d.points)if(!taken[p])left.push_back(p);
    if(left.size()==d.points.size()){remaining.push_back(std::move(d));continue;}
    if(left.size()<3)continue;
    Detection trimmed=make_detection(s,left,config_.weak_min_range);
    if(trimmed.quality!=Quality::Reject)remaining.push_back(std::move(trimmed));
  }
  s.detections=std::move(remaining);
}
double Tracker::pair_cost(const Track& t,const Detection& d,const Scan& s,double chi2,double max_distance) const {
  const V2 centre=s.ego.to_world(d.body_box.center().head<2>());
  if((centre-t.anchor()).norm()>25)return 1e6;
  const double h=heading_for(t,s);
  const AnchorFit fit=rear_anchor(xy_of(s,d.points),h,V2(s.ego.x,s.ego.y),t.width);
  if(t.hits==1){
    const double limit=50*std::max(1e-3,s.stamp-t.last_seen)+1;
    const double moved=(fit.anchor-t.anchor()).norm();
    return moved<=limit?.9*moved/limit:1e6;
  }
  Track::State x;Track::Cov P;peek(t,d.time,x,P);
  Eigen::Matrix2d S;S<<P(0,0),P(0,1),P(1,0),P(1,1);S+=rotated_noise(h,fit.sigma_along,fit.sigma_lateral);
  const V2 innovation=fit.anchor-V2(x[0],x[1]);
  const double mahalanobis=innovation.dot(S.inverse()*innovation);
  if(mahalanobis>chi2||innovation.norm()>max_distance)return 1e6;
  return .7*mahalanobis/chi2+.3*std::min(1.,std::abs(fit.width-t.width)/1.5);
}
void Tracker::assign(Scan& s,const std::vector<int>& track_ids,const std::vector<int>& detection_ids,
                     double chi2,double max_distance,std::vector<bool>& consumed){
  const int n=int(track_ids.size()),m=int(detection_ids.size());
  if(!n||!m)return;
  Eigen::MatrixXd costs=Eigen::MatrixXd::Constant(n,m+n,1.0);
  for(int i=0;i<n;++i)for(int j=0;j<m;++j)
    costs(i,j)=pair_cost(tracks_[track_ids[i]],s.detections[detection_ids[j]],s,chi2,max_distance);
  const auto picks=hungarian(costs);
  for(int i=0;i<n;++i){
    const int j=picks[i];
    if(j<0||j>=m||costs(i,j)>=.95)continue;
    double other=1e6;for(int k=0;k<m;++k)if(k!=j)other=std::min(other,costs(i,k));
    // Two near-identical measurements: use the better one but trust it less.
    const double scale=other-costs(i,j)<.03?4.:1.;
    Track& t=tracks_[track_ids[i]];const Detection& d=s.detections[detection_ids[j]];
    const AnchorFit fit=rear_anchor(xy_of(s,d.points),heading_for(t,s),V2(s.ego.x,s.ego.y),t.width);
    apply(t,d,s,fit,scale,false);
    consumed[detection_ids[j]]=true;
  }
}
void Tracker::birth(const Detection& d,const Scan& s){
  Track t;t.id=next_++;t.t=d.time;t.last_seen=s.stamp;t.hits=1;t.measured=true;
  const auto xy=xy_of(s,d.points);
  V2 centre=V2::Zero();for(const auto& p:xy)centre+=p;centre/=double(xy.size());
  const V2 sensor(s.ego.x,s.ego.y);
  const double h=s.heading_prior?s.heading_prior(centre):std::atan2(centre.y()-sensor.y(),centre.x()-sensor.x());
  const AnchorFit fit=rear_anchor(xy,h,sensor,t.width);
  t.x<<fit.anchor.x(),fit.anchor.y(),0.,h,0.;
  t.P=Track::Cov::Zero();t.P(0,0)=t.P(1,1)=.25;t.P(2,2)=400;t.P(3,3)=.5;t.P(4,4)=.25;
  t.raw_box=d.body_box;t.z=d.body_box.center().z();t.rear_visible=fit.rear_visible;t.observed_width=fit.width;
  if(fit.width>=1.2)t.width=std::min(2.6,fit.width);
  t.weak_born=d.quality==Quality::Weak;t.weak_update=t.weak_born;
  t.vehicle_evidence=full_vehicle_geometry(d.body_box);t.evidence_stamp=s.stamp;
  observe_map(t,d,s);
  tracks_.push_back(std::move(t));
}
std::vector<Track>& Tracker::update(Scan s){
  if(!std::isfinite(s.stamp)){reset();return tracks_;}
  if(s.stamp<=last_||(last_>=0&&s.stamp-last_>config_.memory_s))reset();
  last_=s.stamp;
  for(auto& t:tracks_){t.measured=t.weak_update=t.guided=false;}
  if(config_.guided)gather(s);
  std::vector<bool> consumed(s.detections.size(),false);
  auto pick_tracks=[&](bool confirmed){std::vector<int> ids;
    for(int i=0;i<int(tracks_.size());++i)if(tracks_[i].confirmed==confirmed&&!tracks_[i].measured)ids.push_back(i);return ids;};
  auto pick_detections=[&](bool strong,bool weak){std::vector<int> ids;
    for(int j=0;j<int(s.detections.size());++j)if(!consumed[j]&&((strong&&s.detections[j].quality==Quality::Strong)||(weak&&s.detections[j].quality==Quality::Weak)))ids.push_back(j);return ids;};
  assign(s,pick_tracks(true),pick_detections(true,false),9.21,3.,consumed);
  assign(s,pick_tracks(true),pick_detections(false,true),5.99,3.,consumed);
  assign(s,pick_tracks(false),pick_detections(true,true),9.21,3.,consumed);
  for(int j=0;j<int(s.detections.size());++j){
    if(consumed[j])continue;
    const Detection& d=s.detections[j];
    // A plausible observation of a confirmed track is not a new object.
    bool claimed=false;
    for(const auto& t:tracks_)if(t.confirmed&&pair_cost(t,d,s,9.21,3.)<.95){claimed=true;break;}
    if(claimed)continue;
    if(d.quality==Quality::Strong){birth(d,s);continue;}
    if(d.quality!=Quality::Weak||!s.map_ok)continue;
    if(s.is_structure){int hits=0;for(int p:d.points)hits+=s.is_structure(p);if(hits>=.6*double(d.points.size()))continue;}
    birth(d,s);
  }
  for(auto& t:tracks_){
    if(!t.measured)++t.misses;
    if(t.hits>1||!t.measured)predict_to(t,s.stamp);else t.t=s.stamp;
    t.seen=uint16_t((t.seen<<1)|(t.measured?1:0));
    if(!t.confirmed)t.confirmed=t.weak_born?count_bits(t.seen&0x3FF)>=4:count_bits(t.seen&0xFF)>=3;
  }
  tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[&](const Track& t){
    const double age=s.stamp-t.last_seen;
    return age>config_.memory_s||(!t.confirmed&&age>config_.tentative_s)||(t.hits>1&&t.P(0,0)+t.P(1,1)>4.);}),tracks_.end());
  for(;;){
    int tentative=0;auto oldest=tracks_.end();
    for(auto it=tracks_.begin();it!=tracks_.end();++it)if(it->weak_born&&!it->confirmed){++tentative;
      if(oldest==tracks_.end()||it->last_seen<oldest->last_seen)oldest=it;}
    if(tentative<=config_.max_weak_tracks)break;
    tracks_.erase(oldest);
  }
  return tracks_;
}
double near_surface_x(const Track& t,const Pose2& ego){
  const auto all=corners(t);
  return std::min(ego.to_body(all[0]).x(),ego.to_body(all[1]).x());
}
Box track_body_box(const Track& t,const Pose2& ego){return body_box(t,ego,4);}
Box track_rear_box(const Track& t,const Pose2& ego){return body_box(t,ego,2);}
V2 relative_velocity(const Track& t,const Pose2& ego,const V2& ego_velocity){
  const V2 lead(t.velocity()*std::cos(t.heading()),t.velocity()*std::sin(t.heading()));
  const V2 difference=lead-ego_velocity;
  const double c=std::cos(ego.yaw),s=std::sin(ego.yaw);
  return V2(c*difference.x()+s*difference.y(),-s*difference.x()+c*difference.y());
}
const char* vehicle_geometry_status(const Track& t){
  if(!vehicle_extent(t.raw_box))return "extent_exceeded";
  if(full_vehicle_geometry(t.raw_box))return "full_geometry";
  if(t.vehicle_evidence&&partial_vehicle_geometry(t.raw_box))return "retained_partial";
  const auto d=t.raw_box.hi-t.raw_box.lo;
  if(t.hits>=6&&t.raw_box.center().head<2>().norm()>=20&&partial_vehicle_geometry(t.raw_box)&&d.z()>=.3)return "far_partial";
  return t.weak_update?"weak_surface":"insufficient_geometry";
}
}
