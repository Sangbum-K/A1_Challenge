#include "a1_fusion/camera_evidence.hpp"
#include <algorithm>
#include <cmath>
namespace a1::inspection {
namespace {
constexpr double kBinPixels=32.;
double logit(double p){const double q=std::clamp(p,1e-6,1-1e-6);return std::log(q/(1-q));}
int count_bits(unsigned value){int n=0;for(;value;value&=value-1)++n;return n;}
}
BinSpan bins_for(const Projection& p){
  if(!p.valid)return {};
  return {std::clamp(int(std::floor(p.x0/kBinPixels)),0,39),std::clamp(int(std::floor(p.x1/kBinPixels)),0,39),true};
}
double span_score(const std::array<double,40>& p,const BinSpan& span,const std::array<bool,40>& others,int dilation){
  if(!span.valid)return 0;
  double best=0;
  for(int j=std::max(0,span.left-dilation);j<=std::min(39,span.right+dilation);++j)
    if(!others[j]&&std::isfinite(p[j]))best=std::max(best,p[j]);
  return best;
}
bool assign_ownership(std::vector<TrackView>& views,const std::array<double,40>& p,int dilation){
  std::vector<int> order(views.size());
  for(int i=0;i<int(order.size());++i)order[i]=i;
  std::sort(order.begin(),order.end(),[&](int a,int b){return views[a].range<views[b].range;});
  bool ambiguous=false;
  std::vector<bool> unordered(views.size(),false);
  for(int i=0;i<int(views.size());++i)for(int k=i+1;k<int(views.size());++k){
    const auto& a=views[i];const auto& b=views[k];
    if(!a.span.valid||!b.span.valid||!a.can_own||!b.can_own||std::abs(a.range-b.range)>=3)continue;
    const int shared=std::min(a.span.right,b.span.right)-std::max(a.span.left,b.span.left)+1;
    const int smaller=std::min(a.span.right-a.span.left,b.span.right-b.span.left)+1;
    if(shared>=.7*smaller){unordered[i]=unordered[k]=true;ambiguous=true;}
  }
  std::array<int,40> owner;owner.fill(-1);
  for(int index:order){
    const auto& v=views[index];
    if(!v.span.valid||!v.can_own||unordered[index])continue;
    for(int j=v.span.left;j<=v.span.right;++j)if(owner[j]<0)owner[j]=index;
  }
  for(int index=0;index<int(views.size());++index){
    auto& v=views[index];v.score=0;
    if(!v.span.valid){v.visibility=Visibility::Outside;continue;}
    std::array<bool,40> others{};bool open=false;
    for(int j=0;j<40;++j)others[j]=owner[j]>=0&&owner[j]!=index;
    for(int j=v.span.left;j<=v.span.right;++j)open=open||!others[j];
    if(!open||unordered[index]){v.visibility=Visibility::Occluded;continue;}
    v.visibility=Visibility::Visible;v.score=span_score(p,v.span,others,dilation);
  }
  return ambiguous;
}
void observe(Track& t,double stamp,double score,Visibility visibility,const EvidenceConfig& c){
  if(stamp<=t.evidence_stamp)return;
  const double dt=stamp-t.evidence_stamp;
  t.evidence_stamp=stamp;t.visibility=visibility;t.camera_score=score;
  if(visibility!=Visibility::Visible)return;
  t.evidence*=std::exp(-dt/c.tau);
  t.evidence=std::clamp(t.evidence+c.alpha*std::clamp(logit(score)-logit(c.p0),c.neg,c.pos),c.l_min,c.l_max);
}
void update_far_state(Track& t,const FarInputs& in,const FarConfig& c){
  if(in.range<=c.near_limit||!t.confirmed||!in.map_ok||!t.map_covered||t.map_static){
    t.far_score=std::numeric_limits<double>::quiet_NaN();t.far_vehicle=false;return;
  }
  double score=0;
  const int seen=count_bits(t.seen&0x3FF);
  score+=seen>=8?1.5:seen>=5?1.:0.;
  if(t.velocity()>=3.&&in.speed_sigma<=1.5&&in.ego_compensated)score+=3.;
  score+=.5*std::clamp(t.evidence,0.,3.);
  if(t.observed_width>=1.)score+=.5;
  if(in.raceline_active){const double n=std::abs(in.path_n);if(n<=8.)score+=1.;else if(n>15.)score-=2.;}
  t.far_score=score;
  t.far_vehicle=t.far_vehicle?score>=c.release:score>=c.promote;
}
void EvidenceSelector::reset(){
  primary_=pending_=-1;last_=-1;matched_stamp_=std::numeric_limits<double>::quiet_NaN();
}
TargetSelection EvidenceSelector::update(double stamp,double camera_stamp,const std::vector<TargetObservation>& observations){
  if(!std::isfinite(stamp)||stamp<=last_)reset();
  if(!std::isfinite(stamp))return {};
  last_=stamp;
  const bool silent=!std::isfinite(camera_stamp)||stamp-camera_stamp>config_.camera_silent_s;
  auto far_held=[&](const TargetObservation& o){return o.far_vehicle&&o.range>config_.far_keep_range;};
  auto eligible=[&](const TargetObservation& o){
    if(!o.usable||!o.measured||!o.in_lane||o.map_static||o.range<config_.target_min||o.range>config_.target_max)return false;
    if(o.range>config_.far_entry_range)return o.far_vehicle;
    return !silent&&o.vehicle_shape&&o.evidence>=config_.l_conf;
  };
  auto retained=[&](const TargetObservation& o){
    if(!o.usable||!o.in_lane||o.map_static)return false;
    return far_held(o)||(!silent&&o.evidence>=config_.l_drop);
  };
  std::vector<const TargetObservation*> candidates;
  const TargetObservation* current=nullptr;
  for(const auto& o:observations){
    if(eligible(o))candidates.push_back(&o);
    if(o.id==primary_&&retained(o))current=&o;
  }
  std::sort(candidates.begin(),candidates.end(),[](auto a,auto b){return a->distance==b->distance?a->id<b->id:a->distance<b->distance;});
  const TargetObservation* chosen=current;
  if(!chosen){
    pending_=-1;
    if(!candidates.empty())chosen=candidates.front();
  }else{
    // A closer lane target must take over, or a cut-in stays invisible to P5.
    const TargetObservation* closer=nullptr;
    for(auto o:candidates)if(o->id!=current->id&&o->distance<current->distance){closer=o;break;}
    if(!closer)pending_=-1;
    else{
      if(pending_!=closer->id){pending_=closer->id;pending_since_=stamp;}
      if(stamp-pending_since_>=config_.switch_hold_s){chosen=closer;pending_=-1;}
    }
  }
  TargetSelection result;
  const int64_t previous=primary_;
  primary_=chosen?chosen->id:-1;
  if(primary_!=previous)matched_stamp_=std::numeric_limits<double>::quiet_NaN();
  if(!chosen)return result;
  if(chosen->camera_matched)matched_stamp_=stamp;
  result.primary=chosen->id;
  result.status=!chosen->measured?"predicted":chosen->camera_matched?"measured":"camera_held";
  result.camera_age=stamp-matched_stamp_;
  for(auto o:candidates)if(o->id!=primary_){result.standby=o->id;break;}
  return result;
}
}
