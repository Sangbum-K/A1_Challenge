#include "a1_fusion/map_structure.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
namespace a1::inspection {
std::vector<Eigen::Vector3f> structure_voxels(const std::vector<StructurePoint>& points,int min_keyframes){
  if(min_keyframes<1)throw std::invalid_argument("min_keyframes must be positive");
  using Key=std::tuple<int,int,int>;
  struct Voxel { double x{},y{},z{};int count{};std::set<int> keyframes; };
  std::map<std::pair<int,int>,float> floor;
  std::map<Key,Voxel> voxels;
  for(const auto& p:points){
    if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))continue;
    const std::pair<int,int> cell(int(std::floor(p.x/2)),int(std::floor(p.y/2)));
    auto it=floor.find(cell);
    if(it==floor.end())floor[cell]=p.z;else it->second=std::min(it->second,p.z);
    auto& v=voxels[Key(int(std::floor(p.x/.2)),int(std::floor(p.y/.2)),int(std::floor(p.z/.2)))];
    v.x+=p.x;v.y+=p.y;v.z+=p.z;++v.count;v.keyframes.insert(p.keyframe);
  }
  std::vector<Eigen::Vector3f> out;
  for(const auto& item:voxels){
    const Voxel& v=item.second;
    if(int(v.keyframes.size())<min_keyframes)continue;
    const double x=v.x/v.count,y=v.y/v.count,z=v.z/v.count;
    const auto base=floor.find({int(std::floor(x/2)),int(std::floor(y/2))});
    if(base==floor.end())continue; // Rounding at a cell edge; dropping one voxel is harmless.
    const double height=z-base->second;
    if(height<.25||height>3.5)continue;
    out.emplace_back(float(x),float(y),float(z));
  }
  return out;
}
}
