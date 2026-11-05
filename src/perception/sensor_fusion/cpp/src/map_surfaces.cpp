#include "a1_fusion/map_surfaces.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/point_tests.h>
#include <map>
#include <unordered_map>
#include <cmath>
namespace a1::inspection {
Surfaces::Ptr extract_map_surfaces(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input){
 auto points=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
 // Sparse voxel keys avoid PCL's dense-index overflow on kilometre-scale maps.
 struct Cell{int x,y,z;bool operator==(const Cell& o)const{return x==o.x&&y==o.y&&z==o.z;}};
 struct Hash{size_t operator()(const Cell& c)const{return size_t(c.x)*73856093u^size_t(c.y)*19349663u^size_t(c.z)*83492791u;}};
 struct Accum{Eigen::Vector3d sum=Eigen::Vector3d::Zero();int count{};};
 std::unordered_map<Cell,Accum,Hash> cells;
 for(const auto& p:*input){if(!pcl::isFinite(p))continue;auto& c=cells[{int(std::floor(p.x/.2)),int(std::floor(p.y/.2)),int(std::floor(p.z/.2))}];c.sum+=p.getVector3fMap().cast<double>();++c.count;}
 points->reserve(cells.size());for(const auto& item:cells){Eigen::Vector3d q=item.second.sum/item.second.count;points->push_back({float(q.x()),float(q.y()),float(q.z())});}cells.clear();
 pcl::KdTreeFLANN<pcl::PointXYZ> tree;tree.setInputCloud(points);
 std::map<std::pair<int,int>,float> floor;
 for(const auto& p:*points){auto key=std::make_pair(int(std::floor(p.x/2)),int(std::floor(p.y/2)));auto it=floor.find(key);if(it==floor.end())floor[key]=p.z;else it->second=std::min(it->second,p.z);}
 auto out=pcl::make_shared<Surfaces>();std::vector<int> ids;std::vector<float> ds;
 for(const auto& p:*points){
  if(tree.radiusSearch(p,1.2,ids,ds)<20)continue;
  Eigen::Vector3d mean=Eigen::Vector3d::Zero(),lo=Eigen::Vector3d::Constant(1e20),hi=-lo;
  for(int id:ids){Eigen::Vector3d q=(*points)[id].getVector3fMap().cast<double>();mean+=q;lo=lo.cwiseMin(q);hi=hi.cwiseMax(q);}mean/=ids.size();
  Eigen::Matrix3d cov=Eigen::Matrix3d::Zero();for(int id:ids){Eigen::Vector3d d=(*points)[id].getVector3fMap().cast<double>()-mean;cov+=d*d.transpose();}cov/=ids.size();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(cov);if(eigen.info()!=Eigen::Success)continue;
  auto values=eigen.eigenvalues();if(values[1]<.025||values[0]>.008||values[0]/std::max(1e-9,values.sum())>.02)continue;
  Eigen::Vector3d n=eigen.eigenvectors().col(0);int label=0;
  float low=floor.at({int(std::floor(p.x/2)),int(std::floor(p.y/2))});
  if(std::abs(n.z())>.95&&p.z-low<.3)label=1;
  else if(std::abs(n.z())<.2&&(hi.z()-lo.z())>1.2&&(hi-lo).head<2>().norm()>1.5)label=2;
  if(!label)continue;
  pcl::PointNormal q;q.x=p.x;q.y=p.y;q.z=p.z;q.normal_x=n.x();q.normal_y=n.y();q.normal_z=n.z();q.curvature=label;out->push_back(q);
 }
 return out;
}
void MapSurfaces::set(const Surfaces::ConstPtr& cloud){
 auto valid=pcl::make_shared<Surfaces>();
 for(const auto& p:*cloud){Eigen::Vector3f n(p.normal_x,p.normal_y,p.normal_z);
  if(pcl::isFinite(p)&&n.allFinite()&&std::abs(n.norm()-1)<.01&&(p.curvature==1||p.curvature==2))valid->push_back(p);
 }
 cloud_=valid;if(!cloud_->empty())tree_.setInputCloud(cloud_);
}
int MapSurfaces::match(const Eigen::Vector3d& q)const{
 if(empty()||!q.allFinite())return 0;
 pcl::PointNormal p;p.x=q.x();p.y=q.y();p.z=q.z();std::vector<int> ids(3);std::vector<float> distances(3);
 if(tree_.nearestKSearch(p,3,ids,distances)!=3||distances.back()>.3*.3)return 0;
 int label=int((*cloud_)[ids[0]].curvature);
 for(int id:ids){const auto& s=(*cloud_)[id];Eigen::Vector3d n(s.normal_x,s.normal_y,s.normal_z),v(s.x,s.y,s.z);
  if(int(s.curvature)!=label||std::abs(n.dot(q-v))>.12)return 0;
 }
 return label;
}
}
