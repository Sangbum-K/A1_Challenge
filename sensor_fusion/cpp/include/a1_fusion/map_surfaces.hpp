#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
namespace a1::inspection {
// curvature stores a discrete label: 1=local low horizontal surface, 2=vertical plane,
// 3=static structure of any shape (see map_structure.hpp), with normal (0,0,1).
using Surfaces=pcl::PointCloud<pcl::PointNormal>;
Surfaces::Ptr extract_map_surfaces(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input);
class MapSurfaces {
 public:
  void set(const Surfaces::ConstPtr& cloud);
  // range is the distance from the sensor: heading error grows with it.
  int match(const Eigen::Vector3d& map_point,double range=0.) const;
  bool is_structure(const Eigen::Vector3d& map_point,double range=0.) const;
  // match() returns 0 both on open road and outside the map; this tells them apart.
  bool covered(const Eigen::Vector3d& map_point) const;
  bool empty()const{return !cloud_||cloud_->empty();}
  bool has_structure()const{return structure_&&!structure_->empty();}
 private:
  Surfaces::ConstPtr cloud_,structure_;
  pcl::KdTreeFLANN<pcl::PointNormal> tree_,structure_tree_;
};
}
