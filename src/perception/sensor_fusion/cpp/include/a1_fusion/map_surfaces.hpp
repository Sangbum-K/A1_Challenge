#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
namespace a1::inspection {
// curvature stores a discrete surface label: 1=local low horizontal surface, 2=vertical plane.
using Surfaces=pcl::PointCloud<pcl::PointNormal>;
Surfaces::Ptr extract_map_surfaces(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input);
class MapSurfaces {
 public:
  void set(const Surfaces::ConstPtr& cloud);
  int match(const Eigen::Vector3d& map_point) const;
  bool empty()const{return !cloud_||cloud_->empty();}
 private:
  Surfaces::ConstPtr cloud_;pcl::KdTreeFLANN<pcl::PointNormal> tree_;
};
}
