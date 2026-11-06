#include <pcl/io/pcd_io.h>
#include "a1_fusion/map_surfaces.hpp"
#include <mp2p_icp/metricmap.h>
#include <mola_metric_maps/KeyframePointCloudMap.h>
#include <mrpt/maps/CSimplePointsMap.h>
#include <mrpt/system/os.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <dlfcn.h>
int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("usage: map_surface_export map.mm surfaces.pcd");
 void* plugin=dlopen("libmola_metric_maps.so",RTLD_NOW|RTLD_GLOBAL);if(!plugin)throw std::runtime_error(dlerror());
 mp2p_icp::metric_map_t map;if(!map.load_from_file(argv[1]))throw std::runtime_error("cannot load metric map");
 auto input=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
 for(const auto& [name,layer]:map.layers){
  if(auto kf=std::dynamic_pointer_cast<mola::KeyframePointCloudMap>(layer)){kf->renderOptions.max_points_per_kf=0;kf->renderOptions.max_overall_points=0;}
  const auto* points=layer->getAsSimplePointsMap();if(!points)continue;
  for(size_t i=0;i<points->size();++i){float x,y,z;points->getPointFast(i,x,y,z);if(std::isfinite(x)&&std::isfinite(y)&&std::isfinite(z))input->push_back({x,y,z});}}
 if(input->empty())throw std::runtime_error("map has no exportable points");
 Eigen::Vector3f lo=Eigen::Vector3f::Constant(1e20),hi=-lo;for(const auto& p:*input){lo=lo.cwiseMin(p.getVector3fMap());hi=hi.cwiseMax(p.getVector3fMap());}
 std::cout<<"input_points="<<input->size()<<" min="<<lo.transpose()<<" max="<<hi.transpose()<<std::endl;
 // shortcut: no structure label (curvature=3) is exported, because that needs per-keyframe
 // points and KeyframePointCloudMap access is unverified off the vehicle PC. Feed
 // structure_voxels() from a verified keyframe source; until then map_status stays no_structure.
 auto surfaces=a1::inspection::extract_map_surfaces(input);size_t floor=0,walls=0;for(const auto& p:*surfaces){floor+=p.curvature==1;walls+=p.curvature==2;}
 if(surfaces->empty())throw std::runtime_error("no planar surfaces found");
 if(pcl::io::savePCDFileBinary(argv[2],*surfaces)!=0)throw std::runtime_error("cannot save surfaces");
 std::cout<<"ground="<<floor<<" wall="<<walls<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
