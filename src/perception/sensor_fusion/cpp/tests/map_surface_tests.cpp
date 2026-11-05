#include "a1_fusion/map_surfaces.hpp"
#include <iostream>
#include <stdexcept>
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
int main(){using namespace a1::inspection;
 auto input=pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
 for(int x=-30;x<=30;++x)for(int y=-30;y<=30;++y)input->push_back({x*.1f,y*.1f,0});
 for(int y=-30;y<=30;++y)for(int z=0;z<=40;++z)input->push_back({3,y*.1f,z*.1f});
 auto surfaces=extract_map_surfaces(input);MapSurfaces filter;filter.set(surfaces);
 CHECK(filter.match({0,0,.03})==1);CHECK(filter.match({2.98,0,2})==2);
 CHECK(filter.match({2.5,0,2})==0); // A vehicle half a metre in front of the wall stays.
 CHECK(filter.match({0,0,1})==0);CHECK(filter.match({100,100,0})==0);CHECK(filter.match({NAN,0,0})==0);
 CHECK(filter.match({3.01,0,6})==0); // No infinite-plane extrapolation.
 auto empty=pcl::make_shared<Surfaces>();filter.set(empty);CHECK(filter.match({3,0,2})==0);
 std::cout<<"Map surfaces: floor/wall, nearby vehicle preserved, map boundary, invalid/empty: PASS\n";
}
