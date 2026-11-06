#pragma once
#include <Eigen/Dense>
#include <vector>
namespace a1::inspection {
struct StructurePoint { float x,y,z; int keyframe; };
// Static objects of any shape, including rails too low for the wall label. Keeping
// only voxels seen from several keyframes drops cars that passed during mapping.
std::vector<Eigen::Vector3f> structure_voxels(const std::vector<StructurePoint>&,int min_keyframes);
}
