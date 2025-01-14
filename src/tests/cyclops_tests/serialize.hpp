#pragma once

#include "cyclops/details/type.hpp"

#include <string>
#include <map>
#include <vector>

namespace cyclops {
  struct se3_transform_t;

  std::string serialize(std::vector<Eigen::Vector3d> const&);
  std::string serialize(std::vector<se3_transform_t> const&);
  std::string serialize(
    std::map<landmark_id_t, std::map<frame_id_t, feature_point_t>> const&);
}  // namespace cyclops
