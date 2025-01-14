#pragma once

#include "cyclops/details/initializer/vision/type.hpp"

#include <map>
#include <set>

namespace cyclops::initializer {
  Eigen::Matrix3d refine_epipolar_geometry(
    Eigen::Matrix3d const& E_initial, std::set<landmark_id_t> const& ids,
    std::map<landmark_id_t, two_view_feature_pair_t> const& features);
}  // namespace cyclops::initializer
