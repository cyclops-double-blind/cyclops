#pragma once

#include "cyclops/details/initializer/vision/type.hpp"

#include <set>
#include <map>
#include <vector>

namespace cyclops {
  struct rotation_translation_matrix_pair_t;
}  // namespace cyclops

namespace cyclops::initializer {
  struct epipolar_analysis_t {
    double expected_inliers;

    Eigen::Matrix3d essential_matrix;
    std::set<landmark_id_t> inliers;
  };

  epipolar_analysis_t analyze_two_view_epipolar(
    double sigma, std::vector<std::set<landmark_id_t>> const& ransac_batch,
    std::map<landmark_id_t, two_view_feature_pair_t> const& features);

  std::vector<rotation_translation_matrix_pair_t>
  solve_epipolar_motion_hypothesis(Eigen::Matrix3d const& essential_matrix);
}  // namespace cyclops::initializer
