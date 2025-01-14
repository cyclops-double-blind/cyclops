#pragma once

#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/type.hpp"

#include <map>
#include <set>
#include <vector>

namespace cyclops {
  struct rotation_translation_matrix_pair_t;
}  // namespace cyclops

namespace cyclops::initializer {
  struct homography_analysis_t {
    double expected_inliers;

    Eigen::Matrix3d homography;
    std::set<landmark_id_t> inliers;
  };

  homography_analysis_t analyze_two_view_homography(
    double sigma, std::vector<std::set<landmark_id_t>> const& ransac_batch,
    std::map<landmark_id_t, two_view_feature_pair_t> const& features);

  std::vector<rotation_translation_matrix_pair_t>
  solve_homography_motion_hypothesis(Eigen::Matrix3d const& H);
}  // namespace cyclops::initializer
