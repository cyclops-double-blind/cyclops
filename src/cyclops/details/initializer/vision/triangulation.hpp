#pragma once

#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/type.hpp"

#include <map>
#include <set>

namespace cyclops {
  struct rotation_translation_matrix_pair_t;
}

namespace cyclops::config::initializer {
  struct vision_solver_config_t;
}

namespace cyclops::initializer {
  struct two_view_triangulation_t {
    /* failure statistics. */
    int n_triangulation_failure;
    int n_error_probability_test_failure;

    double expected_inliers;

    /**
     * actually accepted triangulations. i.e. triangulations that passed both
     * reprojection test and direction test, with enough parallax.
     */
    landmark_positions_t landmarks;
  };

  two_view_triangulation_t triangulate_two_view_feature_pairs(
    config::initializer::vision_solver_config_t const& config,
    std::map<landmark_id_t, two_view_feature_pair_t> const& feature_pairs,
    std::set<landmark_id_t> const& feature_ids,
    rotation_translation_matrix_pair_t const& camera_motion);
}  // namespace cyclops::initializer
