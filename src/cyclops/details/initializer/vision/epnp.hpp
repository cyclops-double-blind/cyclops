#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <optional>

namespace cyclops {
  struct rotation_translation_matrix_pair_t;
}  // namespace cyclops

namespace cyclops::initializer {
  struct pnp_image_point_t {
    Eigen::Vector3d position;
    Eigen::Vector2d observation;
  };

  std::optional<rotation_translation_matrix_pair_t> solve_pnp_camera_pose(
    std::map<landmark_id_t, pnp_image_point_t> const& image_point_set,
    int gauss_newton_refinement_iterations = 5);
}  // namespace cyclops::initializer
