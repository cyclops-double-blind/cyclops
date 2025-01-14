#pragma once

#include "cyclops/details/type.hpp"

namespace cyclops::config::initializer {
  struct vision_solver_config_t;
}  // namespace cyclops::config::initializer

namespace cyclops::initializer {
  struct multiview_geometry_t;
  struct vision_bootstrap_solution_t;

  std::optional<vision_bootstrap_solution_t> solve_bundle_adjustment(
    config::initializer::vision_solver_config_t const& config,
    multiview_geometry_t const& guess,
    std::map<frame_id_t, std::map<landmark_id_t, feature_point_t>> const& data);
}  // namespace cyclops::initializer
