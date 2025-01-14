#pragma once

#include "cyclops/details/type.hpp"

#include <random>
#include <map>
#include <memory>
#include <vector>

namespace cyclops {
  struct cyclops_global_config_t;
}  // namespace cyclops

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  struct vision_bootstrap_solution_t;
  struct two_view_imu_rotation_constraint_t;

  class VisionBootstrapSolver {
  public:
    using multiview_image_data_t =
      std::map<frame_id_t, std::map<landmark_id_t, feature_point_t>>;
    using camera_rotations_t =
      std::map<frame_id_t, two_view_imu_rotation_constraint_t>;

  public:
    virtual ~VisionBootstrapSolver() = default;
    virtual void reset() = 0;

    virtual std::vector<vision_bootstrap_solution_t> solve(
      multiview_image_data_t const& image_data,
      camera_rotations_t const& camera_rotation_prior) = 0;

    static std::unique_ptr<VisionBootstrapSolver> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<std::mt19937> rgen,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
