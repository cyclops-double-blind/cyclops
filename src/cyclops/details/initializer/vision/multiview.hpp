#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>
#include <random>
#include <vector>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  struct multiview_geometry_t;
  struct two_view_imu_rotation_constraint_t;

  class MultiviewVisionGeometrySolver {
  public:
    using multiview_image_data_t =
      std::map<frame_id_t, std::map<landmark_id_t, feature_point_t>>;
    using camera_rotation_prior_lookup_t =
      std::map<frame_id_t, two_view_imu_rotation_constraint_t>;

  public:
    virtual ~MultiviewVisionGeometrySolver() = default;
    virtual void reset() = 0;

    // returns a sequence of possible multiview geometries.
    virtual std::vector<multiview_geometry_t> solve(
      multiview_image_data_t const& multiview_data,
      camera_rotation_prior_lookup_t const& camera_rotations) = 0;

    static std::unique_ptr<MultiviewVisionGeometrySolver> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<std::mt19937> rgen,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
