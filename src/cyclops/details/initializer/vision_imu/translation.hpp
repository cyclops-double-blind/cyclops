#pragma once

#include "cyclops/details/measurement/type.hpp"

#include <map>
#include <memory>
#include <optional>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  struct imu_match_camera_translation_prior_t;
  struct imu_match_rotation_solution_t;

  struct imu_match_translation_solution_t {
    double scale;
    double cost;
    Eigen::Vector3d gravity;
    Eigen::Vector3d acc_bias;
    std::map<frame_id_t, Eigen::Vector3d> imu_body_velocities;
    std::map<frame_id_t, Eigen::Vector3d> sfm_positions;
  };

  struct imu_translation_match_t {
    bool accept;
    imu_match_translation_solution_t solution;
  };

  class IMUMatchTranslationSolver {
  public:
    virtual ~IMUMatchTranslationSolver() = default;
    virtual void reset() = 0;

    virtual std::optional<imu_translation_match_t> solve(
      measurement::imu_motion_refs_t const& motions,
      imu_match_rotation_solution_t const& rotations,
      imu_match_camera_translation_prior_t const& camera_prior) = 0;

    static std::unique_ptr<IMUMatchTranslationSolver> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
