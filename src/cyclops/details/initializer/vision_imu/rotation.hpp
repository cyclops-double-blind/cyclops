#pragma once

#include "cyclops/details/measurement/type.hpp"

#include <map>
#include <memory>
#include <optional>

namespace cyclops {
  struct cyclops_global_config_t;
}  // namespace cyclops

namespace cyclops::initializer {
  struct imu_match_camera_rotation_prior_t;

  struct imu_match_rotation_solution_t {
    Eigen::Vector3d gyro_bias;
    std::map<frame_id_t, Eigen::Quaterniond> body_orientations;
  };

  class IMUMatchRotationSolver {
  public:
    virtual ~IMUMatchRotationSolver() = default;
    virtual void reset() = 0;

    virtual std::optional<imu_match_rotation_solution_t> solve(
      measurement::imu_motion_refs_t const& motions,
      imu_match_camera_rotation_prior_t const& prior) = 0;

    static std::unique_ptr<IMUMatchRotationSolver> create(
      std::shared_ptr<cyclops_global_config_t const> config);
  };
}  // namespace cyclops::initializer
