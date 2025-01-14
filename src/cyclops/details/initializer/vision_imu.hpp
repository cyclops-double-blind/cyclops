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
  struct vision_bootstrap_solution_t;

  struct imu_bootstrap_solution_t {
    bool accept;
    double cost;
    double scale;
    Eigen::Vector3d gravity;
    Eigen::Vector3d gyr_bias;
    Eigen::Vector3d acc_bias;

    landmark_positions_t landmarks;
    std::map<frame_id_t, imu_motion_state_t> motions;
  };

  class IMUBootstrapSolver {
  public:
    virtual ~IMUBootstrapSolver() = default;
    virtual void reset() = 0;

    virtual std::optional<imu_bootstrap_solution_t> solve(
      vision_bootstrap_solution_t const& sfm_solution,
      measurement::imu_motion_refs_t const& imu_motions) = 0;

    static std::unique_ptr<IMUBootstrapSolver> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
