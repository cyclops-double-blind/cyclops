#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>

namespace cyclops::measurement {
  struct KeyframeManager;
}

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  class InitializationSolverInternal;

  struct initialization_solution_t {
    Eigen::Vector3d acc_bias;
    Eigen::Vector3d gyr_bias;
    landmark_positions_t landmarks;
    std::map<frame_id_t, imu_motion_state_t> motions;
  };

  class InitializerMain {
  public:
    virtual ~InitializerMain() = default;
    virtual void reset() = 0;

    virtual std::optional<initialization_solution_t> solve() = 0;

    static std::unique_ptr<InitializerMain> create(
      std::unique_ptr<InitializationSolverInternal> solver_internal,
      std::shared_ptr<measurement::KeyframeManager const> keyframe_manager,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
