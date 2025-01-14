#pragma once

#include <memory>
#include <random>
#include <tuple>
#include <vector>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::measurement {
  struct MeasurementDataProvider;
}

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  struct vision_bootstrap_solution_t;
  struct imu_bootstrap_solution_t;

  struct initializer_internal_solution_t {
    std::vector<vision_bootstrap_solution_t> vision_solutions;

    std::vector<std::tuple<
      int,  // index of the vision solution used for solving the IMU match
      imu_bootstrap_solution_t>>
      imu_solutions;
  };

  class InitializationSolverInternal {
  public:
    virtual ~InitializationSolverInternal() = default;
    virtual void reset() = 0;

    virtual initializer_internal_solution_t solve() = 0;

    static std::unique_ptr<InitializationSolverInternal> create(
      std::shared_ptr<std::mt19937> rgen,
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<measurement::MeasurementDataProvider const> data_provider,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
