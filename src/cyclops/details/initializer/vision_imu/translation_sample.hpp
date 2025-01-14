#pragma once

#include "cyclops/details/type.hpp"

#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  struct imu_match_translation_analysis_t;
  struct IMUMatchTranslationAnalysisCache;

  struct imu_match_scale_sample_solution_t {
    double scale;
    double cost;

    Eigen::VectorXd inertial_state;
    Eigen::VectorXd visual_state;

    Eigen::MatrixXd hessian;
  };

  class IMUMatchScaleSampleSolver {
  public:
    virtual ~IMUMatchScaleSampleSolver() = default;
    virtual void reset() = 0;

    virtual std::optional<std::vector<imu_match_scale_sample_solution_t>> solve(
      std::set<frame_id_t> const& motion_frames,
      imu_match_translation_analysis_t const& analysis,
      IMUMatchTranslationAnalysisCache const& cache) const = 0;

    static std::unique_ptr<IMUMatchScaleSampleSolver> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
