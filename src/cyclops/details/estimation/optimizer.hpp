#pragma once

#include "cyclops/details/estimation/sanity.hpp"
#include "cyclops/details/type.hpp"

#include <memory>
#include <set>
#include <optional>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::measurement {
  class MeasurementDataProvider;
}

namespace cyclops::estimation {
  struct gaussian_prior_t;

  class FactorGraphInstance;
  class StateVariableWriteAccessor;
  class OptimizerSolutionGuessPredictor;

  class LikelihoodOptimizer {
  public:
    virtual ~LikelihoodOptimizer() = default;
    virtual void reset() = 0;

    struct optimization_result_t {
      std::unique_ptr<FactorGraphInstance> graph;

      landmark_sanity_statistics_t landmark_sanity_statistics;
      optimizer_sanity_statistics_t optimizer_sanity_statistics;

      double solve_time;
      double optimizer_time;
      std::string optimizer_report;

      std::set<frame_id_t> motion_frames;
      std::set<landmark_id_t> active_landmarks;
      std::set<landmark_id_t> mapped_landmarks;
    };
    virtual std::optional<optimization_result_t> optimize(
      gaussian_prior_t const& prior) = 0;

    static std::unique_ptr<LikelihoodOptimizer> create(
      std::unique_ptr<OptimizerSolutionGuessPredictor> predictor,
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<StateVariableWriteAccessor> state_accessor,
      std::shared_ptr<measurement::MeasurementDataProvider> measurement);
  };
}  // namespace cyclops::estimation
