#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"
#include <memory>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::measurement {
  class MeasurementDataProvider;
}

namespace cyclops::initializer {
  class InitializerMain;
}

namespace cyclops::estimation {
  class StateVariableReadAccessor;

  class OptimizerSolutionGuessPredictor {
  public:
    virtual ~OptimizerSolutionGuessPredictor() = default;
    virtual void reset() = 0;

    struct solution_t {
      motion_frame_parameter_blocks_t motions;
      landmark_parameter_blocks_t landmarks;
    };
    virtual std::optional<solution_t> solve() = 0;

    static std::unique_ptr<OptimizerSolutionGuessPredictor> create(
      std::unique_ptr<initializer::InitializerMain> initializer,
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<StateVariableReadAccessor const> state,
      std::shared_ptr<measurement::MeasurementDataProvider> measurement);
  };
}  // namespace cyclops::estimation
