#pragma once

#include <memory>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::measurement {
  class MeasurementDataQueue;
}

namespace cyclops::estimation {
  class FactorGraphInstance;
  class StateVariableReadAccessor;

  struct gaussian_prior_t;

  class MarginalizationManager {
  public:
    virtual ~MarginalizationManager() = default;
    virtual void reset() = 0;

    virtual void marginalize(FactorGraphInstance& graph_instance) = 0;
    virtual void marginalize() = 0;
    virtual gaussian_prior_t const& prior() const = 0;

    static std::unique_ptr<MarginalizationManager> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<StateVariableReadAccessor const> state,
      std::shared_ptr<measurement::MeasurementDataQueue> data_queue);
  };
}  // namespace cyclops::estimation
