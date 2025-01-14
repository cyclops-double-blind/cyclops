#pragma once

#include <memory>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::telemetry {
  struct OptimizerTelemetry;
}

namespace cyclops::estimation {
  struct landmark_sanity_statistics_t {
    size_t landmark_observations;
    size_t landmark_accepts;
    size_t uninitialized_landmarks;
    size_t depth_threshold_failures;
    size_t mnorm_threshold_failures;
  };

  struct optimizer_sanity_statistics_t {
    double final_cost;
    int num_residuals;
    int num_parameters;
  };

  class EstimationSanityDiscriminator {
  public:
    virtual ~EstimationSanityDiscriminator() = default;

    virtual void reset() = 0;
    virtual void update(
      landmark_sanity_statistics_t const& landmark_sanity,
      optimizer_sanity_statistics_t const& optimizer_sanity) = 0;

    virtual bool sanity() const = 0;

    static std::unique_ptr<EstimationSanityDiscriminator> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<telemetry::OptimizerTelemetry> telemetry = nullptr);
  };
}  // namespace cyclops::estimation
