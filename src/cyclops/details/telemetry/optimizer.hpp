#pragma once

#include <memory>

namespace cyclops::telemetry {
  class OptimizerTelemetry {
  public:
    virtual ~OptimizerTelemetry() = default;
    virtual void reset();

    struct sanity_statistics_t {
      double final_cost;
      double final_cost_significant_probability;

      size_t landmark_observations;
      double landmark_accept_rate;
      double landmark_uninitialized_rate;
      double landmark_depth_threshold_failure_rate;
      double landmark_chi_square_test_failure_rate;
    };
    virtual void onSanityStatistics(sanity_statistics_t const& statistics);

    struct bad_reason_t {
      bool bad_landmark_update;
      bool bad_final_cost;
    };
    virtual void onSanityBad(
      bad_reason_t reason, sanity_statistics_t const& statistics);

    struct failure_reason_t {
      bool continued_bad_landmark_update;
      bool continued_bad_final_cost;
    };
    virtual void onSanityFailure(
      failure_reason_t reason, sanity_statistics_t const& statistics);

    virtual void onUserResetRequest();

    static std::unique_ptr<OptimizerTelemetry> createDefault();
  };
}  // namespace cyclops::telemetry
