#include "cyclops/details/telemetry/optimizer.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>

namespace cyclops::telemetry {
  void OptimizerTelemetry::reset() {
    // Nothing
  }

  void OptimizerTelemetry::onSanityStatistics(
    sanity_statistics_t const& statistics) {
    __logger__->info(
      "================== VIO optimizer sanity statistics: ==================");

    auto p_value = statistics.final_cost_significant_probability;
    auto n_landmarks = statistics.landmark_observations;
    auto r_landmark_accept = statistics.landmark_accept_rate;
    auto r_landmark_outlier = statistics.landmark_chi_square_test_failure_rate;

    __logger__->info(
      "Final cost significance probability: {} %", 100 * p_value);
    __logger__->info(  //
      "Landmark observations:               {}", n_landmarks);
    __logger__->info(
      "Landmark accept rate:                {}", r_landmark_accept);
    __logger__->info(
      "Landmark outlier rate:               {}", r_landmark_outlier);

    __logger__->info(
      "======================================================================");
  }

  static void print_bad_reason(
    bool landmark_bad, bool cost_bad,
    OptimizerTelemetry::sanity_statistics_t const& statistics) {
    int reasons_count = 0;
    if (landmark_bad) {
      __logger__->warn("Reason 1: bad landmark update");
      __logger__->warn(
        "Landmark accept rate: {}", statistics.landmark_accept_rate);

      reasons_count++;
    }
    if (cost_bad) {
      __logger__->warn("Reason {}: bad final cost", reasons_count + 1);
      __logger__->warn("Final cost: {}", statistics.final_cost);
      __logger__->warn(
        "Final cost significant probability: {}%",
        statistics.final_cost_significant_probability * 100);

      reasons_count++;
    }

    if (reasons_count == 0) {
      __logger__->error(
        "Internal error: VIO sanity is bad but no reason is reported");
    }
  }

  void OptimizerTelemetry::onSanityBad(
    bad_reason_t reason, sanity_statistics_t const& statistics) {
    __logger__->warn("VIO optimizer bad sanity.");

    print_bad_reason(
      reason.bad_landmark_update, reason.bad_final_cost, statistics);
  }

  void OptimizerTelemetry::onSanityFailure(
    failure_reason_t reason, sanity_statistics_t const& statistics) {
    __logger__->error("VIO optimization failed.");

    print_bad_reason(
      reason.continued_bad_landmark_update, reason.continued_bad_final_cost,
      statistics);
  }

  void OptimizerTelemetry::onUserResetRequest() {
    __logger__->error("User reset request");
  }

  std::unique_ptr<OptimizerTelemetry> OptimizerTelemetry::createDefault() {
    return std::make_unique<OptimizerTelemetry>();
  }
}  // namespace cyclops::telemetry
