#include "cyclops/details/telemetry/initializer.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

#include <sstream>

namespace cyclops::telemetry {
  namespace views = ranges::views;

  void InitializerTelemetry::reset() {
    // Nothing
  }

  void InitializerTelemetry::onImageObservabilityPretest(
    image_observability_pretest_t const& test) {
    // TODO
  }

  void InitializerTelemetry::onVisionFailure(
    vision_initialization_failure_t const& failure) {
    // TODO
  }

  void InitializerTelemetry::onBestTwoViewSelection(
    best_two_view_selection_t const& selection) {
    // TODO
  }

  void InitializerTelemetry::onTwoViewMotionHypothesis(
    two_view_motion_hypothesis_t const& hypothesis) {
    // TODO
  }

  void InitializerTelemetry::onTwoViewSolverSuccess(
    two_view_solver_success_t const& success) {
    // TODO
  }

  void InitializerTelemetry::onBundleAdjustmentSuccess(
    bundle_adjustment_solution_t const& solution) {
    // TODO
  }

  void InitializerTelemetry::onBundleAdjustmentSanity(
    bundle_adjustment_candidates_sanity_t const& sanity) {
    // TODO
  }

  template <typename container_t, typename element_format_t>
  static std::string formatlist(
    container_t const& container, element_format_t const& element_formatter) {
    std::ostringstream ss;

    ss << "[";
    for (auto const& elem : container | views::drop_last(1)) {
      ss << std::endl;
      element_formatter(ss, elem);
    }

    if (!container.empty()) {
      ss << std::endl;
      element_formatter(ss, container.back());
      ss << std::endl;
    }
    ss << "]";
    return ss.str();
  }

  static std::string format_costpoints(
    std::vector<std::tuple<double, double>> const& costs) {
    return formatlist(costs, [](auto& ss, auto const& point) {
      auto const& [scale, cost] = point;
      ss << "  " << scale << ": " << cost;
    });
  }

  void InitializerTelemetry::onIMUMatchAttempt(
    imu_match_attempt_t const& argument) {
    __logger__->debug("Attempting IMU match");
    __logger__->debug("Local minima: {}", format_costpoints(argument.minima));
  }

  static std::string format_reject_reason(
    InitializerTelemetry::imu_match_candidate_reject_reason_t reason) {
    switch (reason) {
    case InitializerTelemetry::UNCERTAINTY_EVALUATION_FAILED:
      return "Uncertainty evaluation failed";
    case InitializerTelemetry::COST_PROBABILITY_INSIGNIFICANT:
      return "Insignificant cost probability";
    case InitializerTelemetry::UNDERINFORMATIVE_PARAMETER:
      return "Under-informative parameter";
    case InitializerTelemetry::SCALE_LESS_THAN_ZERO:
      return "Scale less than zero";
    }
    return "Unknown";
  }

  void InitializerTelemetry::onIMUMatchAmbiguity(
    imu_match_ambiguity_t const& argument) {
    __logger__->debug("IMU match solution ambiguous");
    __logger__->debug("#solutions: {}", argument.solutions.size());
  }

  void InitializerTelemetry::onIMUMatchReject(
    imu_match_reject_t const& argument) {
    __logger__->debug(
      "Rejecting solution point: s = {}", argument.solution.scale);
    __logger__->debug("Reason: {}", format_reject_reason(argument.reason));
  }

  void InitializerTelemetry::onIMUMatchCandidateReject(
    imu_match_reject_t const& argument) {
    __logger__->debug(
      "Rejecting candidate point: s = {}", argument.solution.scale);
    __logger__->debug("Reason: {}", format_reject_reason(argument.reason));
  }

  void InitializerTelemetry::onIMUMatchAccept(
    imu_match_accept_t const& argument) {
    __logger__->debug(
      "Accepting solution point: s = {}", argument.solution.scale);
  }

  void InitializerTelemetry::onFailure(onfailure_argument_t const& argument) {
    __logger__->info("VIO initialization failed");

    if (argument.vision_solutions.empty()) {
      __logger__->debug("Reason: vision initialization failed");
      return;
    }

    if (argument.imu_solutions.empty()) {
      __logger__->debug("Reason: no IMU match candidate");
      return;
    }

    if (argument.imu_solutions.size() > 1) {
      __logger__->debug("Reason: ambiguous IMU match");
      __logger__->debug("#candidates: {}", argument.imu_solutions.size());
      return;
    }

    auto const& imu_solution = argument.imu_solutions.front();
    auto const& vision_solution =
      argument.vision_solutions.at(imu_solution.vision_solution_index);

    if (!vision_solution.acceptable) {
      __logger__->debug("Unacceptable vision solution");
      return;
    }

    if (!imu_solution.acceptable) {
      __logger__->debug("Unacceptable IMU solution");
      __logger__->debug("Scale: {}", imu_solution.scale);
      return;
    }

    if (imu_solution.keyframes.empty()) {
      __logger__->error("Empty motion in initialization solution");
      return;
    }
  }

  void InitializerTelemetry::onSuccess(onsuccess_argument_t const& argument) {
    __logger__->info("VIO initialization successed");
    __logger__->debug("scale: {}", argument.scale);
  }

  std::unique_ptr<InitializerTelemetry> InitializerTelemetry::createDefault() {
    return std::make_unique<InitializerTelemetry>();
  }
}  // namespace cyclops::telemetry
