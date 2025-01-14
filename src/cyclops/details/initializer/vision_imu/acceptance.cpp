#include "cyclops/details/initializer/vision_imu/acceptance.hpp"
#include "cyclops/details/initializer/vision_imu/translation.hpp"
#include "cyclops/details/initializer/vision_imu/uncertainty.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>

namespace cyclops::initializer {
  static bool check_percent_threshold(
    std::string tag, double value, double threshold) {
    if (value > threshold) {
      __logger__->info(
        "IMU match: {} is uncertain. estimated uncertainty: {}% > {}%.", tag,
        100 * value, 100 * threshold);
      return true;
    }
    return false;
  }

  static auto max_velocity(imu_match_translation_solution_t const& solution) {
    double result = 1e-6;
    for (auto const& [_, v] : solution.imu_body_velocities)
      result = std::max<double>(v.norm(), result);
    return result;
  }

  class IMUTranslationMatchAcceptDiscriminatorImpl:
      public IMUTranslationMatchAcceptDiscriminator {
  private:
    std::shared_ptr<cyclops_global_config_t const> _config;

  public:
    explicit IMUTranslationMatchAcceptDiscriminatorImpl(
      std::shared_ptr<cyclops_global_config_t const> config)
        : _config(config) {
    }
    void reset() override;

    decision_t determineCandidate(
      imu_match_translation_solution_t const& solution,
      imu_match_translation_uncertainty_t const& uncertainty) const override;
    decision_t determineAccept(
      imu_match_translation_solution_t const& solution,
      imu_match_translation_uncertainty_t const& uncertainty) const override;
  };

  void IMUTranslationMatchAcceptDiscriminatorImpl::reset() {
    // Does nothing
  }

  IMUTranslationMatchAcceptDiscriminator::decision_t
  IMUTranslationMatchAcceptDiscriminatorImpl::determineCandidate(
    imu_match_translation_solution_t const& solution,
    imu_match_translation_uncertainty_t const& uncertainty) const {
    if (solution.scale <= 0) {
      __logger__->debug("IMU match scale less than zero");
      return REJECT_SCALE_LESS_THAN_ZERO;
    }

    auto P = uncertainty.final_cost_significant_probability;
    auto rho = _config->initialization.imu.candidate_test.cost_significance;
    if (P < rho) {
      __logger__->debug(
        "Unlikely large IMU match candidate final cost. p value: {} < {}", P,
        rho);
      return REJECT_COST_PROBABILITY_INSIGNIFICANT;
    }

    return ACCEPT;
  }

  IMUTranslationMatchAcceptDiscriminator::decision_t
  IMUTranslationMatchAcceptDiscriminatorImpl::determineAccept(
    imu_match_translation_solution_t const& solution,
    imu_match_translation_uncertainty_t const& uncertainty) const {
    if (solution.scale <= 0) {
      __logger__->debug("IMU match scale less than zero");
      return REJECT_SCALE_LESS_THAN_ZERO;
    }

    auto const& threshold = _config->initialization.imu.acceptance_test;
    auto P = uncertainty.final_cost_significant_probability;
    auto rho = threshold.translation_match_min_p_value;
    if (P < rho) {
      __logger__->debug(
        "Unlikely large IMU match final solution cost. p value: {} < {}", P,
        rho);
      return REJECT_COST_PROBABILITY_INSIGNIFICANT;
    }

    auto sigma_s = uncertainty.scale_log_deviation;
    auto epsilon_s = threshold.max_scale_log_deviation;
    if (check_percent_threshold("Scale", sigma_s, epsilon_s))
      return REJECT_UNDERINFORMATIVE_PARAMETER;

    auto gravity = _config->gravity_norm;
    auto sigma_g = uncertainty.gravity_tangent_deviation(0) / gravity;
    auto epsilon_g = threshold.max_normalized_gravity_deviation;
    if (check_percent_threshold("Gravity direction", sigma_g, epsilon_g))
      return REJECT_UNDERINFORMATIVE_PARAMETER;

    auto sigma_v =
      uncertainty.body_velocity_deviation(1) / max_velocity(solution);
    auto epsilon_v = threshold.max_normalized_velocity_deviation;
    if (check_percent_threshold("Velocity", sigma_v, epsilon_v))
      return REJECT_UNDERINFORMATIVE_PARAMETER;

    return ACCEPT;
  }

  std::unique_ptr<IMUTranslationMatchAcceptDiscriminator>
  IMUTranslationMatchAcceptDiscriminator::create(
    std::shared_ptr<cyclops_global_config_t const> config) {
    return std::make_unique<IMUTranslationMatchAcceptDiscriminatorImpl>(config);
  }
}  // namespace cyclops::initializer
