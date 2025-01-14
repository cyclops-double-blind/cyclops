#pragma once

#include <memory>

namespace cyclops {
  struct cyclops_global_config_t;
}  // namespace cyclops

namespace cyclops::initializer {
  struct imu_match_translation_solution_t;
  struct imu_match_translation_uncertainty_t;

  class IMUTranslationMatchAcceptDiscriminator {
  public:
    virtual ~IMUTranslationMatchAcceptDiscriminator() = default;
    virtual void reset() = 0;

    enum decision_t {
      ACCEPT,
      REJECT_COST_PROBABILITY_INSIGNIFICANT,
      REJECT_UNDERINFORMATIVE_PARAMETER,
      REJECT_SCALE_LESS_THAN_ZERO,
    };

    virtual decision_t determineCandidate(
      imu_match_translation_solution_t const& solution,
      imu_match_translation_uncertainty_t const& uncertainty) const = 0;
    virtual decision_t determineAccept(
      imu_match_translation_solution_t const& solution,
      imu_match_translation_uncertainty_t const& uncertainty) const = 0;

    static std::unique_ptr<IMUTranslationMatchAcceptDiscriminator> create(
      std::shared_ptr<cyclops_global_config_t const> config);
  };
}  // namespace cyclops::initializer
