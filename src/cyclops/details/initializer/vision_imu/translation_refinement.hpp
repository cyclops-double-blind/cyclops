#pragma once

#include <optional>
#include <memory>

namespace cyclops {
  struct cyclops_global_config_t;
}  // namespace cyclops

namespace cyclops::initializer {
  struct IMUMatchScaleEvaluationContext;

  struct imu_match_scale_refinement_t {
    double scale;
    double cost;
  };

  class IMUMatchTranslationLocalOptimizer {
  private:
    std::shared_ptr<cyclops_global_config_t const> _config;

  public:
    explicit IMUMatchTranslationLocalOptimizer(
      std::shared_ptr<cyclops_global_config_t const> config);
    ~IMUMatchTranslationLocalOptimizer();
    void reset();

    std::optional<imu_match_scale_refinement_t> optimize(
      IMUMatchScaleEvaluationContext const& evaluator, double s0);

    static std::unique_ptr<IMUMatchTranslationLocalOptimizer> create(
      std::shared_ptr<cyclops_global_config_t const> config);
  };
}  // namespace cyclops::initializer
