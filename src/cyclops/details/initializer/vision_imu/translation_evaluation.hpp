#pragma once

#include <Eigen/Dense>
#include <optional>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::initializer {
  struct imu_match_translation_analysis_t;
  struct IMUMatchTranslationAnalysisCache;

  struct imu_match_scale_evaluation_t {
    double multiplier;
    double cost;

    /*
     * concatenation of gravity, acc bias error, and IMU body velocities.
     */
    Eigen::VectorXd inertial_solution;

    /*
     * perturbations of vision position estimation, omitting the first frame.
     * defined exponentially ignoring scale gauge; `p_i = p_hat_i + R_i * dp_i`.
     */
    Eigen::VectorXd visual_solution;
  };

  class IMUMatchScaleEvaluationContext {
  private:
    double const gravity_norm;
    imu_match_translation_analysis_t const& analysis;
    IMUMatchTranslationAnalysisCache const& cache;

  public:
    IMUMatchScaleEvaluationContext(
      double gravity_norm, imu_match_translation_analysis_t const& analysis,
      IMUMatchTranslationAnalysisCache const& cache);

    std::optional<imu_match_scale_evaluation_t> evaluate(double scale) const;

    double evaluateDerivative(
      imu_match_scale_evaluation_t const& evaluation, double scale) const;
    Eigen::MatrixXd evaluateHessian(
      imu_match_scale_evaluation_t const& evaluation, double scale) const;
  };
}  // namespace cyclops::initializer
