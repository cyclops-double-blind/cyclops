#pragma once

#include <Eigen/Dense>
#include <optional>

namespace cyclops {
  struct cyclops_global_config_t;
}  // namespace cyclops

namespace cyclops::initializer {
  struct imu_match_translation_analysis_t;
  struct imu_match_scale_sample_solution_t;

  struct imu_match_translation_uncertainty_t {
    double final_cost_significant_probability;
    double scale_log_deviation;
    Eigen::Vector2d gravity_tangent_deviation;
    Eigen::Vector3d bias_deviation;
    Eigen::VectorXd body_velocity_deviation;
    Eigen::VectorXd translation_scale_symmetric_deviation;
  };

  std::optional<imu_match_translation_uncertainty_t>
  imu_match_analyze_translation_uncertainty(
    imu_match_translation_analysis_t const& analysis,
    imu_match_scale_sample_solution_t const& solution);
}  // namespace cyclops::initializer
