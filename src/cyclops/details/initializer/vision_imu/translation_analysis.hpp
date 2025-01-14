#pragma once

#include "cyclops/details/measurement/type.hpp"

#include <Eigen/Dense>
#include <memory>

namespace cyclops {
  struct cyclops_global_config_t;
}  // namespace cyclops

namespace cyclops::initializer {
  struct imu_match_camera_translation_prior_t;
  struct imu_match_rotation_solution_t;

  /*
   * represents biquadratic translational visual-inertial matching cost,
   *
   * p* = min || [A_I,   B_I * s] * x + [ alpha + beta * s ] ||^2
   *      s,x || [  0,       A_V]       [        0         ] ||,
   *
   * where
   *   A_I: inertial weight
   *   B_I: translational weight
   *   A_V: vision weight
   *   alpha: inertial perturbation
   *   beta: translation perturbation,
   *
   *   x = [x_I; x_V],
   *   x_I = [g; db_a; v[1:n]],
   *   x_V = [dp[2:n]],
   *
   *   g: gravity
   *   db_a: accelerometer bias estimation error
   *   v[]: body velocities
   *   dp[]: visual SLAM position estimation errors.
   *
   *  x_V starts from dp[2] to handle global translation symmetry.
   */
  struct imu_match_translation_analysis_t {
    size_t frames_count;
    size_t residual_dimension;
    size_t parameter_dimension;

    Eigen::MatrixXd inertial_weight;
    Eigen::MatrixXd translational_weight;
    Eigen::MatrixXd visual_weight;

    Eigen::VectorXd inertial_perturbation;
    Eigen::VectorXd translation_perturbation;
  };

  class IMUMatchTranslationAnalyzer {
  public:
    virtual ~IMUMatchTranslationAnalyzer() = default;
    virtual void reset() = 0;

    virtual imu_match_translation_analysis_t analyze(
      measurement::imu_motion_refs_t const& motions,
      imu_match_rotation_solution_t const& rotations,
      imu_match_camera_translation_prior_t const& camera_prior) = 0;

    static std::unique_ptr<IMUMatchTranslationAnalyzer> create(
      std::shared_ptr<cyclops_global_config_t const> config);
  };
}  // namespace cyclops::initializer
