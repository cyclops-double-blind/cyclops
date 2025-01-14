#pragma once

#include <Eigen/Dense>

#include <memory>

namespace cyclops::initializer {
  struct imu_match_translation_analysis_t;

  class IMUMatchTranslationAnalysisCache {
  private:
    struct Impl;
    std::unique_ptr<Impl> _pimpl;

  public:
    explicit IMUMatchTranslationAnalysisCache(
      imu_match_translation_analysis_t const& analysis);
    ~IMUMatchTranslationAnalysisCache();

    /*
     * represents the following Schur-decomposed system of linear equations,
     * 1. (H_I_bar + mu * C_I) * x_I + b_I_bar = 0,
     * 2. x_V = -(F_V * x_I + z_V) * s,
     *
     * that is originated from the following system of linear equations,
     *                  (H(s) + mu * C_g) * x + b(s) = 0
     *                                <=>
     * [H_I + mu * C_I,           F_I * s] * [ x_I ] + [ b_I(s) ]  = [ 0 ]
     * [F_I.T * s,        H_V + D_I * s^2]   [ x_V ]   [ b_V(s) ]    [ 0 ].
     */
    struct primal_cache_inflation_t {
      Eigen::MatrixXd H_I_bar;
      Eigen::VectorXd b_I_bar;
      Eigen::MatrixXd F_V;
      Eigen::VectorXd z_V;
    };
    primal_cache_inflation_t inflatePrimal(double scale) const;

    struct derivative_cache_inflation_t {
      double r_s__dot;
      Eigen::VectorXd b_I_s__dot;
      Eigen::VectorXd b_V_s__dot;
      Eigen::MatrixXd F_I;
      Eigen::MatrixXd D_I;
    };
    derivative_cache_inflation_t inflateDerivative(double scale) const;
  };
}  // namespace cyclops::initializer
