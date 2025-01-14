#pragma once

#include "cyclops/details/estimation/type.hpp"

#include <ceres/ceres.h>
#include <Eigen/Dense>

namespace cyclops::estimation {
  class GaussianPriorCost: public ceres::CostFunction {
    gaussian_prior_t _prior;

  public:
    explicit GaussianPriorCost(gaussian_prior_t prior);
    bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const override;
  };
}  // namespace cyclops::estimation
