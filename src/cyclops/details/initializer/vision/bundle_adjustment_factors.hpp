#pragma once

#include "cyclops/details/type.hpp"
#include <ceres/ceres.h>

namespace cyclops::initializer {
  struct LandmarkProjectionCost: public ceres::SizedCostFunction<2, 7, 3> {
    Eigen::Vector2d const u;
    Eigen::Matrix2d const weight_sqrt;

    LandmarkProjectionCost(feature_point_t const& feature);
    bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const override;
  };

  struct BundleAdjustmentScaleConstraintVirtualCost {
    double const weight;
    explicit BundleAdjustmentScaleConstraintVirtualCost(double weight)
        : weight(weight) {
    }

    template <typename scalar_t>
    bool operator()(
      scalar_t const* const x0, scalar_t const* const xn,
      scalar_t* const r) const {
      using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;

      vector3_t const p0(x0 + 4);
      vector3_t const pn(xn + 4);

      *r = ((pn - p0).norm() - scalar_t(1.0)) * scalar_t(weight);
      return true;
    }
  };
}  // namespace cyclops::initializer
