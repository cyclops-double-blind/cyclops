#pragma once

#include "cyclops/details/type.hpp"
#include <Eigen/Dense>

namespace cyclops::estimation {
  struct LandmarkProjectionCostEvaluator {
    Eigen::Vector2d const u;
    Eigen::Matrix2d const weight_sqrt;

    se3_transform_t const& extrinsic;

    LandmarkProjectionCostEvaluator(
      feature_point_t const& feature, se3_transform_t const& extrinsic)
        : u(feature.point),
          weight_sqrt(Eigen::LLT<Eigen::Matrix2d>(feature.weight).matrixU()),
          extrinsic(extrinsic) {
    }

    template <typename scalar_t, int dim>
    using vector_t = Eigen::Matrix<scalar_t, dim, 1>;

    template <typename scalar_t>
    auto computeCameraPoint(
      scalar_t const* const x_b, scalar_t const* const f) const {
      using quaternion_t = Eigen::Quaternion<scalar_t>;
      using vector3_t = vector_t<scalar_t, 3>;
      quaternion_t const q_b(x_b);
      quaternion_t const q_bc = extrinsic.rotation.cast<scalar_t>();
      quaternion_t const q_c = q_b * q_bc;

      vector3_t const p_b(x_b + 4);
      vector3_t const p_bc = extrinsic.translation.cast<scalar_t>();
      vector3_t const p_c = p_b + q_b * p_bc;

      return (q_c.inverse() * (vector3_t(f) - p_c)).eval();
    }

    template <typename scalar_t>
    auto computeProjectionError(vector_t<scalar_t, 3> const& z) const {
      auto const d_min = scalar_t(1e-2);
      auto const d = z.z() < d_min ? d_min : z.z();

      auto const& S = weight_sqrt;
      auto u_hat = (z.template head<2>() / d).eval();
      return (S.cast<scalar_t>() * (u_hat - u.cast<scalar_t>())).eval();
    }

    template <typename scalar_t>
    bool operator()(
      scalar_t const* const x_b, scalar_t const* const f,
      scalar_t* const r) const {
      (Eigen::Map<vector_t<scalar_t, 2>>(r)) =
        computeProjectionError(computeCameraPoint(x_b, f));
      return true;
    }
  };
}  // namespace cyclops::estimation
