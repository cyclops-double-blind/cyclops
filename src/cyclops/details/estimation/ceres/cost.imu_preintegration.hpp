#pragma once

#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/measurement/preintegration.ipp"
#include "cyclops/details/utils/math.hpp"

#include <Eigen/Dense>

namespace cyclops::estimation {
  struct IMUPreintegrationCostEvaluator {
    measurement::IMUPreintegration const* data;
    Eigen::Matrix<double, 9, 9> const weight;

    double const gravity;

    IMUPreintegrationCostEvaluator(
      measurement::IMUPreintegration const* data, double gravity);

    template <typename scalar_t>
    bool operator()(
      scalar_t const* const x0, scalar_t const* const x1,
      scalar_t const* const b, scalar_t* const r_buffer) const {
      using vector9_t = Eigen::Matrix<scalar_t, 9, 1>;
      using matrix9_t = Eigen::Matrix<scalar_t, 9, 9>;

      using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
      using quaternion_t = Eigen::Quaternion<scalar_t>;

      using vector3_cmap_t = Eigen::Map<vector3_t const>;
      using quaternion_cmap_t = Eigen::Map<quaternion_t const>;

      quaternion_cmap_t q0(x0);
      vector3_cmap_t p0(x0 + 4);
      vector3_cmap_t v0(x0 + 7);

      quaternion_cmap_t q1(x1);
      vector3_cmap_t p1(x1 + 4);
      vector3_cmap_t v1(x1 + 7);

      auto g = vector3_t(scalar_t(0), scalar_t(0), scalar_t(gravity));
      auto dt = scalar_t(data->time_delta);
      auto half_dt2 = dt * dt / scalar_t(2);

      quaternion_t y_q = q0.conjugate() * q1;
      vector3_t y_p = q0.conjugate() * (p1 - p0 - v0 * dt + g * half_dt2);
      vector3_t y_v = q0.conjugate() * (v1 - v0 + g * dt);

      vector3_t b_a(b);
      vector3_t b_w(b + 3);

      auto r = Eigen::Map<vector9_t>(r_buffer);
      auto S = static_cast<matrix9_t>(weight.cast<scalar_t>());
      r = S * data->evaluateError<scalar_t>(y_q, y_p, y_v, b_a, b_w);
      return true;
    }
  };
}  // namespace cyclops::estimation
