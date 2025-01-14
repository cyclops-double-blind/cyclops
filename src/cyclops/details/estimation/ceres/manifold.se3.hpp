#pragma once

#include <Eigen/Dense>

namespace cyclops::estimation {
  /*
   * [1] A. Barrau and S. Bonnabel, "A Mathematical Framework for IMU Error
   *     Propagation with Applications to Preintegration", ICRA 2020.
   * [2] A. Barrau, "Non-linear state error based extended Kalman filters with
   *     applications to navigation", PhD thesis, 2015.
   */
  template <bool gauge_constraint, bool extended = true>
  struct ExponentialSE3Plus {
    template <typename scalar_t>
    bool operator()(
      scalar_t const* x, scalar_t const* delta, scalar_t* result) const {
      using Eigen::Map;
      using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
      using matrix3_t = Eigen::Matrix<scalar_t, 3, 3>;
      using quaternion_t = Eigen::Quaternion<scalar_t>;

      auto const w_x = delta[0] / scalar_t(2);
      auto const w_y = delta[1] / scalar_t(2);
      auto const w_z = gauge_constraint ? scalar_t(0.) : delta[2] / scalar_t(2);
      auto const squared_theta = w_x * w_x + w_y * w_y + w_z * w_z;
      if (squared_theta != squared_theta)  // Check for NaN.
        return false;

      auto [a, b, q_delta] = [&]() {
        if (squared_theta <= scalar_t(0.)) {
          auto a = scalar_t(0.5);
          auto b = scalar_t(1. / 6.);
          return std::make_tuple(
            a, b, quaternion_t(scalar_t(1.0), w_x, w_y, w_z));
        }

        auto const theta = sqrt(squared_theta);
        auto const sin_theta_over_theta = sin(theta) / theta;

        // clang-format off
        auto const q_delta = quaternion_t(
          cos(theta),
          sin_theta_over_theta * w_x,
          sin_theta_over_theta * w_y,
          sin_theta_over_theta * w_z);
        // clang-format on

        if (theta < scalar_t(1e-2)) {
          // clang-format off
          auto const a =
            + scalar_t(0.5)
            - scalar_t(1. / 24.) * pow(theta, 2)
            + scalar_t(1. / 720.) * pow(theta, 4);
          auto const b =
            + scalar_t(1. / 6.)
            - scalar_t(1. / 120.) * pow(theta, 2)
            + scalar_t(1. / 5040.) * pow(theta, 4);
          // clang-format on
          return std::make_tuple(a, b, q_delta);
        }

        auto const a = (scalar_t(1.0) - cos(theta)) / theta / theta;
        auto const b = (theta - sin(theta)) / theta / theta / theta;
        return std::make_tuple(a, b, q_delta);
      }();

      // clang-format off
      matrix3_t const S_w =
        (matrix3_t() <<
          scalar_t(+0.0), -w_z, +w_y,
          +w_z, scalar_t(+0.0), -w_x,
          -w_y, +w_x, scalar_t(+0.0)
        ).finished();
      // clang-format on

      matrix3_t const N = matrix3_t::Identity() + a * S_w + b * S_w * S_w;

      Map<quaternion_t> q_result(result);
      Map<quaternion_t const> q(x);
      q_result = q * q_delta;

      Map<vector3_t> p_result(result + 4);
      p_result = Map<vector3_t const>(x + 4);
      if (!gauge_constraint)
        p_result += q * (N * (Map<vector3_t const>(delta + 3)));

      if (extended) {
        Map<vector3_t> v_result(result + 7);
        Map<vector3_t const> v(x + 7);
        Map<vector3_t const> dv(delta + (gauge_constraint ? 2 : 6));
        v_result = v + q * (N * dv);
      }

      return true;
    }
  };
}  // namespace cyclops::estimation
