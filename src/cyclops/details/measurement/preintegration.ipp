#pragma once

#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/utils/math.hpp"

namespace cyclops::measurement {
  template <typename T>
  IMUPreintegration::Vector9<T> IMUPreintegration::evaluateError(
    Quaternion<T> const& y_q, Vector3<T> const& y_p, Vector3<T> const& y_v,
    Vector3<T> const& b_a, Vector3<T> const& b_w) const {
    using Matrix3 = Eigen::Matrix<T, 3, 3>;
    using Matrix9x6 = Eigen::Matrix<T, 9, 6>;

    Quaternion<T> y_q_bar = rotation_delta.cast<T>();
    Vector3<T> y_p_bar = position_delta.cast<T>();
    Vector3<T> y_v_bar = velocity_delta.cast<T>();

    Vector3<T> db_a = (b_a - accBias().cast<T>()).eval();
    Vector3<T> db_w = (b_w - gyrBias().cast<T>()).eval();

    Quaternion<T> dy_q = y_q_bar.conjugate() * y_q;
    Vector3<T> dy_p = y_q_bar.conjugate() * (y_p - y_p_bar);
    Vector3<T> dy_v = y_q_bar.conjugate() * (y_v - y_v_bar);

    Vector3<T> delta_theta = so3_logmap(dy_q);
    Matrix3 N_inv = so3_left_jacobian_inverse(delta_theta);

    Vector3<T> delta_p = N_inv * dy_p;
    Vector3<T> delta_v = N_inv * dy_v;

    Vector9<T> r;
    r << delta_theta, delta_p, delta_v;

    Matrix9x6 G = bias_jacobian.cast<T>();
    r -= G.middleCols(0, 3) * db_a;
    r -= G.middleCols(3, 3) * db_w;

    return r;
  }
}  // namespace cyclops::measurement
