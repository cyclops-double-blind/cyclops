#pragma once

#include "cyclops_tests/signal.hpp"

namespace cyclops {
  template <typename vector_t>
  static auto numeric_derivative(
    std::function<vector_t(timestamp_t)> f, double const h = 1e-6) {
    return [f, h](timestamp_t const t) -> vector_t {
      return (f(t + h) - f(t - h)) / (2 * h);
    };
  }

  template <>
  auto numeric_derivative<Eigen::Quaterniond>(
    std::function<Eigen::Quaterniond(timestamp_t)> f, double const h) {
    return [f, h](timestamp_t const t) -> Eigen::Vector3d {
      auto const q1 = f(t - h);
      auto const q2 = f(t + h);
      return (q1.inverse() * q2).vec() / h;
    };
  }

  template <typename vector_t>
  static auto numeric_second_derivative(
    std::function<vector_t(timestamp_t)> f, double const h = 1e-6) {
    return [f, h](timestamp_t const t) -> vector_t {
      return (f(t + h) - 2 * f(t) + f(t - h)) / h / h;
    };
  }

  static quaternion_signal_t yaw_rotation(scalar_signal_t phi) {
    return [phi](timestamp_t t) -> Eigen::Quaterniond {
      return Eigen::Quaterniond(
        Eigen::AngleAxisd(phi(t), Eigen::Vector3d::UnitZ()));
    };
  }

  static quaternion_signal_t operator>>=(
    quaternion_signal_t q1, quaternion_signal_t q2) {
    return
      [q1, q2](timestamp_t t) -> Eigen::Quaterniond { return q1(t) * q2(t); };
  }

  static quaternion_signal_t just(Eigen::Quaterniond const& q) {
    return [q](timestamp_t _) -> Eigen::Quaterniond { return q; };
  }
}  // namespace cyclops
