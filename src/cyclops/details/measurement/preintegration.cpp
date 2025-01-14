#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/utils/math.hpp"

namespace cyclops::measurement {
  using Eigen::AngleAxisd;
  using Eigen::Matrix;
  using Eigen::Matrix3d;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix9d = Eigen::Matrix<double, 9, 9>;
  using Vector12d = Eigen::Matrix<double, 12, 1>;

  static AngleAxisd exp(Vector3d const& v) {
    double const w = v.norm();
    if (w == 0)
      return AngleAxisd(w, Vector3d::UnitZ());
    return AngleAxisd(w, v.normalized());
  }

  static int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++)
      result *= i;
    return result;
  }

  struct imu_single_step_propagation_t {
    Quaterniond rotation_increment;
    Vector3d position_increment;
    Vector3d velocity_increment;
  };

  static imu_single_step_propagation_t propagate_single_step(
    double dt, Vector3d const& a, Vector3d const& w) {
    auto theta = w.norm() * dt;
    auto S = skew3d((w * dt).eval());
    auto S_2 = (S * S).eval();

    auto [alpha, beta, gamma] = [theta]() {
      if (theta > 0.1) {
        auto alpha = (1 - cos(theta)) / theta / theta;
        auto beta = (theta - sin(theta)) / theta / theta / theta;
        auto gamma = (cos(theta) - 1 + 0.5 * theta * theta) / theta / theta /
          theta / theta;
        return std::make_tuple(alpha, beta, gamma);
      }

      auto compute_alternating_taylor_sum = [theta](auto start, auto order) {
        auto x = 1. / factorial(start);
        auto result = x;
        for (int n = 0; n < order; n++) {
          x *= -theta * theta / (start + 2 * n + 1) / (start + 2 * n + 2);
          result += x;
        }
        return result;
      };
      auto alpha = compute_alternating_taylor_sum(2, 5);
      auto beta = compute_alternating_taylor_sum(3, 5);
      auto gamma = compute_alternating_taylor_sum(4, 4);
      return std::make_tuple(alpha, beta, gamma);
    }();

    auto V1 = (Matrix3d::Identity() + alpha * S + beta * S_2).eval();
    auto V2 = (Matrix3d::Identity() + 2 * beta * S + 2 * gamma * S_2).eval();
    return {
      .rotation_increment = Quaterniond(exp(w * dt)),
      .position_increment = V2 * 0.5 * a * dt * dt,
      .velocity_increment = V1 * a * dt,
    };
  }

  void IMUPreintegration::integrate(imu_data_t const& data) {
    auto dt = data.dt;
    time_delta += dt;

    auto a_bar = (data.a - _b_a).eval();
    auto w_bar = (data.w - _b_w).eval();

    auto& y_q = rotation_delta;
    auto& y_p = position_delta;
    auto& y_v = velocity_delta;

    // local increments of the preintegration delta.
    auto [dy_q, dy_p, dy_v] = propagate_single_step(dt, a_bar, w_bar);
    auto dy_R_T = dy_q.matrix().transpose().eval();

    y_q = y_q * dy_q;
    y_p = y_p + y_v * dt + y_q * dy_p;
    y_v = y_v + y_q * dy_v;

#define I3 (Matrix3d::Identity())
    auto F = Matrix<double, 9, 9>::Zero().eval();
    F.block<3, 3>(0, 0) = dy_R_T;
    F.block<3, 3>(3, 0) = -dy_R_T * skew3d(dy_p);
    F.block<3, 3>(3, 3) = dy_R_T;
    F.block<3, 3>(3, 6) = dy_R_T * dt;
    F.block<3, 3>(6, 0) = -dy_R_T * skew3d(dy_v);
    F.block<3, 3>(6, 6) = dy_R_T;
#undef I3
    updateJacobians(F, dt);
    updateCovariance(F, dt);
  }

  void IMUPreintegration::updateJacobians(
    Eigen::Matrix<double, 9, 9> const& F, double dt) {
    auto& G = bias_jacobian;

    G = F * G;
    G.block<3, 3>(0, 3) -= Matrix3d::Identity() * dt;
    G.block<3, 3>(6, 0) -= Matrix3d::Identity() * dt;
  }

  void IMUPreintegration::updateCovariance(
    Eigen::Matrix<double, 9, 9> const& F, double dt) {
    auto const& n_a = _noise.acc_white_noise;
    auto const& n_w = _noise.gyr_white_noise;

#define I3 (Matrix3d::Identity())
    auto N = Matrix<double, 9, 6>::Zero().eval();
    N.block<3, 3>(0, 3) = I3 * dt;
    N.block<3, 3>(6, 0) = I3 * dt;
#undef I3

    Vector6d Q;
    // clang-format off
    Q <<
      n_a * n_a, n_a * n_a, n_a * n_a,
      n_w * n_w, n_w * n_w, n_w * n_w
    ;
    // clang-format on
    covariance =
      F * covariance * F.transpose() + N * Q.asDiagonal() * N.transpose();
  }

  void IMUPreintegration::reset() {
    time_delta = 0;
    rotation_delta.setIdentity();
    position_delta.setZero();
    velocity_delta.setZero();
    covariance.setZero();
    bias_jacobian.setZero();
  }

  void IMUPreintegration::propagate(
    double dt, Vector3d const& a_hat, Vector3d const& w_hat) {
    _history.push_back({dt, a_hat, w_hat});
    integrate(_history.back());
  }

  void IMUPreintegration::repropagate() {
    reset();
    for (auto const& data : _history)
      integrate(data);
  }

  void IMUPreintegration::updateBias(Vector3d const& b_a, Vector3d const& b_w) {
    _b_a = b_a;
    _b_w = b_w;
    repropagate();
  }

  Vector3d const& IMUPreintegration::accBias() const {
    return _b_a;
  }

  Vector3d const& IMUPreintegration::gyrBias() const {
    return _b_w;
  }
}  // namespace cyclops::measurement
