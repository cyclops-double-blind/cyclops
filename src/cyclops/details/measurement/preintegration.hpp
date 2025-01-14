#pragma once

#include <Eigen/Dense>

#include <memory>
#include <vector>

namespace cyclops::measurement {
  struct imu_noise_t {
    double acc_white_noise;
    double gyr_white_noise;
  };

  class IMUPreintegration {
  private:
    imu_noise_t const _noise;

    Eigen::Vector3d _b_a;
    Eigen::Vector3d _b_w;

    struct imu_data_t {
      double dt;
      Eigen::Vector3d a;
      Eigen::Vector3d w;
    };
    std::vector<imu_data_t> _history;

    void integrate(imu_data_t const& data);
    void reset();
    void repropagate();
    void updateCovariance(Eigen::Matrix<double, 9, 9> const& F, double dt);
    void updateJacobians(Eigen::Matrix<double, 9, 9> const& F, double dt);

    template <typename scalar_t>
    using Vector3 = Eigen::Matrix<scalar_t, 3, 1>;

    template <typename scalar_t>
    using Vector9 = Eigen::Matrix<scalar_t, 9, 1>;

    template <typename scalar_t>
    using Quaternion = Eigen::Quaternion<scalar_t>;

  public:
    using UniquePtr = std::unique_ptr<IMUPreintegration>;

    IMUPreintegration(
      Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
      imu_noise_t const& noise)
        : _noise(noise), _b_a(bias_acc), _b_w(bias_gyr) {
      reset();
    }

    double time_delta;

    Eigen::Quaterniond rotation_delta;
    Eigen::Vector3d position_delta;
    Eigen::Vector3d velocity_delta;

    Eigen::Matrix<double, 9, 9> covariance;
    Eigen::Matrix<double, 9, 6> bias_jacobian;

    void propagate(
      double dt, Eigen::Vector3d const& a_hat, Eigen::Vector3d const& w_hat);
    void updateBias(Eigen::Vector3d const& b_a, Eigen::Vector3d const& b_w);

    Eigen::Vector3d const& accBias() const;
    Eigen::Vector3d const& gyrBias() const;

    template <typename T>
    Vector9<T> evaluateError(
      Quaternion<T> const& y_q, Vector3<T> const& y_p, Vector3<T> const& y_v,
      Vector3<T> const& b_a, Vector3<T> const& b_w) const;
  };
}  // namespace cyclops::measurement
