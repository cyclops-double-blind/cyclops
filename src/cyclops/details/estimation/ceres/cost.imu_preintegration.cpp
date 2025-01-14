#include "cyclops/details/estimation/ceres/cost.imu_preintegration.hpp"

namespace cyclops::estimation {
  using Matrix9d = Eigen::Matrix<double, 9, 9>;

  static Matrix9d make_imu_preintegration_residual_weight(
    measurement::IMUPreintegration const* data) {
    return Eigen::LLT<Matrix9d>(data->covariance.inverse()).matrixU();
  }

  IMUPreintegrationCostEvaluator::IMUPreintegrationCostEvaluator(
    measurement::IMUPreintegration const* data, double gravity)
      : data(data),
        weight(make_imu_preintegration_residual_weight(data)),
        gravity(gravity) {
  }
}  // namespace cyclops::estimation
