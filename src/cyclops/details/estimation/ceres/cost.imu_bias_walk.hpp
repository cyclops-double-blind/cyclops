#pragma once

#include <Eigen/Dense>

namespace cyclops::estimation {
  struct IMUBiasRandomWalkCostEvaluator {
    double _dt;
    double _acc_walk_stddev;
    double _gyr_walk_stddev;

    IMUBiasRandomWalkCostEvaluator(
      double dt, double acc_walk_stddev, double gyr_walk_stddev)
        : _dt(std::max(dt, 0.)),
          _acc_walk_stddev(acc_walk_stddev),
          _gyr_walk_stddev(gyr_walk_stddev) {
    }

    template <typename scalar_t>
    bool operator()(
      scalar_t const* const b0, scalar_t const* const b1,
      scalar_t* const r) const {
      using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
      auto r_acc = Eigen::Map<vector3_t>(r + 0);
      auto r_gyr = Eigen::Map<vector3_t>(r + 3);

      auto sqrt_dt = std::sqrt(_dt);
      r_acc =
        (vector3_t(b1 + 0) - vector3_t(b0 + 0)) / sqrt_dt / _acc_walk_stddev;
      r_gyr =
        (vector3_t(b1 + 3) - vector3_t(b0 + 3)) / sqrt_dt / _gyr_walk_stddev;
      return true;
    }
  };
}  // namespace cyclops::estimation
