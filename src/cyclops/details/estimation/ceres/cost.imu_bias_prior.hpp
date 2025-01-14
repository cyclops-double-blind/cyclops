#pragma once

#include <Eigen/Dense>

namespace cyclops::estimation {
  struct IMUBiasPriorCostEvaluator {
    double _acc_prior_stddev;
    double _gyr_prior_stddev;

    IMUBiasPriorCostEvaluator(double acc_prior_stddev, double gyr_prior_stddev)
        : _acc_prior_stddev(acc_prior_stddev),
          _gyr_prior_stddev(gyr_prior_stddev) {
    }

    template <typename scalar_t>
    bool operator()(scalar_t const* const b, scalar_t* const r) const {
      using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
      auto r_acc = Eigen::Map<vector3_t>(r + 0);
      auto r_gyr = Eigen::Map<vector3_t>(r + 3);

      r_acc = vector3_t(b + 0) / _acc_prior_stddev;
      r_gyr = vector3_t(b + 3) / _gyr_prior_stddev;
      return true;
    }
  };
}  // namespace cyclops::estimation
