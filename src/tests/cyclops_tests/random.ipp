#pragma once

#include "cyclops_tests/random.hpp"

namespace cyclops {
  template <int n, int m>
  static Eigen::Matrix<double, n, m> make_random_matrix(
    std::mt19937& rgen, double s) {
    std::normal_distribution<double> random(0, s);

    Eigen::Matrix<double, n, m> M;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        M(i, j) = random(rgen);
      }
    }
    return M;
  }

  static Eigen::MatrixXd make_random_matrix(
    std::mt19937& rgen, int n, int m, double s) {
    std::normal_distribution<double> random(0, s);

    Eigen::MatrixXd M(n, m);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        M(i, j) = random(rgen);
      }
    }
    return M;
  }
}  // namespace cyclops
