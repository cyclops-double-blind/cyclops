#pragma once

#include <Eigen/Dense>

namespace cyclops {
  struct qcqp1_solution_t {
    bool success;
    double multiplier;
    Eigen::Vector3d x;
  };

  qcqp1_solution_t solve_norm_constrained_qcqp1(
    Eigen::Matrix3d const& H, Eigen::Vector3d const& b, double norm_sqr,
    double multiplier_min, size_t max_iterations = 100,
    double constraint_violation_tolerance = 1e-6,
    double multiplier_safeguard_margin = 1e-6);
}  // namespace cyclops
