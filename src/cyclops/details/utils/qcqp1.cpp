#include "cyclops/details/utils/qcqp1.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>

namespace cyclops {
  using Eigen::Matrix3d;
  using Eigen::Vector3d;

  qcqp1_solution_t solve_norm_constrained_qcqp1(
    Matrix3d const& H, Vector3d const& b, double norm_sqr,
    double multiplier_min, size_t max_iterations,
    double constraint_violation_tolerance, double multiplier_safeguard_margin) {
    auto rho_curr = multiplier_min + 0.1;

    size_t i = 0;
    while (true) {
      Matrix3d H_damped = H;
      H_damped.diagonal() = H.diagonal().array() + rho_curr;

      Eigen::LDLT<Matrix3d> H_damped_inv(H_damped);

      auto x = (-0.5 * H_damped_inv.solve(b)).eval();
      auto phi = x.dot(x) - norm_sqr;
      auto phi_deriv = -2 * x.dot(H_damped_inv.solve(x));

      rho_curr = std::max(
        multiplier_min + multiplier_safeguard_margin,
        rho_curr - phi / phi_deriv);

      if (std::abs(phi) < constraint_violation_tolerance) {
        return {true, rho_curr, x};
      }

      if (i >= max_iterations) {
        __logger__->debug("Norm constrained qcqp failed. ");
        __logger__->debug("Iterations = {}, constraint violation = {}", i, phi);
        return {false, rho_curr, x};
      }

      i++;
    }
  }
}  // namespace cyclops
