#include "cyclops/details/initializer/vision_imu/uncertainty.hpp"
#include "cyclops/details/initializer/vision_imu/translation.hpp"
#include "cyclops/details/initializer/vision_imu/translation_analysis.hpp"
#include "cyclops/details/initializer/vision_imu/translation_sample.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/logging.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace cyclops::initializer {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  /*
   * for positive-definite 2-by-2 block matrix H, of which components are
   * ```
   * (1) H = [H_a,    H_r;
   *          H_r^T,  H_b],
   * ```
   * compute and return a pair of Schur-complements <H_a/H_b, H_b/H_a>.
   *
   * H: positive definite matrix.
   * p: dimension of the upper left block.
   */
  static std::optional<std::tuple<MatrixXd, MatrixXd>>
  compute_marginal_information_pair(MatrixXd const& H, int p) {
    __logger__->trace("Computing marginal information pair");
    __logger__->trace("Matrix size: <{}, {}>", H.rows(), H.cols());
    __logger__->trace("Partition dimension: {}", p);

    int n = H.rows();
    int m = H.cols();

    if (n != m) {
      __logger__->error(
        "Input matrix is not square during marginal information pair "
        "computation");
      return std::nullopt;
    }

    if (n < p) {
      __logger__->error(
        "Partition dimension exceeds the matrix dimension during marginal "
        "information pair computation");
      return std::nullopt;
    }

    auto q = n - p;

#define H_a (H.topLeftCorner(p, p))
#define H_b (H.bottomRightCorner(q, q))
#define H_r (H.topRightCorner(p, q))

    Eigen::LLT<MatrixXd> H_b_llt(H_b);
    if (H_b_llt.info() != Eigen::Success) {
      __logger__->debug("Cholesky decomposition failed.");
      return std::nullopt;
    }
    MatrixXd H_a_bar = H_a - H_r * H_b_llt.solve(H_r.transpose());

    Eigen::LLT<MatrixXd> H_a_llt(H_a);
    if (H_a_llt.info() != Eigen::Success) {
      __logger__->debug("Cholesky decomposition failed.");
      return std::nullopt;
    }
    MatrixXd H_b_bar = H_b - H_r.transpose() * H_a_llt.solve(H_r);

#undef H_r
#undef H_b
#undef H_a

    return std::make_tuple(H_a_bar, H_b_bar);
  }

  static std::optional<VectorXd> compute_positive_eigenvalues(
    MatrixXd const& matrix) {
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigen(matrix);
    if (eigen.info() != Eigen::Success) {
      __logger__->debug(
        "Eigendecomposition failed during IMU match uncertainty analysis");
      return std::nullopt;
    }

    VectorXd lambda = eigen.eigenvalues();
    if (lambda.size() != 0 && lambda(0) <= 0) {
      __logger__->debug(
        "Semi-or-indefinite information matrix during IMU match uncertainty "
        "analysis");
      return std::nullopt;
    }

    return lambda;
  }

  static bool check_hessian_dimension(MatrixXd const& H, int frames) {
    if (H.rows() != H.cols()) {
      __logger__->error("IMU match hessian is not a square matrix");
      return false;
    }

    auto n = H.cols();
    if (n != 6 * frames + 3) {
      __logger__->error("IMU match hessian dimension mismatch");
      __logger__->debug("{} vs {}", n, 6 * frames + 3);
      return false;
    }
    return true;
  }

  std::optional<imu_match_translation_uncertainty_t>
  imu_match_analyze_translation_uncertainty(
    imu_match_translation_analysis_t const& analysis,
    imu_match_scale_sample_solution_t const& solution) {
    __logger__->debug("Analyzing the uncertainty of imu match");

    int degrees_of_freedom =
      analysis.residual_dimension - analysis.parameter_dimension;
    int frames = analysis.frames_count;

    if (degrees_of_freedom <= 0)
      return std::nullopt;

    auto const& [_1, _2, _3, A_I, B_I, A_V, alpha, beta] = analysis;
    auto s = solution.scale;
    auto const& x_I = solution.inertial_state;
    auto const& x_V = solution.visual_state;

    auto const& H = solution.hessian;
    if (!check_hessian_dimension(H, frames))
      return std::nullopt;

    auto r_I = (A_I * x_I + B_I * x_V * s + alpha + beta * s).eval();
    auto r_V = (A_V * x_V).eval();
    auto cost = r_I.dot(r_I) + r_V.dot(r_V);

    __logger__->debug("IMU match (s = {}) solution residual:", s);
    __logger__->debug("r_I = {}", r_I.transpose());
    __logger__->debug("r_V = {}", r_V.transpose());

    __logger__->debug("Degrees of freedom: {}", degrees_of_freedom);
    auto cost_probability = 1 - chi_squared_cdf(degrees_of_freedom, cost);

    auto marginalize_or_die = [](auto const& matrix, auto dim, auto tag) {
      auto result = compute_marginal_information_pair(matrix, dim);
      if (!result)
        __logger__->debug("{} marginalization failed", tag);
      return result;
    };

    auto scale_marginalization = marginalize_or_die(H, 1, "scale");
    if (!scale_marginalization)
      return std::nullopt;
    auto const& [H_s, H_x] = *scale_marginalization;

    auto gravity_marginalization = marginalize_or_die(H_x, 2, "gravity");
    if (!gravity_marginalization)
      return std::nullopt;
    auto const& [H_g, H_q] = *gravity_marginalization;

    auto bias_marginalization = marginalize_or_die(H_q, 3, "bias");
    if (!bias_marginalization)
      return std::nullopt;
    auto const& [H_b, H_vx] = *bias_marginalization;

    auto v_dim = 3 * frames;
    auto velocity_marginalization = marginalize_or_die(H_vx, v_dim, "velocity");
    if (!velocity_marginalization)
      return std::nullopt;
    auto const& [H_v, H_p] = *velocity_marginalization;

    auto lambda_s = H_s(0, 0);
    if (lambda_s <= 0) {
      __logger__->debug("Scale information is negative definite.");
      return std::nullopt;
    }

    auto eigenvalue_or_die = [](auto const& matrix, auto tag) {
      auto maybe_lambda = compute_positive_eigenvalues(matrix);
      if (!maybe_lambda)
        __logger__->debug("{} eigenvalue computation failed.", tag);
      return maybe_lambda;
    };
    auto maybe_lambda_g = eigenvalue_or_die(H_g, "Gravity");
    if (!maybe_lambda_g)
      return std::nullopt;

    auto maybe_lambda_b = eigenvalue_or_die(H_b, "Bias");
    if (!maybe_lambda_b)
      return std::nullopt;

    auto maybe_lambda_v = eigenvalue_or_die(H_v, "Velocity");
    if (!maybe_lambda_v)
      return std::nullopt;

    auto maybe_lambda_p = eigenvalue_or_die(H_p, "Position");
    if (!maybe_lambda_p)
      return std::nullopt;

    return imu_match_translation_uncertainty_t {
      .final_cost_significant_probability = cost_probability,
      .scale_log_deviation = 1 / std::sqrt(lambda_s),
      .gravity_tangent_deviation = maybe_lambda_g->cwiseSqrt().cwiseInverse(),
      .bias_deviation = maybe_lambda_b->cwiseSqrt().cwiseInverse(),
      .body_velocity_deviation = maybe_lambda_v->cwiseSqrt().cwiseInverse(),
      .translation_scale_symmetric_deviation =
        maybe_lambda_p->cwiseSqrt().cwiseInverse(),
    };
  }
}  // namespace cyclops::initializer
