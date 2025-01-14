#include "cyclops/details/estimation/ceres/cost.gaussian_prior.hpp"
#include "cyclops/details/estimation/graph/node.hpp"

#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/utils/type.hpp"

namespace cyclops::estimation {
  using Eigen::Map;
  using Eigen::Matrix3d;
  using Eigen::Matrix4d;
  using Eigen::MatrixXd;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;
  using Eigen::VectorXd;

  using Vector9d = Eigen::Matrix<double, 9, 1>;

  using Matrix3x4d = Eigen::Matrix<double, 3, 4>;
  using Matrix9x10d = Eigen::Matrix<double, 9, 10>;
  using MatrixXdR =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  static Matrix3x4d axisangle_quaternion_derivative(Vector3d const& theta) {
    auto a = theta.norm();
    auto s = std::sin(a / 2);
    auto c = std::cos(a / 2);

    auto const [s_over_a, alpha, beta, gamma] = [&]() {
      if (a == 0)
        return std::make_tuple(1 / 2., 1 / 48., 1 / 4., -1 / 24.);

      auto s_over_a = s / a;
      auto beta = s_over_a * s_over_a;

      if (a < 1e-2) {
        // clang-format off
        auto alpha =
          (
            + (1 / 48.)
            - (1 / 3840.) * pow(a, 2)
            + (1 / 645120.) * pow(a, 4)
          )
          *
          (
            + (1.)
            - (1 / 48.) * pow(a, 2)
            + (1 / 3840.) * pow(a, 4)
            - (1 / 645120.) * pow(a, 6)
          );
        // clang-format on
        auto gamma =
          -(1. / 24.) + (1 / 960.) * pow(a, 2) - (1 / 167520.) * pow(a, 4);
        return std::make_tuple(s_over_a, alpha, beta, gamma);
      } else {
        auto alpha = (0.25 - beta) / a / a;
        auto gamma = (0.5 * c - s_over_a) / a / a;
        return std::make_tuple(s_over_a, alpha, beta, gamma);
      }
    }();
#define __I3__ (Matrix3d::Identity())
    Matrix3d const M = beta * __I3__ + alpha * theta * theta.transpose();
    Matrix3d const B = gamma * theta * theta.transpose() + s_over_a * __I3__;
    Matrix3x4d const A_T =
      (Matrix3x4d() << B, -0.5 * s_over_a * theta).finished();
#undef __I3__

    return M.colPivHouseholderQr().solve(A_T);
  }

  static Matrix3d se3_log_translation_rotation_derivative(
    Vector3d const& theta, Vector3d const& p) {
    auto a = theta.norm();
    auto s = std::sin(a);
    auto c = std::cos(a);

    auto const [alpha, beta, gamma, delta, epsilon] = [&]() {
      if (a < 1e-2) {
        // clang-format off
        auto alpha =
          + (1 / 3.)
          - (1 / 60.) * pow(a, 2)
          + (1 / 2520.) * pow(a, 4);
        auto beta =
          + (1 / 12.)
          - (1 / 180.) * std::pow(a, 2)
          + (1 / 6720.) * std::pow(a, 4);
        auto gamma =
          + (1 / 60.)
          - (1 / 1260.) * std::pow(a, 2)
          + (1 / 60480.) * std::pow(a, 4);
        auto delta =
          + (1 / 2.)
          - (1 / 24.) * std::pow(a, 2)
          + (1 / 720.) * std::pow(a, 4);
        auto epsilon =
          - (1 / 6.)
          + (1 / 120.) * std::pow(a, 2)
          - (1 / 5040.) * std::pow(a, 4);
        // clang-format on
        return std::make_tuple(alpha, beta, gamma, delta, epsilon);
      } else {
        auto alpha = 2 * (a - s) / a / a / a;
        auto beta = -(s * a - 2 * (1 - c)) / a / a / a / a;
        auto gamma = -((1 - c) * a - 3 * (a - s)) / a / a / a / a / a;
        auto delta = (1 - c) / a / a;
        auto epsilon = -(a - s) / a / a / a;
        return std::make_tuple(alpha, beta, gamma, delta, epsilon);
      }
    }();

    Matrix3d const S = skew3d(theta);

    // clang-format off
    return
      + alpha * p * theta.transpose()
      + beta * S * p * theta.transpose()
      + gamma * S * S * p * theta.transpose()
      + delta * skew3d(p)
      + epsilon * (
          + theta.dot(p) * Matrix3d::Identity()
          + theta * p.transpose()
        );
    // clang-format on
  }

  static Matrix4d quaternion_left_multiplication_matrix(Quaterniond const& q) {
    Matrix4d L = q.w() * Matrix4d::Identity();
    L.block<3, 3>(0, 0) += skew3d<double>(q.vec());
    L.block<3, 1>(0, 3) += q.vec();
    L.block<1, 3>(3, 0) -= q.vec().transpose();
    return L;
  }

  static Vector9d evaluate_motion_error(
    double* maybe_jacobian, MatrixXd const& nominal_jacobian, int param_index,
    double const* x, double const* x_hat) {
    Map<Quaterniond const> q(x);
    Map<Quaterniond const> q_hat(x_hat);
    auto delta_theta = so3_logmap(q_hat.conjugate() * q);
    auto N_inv = so3_left_jacobian_inverse(delta_theta);

    Map<Vector3d const> p(x + 4);
    Map<Vector3d const> p_hat(x_hat + 4);
    auto delta_p = (N_inv * (q_hat.conjugate() * (p - p_hat))).eval();

    Map<Vector3d const> v(x + 7);
    Map<Vector3d const> v_hat(x_hat + 7);
    auto delta_v = (N_inv * (q_hat.conjugate() * (v - v_hat))).eval();

    if (maybe_jacobian != nullptr) {
      Matrix3x4d const delta_theta_over_q =
        axisangle_quaternion_derivative(delta_theta) *
        quaternion_left_multiplication_matrix(q_hat.conjugate());

      Matrix3x4d const delta_p_over_q = N_inv *
        se3_log_translation_rotation_derivative(delta_theta, delta_p) *
        delta_theta_over_q;
      Matrix3d const delta_p_over_p = N_inv * q_hat.matrix().transpose();

      Matrix3x4d const delta_v_over_q = N_inv *
        se3_log_translation_rotation_derivative(delta_theta, delta_v) *
        delta_theta_over_q;
      Matrix3d const delta_v_over_v = N_inv * q_hat.matrix().transpose();

      Matrix9x10d delta_x_over_x;
      // clang-format off
      delta_x_over_x <<
        delta_theta_over_q, Matrix3d::Zero(), Matrix3d::Zero(),
        delta_p_over_q,     delta_p_over_p,   Matrix3d::Zero(),
        delta_v_over_q,     Matrix3d::Zero(), delta_v_over_v;
      // clang-format on

      auto rows = nominal_jacobian.rows();
      Map<MatrixXdR> jacobian(maybe_jacobian, rows, 10);
      jacobian = nominal_jacobian.middleCols(param_index, 9) * delta_x_over_x;
    }
    return (Vector9d() << delta_theta, delta_p, delta_v).finished();
  }

  template <int dimension>
  static auto evaluate_euclidean_error(
    double* maybe_jacobian, MatrixXd const& nominal_jacobian, int param_index,
    double const* param, double const* param_hat) {
    using vector_t = Eigen::Matrix<double, dimension, 1>;

    if (maybe_jacobian != nullptr) {
      auto rows = nominal_jacobian.rows();
      Map<MatrixXdR>(maybe_jacobian, rows, dimension) =
        nominal_jacobian.middleCols(param_index, dimension);
    }
    vector_t error =
      Map<vector_t const>(param) - Map<vector_t const>(param_hat);
    return error;
  }

  GaussianPriorCost::GaussianPriorCost(gaussian_prior_t prior)
      : _prior(std::move(prior)) {
    auto& parameter_sizes = *mutable_parameter_block_sizes();

    parameter_sizes.reserve(_prior.input_nodes.size());
    for (auto const& node : _prior.input_nodes)
      parameter_sizes.push_back(node.dimension());
    set_num_residuals(_prior.residual.rows());
  }

  bool GaussianPriorCost::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
    VectorXd error(_prior.jacobian.cols());

    int i = 0;
    int index_param = 0;
    int index_manif = 0;
    for (auto const& node : _prior.input_nodes) {
      auto visitor_common = [&](auto const& node, auto const& evaluator) {
        auto x = parameters[i];
        auto x_hat = &_prior.nominal_parameters.at(index_param);

        auto n_manif = node.manifold_dimension();
        auto n_param = node.dimension();
        auto jacobian = jacobians != nullptr ? jacobians[i] : nullptr;
        error.segment(index_manif, n_manif) =
          evaluator(jacobian, _prior.jacobian, index_manif, x, x_hat);

        i++;
        index_manif += n_manif;
        index_param += n_param;
      };

      std::visit(
        overloaded {
          [&](node_t::frame_t const& _) {
            visitor_common(_, evaluate_motion_error);
          },
          [&](node_t::bias_t const& _) {
            auto evaluator = evaluate_euclidean_error<6>;
            visitor_common(_, evaluator);
          },
          [&](node_t::landmark_t const& _) {
            auto evaluator = evaluate_euclidean_error<3>;
            visitor_common(_, evaluator);
          },
        },
        node.variant);
    }

    (Map<VectorXd>(residuals, _prior.residual.rows())) =
      _prior.residual + _prior.jacobian * error;

    return true;
  }
}  // namespace cyclops::estimation
