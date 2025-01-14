#include "cyclops/details/estimation/marginalizer/marginalizer_helper.hpp"
#include "cyclops/details/estimation/graph/graph.hpp"
#include "cyclops/details/estimation/graph/node.hpp"
#include "cyclops/details/estimation/graph/factor.hpp"

#include "cyclops/details/utils/debug.hpp"
#include "cyclops/details/utils/type.hpp"

#include "cyclops/details/estimation/state/accessor_read.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

#include <set>
#include <vector>

namespace cyclops::estimation {
  using std::set;
  using std::vector;

  using Eigen::Matrix3d;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  using Matrix3xXd = Eigen::Matrix<double, 3, Eigen::Dynamic>;

  namespace views = ranges::views;

  struct schur_complement_t {
    MatrixXd jacobian;
    VectorXd residual;
  };

  static schur_complement_t decompose(MatrixXd const& P, VectorXd const& s) {
    auto __tic__ = tic();

    Eigen::SelfAdjointEigenSolver<MatrixXd> eigen(P);
    __logger__->debug(
      "Marginalizer eigendecomposition time: {}[s]", toc(__tic__));

#define D (eigen.eigenvalues())
#define Q (eigen.eigenvectors())
    double constexpr epsilon = 1e-4;
    auto rank = (D.array() >= epsilon).count();

#define Q_bar (Q.rightCols(rank))
#define d_bar (D.tail(rank).cwiseSqrt())
    MatrixXd J = d_bar.asDiagonal() * Q_bar.transpose();
    VectorXd r = d_bar.cwiseInverse().asDiagonal() * Q_bar.transpose() * s;
#undef d_bar
#undef Q_bar
    r = 0.98 * r;  // XXX

    __logger__->debug(
      "Marginalizer total decomposition time: {}[s]", toc(__tic__));
    return {J, r};
  }

  static vector<Matrix3d> compile_landmark_self_information(
    EigenCRSMatrix const& J_f, int n_landmarks) {
    vector<Matrix3d> H_ff;
    H_ff.reserve(n_landmarks);

    EigenCCSMatrix const H_ff_raw = J_f.transpose() * J_f;
    for (int k = 0; k < n_landmarks; k++)
      H_ff.emplace_back(H_ff_raw.block(3 * k, 3 * k, 3, 3));
    return H_ff;
  }

  static std::tuple<MatrixXd, MatrixXd, MatrixXd, VectorXd, VectorXd>
  marginalize_landmark_information(
    EigenCRSMatrix const& J_n, EigenCRSMatrix const& J_f,
    EigenCRSMatrix const& J_k, VectorXd const& r) {
    __logger__->debug(
      "Marginalizing landmark information. dimension: {}", J_f.cols());

    MatrixXd H_nn = J_n.transpose() * J_n;
    MatrixXd H_kk = J_k.transpose() * J_k;
    Eigen::SelfAdjointView<MatrixXd, Eigen::Upper> H_nn_s(H_nn);
    Eigen::SelfAdjointView<MatrixXd, Eigen::Upper> H_kk_s(H_kk);

    MatrixXd H_nf = J_n.transpose() * J_f;
    MatrixXd H_kf = J_k.transpose() * J_f;
    MatrixXd H_nk = J_n.transpose() * J_k;
    VectorXd s_n = J_n.transpose() * r;
    VectorXd s_k = J_k.transpose() * r;

    int n_landmarks = J_f.cols() / 3;
    auto H_ff = compile_landmark_self_information(J_f, n_landmarks);
    auto H_ff_eigens =
      vector<Eigen::SelfAdjointEigenSolver<Matrix3d>>(n_landmarks);
    for (int k = 0; k < n_landmarks; k++)
      H_ff_eigens.at(k).compute(H_ff.at(k));

    __logger__->debug("Landmark self information eigendecomposition complete");

    auto A_ff_workspace_start = 0;
    auto B_nf_workspace_start = A_ff_workspace_start + 9;
    auto B_kf_workspace_start = B_nf_workspace_start + J_n.cols() * 3;
    auto s_f_brev_workspace_start = B_kf_workspace_start + J_k.cols() * 3;
    auto workspace_size = s_f_brev_workspace_start + 3;

    std::vector<double> __workspace__(workspace_size);

    for (int k = 0; k < n_landmarks; k++) {
      auto const& eigen = H_ff_eigens.at(k);
#define l_ff (eigen.eigenvalues())
#define Q_ff (eigen.eigenvectors())
      auto rank = (l_ff.array() > 1e-6).count();
      if (rank == 0)
        continue;
#define l_ff_bar (l_ff.tail(rank))
#define Q_ff_bar (Q_ff.rightCols(rank))
#define d_ff_bar (l_ff_bar.cwiseSqrt().cwiseInverse().asDiagonal())
      Eigen::Map<Matrix3xXd> A_ff_bar(__workspace__.data(), 3, rank);
      A_ff_bar = Q_ff_bar * d_ff_bar;
#undef d_ff_bar
#undef Q_ff_bar
#undef l_ff_bar
#undef l_ff
#undef Q_ff

      Eigen::Map<MatrixXd> B_nf_k(
        __workspace__.data() + B_nf_workspace_start, J_n.cols(), rank);
      Eigen::Map<MatrixXd> B_kf_k(
        __workspace__.data() + B_kf_workspace_start, J_k.cols(), rank);
#define H_nf_k (H_nf.middleCols(3 * k, 3))
#define H_kf_k (H_kf.middleCols(3 * k, 3))
      B_nf_k = H_nf_k * A_ff_bar;
      B_kf_k = H_kf_k * A_ff_bar;
#undef H_nf_k
#undef H_kf_k
      H_nn_s.rankUpdate(B_nf_k, -1);
      H_kk_s.rankUpdate(B_kf_k, -1);
      H_nk -= B_nf_k * B_kf_k.transpose();

      Eigen::Map<VectorXd> s_f_brev(
        __workspace__.data() + s_f_brev_workspace_start, rank);
#define J_f_k (J_f.middleCols(3 * k, 3))
      s_f_brev = A_ff_bar.transpose() * J_f_k.transpose() * r;
#undef J_f_k
      s_n -= B_nf_k * s_f_brev;
      s_k -= B_kf_k * s_f_brev;
    }
    __logger__->debug("Landmark marginalization complete");

    return std::make_tuple(H_nn_s, H_kk_s, H_nk, s_n, s_k);
  }

  static schur_complement_t compute_schur_complement(
    int drop_frame_dimension, int drop_landmark_dimension, int keep_dimension,
    EigenCRSMatrix const& J, VectorXd const& r) {
    auto __tic__ = tic();

    int n = drop_frame_dimension;
    int f = drop_landmark_dimension;
    int m = n + f;
    int k = keep_dimension;

    EigenCRSMatrix const J_n = J.middleCols(0, n);
    EigenCRSMatrix const J_f = J.middleCols(n, f);
    EigenCRSMatrix const J_k = J.middleCols(m, k);

    size_t constexpr node_variants =
      std::variant_size_v<decltype(node_t::variant)>;
    static_assert(
      std::is_same_v<
        node_t::variant_t_at<node_variants - 1>, node_t::landmark_t>,
      "Landmark node should be ordered to the last of the variants");

    auto [H_nn, H_kk, H_nk, s_n, s_k] =
      marginalize_landmark_information(J_n, J_f, J_k, r);
    Eigen::SelfAdjointEigenSolver<MatrixXd> H_nn_eigen(H_nn);
    __logger__->debug(
      "Schur complement eigendecomposition time: {}[s]", toc(__tic__));

    auto rank = (H_nn_eigen.eigenvalues().array() > 1e-6).count();
#define l_nn (H_nn_eigen.eigenvalues())
#define Q_nn (H_nn_eigen.eigenvectors())
#define l_nn_bar (l_nn.tail(rank))
#define Q_nn_bar (Q_nn.rightCols(rank))
#define d_nn_bar (l_nn_bar.cwiseSqrt().cwiseInverse().asDiagonal())
    MatrixXd L_nn_bar = Q_nn_bar * d_nn_bar;
    MatrixXd B_kn_bar = H_nk.transpose() * L_nn_bar;
    VectorXd t_n = B_kn_bar * L_nn_bar.transpose() * s_n;
#undef d_nn_bar
#undef Q_nn_bar
#undef l_nn_bar
#undef l_nn
#undef Q_nn
    MatrixXd P = H_kk;
    Eigen::SelfAdjointView<MatrixXd, Eigen::Upper> P_s(P);
    P_s.rankUpdate(B_kn_bar, -1);
    VectorXd s = s_k - t_n;
    __logger__->debug("Schur complement time: {}[s]", toc(__tic__));

    return decompose(P_s, s);
  }

  template <typename maybe_block_reference_t>
  static auto& get_or_die(maybe_block_reference_t const& maybe_ref) {
    return maybe_ref.value().get();
  }

  static vector<double> compile_node_nominals(
    StateVariableReadAccessor const& states, set<node_t> const& nodes) {
    vector<double> result;
    for (auto const& node : nodes) {
      auto visitor = overloaded {
        [&](node_t::frame_t const& _) {
          auto& x = get_or_die(states.motionFrame(_.id));
          auto s = x.begin();
          auto e = s + 10;
          result.insert(result.end(), s, e);
        },
        [&](node_t::bias_t const& _) {
          auto& x = get_or_die(states.motionFrame(_.id));
          auto s = x.begin() + 10;
          auto e = x.end();
          result.insert(result.end(), s, e);
        },
        [&](node_t::landmark_t const& _) {
          auto& f = get_or_die(states.landmark(_.id));
          result.insert(result.end(), f.begin(), f.end());
        },
      };
      std::visit(visitor, node.variant);
    }
    return result;
  }

  static gaussian_prior_t compute_prior(
    StateVariableReadAccessor const& state_accessor,
    EigenCRSMatrix const& jacobian, VectorXd const& residual, int n_drop_f,
    int n_drop_l, int n_keep, set<node_t> const& keep_nodes) {
    auto [J_pi, r_pi] =
      compute_schur_complement(n_drop_f, n_drop_l, n_keep, jacobian, residual);

    return {
      .jacobian = std::move(J_pi),
      .residual = std::move(r_pi),
      .input_nodes = keep_nodes,
      .nominal_parameters = compile_node_nominals(state_accessor, keep_nodes),
    };
  }

  static auto compile_parameter_index(
    marginalization_subgraph_t const& subgraph) {
    int n_drop_keyframes = 0;
    int n_drop_landmarks = 0;
    for (auto const& node : subgraph.drop_nodes) {
      auto visitor = overloaded {
        [&](node_t::landmark_t const& _) {
          n_drop_landmarks += _.manifold_dimension();
        },
        [&](auto const& _) { n_drop_keyframes += _.manifold_dimension(); },
      };
      std::visit(visitor, node.variant);
    }

    int n_keep = 0;
    for (auto const& node : subgraph.keep_nodes)
      n_keep += node.manifold_dimension();

    return std::make_tuple(n_drop_keyframes, n_drop_landmarks, n_keep);
  }

  gaussian_prior_t evaluate_gaussian_prior(
    FactorGraphInstance& graph, StateVariableReadAccessor const& state_accessor,
    marginalization_subgraph_t const& drop_subgraph) {
    auto factor_ptrs = drop_subgraph.factors | views::values |
      views::transform([](auto const& _) { return std::get<0>(_); }) |
      ranges::to_vector;
    auto nodes =
      views::concat(drop_subgraph.drop_nodes, drop_subgraph.keep_nodes) |
      ranges::to<vector<node_t>>;
    auto [jacobian, residual] = graph.evaluate(nodes, factor_ptrs);

    auto [n_drop_f, n_drop_l, n_keep] = compile_parameter_index(drop_subgraph);
    return compute_prior(
      state_accessor, jacobian, residual, n_drop_f, n_drop_l, n_keep,
      drop_subgraph.keep_nodes);
  }
}  // namespace cyclops::estimation
