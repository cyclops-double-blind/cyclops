#include "cyclops/details/initializer/vision_imu/translation_cache.hpp"
#include "cyclops/details/initializer/vision_imu/translation_analysis.hpp"

#include <algorithm>

namespace cyclops::initializer {
  using Eigen::JacobiSVD;

  using Eigen::ArrayXd;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  struct IMUMatchTranslationAnalysisCache::Impl {
    JacobiSVD<MatrixXd> _core_svd;

    int _n;
    int _m;
    int _k;

    MatrixXd A_I__T__U;
    MatrixXd A_V__inv__V_bar;
    MatrixXd U_bar__T__A_I;

    MatrixXd F_I;
    MatrixXd D_I;

    VectorXd B_I__T__alpha;
    VectorXd A_I__T__beta;
    VectorXd B_I__T__beta;

    VectorXd U__T__alpha;
    VectorXd U__T__beta;

    double alpha__dot__beta;
    double beta__dot__beta;

    explicit Impl(imu_match_translation_analysis_t const& analysis);
    primal_cache_inflation_t inflatePrimal(double s) const;
    derivative_cache_inflation_t inflateDerivative(double s) const;
  };

  IMUMatchTranslationAnalysisCache::Impl::Impl(
    imu_match_translation_analysis_t const& analysis) {
    auto const& [_1, _2, _3, A_I, B_I, A_V, alpha, beta] = analysis;

#define A_V_ (A_V.triangularView<Eigen::Upper>())
#define A_V__T_inv (A_V_.transpose().solve)
    MatrixXd L_I = A_V__T_inv(B_I.transpose()).transpose();
#undef A_V__T_inv
    _core_svd.compute(L_I, Eigen::ComputeFullU | Eigen::ComputeFullV);

    auto const& U = _core_svd.matrixU();
    auto const& V = _core_svd.matrixV();

#define U_bar (U.leftCols(_k))
#define V_bar (V.leftCols(_k))
    _n = U.rows();
    _m = V.rows();
    _k = std::min<int>(_n, _m);

    A_I__T__U = A_I.transpose() * U;
    A_V__inv__V_bar = A_V_.solve(V_bar);
    U_bar__T__A_I = U_bar.transpose() * A_I;
#undef A_V_
#undef U_bar
#undef V_bar

    F_I = A_I.transpose() * B_I;
    D_I = B_I.transpose() * B_I;

    B_I__T__alpha = B_I.transpose() * alpha;
    A_I__T__beta = A_I.transpose() * beta;
    B_I__T__beta = B_I.transpose() * beta;

    U__T__alpha = U.transpose() * alpha;
    U__T__beta = U.transpose() * beta;

    alpha__dot__beta = alpha.dot(beta);
    beta__dot__beta = beta.dot(beta);
  }

  static ArrayXd extend(ArrayXd const& x, int n, double v) {
    auto r = ArrayXd(n);
    auto k = std::min<int>(x.size(), n);

    r.head(k) = x.head(k);
    if (k < n)
      r.tail(n - k) = v;
    return r;
  }

  IMUMatchTranslationAnalysisCache::primal_cache_inflation_t
  IMUMatchTranslationAnalysisCache::Impl::inflatePrimal(double s) const {
    auto const& U = _core_svd.matrixU();
    auto const& V = _core_svd.matrixV();
    auto const& sigma = _core_svd.singularValues();

    ArrayXd sigma_s = sigma.array() * s;
    ArrayXd K_bar = (1 + sigma_s.square()).inverse();
    ArrayXd K = extend(K_bar, _n, 1.0);

    VectorXd U__T__gamma = U__T__alpha + U__T__beta * s;
    VectorXd U_bar__T__gamma = U__T__gamma.head(_k);

    ArrayXd K_bar__sigma = K_bar * sigma.array();

#define diag(x) (x.matrix().asDiagonal())
    MatrixXd A_I__T__U__K = A_I__T__U * diag(K);
    MatrixXd A_V__inv__V_bar__K_bar__sigma =
      A_V__inv__V_bar * diag(K_bar__sigma);
#undef diag

    return primal_cache_inflation_t {
      .H_I_bar = A_I__T__U__K * A_I__T__U.transpose(),
      .b_I_bar = A_I__T__U__K * U__T__gamma,
      .F_V = A_V__inv__V_bar__K_bar__sigma * U_bar__T__A_I,
      .z_V = A_V__inv__V_bar__K_bar__sigma * U_bar__T__gamma,
    };
  }

  IMUMatchTranslationAnalysisCache::derivative_cache_inflation_t
  IMUMatchTranslationAnalysisCache::Impl::inflateDerivative(double s) const {
    return derivative_cache_inflation_t {
      .r_s__dot = 2 * (alpha__dot__beta + beta__dot__beta * s),
      .b_I_s__dot = A_I__T__beta,
      .b_V_s__dot = B_I__T__alpha + 2 * s * B_I__T__beta,
      .F_I = F_I,
      .D_I = D_I,
    };
  }

  IMUMatchTranslationAnalysisCache::IMUMatchTranslationAnalysisCache(
    imu_match_translation_analysis_t const& analysis)
      : _pimpl(std::make_unique<Impl>(analysis)) {
  }

  IMUMatchTranslationAnalysisCache::~IMUMatchTranslationAnalysisCache() =
    default;

  IMUMatchTranslationAnalysisCache::primal_cache_inflation_t
  IMUMatchTranslationAnalysisCache::inflatePrimal(double s) const {
    return _pimpl->inflatePrimal(s);
  }

  IMUMatchTranslationAnalysisCache::derivative_cache_inflation_t
  IMUMatchTranslationAnalysisCache::inflateDerivative(double s) const {
    return _pimpl->inflateDerivative(s);
  }
}  // namespace cyclops::initializer
