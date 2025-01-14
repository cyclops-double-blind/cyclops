#include "cyclops/details/initializer/vision/twoview_selection.hpp"
#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  using Eigen::Matrix2d;
  using Eigen::Matrix3d;
  using Eigen::Vector2d;
  using Eigen::Vector3d;

  using Matrix2x3d = Eigen::Matrix<double, 2, 3>;

  namespace views = ranges::views;

  struct two_view_selection_geometry_estimation_t {
    Vector3d translation;
    std::vector<std::optional<double>> second_view_depths;
  };

  template <typename value_t, typename outvalue_t>
  static auto fmap_partial(std::function<outvalue_t(value_t)> f) {
    return [f](std::optional<value_t> const& x) -> std::optional<outvalue_t> {
      if (!x.has_value())
        return std::nullopt;
      return f(*x);
    };
  }

  /*
   * Roughly estimate the two-view geometry for an observability analysis of the
   * camera motion.
   *
   * Let us assume u^(h) // f, v^(h) // R^T * (f - p), where u and v are
   * 2D feature observations of each two view frames, f is 3D position of the
   * landmark, R is camera rotation and p is camera translation. u^(h) indicates
   * a homogeneous representation of 2D vector u, u^(h) := [u; 1]. a // b
   * indicates vectors a and b are parallel, i.e., ∃r. a = r * b.
   *
   * Now since u^(h) // f and v^(h) // R^T * (f - p), there exists
   * ∃r, s.
   *   f = u^(h) * r and
   *   R^T * (f - p) = v^(h) * s
   * <=>
   *   p + R * v^(h) * s - u^(h) * r = 0
   * <=>
   *   p + w * s - u^(h) * r = 0,
   *
   * where w := R * v^(h).
   *
   * Now conveniently re-defining u <- u^(h) and iterating over all two-view
   * feature correspondences u_i, v_i,
   *
   * [I, -u1, +w1, +0., +0., +0., +0., ...] [    p   ]
   * [I, +0., +0., -u2, +w2, +0., +0., ...] [ r1; s1 ]
   * [I, +0., +0., +0., +0., -u3, +w3, ...] [ r2; s2 ] = 0.   ........ (*)
   * [                ...                 ] [   ...  ]
   *
   * Solving null space problem of equation (*) yields the estimation of
   * scaleless two-view geometry.
   *
   * Note that this estimated geometry can be inaccurate, due to the presence of
   * noise and outliers in feature tracking, and the presence of inaccuracy in
   * the rotation estimation R which is obtained by integration of the IMU.
   * This is sort-of okay since the purpose of this function is to roughly
   * estimate the geometry for computing a score of the two-view selection.
   * The consequence of the inaccurate geometry estimation is nothing more than
   * a suboptimal selection of the initialization view, which is anyway expected
   * to be better than a heuristic selection.
   */
  static std::optional<two_view_selection_geometry_estimation_t>
  two_view_selection_estimate_geometry(
    two_view_correspondence_data_t const& view) {
    Matrix3d translation_schur_complement = Matrix3d::Zero();

    std::vector<std::optional<Matrix2x3d>> depth_translation_correlations;
    depth_translation_correlations.reserve(view.features.size());

    for (auto const& [u, v] : view.features | views::values) {
      auto w = (view.rotation_prior.value * v.homogeneous()).eval();

      auto d1 = u.homogeneous().dot(u.homogeneous());
      auto d2 = -u.homogeneous().dot(w);
      auto d4 = w.dot(w);
      Matrix2d D = (Matrix2d() << d1, d2, d2, d4).finished();
      Eigen::FullPivLU<Matrix2d> lu(D);

      if (lu.determinant() < 1e-4) {
        depth_translation_correlations.emplace_back(std::nullopt);
        continue;
      }

      Matrix2x3d J;
      J << -u.homogeneous().transpose(), w.transpose();

      depth_translation_correlations.emplace_back(lu.solve(J));
      translation_schur_complement += Matrix3d::Identity() -
        J.transpose() * (*depth_translation_correlations.back());
    }

    Eigen::SelfAdjointEigenSolver<Matrix3d> translation_eigensolver;
    translation_eigensolver.compute(translation_schur_complement);
    if (translation_eigensolver.info() != Eigen::Success)
      return std::nullopt;

    auto p = translation_eigensolver.eigenvectors().col(0).eval();
    auto second_view_depths = depth_translation_correlations |
      views::transform(fmap_partial<Matrix2x3d, Vector2d>(
        [&](auto const& L) { return L * p; })) |
      views::transform(fmap_partial<Vector2d, double>(
        [&](auto const& depth) { return depth.y(); })) |
      ranges::to_vector;
    return two_view_selection_geometry_estimation_t {p, second_view_depths};
  }

  static void update_rotation_fisher_information(
    Matrix3d& information, Vector2d const& feature) {
    auto ux = feature.x();
    auto uy = feature.y();
    Matrix2x3d J;
    // clang-format off
    J <<
      ux * uy,      -1 - ux * ux, +uy,
      1 + uy * uy,  -ux * uy,     -ux;
    // clang-format on
    information += J.transpose() * J;
  }

  static void update_position_fisher_information(
    Matrix3d& information, double s, Vector2d const& v) {
    information.topLeftCorner<2, 2>() += Matrix2d::Identity() / s / s;
    information.topRightCorner<2, 1>() -= v / s / s;
    information.bottomLeftCorner<1, 2>() -= v.transpose() / s / s;
    information(2, 2) += v.dot(v) / s / s;
  }

  static double analyze_two_view_observability_score(
    two_view_correspondence_data_t const& two_view) {
    auto maybe_geometry = two_view_selection_estimate_geometry(two_view);
    if (!maybe_geometry.has_value())
      return -1.0;
    auto const& geometry = *maybe_geometry;

    Matrix3d rotation_information = Matrix3d::Zero();
    for (auto const& feature : two_view.features | views::values) {
      auto const& [u, v] = feature;
      update_rotation_fisher_information(rotation_information, v);
    }

    Matrix3d position_information = Matrix3d::Zero();
    for (auto const& [feature, maybe_depth] : views::zip(
           two_view.features | views::values, geometry.second_view_depths)) {
      if (!maybe_depth.has_value())
        continue;

      auto const& [u, v] = feature;
      auto s = *maybe_depth;
      update_position_fisher_information(position_information, s, v);
    }

    auto min_eigenvalue = [](Matrix3d const& matrix) {
      return matrix.selfadjointView<Eigen::Upper>().eigenvalues().x();
    };

    auto rotation_score = std::max(0., min_eigenvalue(rotation_information));
    auto position_score = std::max(0., min_eigenvalue(position_information));
    return std::sqrt(rotation_score * position_score);
  }

  std::optional<std::reference_wrapper<
    std::pair<frame_id_t const, two_view_correspondence_data_t> const>>
  select_best_two_view_pair(multiview_correspondences_t const& multiviews) {
    auto n_views = multiviews.view_frames.size() + 1;
    if (n_views < 2) {
      __logger__->error("Not enough motion frames ({} < 2).", n_views);
      return std::nullopt;
    }

    auto view_scores = multiviews.view_frames | views::values |
      views::transform(analyze_two_view_observability_score);

    auto best_candidates = views::zip(multiviews.view_frames, view_scores) |
      views::filter([&](auto const& pair) { return std::get<1>(pair) >= 0; });

    auto best_i =
      ranges::max_element(best_candidates, [](auto const& a, auto const& b) {
        auto const& [_1, score_a] = a;
        auto const& [_2, score_b] = b;
        return score_a < score_b;
      });
    if (best_i == best_candidates.end())
      return std::nullopt;

    auto const& [best_ref, _] = *best_i;
    return std::cref(best_ref);
  }
}  // namespace cyclops::initializer
