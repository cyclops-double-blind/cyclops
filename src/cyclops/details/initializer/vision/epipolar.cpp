#include "cyclops/details/initializer/vision/epipolar.hpp"
#include "cyclops/details/initializer/vision/epipolar_refine.hpp"
#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/utils/vision.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

#include <cmath>

namespace cyclops::initializer {
  using std::map;
  using std::set;
  using std::vector;

  using Eigen::Matrix3d;
  using Eigen::Vector3d;

  using MatrixX9d = Eigen::Matrix<double, Eigen::Dynamic, 9>;
  using Vector9d = Eigen::Matrix<double, 9, 1>;

  /*
   * See section 11.1 of [1].
   *
   * [1] R. Hartley and A. Zisserman, "Multiple View Geometry in Computer
   * Vision", 2nd ed. Cambridge: Cambridge University Press, 2004.
   */
  static std::optional<Matrix3d> compute_essential_matrix(
    map<landmark_id_t, two_view_feature_pair_t> const& features,
    set<landmark_id_t> const& inliers) {
    auto n = inliers.size();
    if (n < 8) {
      __logger__->warn("Degenerate Epipolar geometry (n_features = {})", n);
      return std::nullopt;
    }

    MatrixX9d A(n, 9);
    for (auto const& [i, feature_id] : ranges::views::enumerate(inliers)) {
      auto const& [f1, f2] = features.at(feature_id);
      auto x1 = f1.x();
      auto y1 = f1.y();
      auto x2 = f2.x();
      auto y2 = f2.y();

      Vector9d Ai;
      Ai << x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1;
      A.row(i) = Ai.transpose();
    }

    Eigen::JacobiSVD<MatrixX9d> e_svd(A, Eigen::ComputeFullV);
    Vector9d const e = e_svd.matrixV().col(8);
    Matrix3d E0;
    // clang-format off
    E0 <<
      e.segment<3>(0).transpose(),
      e.segment<3>(3).transpose(),
      e.segment<3>(6).transpose()
    ;
    // clang-format on

    // enforce singularity constraint of E
    Eigen::JacobiSVD<Matrix3d> E0_svd(
      E0, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vector3d s = E0_svd.singularValues();
    s.z() = 0;

    Matrix3d const& U = E0_svd.matrixU();
    Matrix3d const& V = E0_svd.matrixV();
    return U * s.asDiagonal() * V.transpose();
  }

  static auto analyze_inliers(
    double sigma, Matrix3d const& E,
    map<landmark_id_t, two_view_feature_pair_t> const& features) {
    auto expected_inliers_count = 0.;
    auto inliers = set<landmark_id_t>();

    for (auto const& [feature_id, feature_pair] : features) {
      auto const& [f1, f2] = feature_pair;
      auto e1 = (E.transpose() * f2.homogeneous()).eval();
      auto e2 = (E * f1.homogeneous()).eval();

      auto a1 = e1.head<2>().eval();
      auto a2 = e2.head<2>().eval();
      if (a1.norm() < 1e-4 || a2.norm() < 1e-4)
        continue;

      auto g = f2.homogeneous().dot(e2);

      auto error_1 = std::pow(g / a1.norm() / sigma, 2);
      auto error_2 = std::pow(g / a2.norm() / sigma, 2);

      // inlier probability of f1 and f2
      auto p1 = std::max(0., 1 - chi_squared_cdf(2, error_1));
      auto p2 = std::max(0., 1 - chi_squared_cdf(2, error_2));

      if (p1 >= 0.05 && p2 >= 0.05) {
        expected_inliers_count += std::sqrt(p1 * p2);
        inliers.emplace(feature_id);
      }
    }
    return std::make_tuple(expected_inliers_count, inliers);
  }

  static epipolar_analysis_t analyze_two_view_epipolar(
    double sigma, set<landmark_id_t> const& ransac_selection,
    map<landmark_id_t, two_view_feature_pair_t> const& features) {
    auto maybe_E = compute_essential_matrix(features, ransac_selection);
    if (!maybe_E)
      return {-1, Matrix3d::Identity(), {}};
    auto const& E = *maybe_E;

    auto [n_inliers, inliers] = analyze_inliers(sigma, E, features);
    return {n_inliers, E, std::move(inliers)};
  }

  epipolar_analysis_t analyze_two_view_epipolar(
    double sigma, vector<set<landmark_id_t>> const& ransac_batch,
    map<landmark_id_t, two_view_feature_pair_t> const& features) {
    if (sigma <= 0) {
      __logger__->warn("Invalid landmark noise (<= 0)");
      return {-1, Matrix3d::Identity(), {}};
    }

    epipolar_analysis_t best_epipolar = {0., Matrix3d::Zero(), {}};
    for (auto const& subset : ransac_batch) {
      auto geometry = analyze_two_view_epipolar(sigma, subset, features);
      if (geometry.expected_inliers > best_epipolar.expected_inliers)
        best_epipolar = std::move(geometry);
    }
    if (best_epipolar.inliers.size() < 8)
      return {-1, Matrix3d::Identity(), {}};

    auto E_refined = refine_epipolar_geometry(
      best_epipolar.essential_matrix, best_epipolar.inliers, features);

    auto [n_inliers, inliers] = analyze_inliers(sigma, E_refined, features);
    return {n_inliers, E_refined, inliers};
  }

  static Matrix3d absdet(Matrix3d const& U) {
    return U.determinant() < 0 ? -U : U;
  }

  /*
   * see: section 9.6 of [1]
   *
   * [1] R. Hartley and A. Zisserman, "Multiple View Geometry in Computer
   * Vision", 2nd ed. Cambridge: Cambridge University Press, 2004.
   */
  vector<rotation_translation_matrix_pair_t> solve_epipolar_motion_hypothesis(
    Matrix3d const& E) {
    Eigen::JacobiSVD<Matrix3d> svd(
      E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Matrix3d U = absdet(svd.matrixU());
    Matrix3d V = absdet(svd.matrixV());

    Matrix3d W;
    // clang-format off
    W <<
      +0, -1, +0,
      +1, +0, +0,
      +0, +0, +1
    ;
    // clang-format on

    Matrix3d R1 = U * W * V.transpose();
    Matrix3d R2 = U * W.transpose() * V.transpose();

    Vector3d R1T_t = R1.transpose() * U.col(2);
    Vector3d R2T_t = R2.transpose() * U.col(2);

    return {
      {R1.transpose(), -R1T_t},
      {R2.transpose(), -R2T_t},
      {R1.transpose(), R1T_t},
      {R2.transpose(), R2T_t},
    };
  }
}  // namespace cyclops::initializer
