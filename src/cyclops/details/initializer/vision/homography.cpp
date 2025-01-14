#include "cyclops/details/initializer/vision/homography.hpp"
#include "cyclops/details/initializer/vision/homography_refinement.hpp"
#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/utils/vision.hpp"

#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  using std::map;
  using std::set;
  using std::vector;

  namespace views = ranges::views;

  using ranges::views::cartesian_product;
  using ranges::views::enumerate;
  using ranges::views::transform;

  using Eigen::Matrix3d;
  using Eigen::Vector2d;
  using Eigen::Vector3d;
  using MatrixX9d = Eigen::Matrix<double, Eigen::Dynamic, 9>;
  using Matrix29d = Eigen::Matrix<double, 2, 9>;
  using Vector9d = Eigen::Matrix<double, 9, 1>;

  static Vector2d project(Vector3d const& h) {
    return h.head<2>() / h.z();
  }

  /*
   * See section 4.1 of [1].
   *
   * [1] R. Hartley and A. Zisserman, "Multiple View Geometry in Computer
   * Vision", 2nd ed. Cambridge: Cambridge University Press, 2004.
   */
  static std::optional<Matrix3d> compute_homography_matrix(
    map<landmark_id_t, two_view_feature_pair_t> const& features,
    set<landmark_id_t> const& inliers) {
    auto n = inliers.size();
    if (n < 4) {
      __logger__->warn("Degenerate homography (n_features = {})", n);
      return std::nullopt;
    }

    MatrixX9d P(2 * n, 9);
    for (auto const& [i, feature_id] : enumerate(inliers)) {
      auto const& [f1, f2] = features.at(feature_id);
      auto x1 = f1.x();
      auto y1 = f1.y();
      auto x2 = f2.x();
      auto y2 = f2.y();

      Matrix29d Pi;

      // clang-format off
      Pi <<
        +0., +0., +0., -x1, -y1, -1., +x1 * y2, +y1 * y2, +y2,
        +x1, +y1, +1., +0., +0., +0., -x1 * x2, -y1 * x2, -x2
      ;
      // clang-format on
      P.block<2, 9>(2 * i, 0) = Pi;
    }

    Eigen::JacobiSVD<MatrixX9d> svd(P, Eigen::ComputeFullV);
    Vector9d const h = svd.matrixV().col(8);
    Matrix3d H21;
    // clang-format off
    H21 <<
      h.segment<3>(0).transpose(),
      h.segment<3>(3).transpose(),
      h.segment<3>(6).transpose()
    ;
    // clang-format on

    auto det = H21.determinant();
    if (det < 0)
      H21 = -H21;

    if (std::abs(det) <= 1e-4) {
      __logger__->trace("Collinear homography points");
      return std::nullopt;
    }
    return H21;
  }

  static auto analyze_inliers(
    double sigma, Matrix3d const& H21,
    map<landmark_id_t, two_view_feature_pair_t> const& features) {
    Matrix3d H12 = H21.inverse();

    auto expected_inliers_number = 0.;
    auto inliers = set<landmark_id_t>();
    for (auto const& [feature_id, feature_pair] : features) {
      auto const& [f1, f2] = feature_pair;
      auto f2_hat = project(H21 * f1.homogeneous());
      auto f1_hat = project(H12 * f2.homogeneous());

      auto r1 = ((f1 - f1_hat) / sigma).eval();
      auto r2 = ((f2 - f2_hat) / sigma).eval();

      // inlier probability of f1 and f2
      auto p1 = std::max(0., 1 - chi_squared_cdf(2, r1.dot(r1)));
      auto p2 = std::max(0., 1 - chi_squared_cdf(2, r2.dot(r2)));

      if (p1 >= 0.05 && p2 >= 0.05) {
        expected_inliers_number += std::sqrt(p1 * p2);
        inliers.emplace(feature_id);
      }
    }
    return std::make_tuple(expected_inliers_number, inliers);
  }

  static homography_analysis_t analyze_two_view_homography(
    double sigma, set<landmark_id_t> const& ransac_selection,
    map<landmark_id_t, two_view_feature_pair_t> const& features) {
    auto maybe_H21 = compute_homography_matrix(features, ransac_selection);
    if (!maybe_H21)
      return {-1, Matrix3d::Identity(), {}};
    auto const& H21 = *maybe_H21;

    auto [n_inliers, inliers] = analyze_inliers(sigma, H21, features);
    return {n_inliers, H21, inliers};
  }

  homography_analysis_t analyze_two_view_homography(
    double sigma, vector<set<landmark_id_t>> const& ransac_batch,
    map<landmark_id_t, two_view_feature_pair_t> const& features) {
    if (sigma <= 0)
      return {-1, Matrix3d::Identity(), {}};

    homography_analysis_t best_homography = {0., Matrix3d::Zero(), {}};
    for (auto const& selection : ransac_batch) {
      auto homography = analyze_two_view_homography(sigma, selection, features);
      if (homography.expected_inliers > best_homography.expected_inliers)
        best_homography = std::move(homography);
    }
    if (best_homography.inliers.size() < 4)
      return {-1, Matrix3d::Identity(), {}};

    auto H_refinement = refine_homography_geometry(
      sigma, best_homography.homography, best_homography.inliers, features);

    auto [n_inliers, inliers] = analyze_inliers(sigma, H_refinement, features);
    return {n_inliers, H_refinement, inliers};
  }

  static Matrix3d absdet(Matrix3d const& U) {
    return U.determinant() < 0 ? -U : U;
  }

  static bool isclose(double a, double b, double h) {
    return std::abs((a - b) / a) < h;
  }

  static std::tuple<double, double> analyze_plane_normal_components(
    double s1, double s2, double s3) {
    auto s1s1_minus_s3s3 = s1 * s1 - s3 * s3;
    auto s1s1_minus_s2s2 = s1 * s1 - s2 * s2;
    auto s2s2_minus_s3s3 = s2 * s2 - s3 * s3;

    auto x1 = std::sqrt(s1s1_minus_s2s2 / s1s1_minus_s3s3);
    auto x3 = std::sqrt(s2s2_minus_s3s3 / s1s1_minus_s3s3);

    return std::make_tuple(x1, x3);
  }

  template <typename app_t>
  static auto foreach_sign_permutations(app_t const& app) {
    auto signs = std::array {-1., 1.};
    auto pair_app = [&](auto const& pair) {
      auto const& [e1, e3] = pair;
      return app(e1, e3);
    };
    return cartesian_product(signs, signs) | transform(pair_app) |
      ranges::to_vector;
  }

  class HomographyMotionDecompositionContext {
  private:
    Matrix3d const& U;
    Matrix3d const& V;
    Vector3d const& s;

  public:
    HomographyMotionDecompositionContext(
      Matrix3d const& U, Matrix3d const& V, Vector3d const& s)
        : U(U), V(V), s(s) {
    }

    auto handlePositiveDPrime(double x1, double x3) {
      return foreach_sign_permutations([&](auto e1, auto e3) {
        auto const sin = e1 * e3 * (s.x() - s.z()) * x1 * x3 / s.y();
        auto const cos = (s.x() * x3 * x3 + s.z() * x1 * x1) / s.y();

        Matrix3d R;
        // clang-format off
        R <<
          +cos, +0, -sin,
          +0,   +1, +0,
          +sin, +0, +cos
        ;
        // clang-format on
        Vector3d const t = (s.x() - s.z()) * Vector3d(e1 * x1, 0, -e3 * x3);

        return rotation_translation_matrix_pair_t {
          .rotation = V * R.transpose() * U.transpose(),
          .translation = -V * R.transpose() * t,
        };
      });
    }

    auto handleNegativeDPrime(double x1, double x3) {
      return foreach_sign_permutations([&](auto e1, auto e3) {
        auto const sin = e1 * e3 * (s.x() + s.z()) * x1 * x3 / s.y();
        auto const cos = (s.z() * x1 * x1 - s.x() * x3 * x3) / s.y();

        Matrix3d R;
        // clang-format off
        R <<
          +cos, +0, +sin,
          +0,   -1, +0,
          +sin, +0, -cos
        ;
        // clang-format on
        Vector3d const t = (s.x() + s.z()) * Vector3d(e1 * x1, 0, e3 * x3);

        return rotation_translation_matrix_pair_t {
          .rotation = V * R.transpose() * U.transpose(),
          .translation = -V * R.transpose() * t,
        };
      });
    }
  };

  /*
   * extract motion hypothesis from homography as in [1]
   *
   * [1] O. D. Faugeras and F. Lustman, "Motion and structure from motion in a
   * piecewise planar environment", in International Journal of Pattern
   * Recognition and Artificial Intelligence, vol. 2, no. 3, pp. 485-508, 1988
   */
  vector<rotation_translation_matrix_pair_t> solve_homography_motion_hypothesis(
    Matrix3d const& homography) {
    Eigen::JacobiSVD<Matrix3d> svd(
      homography, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Matrix3d U = absdet(svd.matrixU());
    Matrix3d V = absdet(svd.matrixV());

    if (svd.singularValues().norm() < 1e-3) {
      __logger__->warn("Homography matrix ill-conditioned");
      return {};
    }
    Vector3d s = svd.singularValues().normalized();

    // test for multiplicity of singular values
    if (isclose(s.z(), s.x(), 1e-4)) {
      __logger__->warn("Undetermined homography: multiplicity = 3");
      return {};
    }

    HomographyMotionDecompositionContext context(U, V, s);
    auto [x1, x3] = analyze_plane_normal_components(s.x(), s.y(), s.z());

    auto h1 = context.handlePositiveDPrime(x1, x3);
    auto h2 = context.handleNegativeDPrime(x1, x3);
    return views::concat(h1, h2) | ranges::to_vector;
  }
}  // namespace cyclops::initializer
