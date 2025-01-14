#include "cyclops/details/initializer/vision/triangulation.hpp"
#include "cyclops/details/utils/vision.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"

namespace cyclops::initializer {
  using Eigen::Matrix3d;
  using Eigen::Vector2d;
  using Eigen::Vector3d;
  using Eigen::Vector4d;

  using Matrix34d = Eigen::Matrix<double, 3, 4>;

  static std::tuple<Vector2d, Vector2d> correct_triangulation_point(
    Vector2d const& u1_hat, Vector2d const& u2_hat, Matrix3d const& R,
    Vector3d const& p, int max_iterations = 4) {
    auto u1 = u1_hat;
    auto u2 = u2_hat;
    for (int i = 0; i < max_iterations; i++) {
      auto u1_h = [&]() { return u1.homogeneous(); };
      auto u2_h = [&]() { return u2.homogeneous(); };

      Vector2d g1 = R.topRows<2>() * (R.transpose() * p).cross(u2_h());

      Vector3d l12 = -R.transpose() * p.cross(u1_h());
      Vector2d g2 = l12.head<2>();

      auto g = (Vector4d() << g1, g2).finished().eval();
      auto r = (Vector4d() << u1 - u1_hat, u2 - u2_hat).finished().eval();

      auto d = u2_h().dot(l12);
      auto multiplier = (d - g.dot(r)) / g.dot(g);
      u1 = u1_hat - g.head<2>() * multiplier;
      u2 = u2_hat - g.tail<2>() * multiplier;
    }
    return std::make_tuple(u1, u2);
  }

  //
  // Cost function that takes the distance `lambda` of the feature u1,
  // then yields the projection error in the second view,
  //
  //     Q(lambda) = || project(R^T * (u1^(h) * lambda - p)) - u2 ||. ...... (*)
  //
  // Here, R, p are the rotation and translation of the camera motion from the
  // first view to the second view, x^(h) is a homogeneous representation of a
  // vector x, x^(h) := [x; 1].
  //
  // As it turns out later, the resulting cost is a fraction of two quadratics,
  //
  //                  n[0] + n[1] * lambda + n[2] * lambda^2
  //     Q(lambda) = ----------------------------------------  ............. (†)
  //                  d[0] + d[1] * lambda + d[2] * lambda^2 .
  //
  // `numerator`: the numerator quadratic n[0:2] in (†).
  // `denominator`: the denominator quadratic d[0:2] in (†).
  //
  struct triangulation_second_view_projection_error_cost_t {
    std::array<double, 3> numerator;
    std::array<double, 3> denominator;

    double evaluate(double distance) {
      auto const& [n0, n1, n2] = numerator;
      auto const& [d0, d1, d2] = denominator;

      auto N = n0 + distance * (n1 + distance * n2);
      auto D = d0 + distance * (d1 + distance * d2);
      return N / D;
    }
  };

  //
  // The cost function is obtained by the following process.
  //
  // Let us de-structure R = [A, b], where A ∈ ℝ^3x2, b ∈ ℝ^3.
  // The second-view projection is obtained by
  //
  //                                           A^T * (u1^(h) * lambda - p)
  //   project(R^T * (u1^(h) * lambda - p)) = -----------------------------
  //                                           b^T * (u1^(h) * lambda - p) .
  //
  // Substituting this into (*), one obtains
  //
  // Q(lambda) =
  //    ||  A^T * (u1^(h) * lambda - p) - u2 * b^T * (u1^(h) * lambda - p)  ||^2
  //    || ---------------------------------------------------------------- ||
  //    ||                   b^T * (u1^(h) * lambda - p)                    ||
  //
  //    ||  (A^T - u2 * b^T) * (u1^(h) * lambda - p)  ||^2
  //  = || ------------------------------------------ ||
  //    ||        b^T * (u1^(h) * lambda - p)         ||
  //
  //    || u1^(h) * lambda - p ||_W^2
  // := -----------------------------   .................................... (‡)
  //    || u1^(h) * lambda - p ||_B^2
  //
  // where
  //   W := (A - b * u2^2) * (A^2 - u2 * b^T),
  //   B := b * b^T.
  //
  // Here, both the numerator and the denominator of (‡) are quadratic
  // polynomials.
  //
  static triangulation_second_view_projection_error_cost_t
  analyze_second_view_projection_error_cost(
    Vector2d const& u1, Vector2d const& u2, Matrix3d const& R,
    Vector3d const& p) {
    auto A = R.leftCols<2>().eval();
    auto b = R.rightCols<1>().eval();

    auto L = (A.transpose() - u2 * b.transpose()).eval();
    auto w1 = (L * p).eval();
    auto w2 = (L * u1.homogeneous()).eval();

    auto n0 = w1.dot(w1);
    auto n1 = -2 * w1.dot(w2);
    auto n2 = w2.dot(w2);

    auto v1 = b.dot(p);
    auto v2 = b.dot(u1.homogeneous());

    auto d0 = v1 * v1;
    auto d1 = -2 * v1 * v2;
    auto d2 = v2 * v2;

    return {
      .numerator = {n0, n1, n2},
      .denominator = {d0, d1, d2},
    };
  }

  static auto analyze_expected_feature_distance_stddev(
    double sigma, Vector2d const& u1, double distance, Matrix3d const& R,
    Vector3d const& p) {
    auto z = (R.transpose() * (distance * u1.homogeneous() - p)).eval();
    auto m = z.z();
    auto v = (z.head<2>() / m).eval();

    auto y = (R.transpose() * u1.homogeneous()).eval();
    return sigma / ((y.head<2>() - y.z() * v) / m).norm();
  }

  //
  // The derivative of the cost Q(lambda) is a
  // quadratic-over-square-of-quadratic polynomial fraction. Specifically,
  //
  //                   a[0] + 2 * a[1] * lambda + a[2] * lambda^2
  //     Q'(lambda) = --------------------------------------------
  //                   (d[0] + d[1] * lambda + d[2] * lambda^2)^2  ,
  //
  // where
  //   a[0] = d[0] * n[1] - d[1] * n[0],
  //   a[1] = d[0] * n[2] - d[2] * n[0],
  //   a[2] = d[1] * n[2] - d[2] * n[1].
  //
  // Now we can analytically solve the critical point of Q(lambda) by the
  // quadratic formula. The minima is in either case a[2] > 0 or a[2] < 0,
  //
  //                 -a[1] + √(a[1]^2 - a[0] * a[2])
  //     lambda^* = ---------------------------------
  //                              a[2]               .
  //
  // We also evaluate lambda = ∞ and lambda = 0, then returns the optimal lambda
  // in the three optima candidates, (0, ∞, lambda^*).
  //
  static std::optional<Vector3d> triangulate_point(
    config::initializer::vision_solver_config_t const& config,
    Vector2d const& u1_hat, Vector2d const& u2_hat,
    rotation_translation_matrix_pair_t const& motion) {
    auto const& sigma = config.feature_point_isotropic_noise;
    auto const& p_value_threshold =
      config.two_view.triangulation_acceptance.min_p_value;

    auto const& R = motion.rotation;
    auto p = motion.translation.normalized().eval();
    auto scale = motion.translation.norm();

    auto [u1, u2] = correct_triangulation_point(u1_hat, u2_hat, R, p);
    auto du1 = (u1 - u1_hat).eval();
    auto du2 = (u2 - u2_hat).eval();

    auto correction_error = (du1.dot(du1) + du2.dot(du2)) / sigma / sigma;
    auto correction_p_value = 1. - chi_squared_cdf(1, correction_error);

    if (correction_p_value < p_value_threshold)
      return std::nullopt;

    auto cost = analyze_second_view_projection_error_cost(u1, u2, R, p);

    auto const& [n0, n1, n2] = cost.numerator;
    auto const& [d0, d1, d2] = cost.denominator;
    auto a0 = d0 * n1 - d1 * n0;
    auto a1 = d0 * n2 - d2 * n0;
    auto a2 = d1 * n2 - d2 * n1;

    auto distance_min = 1e-4;  // a very small distance - 0.1mm close.
    auto distance_max = 1e+4;  // a very large distance - 10km far.
    auto safeguard = [&](auto x) {
      return std::min(distance_max, std::max(distance_min, x));
    };

    auto distance = safeguard([&]() -> double {
      auto discriminant = a1 * a1 - a2 * a0;
      if (discriminant < 0) {
        if (a2 > 0)
          return distance_min;
        return distance_max;
      }

      // If a2 is small, use a taylor expansion. Note that `a1 ≈ 0` is already
      // handled above by checking the `discriminator < 0` condition.
      if (std::abs(a2) < 1e-5)
        return -a0 / 2 / a1;
      return (-a1 + std::sqrt(a1 * a1 - a0 * a2)) / a2;
    }());

    auto Q0 = cost.evaluate(distance_min);
    auto Q1 = cost.evaluate(distance);
    auto Q2 = cost.evaluate(distance_max);

    auto [Q_optimal, distance_optimal] = [&]() {
      if (Q0 < Q1) {
        if (Q0 < Q2) {
          // Q0 < Q1, Q2
          return std::make_tuple(Q0, distance_min);
        } else {
          // Q2 < Q0 < Q1
          return std::make_tuple(Q2, distance_max);
        }
      } else {
        if (Q2 < Q1) {
          // Q2 < Q1 < Q0
          return std::make_tuple(Q2, distance_max);
        } else {
          // Q1 < Q2, Q0
          return std::make_tuple(Q1, distance);
        }
      }
    }();

    auto depth_p_value = 1. - chi_squared_cdf(1, Q_optimal / sigma / sigma);
    if (depth_p_value < p_value_threshold)
      return std::nullopt;

    auto const& deviation_threshold =
      config.two_view.triangulation_acceptance.max_normalized_deviation;

    // Reject if the expected deviation of the distance is large.
    auto distance_stddev = analyze_expected_feature_distance_stddev(
      sigma, u1, distance_optimal, R, p);
    if (distance_stddev > deviation_threshold * distance_optimal)
      return std::nullopt;

    return scale * distance_optimal * u1.homogeneous();
  }

  two_view_triangulation_t triangulate_two_view_feature_pairs(
    config::initializer::vision_solver_config_t const& config,
    std::map<landmark_id_t, two_view_feature_pair_t> const& features,
    std::set<landmark_id_t> const& feature_ids,
    rotation_translation_matrix_pair_t const& motion) {
    auto const& sigma = config.feature_point_isotropic_noise;
    auto const& p_value_threshold =
      config.two_view.triangulation_acceptance.min_p_value;

    two_view_triangulation_t result = {0, 0, 0., {}};
    for (auto feature_id : feature_ids) {
      auto const& [u1_hat, u2_hat] = features.at(feature_id);

      auto maybe_f = triangulate_point(config, u1_hat, u2_hat, motion);
      if (!maybe_f) {
        result.n_triangulation_failure++;
        continue;
      }

      auto const& f = *maybe_f;
      auto const& [R, p] = motion;
      Vector3d z = R.transpose() * (f - p);

      Vector2d u1 = f.head<2>() / f.z();
      Vector2d u2 = z.head<2>() / z.z();

      auto r1 = (u1_hat - u1).eval();
      auto r2 = (u2_hat - u2).eval();

      auto residual = (Vector4d() << r1, r2).finished().eval();
      auto cost = residual.dot(residual) / sigma / sigma;
      auto cost_probability = 1. - chi_squared_cdf(1, cost);

      if (cost_probability < p_value_threshold) {
        result.n_error_probability_test_failure++;
        continue;
      }

      result.expected_inliers += cost_probability;
      result.landmarks.emplace(feature_id, f);
    }
    return result;
  }
}  // namespace cyclops::initializer
