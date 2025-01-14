#include "cyclops/details/initializer/vision/hypothesis.hpp"
#include "cyclops/details/initializer/vision/epipolar.hpp"
#include "cyclops/details/initializer/vision/homography.hpp"
#include "cyclops/details/utils/vision.hpp"
#include "cyclops/details/config.hpp"

#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/default.hpp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops::initializer {
  using Eigen::Matrix3d;
  using Eigen::Vector3d;

  namespace views = ranges::views;

  static auto project(Vector3d const& x) {
    return (x.head<2>() / x.z()).eval();
  }

  static std::map<landmark_id_t, two_view_feature_pair_t> make_feature_pairs(
    Matrix3d const& R, Vector3d const& p,
    std::map<landmark_id_t, Vector3d> const& landmarks) {
    auto feature_pairs =
      views::for_each(landmarks, [&](auto const& id_landmark) {
        auto const& [id, x1] = id_landmark;
        Vector3d const x2 = R.transpose() * (x1 - p);

        auto const u1 = project(x1);
        auto const u2 = project(x2);
        auto&& contained = [](auto x, auto a, auto b) {
          return x > a && x < b;
        };
        auto u1_in_fov =
          x1.z() > 1e-3 && contained(u1.x(), -1, 1) && contained(u1.y(), -1, 1);
        auto u2_in_fov =
          x2.z() > 1e-3 && contained(u2.x(), -1, 1) && contained(u2.y(), -1, 1);

        return ranges::yield_if(
          u1_in_fov && u2_in_fov, std::make_pair(id, std::make_tuple(u1, u2)));
      });
    return feature_pairs |
      ranges::to<std::map<landmark_id_t, two_view_feature_pair_t>>;
  }

  // Artificially generates landmark observation data.
  static auto make_artificial_two_view_geometry(
    std::mt19937& rgen, Matrix3d const& R, Vector3d const& p,
    Vector3d const& landmark_dispersion) {
    auto landmark_ids =
      views::ints(0, 200) | ranges::to<std::set<landmark_id_t>>;
    auto uniform_distribution = std::uniform_real_distribution<>(-1, 1);
    auto rnd = [&]() { return 0.5 * uniform_distribution(rgen); };

    auto landmarks = generate_landmarks(landmark_ids, [&](auto _) -> Vector3d {
      auto dx = landmark_dispersion.x() * rnd();
      auto dy = landmark_dispersion.y() * rnd();
      auto dz = landmark_dispersion.z() * rnd();
      return Vector3d(0.5, 0., 1.5) + Vector3d(dx, dy, dz);
    });
    auto common_features = make_feature_pairs(R, p, landmarks);

    auto ransac_batch =
      views::ints(0, 200) | views::transform([&](auto _) {
        return landmark_ids | views::sample(8, rgen) | ranges::to<std::set>;
      }) |
      ranges::to_vector;
    return std::make_tuple(landmarks, common_features, ransac_batch);
  }

  static auto test_motion_hypothesis_selection(
    Matrix3d const& R, Vector3d const& p, landmark_positions_t const& landmarks,
    std::map<landmark_id_t, two_view_feature_pair_t> const& two_view_features,
    std::vector<rotation_translation_matrix_pair_t> const& motion_hypotheses) {
    auto config = make_default_config();
    auto selector = TwoViewMotionHypothesisSelector::create(config);

    auto rotation_prior = two_view_imu_rotation_data_t {
      .value = Eigen::Quaterniond(R),
      .covariance = 1e-6 * Matrix3d::Identity(),
    };

    auto selections = selector->selectPossibleMotions(
      motion_hypotheses, two_view_features,
      landmarks | views::keys | ranges::to<std::set>, rotation_prior);
    REQUIRE_FALSE(selections.empty());

    auto const& result = *std::max_element(
      selections.begin(), selections.end(), [](auto const& a, auto const& b) {
        return a.landmarks.size() < b.landmarks.size();
      });

    THEN("The rotation matrix is correct") {
      CHECK(result.camera_motion.rotation.matrix().isApprox(R));

      AND_THEN("The translation is up-to-scale correct") {
        Vector3d p_unit_truth = p.normalized();
        Vector3d p_unit_got = result.camera_motion.translation.normalized();
        CHECK(p_unit_truth.isApprox(p_unit_got));

        auto scale = p.norm() / result.camera_motion.translation.norm();
        AND_THEN("Landmark positions are up-to-scale correct") {
          for (auto const& [id, x] : landmarks)
            CHECK(x.isApprox(scale * result.landmarks.at(id)));
        }
      }
    }
  }

  TEST_CASE("Motion hypothesis selection") {
    std::mt19937 rgen(20240513003);

    GIVEN("Random camera motion") {
      auto p = Vector3d(1.0, 0, 0);
      auto R = Eigen::AngleAxisd(-M_PI / 4, Vector3d::UnitY()).matrix().eval();

      GIVEN("Planary distributed landmark positions") {
        auto landmark_dispersion = Vector3d(1, 1, 0);
        auto [landmarks, two_view_features, ransac_batch] =
          make_artificial_two_view_geometry(rgen, R, p, landmark_dispersion);

        REQUIRE(landmarks.size() == 200);
        REQUIRE(
          (landmarks | views::keys | ranges::to<std::set>) ==
          (two_view_features | views::keys | ranges::to<std::set>));

        WHEN("Solve two-view geometry by the homography model") {
          auto geometry =
            analyze_two_view_homography(0.001, ransac_batch, two_view_features);
          REQUIRE(geometry.expected_inliers > 0);
          REQUIRE(geometry.inliers.size() == 200);

          auto motion_hypotheses =
            solve_homography_motion_hypothesis(geometry.homography);
          REQUIRE(motion_hypotheses.size() == 8);

          test_motion_hypothesis_selection(
            R, p, landmarks, two_view_features, motion_hypotheses);
        }
      }

      GIVEN("Cube-distributed landmark positions") {
        auto landmark_dispersion = Vector3d(1, 1, 1);
        auto [landmarks, two_view_features, ransac_batch] =
          make_artificial_two_view_geometry(rgen, R, p, landmark_dispersion);

        REQUIRE(landmarks.size() == 200);
        REQUIRE(
          (landmarks | views::keys | ranges::to<std::set>) ==
          (two_view_features | views::keys | ranges::to<std::set>));

        WHEN("Solve two-view geometry by the epipolar model") {
          auto geometry =
            analyze_two_view_epipolar(0.001, ransac_batch, two_view_features);
          REQUIRE(geometry.expected_inliers > 0);
          REQUIRE(geometry.inliers.size() == 200);

          auto motion_hypotheses =
            solve_epipolar_motion_hypothesis(geometry.essential_matrix);
          REQUIRE(motion_hypotheses.size() == 4);

          test_motion_hypothesis_selection(
            R, p, landmarks, two_view_features, motion_hypotheses);
        }
      }
    }
  }
}  // namespace cyclops::initializer
