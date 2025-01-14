#include "cyclops/details/initializer/vision/twoview.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/utils/vision.hpp"

#include "cyclops/details/config.hpp"

#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/default.hpp"
#include "cyclops_tests/random.hpp"

#include <range/v3/all.hpp>
#include <iostream>

#include <doctest/doctest.h>

namespace cyclops::initializer {
  using std::map;
  using std::set;
  using std::vector;

  using Eigen::AngleAxisd;
  using Eigen::Matrix3d;
  using Eigen::Quaterniond;
  using Eigen::Vector2d;
  using Eigen::Vector3d;

  namespace views = ranges::views;

  static Vector2d project(Vector3d const& x) {
    return x.head<2>() / x.z();
  }

  TEST_CASE("Two-view vision pair reconstruction") {
    GIVEN("Random camera motion") {
      auto p = Vector3d(1.0, 0, 0);
      auto R = AngleAxisd(-M_PI / 4, Vector3d::UnitY()).matrix().eval();

      GIVEN("Random-generated landmarks and their two-view observations") {
        auto rgen = std::make_shared<std::mt19937>(20240513004);
        auto landmarks = generate_landmarks(
          views::ints(0, 200) | ranges::to<set<landmark_id_t>>,
          [&](auto _) -> Vector3d {
            auto dist = std::uniform_real_distribution<>(-1, 1);
            Vector3d const origin = Vector3d(0.5, 0., 1);
            return origin +
              0.5 * Vector3d(dist(*rgen), dist(*rgen), dist(*rgen));
          });

        auto view0 = generate_landmark_observations(
          Matrix3d::Identity(), Vector3d::Zero(), landmarks);
        auto view1 = generate_landmark_observations(R, p, landmarks);

        auto correspondence = two_view_correspondence_data_t {
          .rotation_prior =
            {Eigen::Quaterniond(R), 1e-6 * Matrix3d::Identity()},
          .features =
            views::set_intersection(view0 | views::keys, view1 | views::keys) |
            views::transform([&](auto landmark_id) {
              auto const& u0 = view0.at(landmark_id).point;
              auto const& u1 = view1.at(landmark_id).point;
              auto feature_pair = std::make_tuple(u0, u1);

              return std::make_pair(landmark_id, feature_pair);
            }) |
            ranges::to<std::map<landmark_id_t, two_view_feature_pair_t>>,
        };

        WHEN("Solved for possible motion geometry") {
          auto config = make_default_config();
          auto solver = TwoViewVisionGeometrySolver::create(config, rgen);
          auto possible_solutions = solver->solve(correspondence);
          REQUIRE_FALSE(possible_solutions.empty());

          AND_WHEN(
            "Polled the solution with the maximum landmark reconstruction") {
            auto const& result = *std::max_element(
              possible_solutions.begin(), possible_solutions.end(),
              [](auto const& a, auto const& b) {
                return a.landmarks.size() < b.landmarks.size();
              });
            REQUIRE(result.landmarks.size() >= 50);

            THEN("The rotation matrix is correct") {
              CHECK(result.camera_motion.rotation.matrix().isApprox(R));

              AND_THEN("The translation is up-to-scale correct") {
                Vector3d const p_unit_truth = p.normalized();
                Vector3d const p_unit_got =
                  result.camera_motion.translation.normalized();
                CHECK(p_unit_truth.isApprox(p_unit_got));

                double const scale =
                  p.norm() / result.camera_motion.translation.norm();
                AND_THEN("Landmark positions are up-to-scale correct") {
                  for (auto const& [id, x] : result.landmarks) {
                    CHECK(x.isApprox(scale * landmarks.at(id)));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}  // namespace cyclops::initializer
