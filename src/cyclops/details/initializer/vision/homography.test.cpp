#include "cyclops/details/initializer/vision/homography.cpp"
#include "cyclops_tests/random.hpp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops::initializer {
  using Eigen::Vector2d;
  using Eigen::Vector3d;

  namespace views = ranges::views;

  TEST_CASE("Homography matrix computation") {
    std::mt19937 rgen(20240513002);

    GIVEN("Random-generated homography matrix") {
      Eigen::Matrix3d H;
      for (auto [i, j] :
           views::cartesian_product(views::ints(0, 3), views::ints(0, 3))) {
        std::uniform_real_distribution<double> dist(-1, 1);
        H(i, j) = dist(rgen);
      }
      H.normalize();
      if (H.determinant() < 0)
        H = -H;
      CAPTURE(H);

      GIVEN(
        "Two-view feature pairs corresponding to the homography transform") {
        std::map<landmark_id_t, two_view_feature_pair_t> feature_frame;
        for (auto feature_id : views::ints(0, 32)) {
          std::uniform_real_distribution<double> dist(-1, 1);
          auto const f_1 = Vector2d(dist(rgen), dist(rgen));
          auto const f_2 = project(H * f_1.homogeneous());

          feature_frame.emplace(feature_id, std::make_tuple(f_1, f_2));
        }

        WHEN("Solved homography by the random consensus") {
          std::vector<std::set<landmark_id_t>> batch = {
            {0, 7, 25, 28}, {3, 8, 19, 26},  {5, 12, 24, 31},
            {4, 9, 17, 27}, {2, 15, 18, 23}, {11, 13, 21, 29},
          };

          auto const result =
            analyze_two_view_homography(1e-6, batch, feature_frame);

          THEN("The result is same to the initially given homography matrix") {
            CAPTURE(result.homography);
            CHECK(result.expected_inliers > 0.);
            CHECK(result.homography.isApprox(H));
            CHECK(result.inliers.size() == 32);
          }
        }
      }
    }
  }
}  // namespace cyclops::initializer
