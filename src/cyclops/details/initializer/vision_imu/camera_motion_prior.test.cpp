#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include <range/v3/all.hpp>
#include <set>

#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using std::set;

  using Eigen::Matrix3d;
  using Eigen::MatrixXd;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  static Matrix3d make_random_positive_definite_matrix(std::mt19937& rgen) {
    Matrix3d S;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++)
        S(i, j) = std::uniform_real_distribution(-1.0, 1.0)(rgen);
    }
    return S.transpose() * S;
  }

  TEST_CASE("test camera motion prior generation") {
    vision_bootstrap_solution_t monocular_sfm;
    monocular_sfm.geometry.camera_motions = {
      {0, se3_transform_t {Vector3d::UnitX(), Quaterniond(1, 0, 0, 0)}},
      {1, se3_transform_t {Vector3d::UnitY(), Quaterniond(0, 1, 0, 0)}},
    };
    monocular_sfm.motion_information_weight = MatrixXd::Zero(12, 12);

    std::mt19937 rgen(20220510);
    auto W_R2 = make_random_positive_definite_matrix(rgen);
    auto W_p2 = make_random_positive_definite_matrix(rgen);

    monocular_sfm.motion_information_weight.block(6, 6, 3, 3) = W_R2;
    monocular_sfm.motion_information_weight.block(9, 9, 3, 3) = W_p2;

    auto [rotation_prior, translation_prior] =
      make_imu_match_camera_motion_prior(monocular_sfm);

    REQUIRE(
      (rotation_prior.rotations | views::keys | ranges::to<set>) ==
      set<frame_id_t> {0, 1});
    REQUIRE(
      (translation_prior.translations | views::keys | ranges::to<set>) ==
      set<frame_id_t> {0, 1});

    CHECK(rotation_prior.rotations.at(0).isApprox(Quaterniond(1, 0, 0, 0), 0));
    CHECK(rotation_prior.rotations.at(1).isApprox(Quaterniond(0, 1, 0, 0), 0));
    CHECK(translation_prior.translations.at(0) == Vector3d(1, 0, 0));
    CHECK(translation_prior.translations.at(1) == Vector3d(0, 1, 0));

    CHECK(translation_prior.weight == W_p2);
    CHECK(rotation_prior.weight == W_R2);
  }
}  // namespace cyclops::initializer
