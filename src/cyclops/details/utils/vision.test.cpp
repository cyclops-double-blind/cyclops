#include "cyclops/details/utils/vision.cpp"
#include "cyclops_tests/random.hpp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops {
  namespace views = ranges::views;

  auto constexpr n_frames = 10;

  static auto project(Vector3d const& h) {
    return (h.head<2>() / h.z()).eval();
  }

  TEST_CASE("Triangulation from multi-view features") {
    auto pose_sequence =
      views::ints(0, n_frames) | views::transform([&](auto i) {
        auto t = static_cast<double>(i) / n_frames;
        auto theta = (-M_PI / 4) * t + 0.5;
        auto x = t + 0.5;

        auto rotation = Eigen::AngleAxisd(theta, Vector3d::UnitY());
        auto position = Vector3d(x, 0, 0);
        auto pose =
          rotation_translation_matrix_pair_t {rotation.matrix(), position};

        return std::make_pair(i, pose);
      }) |
      ranges::to<std::map<frame_id_t, rotation_translation_matrix_pair_t>>;

    auto f = Vector3d(1.0, 0, 1);
    auto features =  //
      pose_sequence | views::transform([&](auto const& id_pose) {
        auto const& [id, pose] = id_pose;
        auto const& [R, p] = pose;
        auto point = project(R.transpose() * (f - p));
        return std::make_pair(
          id, feature_point_t {point, Eigen::Matrix2d::Identity()});
      }) |
      ranges::to<std::map<frame_id_t, feature_point_t>>;

    auto const maybe_f_got = triangulate_point(features, pose_sequence);
    REQUIRE(static_cast<bool>(maybe_f_got));

    auto const& f_got = *maybe_f_got;

    CAPTURE(f.transpose());
    CAPTURE(f_got.transpose());
    CHECK(f.isApprox(f_got));
  }
}  // namespace cyclops
