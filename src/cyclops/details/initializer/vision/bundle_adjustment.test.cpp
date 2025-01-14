#include "cyclops/details/initializer/vision/bundle_adjustment.cpp"
#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"

#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/default.hpp"
#include "cyclops_tests/random.hpp"
#include "cyclops_tests/range.ipp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  static auto constexpr n_frames = 8;

  static auto position_signal(timestamp_t t) {
    auto x = 3 * (1 - std::cos(t));
    return Vector3d(x, 0, 0);
  }

  static auto orientation_signal(timestamp_t t) -> Quaterniond {
    auto theta = atan2(1, cos(t));
    auto q0 = make_default_camera_rotation();
    return Eigen::AngleAxisd(theta, Vector3d::UnitZ()) * q0;
  }

  static auto make_multiview_landmark_observation(
    std::mt19937& rgen, pose_signal_t pose_signal,
    std::map<frame_id_t, timestamp_t> motion_timestamps) {
    auto landmarks = generate_landmarks(
      rgen, {200, Vector3d(3, 3, 0), Vector3d(1, 1, 1).asDiagonal()});
    auto image_data = make_landmark_multiview_observation(
      pose_signal, se3_transform_t::Identity(), landmarks, motion_timestamps);
    return std::make_tuple(landmarks, image_data);
  }

  static auto make_multiview_geometry_guess(
    std::mt19937& rgen, std::map<frame_id_t, timestamp_t> motion_timestamps,
    pose_signal_t pose_signal, landmark_positions_t const& landmarks) {
    REQUIRE_FALSE(motion_timestamps.empty());
    auto [_, init_time] = *motion_timestamps.begin();
    auto init_pose = pose_signal.evaluate(init_time);

    auto camera_motions_estimated =  //
      motion_timestamps | views::transform([&](auto _) {
        auto [frame_id, time] = _;
        auto [p, q] = compose(inverse(init_pose), pose_signal.evaluate(time));
        auto p_perturbed = perturbate((2 * p).eval(), 0.1, rgen);
        auto q_perturbed = perturbate(q, 0.1, rgen);

        return std::make_pair(frame_id, se3_transform_t {p, q});
      }) |
      ranges::to<std::map<frame_id_t, se3_transform_t>>;

    auto landmarks_estimated =  //
      landmarks | views::transform([&](auto const& id_landmark) {
        auto const& [landmark_id, landmark] = id_landmark;
        auto const& [p0, q0] = init_pose;
        auto f = (q0.conjugate() * (landmark - p0)).eval();

        return std::make_pair(
          landmark_id, perturbate((2 * f).eval(), 0.1, rgen));
      }) |
      ranges::to<landmark_positions_t>;

    return multiview_geometry_t {camera_motions_estimated, landmarks_estimated};
  }

  TEST_CASE("Bundle adjustment") {
    std::mt19937 rgen(20240513007);

    auto timestamps = linspace(0, M_PI_2, n_frames) | ranges::to_vector;
    auto motion_frames = views::ints(0, n_frames) | ranges::to_vector;
    auto motion_timestamps = make_dictionary<frame_id_t, timestamp_t>(
      views::zip(motion_frames, timestamps));
    auto pose_signal = pose_signal_t {position_signal, orientation_signal};

    auto [landmarks, image_data] =
      make_multiview_landmark_observation(rgen, pose_signal, motion_timestamps);
    auto geometry_guess = make_multiview_geometry_guess(
      rgen, motion_timestamps, pose_signal, landmarks);

    auto config = config::initializer::vision_solver_config_t::createDefault();
    auto maybe_solution =
      solve_bundle_adjustment(config, geometry_guess, image_data);
    REQUIRE(maybe_solution.has_value());

    auto const& camera_motions = maybe_solution->geometry.camera_motions;
    REQUIRE(camera_motions.size() == n_frames);
    REQUIRE(
      (camera_motions | views::keys | ranges::to<std::set>) ==
      (motion_timestamps | views::keys | ranges::to<std::set>));

    THEN("The resulting Fisher information is positive semidefinite") {
      auto const& H = maybe_solution->motion_information_weight;
      REQUIRE(H.rows() == n_frames * 6);
      REQUIRE(H.cols() == n_frames * 6);
      REQUIRE(H.isApprox(H.transpose()));

      auto lambda = H.selfadjointView<Eigen::Upper>().eigenvalues().eval();

      // 1e-6: Margin to avoid the numerical inaccuracy
      REQUIRE(lambda.x() > -1e-6);

      AND_THEN("The resulting Fisher information experiences 6-DoF symmetry") {
        REQUIRE(lambda.head(6).norm() < 1e-6);
        REQUIRE(lambda.head(7).norm() > 1e-6);
      }
    }

    auto distance = [](auto const& a, auto const& b) { return (a - b).norm(); };

    auto init_time = timestamps.front();
    auto last_time = timestamps.back();
    auto true_travel =
      distance(position_signal(init_time), position_signal(last_time));

    auto init_frame = motion_frames.front();
    auto last_frame = motion_frames.back();
    auto result_travel = distance(
      camera_motions.at(init_frame).translation,
      camera_motions.at(last_frame).translation);
    auto scale = true_travel / result_travel;

    for (auto const& [frame_id, time] : motion_timestamps) {
      auto result_motion = compose(
        inverse(camera_motions.at(init_frame)), camera_motions.at(frame_id));
      auto true_motion = compose(
        inverse(pose_signal.evaluate(init_time)), pose_signal.evaluate(time));

      auto const& q_true = true_motion.rotation;
      auto const& p_true = true_motion.translation;
      auto const& q_result = result_motion.rotation;
      auto const& p_result = result_motion.translation;

      CAPTURE(frame_id);
      CAPTURE(q_true.coeffs().transpose());
      CAPTURE(q_result.coeffs().transpose());
      CHECK(q_true.isApprox(q_result, 1e-6));

      CAPTURE(p_true.transpose());
      CAPTURE(scale * p_result.transpose());
      CHECK(p_true.isApprox(p_result * scale, 1e-6));
    }
  }
}  // namespace cyclops::initializer
