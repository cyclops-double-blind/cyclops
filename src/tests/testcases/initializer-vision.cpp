#include "cyclops_tests/data/imu.hpp"
#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/data/rotation.hpp"
#include "cyclops_tests/default.hpp"
#include "cyclops_tests/random.hpp"
#include "cyclops_tests/range.ipp"

#include "cyclops/details/initializer/vision.hpp"

#include "cyclops/details/initializer/vision/multiview.hpp"
#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/telemetry/initializer.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using cyclops::telemetry::InitializerTelemetry;

  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  static auto position_signal(timestamp_t t) {
    auto x = 3 * (1 - std::cos(t));
    return Vector3d(x, 0, 0);
  }

  static auto orientation_signal(timestamp_t t) -> Quaterniond {
    auto theta = atan2(1, cos(t));
    auto q0 = make_default_camera_rotation();
    return Eigen::AngleAxisd(theta, Vector3d::UnitZ()) * q0;
  }

  TEST_CASE("Vision bootstrap solver") {
    auto rgen = std::make_shared<std::mt19937>(2021052403);

    auto pose_signal = pose_signal_t {
      .position = position_signal,
      .orientation = orientation_signal,
    };
    auto extrinsic = se3_transform_t::Identity();
    auto timestamps = linspace(0, M_PI_2, 16) | ranges::to_vector;
    auto timestamp_lookup =
      make_dictionary<frame_id_t, timestamp_t>(views::enumerate(timestamps));

    auto motion_frames = timestamp_lookup | views::keys | ranges::to_vector;

    auto landmarks = generate_landmarks(
      *rgen, {200, Vector3d(3, 3, 0), Vector3d(1, 1, 1).asDiagonal()});

    auto multiview_image_data = make_landmark_multiview_observation(
      pose_signal, extrinsic, landmarks, timestamp_lookup);
    auto multiview_rotation_prior =
      make_multiview_rotation_prior(pose_signal, extrinsic, timestamp_lookup);

    auto config = make_default_config();
    config->initialization.vision.feature_point_isotropic_noise = 0.005;
    auto solver = VisionBootstrapSolver::create(
      config, rgen, InitializerTelemetry::createDefault());
    auto solutions =
      solver->solve(multiview_image_data, multiview_rotation_prior);
    REQUIRE_FALSE(solutions.empty());

    auto const& best_solution = *std::max_element(
      solutions.begin(), solutions.end(), [](auto const& a, auto const& b) {
        return a.geometry.landmarks.size() < b.geometry.landmarks.size();
      });
    auto const& result = best_solution.geometry;

    REQUIRE(result.camera_motions.size() != 0);
    REQUIRE(
      (result.camera_motions | views::keys | ranges::to_vector) ==
      motion_frames);

    auto init_frame = motion_frames.front();
    auto last_frame = motion_frames.back();

    auto distance = [](auto const& a, auto const& b) { return (a - b).norm(); };
    auto result_travel = distance(
      result.camera_motions.at(last_frame).translation,
      result.camera_motions.at(init_frame).translation);

    auto init_time = timestamp_lookup.at(init_frame);
    auto last_time = timestamp_lookup.at(last_frame);
    auto truth_travel =
      distance(position_signal(last_time), position_signal(init_time));

    REQUIRE(result_travel != 0);
    REQUIRE(truth_travel != 0);

    auto scale = truth_travel / result_travel;
    for (auto const& [frame_id, time] : timestamp_lookup) {
      auto const& result_motion = result.camera_motions.at(frame_id);
      auto truth_motion = compose(
        inverse(pose_signal.evaluate(init_time)), pose_signal.evaluate(time));

      auto const& q_truth = truth_motion.rotation;
      auto const& q_result = result_motion.rotation;
      CAPTURE(q_truth.coeffs().transpose());
      CAPTURE(q_result.coeffs().transpose());
      CHECK(q_truth.isApprox(q_result, 0.01));

      auto const& p_truth = truth_motion.translation;
      auto const& p_result = result_motion.translation;
      CAPTURE(p_truth.transpose());
      CAPTURE(p_result.transpose());
      CHECK(p_truth.isApprox(p_result * scale, 0.05));
    }
  }
}  // namespace cyclops::initializer
