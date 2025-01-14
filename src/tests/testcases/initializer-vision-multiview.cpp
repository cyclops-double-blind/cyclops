#include "cyclops/details/initializer/vision/multiview.hpp"
#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/telemetry/initializer.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"

#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/data/rotation.hpp"
#include "cyclops_tests/default.hpp"
#include "cyclops_tests/range.ipp"

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

  static auto make_multiview_data(
    std::mt19937& rgen, pose_signal_t pose, se3_transform_t const& extrinsic) {
    auto timestamps = linspace(0, M_PI_2, 8) | ranges::to_vector;
    auto motion_frame_ids = views::ints(0, 8) | ranges::to_vector;
    auto motion_timestamps = make_dictionary<frame_id_t, timestamp_t>(
      views::zip(motion_frame_ids, timestamps));

    auto landmarks = generate_landmarks(
      rgen, {200, Vector3d(3, 3, 0), Vector3d(1, 1, 1).asDiagonal()});
    auto image_data = make_landmark_multiview_observation(
      pose, extrinsic, landmarks, motion_timestamps);

    return std::make_tuple(motion_frame_ids, motion_timestamps, image_data);
  }

  static auto resolve_vision_pose_gauge(
    se3_transform_t const& se3_gauge, double scale_gauge,
    se3_transform_t const& camera_pose) {
    auto [p, q] = compose(inverse(se3_gauge), camera_pose);
    return se3_transform_t {p * scale_gauge, q};
  }

  TEST_CASE("Multiview vision reconstruction") {
    auto rgen = std::make_shared<std::mt19937>(20240513006);

    auto config = make_default_config();
    config->extrinsics.imu_camera_transform = se3_transform_t::Identity();
    auto const& camera_extrinsic = config->extrinsics.imu_camera_transform;

    GIVEN("Random generated multi-view data") {
      auto pose_signal = pose_signal_t {position_signal, orientation_signal};
      auto [motion_frames, motion_timestamps, image_data] =
        make_multiview_data(*rgen, pose_signal, camera_extrinsic);
      auto rotation_prior = make_multiview_rotation_prior(
        pose_signal, camera_extrinsic, motion_timestamps);

      WHEN("Solved multi-view geometry") {
        auto solver = MultiviewVisionGeometrySolver::create(
          config, rgen, InitializerTelemetry::createDefault());
        auto possible_solutions = solver->solve(image_data, rotation_prior);
        REQUIRE_FALSE(possible_solutions.empty());

        auto solution = ranges::max_element(
          possible_solutions, [](auto const& a, auto const& b) {
            return a.landmarks.size() < b.landmarks.size();
          });
        REQUIRE(solution != possible_solutions.end());
        auto const& camera_motions = solution->camera_motions;

        REQUIRE(
          (camera_motions | views::keys | ranges::to<std::set>) ==
          (image_data | views::keys | ranges::to<std::set>));
        REQUIRE(camera_motions.size() == motion_frames.size());

        auto init_frame_id = motion_frames.front();
        auto last_frame_id = motion_frames.back();
        auto init_time = motion_timestamps.at(init_frame_id);
        auto last_time = motion_timestamps.at(last_frame_id);

        auto distance = [](auto const& x, auto const& y) {
          return (x - y).norm();
        };
        auto result_travel = distance(
          camera_motions.at(init_frame_id).translation,
          camera_motions.at(last_frame_id).translation);
        auto truth_travel =
          distance(position_signal(init_time), position_signal(last_time));

        REQUIRE(result_travel != 0);
        REQUIRE(truth_travel != 0);

        auto scale = truth_travel / result_travel;

        THEN("The result camera motions are up-to-gauge correct to the truth") {
          for (auto const& [motion_frame_id, time] : motion_timestamps) {
            auto result_motion = resolve_vision_pose_gauge(
              camera_motions.at(last_frame_id), scale,
              camera_motions.at(motion_frame_id));
            auto true_motion = resolve_vision_pose_gauge(
              pose_signal.evaluate(last_time), 1, pose_signal.evaluate(time));

            auto const& q_truth = true_motion.rotation;
            auto const& p_truth = true_motion.translation;
            auto const& q_result = result_motion.rotation;
            auto const& p_result = result_motion.translation;

            CAPTURE(q_truth.coeffs().transpose());
            CAPTURE(q_result.coeffs().transpose());
            CHECK(q_truth.isApprox(q_result));

            CAPTURE(p_truth.transpose());
            CAPTURE(p_result.transpose());
            CHECK(p_truth.isApprox(p_result));
          }
        }
      }
    }
  }
}  // namespace cyclops::initializer
