#include "cyclops/details/initializer/initializer.hpp"
#include "cyclops/details/initializer/solver.hpp"
#include "cyclops/details/initializer/vision_imu.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/telemetry/initializer.hpp"
#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/config.hpp"

#include "cyclops_tests/mockups/data_provider.hpp"
#include "cyclops_tests/mockups/keyframe_manager.hpp"
#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/default.hpp"
#include "cyclops_tests/range.ipp"

#include <range/v3/all.hpp>

#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  using measurement::KeyframeManagerMock;
  using measurement::make_measurement_provider_mockup;
  using telemetry::InitializerTelemetry;

  static auto make_position_signal(Vector3d const& axis) {
    return [axis](timestamp_t t) -> Vector3d {
      return Vector3d(-1.3, +1.3, 0.) + 0.5 * std::sin(t * 0.57 * M_PI) * axis;
    };
  }

  static auto make_orientation_signal(
    Quaterniond const& q_bc, Vector3d const& axis) {
    return [q_bc, axis](timestamp_t t) -> Quaterniond {
      auto angle = 0.5 * t;
      return q_bc * Eigen::AngleAxisd(angle, axis) * q_bc.conjugate();
    };
  }

  static auto make_data_provider_mockup(
    std::mt19937& rgen, se3_transform_t const& extrinsic,
    pose_signal_t pose_signal, landmark_positions_t const& landmarks) {
    auto timestamps = linspace(0., 1., 10) | ranges::to_vector;
    auto motion_timestamp_lookup =
      make_dictionary<frame_id_t, timestamp_t>(views::enumerate(timestamps));

    std::shared_ptr data_provider = make_measurement_provider_mockup(
      pose_signal, extrinsic, landmarks, motion_timestamp_lookup);

    return std::make_tuple(motion_timestamp_lookup, data_provider);
  }

  static auto as_se3_transform(imu_motion_state_t const& x) {
    auto const& [q, p, v] = x;
    return se3_transform_t {p, q};
  }

  TEST_CASE("Initializer main") {
    auto rgen = std::make_shared<std::mt19937>(20220803);

    auto config = make_default_config();
    auto const& extrinsic = config->extrinsics.imu_camera_transform;

    GIVEN("Vision and IMU data from the sinusoidal body motion") {
      auto translation_axis = Vector3d(1, 1, 0.5).normalized().eval();
      auto orientation_axis = Vector3d(1, 1, 0.5).normalized().eval();

      auto pose_signal = pose_signal_t {
        make_position_signal(translation_axis),
        make_orientation_signal(extrinsic.rotation, orientation_axis),
      };

      GIVEN("3D-uniformly distributed landmark positions") {
        auto landmarks = generate_landmarks(
          *rgen, {200, Vector3d(0, 0, 0), Vector3d(1, 1, 1).asDiagonal()});
        auto [motion_timestamps, data_provider] =
          make_data_provider_mockup(*rgen, extrinsic, pose_signal, landmarks);

        std::shared_ptr telemetry = InitializerTelemetry::createDefault();
        auto solver_internal = InitializationSolverInternal::create(
          rgen, config, data_provider, telemetry);

        WHEN("Invoked the initialization solver core") {
          auto solution = solver_internal->solve();

          THEN(
            "Despite the unprovided guarantee, the acceptable vision solution "
            "from the epipolar geometry model is only one in almost cases") {
            CHECK(solution.vision_solutions.size() == 1);

            AND_THEN("Therefore, the IMU match solution is also only one") {
              CHECK(solution.imu_solutions.size() == 1);

              auto [imu_match_vision_solution_index, _] =
                solution.imu_solutions.front();
              CHECK(imu_match_vision_solution_index == 0);
            }
          }
        }

        WHEN("Invoked the main initializer") {
          auto keyframe_manager = std::make_shared<KeyframeManagerMock>();
          keyframe_manager->_keyframes = motion_timestamps;

          auto initializer = InitializerMain::create(
            std::move(solver_internal), keyframe_manager, telemetry);

          auto maybe_solution = initializer->solve();
          THEN("The solution comes out") {
            REQUIRE(maybe_solution.has_value());
            auto const& solution = maybe_solution.value();

            AND_THEN("The bias estimation is correct") {
              // The accelerometer bias is not perfectly observable.
              CHECK(solution.acc_bias.norm() < 0.01);
              CHECK(solution.gyr_bias.norm() == doctest::Approx(0));
            }

            AND_THEN("The motion estimations are correct up to SE(3) gauge") {
              auto init_frame =
                (motion_timestamps | views::keys | ranges::to_vector).front();
              auto init_time = motion_timestamps.at(init_frame);

              auto result_init_motion =
                as_se3_transform(solution.motions.at(init_frame));
              auto actual_init_motion = pose_signal.evaluate(init_time);

              for (auto [frame_id, time] : motion_timestamps) {
                auto result_motion = compose(
                  inverse(result_init_motion),
                  as_se3_transform(solution.motions.at(frame_id)));
                auto actual_motion = compose(
                  inverse(pose_signal.evaluate(init_time)),
                  pose_signal.evaluate(time));

                CAPTURE(result_motion.translation.transpose());
                CAPTURE(result_motion.rotation.coeffs().transpose());
                CAPTURE(actual_motion.translation.transpose());
                CAPTURE(actual_motion.rotation.coeffs().transpose());

                auto const& [p_result, q_result] = result_motion;
                auto const& [p_actual, q_actual] = actual_motion;
                CHECK(p_result.isApprox(p_actual, 1e-3));
                CHECK(q_result.isApprox(q_actual, 1e-3));
              }

              AND_THEN(
                "The rotation gauge is correct up to the gravity direction") {
                for (auto [frame_id, time] : motion_timestamps) {
                  auto x_result = solution.motions.at(frame_id);
                  auto x_actual = pose_signal.evaluate(time);

                  auto const& q_result = x_result.orientation;
                  auto const& q_actual = x_actual.rotation;

                  Vector3d g_result = q_result.conjugate() * Vector3d::UnitZ();
                  Vector3d g_actual = q_actual.conjugate() * Vector3d::UnitZ();

                  CAPTURE(g_result.transpose());
                  CAPTURE(g_actual.transpose());
                  auto error_angle = std::acos(g_result.dot(g_actual));

                  CHECK(error_angle < 5e-3);
                }
              }
            }
          }
        }
      }
    }
  }
}  // namespace cyclops::initializer
