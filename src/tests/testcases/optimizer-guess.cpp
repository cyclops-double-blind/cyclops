#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/mockups/data_provider.hpp"
#include "cyclops_tests/mockups/keyframe_manager.hpp"
#include "cyclops_tests/default.hpp"
#include "cyclops_tests/range.ipp"
#include "cyclops_tests/signal.ipp"

#include "cyclops/details/estimation/optimizer_guess.hpp"
#include "cyclops/details/estimation/state/accessor_read.hpp"
#include "cyclops/details/estimation/state/accessor_write.hpp"
#include "cyclops/details/estimation/state/state_internal.hpp"

#include "cyclops/details/initializer/initializer.hpp"
#include "cyclops/details/initializer/solver.hpp"

#include "cyclops/details/measurement/preintegration.hpp"

#include "cyclops/details/telemetry/initializer.hpp"
#include "cyclops/details/config.hpp"

#include <range/v3/all.hpp>
#include <fstream>

#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  using estimation::landmark_parameter_blocks_t;
  using estimation::motion_frame_parameter_blocks_t;

  static auto position_signal(timestamp_t t) {
    auto x = t - sin(2 * M_PI * t) / (4 * M_PI);
    auto y = cos(2 * M_PI * t) / 10;
    auto z = sin(2 * M_PI * t + 0.3) / 10;
    return Vector3d(x, y, z);
  }

  static auto orientation_signal(timestamp_t t) {
    auto const theta = 0.1 * std::sin(2 * M_PI * t) + M_PI_2;
    return Quaterniond(Eigen::AngleAxisd(theta, Vector3d::UnitZ()));
  }

  static auto evaluate_motion_states(
    pose_signal_t pose_signal, std::map<frame_id_t, timestamp_t> timestamps) {
    auto velocity_signal = numeric_derivative<Vector3d>(pose_signal.position);

    return  //
      timestamps | views::transform([&](auto pair) {
        auto [frame_id, t] = pair;
        auto [p, q] = pose_signal.evaluate(t);
        auto v = velocity_signal(t);
        auto motion =
          imu_motion_state_t {.orientation = q, .position = p, .velocity = v};
        return std::make_pair(frame_id, motion);
      }) |
      ranges::to<std::map<frame_id_t, imu_motion_state_t>>;
  }

  static auto make_initializer(
    std::shared_ptr<std::mt19937> rgen, landmark_positions_t const& landmarks,
    pose_signal_t pose_signal, std::map<frame_id_t, timestamp_t> timestamps) {
    auto config = make_default_config();

    std::shared_ptr mprovider = measurement::make_measurement_provider_mockup(
      pose_signal, config->extrinsics.imu_camera_transform, landmarks,
      timestamps);

    auto state = std::make_shared<estimation::StateVariableInternal>();
    auto state_reader =
      std::make_shared<estimation::StateVariableReadAccessor>(state, nullptr);
    auto state_writer =
      std::make_shared<estimation::StateVariableWriteAccessor>(state);

    auto keyframe_manager =
      std::make_shared<measurement::KeyframeManagerMock>();
    keyframe_manager->_keyframes = timestamps;

    std::shared_ptr telemetry =
      telemetry::InitializerTelemetry::createDefault();

    auto bootstrap_internal =
      InitializationSolverInternal::create(rgen, config, mprovider, telemetry);
    auto bootstrap_solver = InitializerMain::create(
      std::move(bootstrap_internal), keyframe_manager, telemetry);

    auto initializer = estimation::OptimizerSolutionGuessPredictor::create(
      std::move(bootstrap_solver), config, state_reader, mprovider);

    return std::make_tuple(state_reader, state_writer, std::move(initializer));
  }

  static auto CHECK_MOTION_STATE_VALID(
    se3_transform_t const& our_origin, se3_transform_t const& sol_origin,
    motion_frame_parameter_blocks_t const& motion_ours,
    std::map<frame_id_t, imu_motion_state_t> const& motion_sols) {
    auto normalize_gauge = [](auto const& origin, auto const& motion) {
      auto const& [q, p, v] = motion;
      auto const& [p_origin, q_origin] = origin;
      return imu_motion_state_t {
        .orientation = q_origin.conjugate() * q,
        .position = q_origin.conjugate() * (p - p_origin),
        .velocity = q_origin.conjugate() * v,
      };
    };

    for (auto const& [frame_id, motion_our_block] : motion_ours) {
      auto x_our = normalize_gauge(
        our_origin,
        estimation::motion_state_of_motion_frame_block(motion_our_block));
      auto x_sol = normalize_gauge(sol_origin, motion_sols.at(frame_id));

      CAPTURE(x_sol.orientation.coeffs().transpose());
      CAPTURE(x_our.orientation.coeffs().transpose());
      CHECK(x_our.orientation.isApprox(x_sol.orientation, 1e-2));

      CAPTURE(x_sol.position.transpose());
      CAPTURE(x_our.position.transpose());
      CHECK(x_our.position.isApprox(x_sol.position, 1e-2));

      CAPTURE(x_sol.velocity.transpose());
      CAPTURE(x_our.velocity.transpose());
      CHECK(x_our.velocity.isApprox(x_sol.velocity, 1e-2));
    }
  }

  static auto CHECK_LANDMARK_STATE_VALID(
    se3_transform_t const& our_origin, se3_transform_t const& sol_origin,
    landmark_parameter_blocks_t const& landmarks_our,
    landmark_positions_t const& landmarks_sol) {
    auto normalize_gauge = [](auto const& origin, auto const& f) {
      auto const& [p, q] = origin;
      return (q.conjugate() * (f - p)).eval();
    };

    for (auto const& [landmark_id, landmark_block] : landmarks_our) {
      auto f_our = normalize_gauge(our_origin, Vector3d(landmark_block.data()));
      auto f_sol = normalize_gauge(sol_origin, landmarks_sol.at(landmark_id));

      CAPTURE(f_our.transpose());
      CAPTURE(f_sol.transpose());
      CHECK(f_sol.isApprox(f_our, 1e-2));
    }
  }

  TEST_CASE("Optimizer solution predictor") {
    GIVEN("Well-excited body motion") {
      auto rgen = std::make_shared<std::mt19937>(20220803);
      auto landmarks_sol = generate_landmarks(
        *rgen, {1000, Vector3d(0.5, 3, 0), Vector3d(0.5, 0.5, 1).asDiagonal()});

      auto pose_signal = pose_signal_t {position_signal, orientation_signal};
      auto timestamps = make_dictionary<frame_id_t, timestamp_t>(
        views::enumerate(linspace(0., 1., 10)));
      auto [state_reader, state_writer, initializer] =
        make_initializer(rgen, landmarks_sol, pose_signal, timestamps);

      auto motions_sol = evaluate_motion_states(pose_signal, timestamps);

      GIVEN("Uninitialized estimator") {
        REQUIRE(state_reader->motionFrames().empty());

        WHEN("Solved initial bootstrap") {
          auto maybe_solution = initializer->solve();
          REQUIRE(maybe_solution.has_value());

          auto const& motions_our = maybe_solution->motions;
          auto const& landmarks_our = maybe_solution->landmarks;

          frame_id_t init_frame_id = 0;
          auto our_origin = estimation::se3_of_motion_frame_block(
            motions_our.at(init_frame_id));
          auto sol_origin = pose_signal.evaluate(timestamps.at(init_frame_id));

          THEN("The obtained solution is not empty") {
            REQUIRE_FALSE(motions_our.empty());
            REQUIRE_FALSE(landmarks_our.empty());
            REQUIRE(landmarks_our.size() >= 150);

            AND_THEN("The obtained solution is correct") {
              CHECK_MOTION_STATE_VALID(
                our_origin, sol_origin, motions_our, motions_sol);
              CHECK_LANDMARK_STATE_VALID(
                our_origin, sol_origin, landmarks_our, landmarks_sol);
            }
          }

          AND_WHEN(
            "Partly initialized the estimator with some of the motion frame "
            "states and the landmarks states are unknown") {
            REQUIRE(motions_our.size() == 10);
            state_writer->updateMotionFrameGuess(
              views::ints(0, 9) | views::transform([&](auto frame_id) {
                return std::make_pair(frame_id, motions_our.at(frame_id));
              }) |
              ranges::to<motion_frame_parameter_blocks_t>);

            REQUIRE(state_reader->motionFrames().size() != 10);

            REQUIRE(landmarks_our.size() > 100);
            state_writer->updateLandmarkGuess(
              landmarks_our | views::keys |
              views::filter([](auto _) { return _ < 100; }) |
              views::transform([&](auto id) {
                return std::make_pair(id, landmarks_our.at(id));
              }) |
              ranges::to<landmark_parameter_blocks_t>);

            AND_WHEN("Solved solution prediction again") {
              auto maybe_solution = initializer->solve();
              REQUIRE(maybe_solution.has_value());

              auto const& motions_our = maybe_solution->motions;
              auto const& landmarks_our = maybe_solution->landmarks;

              THEN("The obtained solution is not empty") {
                REQUIRE_FALSE(motions_our.empty());
                REQUIRE_FALSE(landmarks_our.empty());
                REQUIRE(landmarks_our.size() >= 50);

                THEN(
                  "The obtained motion solution contains the previously "
                  "uninitialized motion states") {
                  REQUIRE(motions_our.find(9) != motions_our.end());
                }

                AND_THEN("The obtained solution is correct") {
                  CHECK_MOTION_STATE_VALID(
                    our_origin, sol_origin, motions_our, motions_sol);
                  CHECK_LANDMARK_STATE_VALID(
                    our_origin, sol_origin, landmarks_our, landmarks_sol);
                }
              }
            }
          }
        }
      }
    }
  }
}  // namespace cyclops::initializer
