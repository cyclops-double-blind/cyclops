#include "cyclops/details/estimation/optimizer.hpp"
#include "cyclops/details/estimation/type.hpp"
#include "cyclops/details/estimation/graph/graph.hpp"
#include "cyclops/details/estimation/graph/node.hpp"
#include "cyclops/details/estimation/state/accessor_read.hpp"
#include "cyclops/details/estimation/state/accessor_write.hpp"
#include "cyclops/details/estimation/state/state_internal.hpp"

#include "cyclops/details/measurement/preintegration.hpp"

#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/config.hpp"

#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/mockups/data_provider.hpp"
#include "cyclops_tests/mockups/initializer.hpp"

#include "cyclops_tests/default.hpp"
#include "cyclops_tests/range.ipp"
#include "cyclops_tests/signal.ipp"

#include <range/v3/all.hpp>

#include <doctest/doctest.h>

namespace cyclops::estimation {
  namespace views = ranges::views;

  using Eigen::Matrix3d;
  using Eigen::Vector3d;

  TEST_CASE("Measurement probability optimization with mockup data") {
    auto rgen = std::make_shared<std::mt19937>(20240516001);
    auto config = make_default_config();

    GIVEN("Random-sampled landmark positions and sinusoidal inertial motion") {
      auto landmarks = generate_landmarks(
        *rgen, {50, Vector3d(3, 3, 0), 2 * Matrix3d::Identity()});

      auto pose_signal = pose_signal_t {
        .position =
          [](auto t) { return Vector3d(3 * (1 - std::cos(t)), 0, 0); },
        .orientation = yaw_rotation([](auto t) { return atan2(1, cos(t)); }),
      };
      auto timestamps = make_dictionary<frame_id_t, timestamp_t>(
        views::enumerate(linspace(0, M_PI_2, 16)));

      auto state = std::make_shared<StateVariableInternal>();
      auto state_reader =
        std::make_shared<StateVariableReadAccessor>(state, nullptr);
      auto state_writer = std::make_shared<StateVariableWriteAccessor>(state);

      GIVEN("Perfectly correct visual-inertial measurement data") {
        auto data_provider = measurement::make_measurement_provider_mockup(
          pose_signal, config->extrinsics.imu_camera_transform, landmarks,
          timestamps);

        GIVEN("Solution guess predictor of reasonable accuracy") {
          auto initializer =
            std::make_unique<OptimizerSolutionGuessPredictorMock>(
              rgen, landmarks, pose_signal, timestamps);
          auto optimizer = LikelihoodOptimizer::create(
            std::move(initializer), config, state_writer,
            std::move(data_provider));

          WHEN("Optimized measurement probability") {
            optimizer->optimize({});

            THEN("Frame state estimations are effectively correct") {
              auto x0 = pose_signal.evaluate(0);

              for (auto const& [frame_id, t] : timestamps) {
                auto x_sol = compose(inverse(x0), pose_signal.evaluate(t));

                auto maybe_frame = state_reader->motionFrame(frame_id);
                REQUIRE(maybe_frame.has_value());

                auto x_got = se3_of_motion_frame_block(maybe_frame->get());

                CAPTURE(x_got.translation.transpose());
                CAPTURE(x_sol.translation.transpose());
                CHECK(x_got.translation.isApprox(x_sol.translation, 1e-2));

                CAPTURE(x_got.rotation.coeffs().transpose());
                CAPTURE(x_sol.rotation.coeffs().transpose());
                CHECK(x_got.rotation.isApprox(x_sol.rotation, 1e-2));
              }
            }
          }
        }
      }
    }
  }
}  // namespace cyclops::estimation
