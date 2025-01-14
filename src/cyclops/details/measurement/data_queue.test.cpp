#include "cyclops/details/measurement/data_queue.cpp"
#include "cyclops/details/measurement/data_provider.hpp"
#include "cyclops/details/measurement/keyframe.hpp"

#include "cyclops/details/estimation/state/state_internal.hpp"
#include "cyclops/details/estimation/state/accessor_read.hpp"

#include "cyclops/details/telemetry/keyframe.hpp"
#include "cyclops/details/config.hpp"

#include "cyclops_tests/default.hpp"
#include "cyclops_tests/range.ipp"

#include <range/v3/all.hpp>
#include <memory>

#include <doctest/doctest.h>

namespace cyclops::measurement {
  namespace views = ranges::views;

  using estimation::StateVariableInternal;
  using estimation::StateVariableReadAccessor;

  template <typename container_t, typename key_t>
  static bool contains(container_t const& container, key_t const& key) {
    return container.find(key) != container.end();
  }

  template <typename int_range_t>
  static auto make_landmark_measurement(
    timestamp_t time, int_range_t const& id_range) {
    auto result = image_data_t {time};
    for (auto id : id_range) {
      result.features.emplace(
        id,
        feature_point_t {Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()});
    }
    return result;
  }

  static auto CHECK_IMU_MOTION_FRAME_CONTINUOUS(
    imu_motions_t const& imu_motions) {
    for (auto const& motion : imu_motions)
      CHECK(motion.from < motion.to);

    for (int i = 0; i + 1 < imu_motions.size(); i++) {
      auto const& prev_motion = imu_motions.at(i);
      auto const& next_motion = imu_motions.at(i + 1);

      CHECK(prev_motion.to == next_motion.from);
    }
  }

  static auto CHECK_IMU_MOTION_FRAME_NOT_CONTAINS(
    imu_motions_t const& imu_motions, frame_id_t frame) {
    CHECK_FALSE(contains(
      imu_motions | views::transform([](auto const& _) { return _.from; }) |
        ranges::to<std::set>,
      frame));
    CHECK_FALSE(contains(
      imu_motions | views::transform([](auto const& _) { return _.to; }) |
        ranges::to<std::set>,
      frame));
  }

  TEST_CASE("Measurement queue data update and marginalization") {
    auto config = make_default_config();
    config->keyframe_window.optimization_phase_max_keyframes = 5;

    GIVEN("Initialized estimator") {
      auto state_internal = std::make_shared<StateVariableInternal>();
      auto state_accessor =
        std::make_shared<StateVariableReadAccessor>(state_internal, nullptr);
      state_internal->motionFrames() = {{0, {}}};

      std::shared_ptr mprovider =
        MeasurementDataProvider::create(config, state_accessor);

      auto telemetry = telemetry::KeyframeTelemetry::createDefault();
      auto mqueue = MeasurementDataQueue::create(
        config, mprovider, KeyframeManager::create(std::move(telemetry)),
        state_accessor);

      WHEN(
        "Updated measurement queue from t = 0 to t = 0.6, landmark by 10Hz and "
        "IMU by 100Hz") {
        for (int i = 0; i < 7; i++) {
          auto t_curr = 0.1 * i;
          auto t_next = t_curr + 0.1;
          for (auto t : linspace(t_curr, t_next, 10)) {
            mqueue->updateImu(
              {t, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});
          }

          auto landmark =
            make_landmark_measurement(t_curr, std::vector {1, 2, 3, 4, 5});
          mqueue->updateLandmark(landmark);
        }

        THEN("Only the first frame is the keyframe") {
          REQUIRE(mqueue->keyframes().size() == 1);
          REQUIRE(contains(mqueue->keyframes(), 0));

          AND_THEN("Every else are the pending frames") {
            REQUIRE(mqueue->pendingFrames().size() == 6);
          }
        }

        AND_WHEN("Accepted the pending frame twice") {
          mqueue->acceptCurrentPendingKeyframe();
          mqueue->acceptCurrentPendingKeyframe();

          THEN("The two accepted becomes the keyframe") {
            REQUIRE(mqueue->keyframes().size() == 3);
            REQUIRE(
              (mqueue->keyframes() | views::keys | ranges::to_vector) ==
              std::vector<frame_id_t> {0, 1, 2});

            AND_THEN("And the rest are still pending") {
              REQUIRE(mqueue->pendingFrames().size() == 4);
              REQUIRE(
                (mqueue->pendingFrames() | views::keys | ranges::to_vector) ==
                std::vector<frame_id_t> {3, 4, 5, 6});
            }
          }

          THEN("Regardless, the measurement frames are correctly collected") {
            auto const& imu_motions = mprovider->imu();
            auto const& landmarks = mprovider->tracks();

            CHECK(imu_motions.size() == 6);
            CHECK_IMU_MOTION_FRAME_CONTINUOUS(imu_motions);

            CHECK(landmarks.size() == 5);
            for (int i = 1; i <= 5; i++)
              CHECK(landmarks.at(i).size() == 7);
            CHECK(contains(mprovider->tracks().at(1), 0));
            CHECK(contains(mprovider->tracks().at(4), 1));
          }

          AND_WHEN(
            "Marginalized the keyframe #0, specifying the pending frame #3 as "
            "the replacement frame, and also dropping the landmark #1") {
            frame_id_t drop_frame = 0;
            frame_id_t replacement_frame = 3;
            landmark_id_t drop_landmark = 1;

            mqueue->marginalizeKeyframe(
              drop_frame, {drop_landmark}, replacement_frame);

            THEN("keyframes are now [#1, #2, #3]") {
              REQUIRE(
                (mqueue->keyframes() | views::keys | ranges::to_vector) ==
                std::vector<frame_id_t> {1, 2, replacement_frame});

              AND_THEN("pending frames are now [#4, #5, #6]") {
                REQUIRE(
                  (mqueue->pendingFrames() | views::keys | ranges::to_vector) ==
                  std::vector<frame_id_t> {4, 5, 6});
              }
            }

            THEN(
              "The IMU motion frame corresponding to the marginalization frame "
              "is correctly removed") {
              auto const& imu_motions = mprovider->imu();

              CHECK(imu_motions.size() == 5);
              CHECK_IMU_MOTION_FRAME_CONTINUOUS(imu_motions);
              CHECK_IMU_MOTION_FRAME_NOT_CONTAINS(imu_motions, drop_frame);
            }

            THEN(
              "The landmark track corresponding to the drop landmark is "
              "correctly removed") {
              REQUIRE(mprovider->tracks().size() == 4);
              REQUIRE_FALSE(contains(mprovider->tracks(), drop_landmark));

              AND_THEN(
                "The landmark observation corresponding to the drop frame is "
                "correctly removed") {
                for (auto const& [_, track] : mprovider->tracks())
                  REQUIRE_FALSE(contains(track, drop_frame));
              }
            }

            AND_WHEN("Marginalized the pending frame #4") {
              frame_id_t drop_frame = 4;
              mqueue->marginalizePendingFrame(drop_frame, {});

              THEN("Keyframes are still [#1, #2, #3]") {
                REQUIRE(
                  (mqueue->keyframes() | views::keys | ranges::to_vector) ==
                  std::vector<frame_id_t> {1, 2, 3});

                AND_THEN("Pending frames are now [#5, #6]") {
                  REQUIRE(
                    (mqueue->pendingFrames() | views::keys |
                     ranges::to_vector) == std::vector<frame_id_t> {5, 6});
                }
              }

              THEN(
                "The drop frame is at the middle of the IMU motions. This time "
                "two IMU motions are dropped: that ends at the drop frame, and "
                "that starts from the drop frame") {
                auto const& imu_motions = mprovider->imu();

                REQUIRE(imu_motions.size() == 3);
                for (auto const& motion : imu_motions)
                  CHECK(motion.from < motion.to);

                CHECK_IMU_MOTION_FRAME_NOT_CONTAINS(imu_motions, drop_frame);
              }

              THEN(
                "The landmark observation corresponding to the drop frame is "
                "correctly removed") {
                REQUIRE(mprovider->tracks().size() == 4);

                auto n_keyframes = mqueue->keyframes().size();
                auto n_pending_frames = mqueue->pendingFrames().size();
                for (auto const& [_, track] : mprovider->tracks()) {
                  CHECK(track.size() == (n_keyframes + n_pending_frames));
                  CHECK_FALSE(contains(track, drop_frame));
                }
              }
            }
          }
        }
      }
    }
  }
}  // namespace cyclops::measurement
