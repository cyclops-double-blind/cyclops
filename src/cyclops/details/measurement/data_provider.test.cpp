#include "cyclops/details/measurement/data_provider.cpp"
#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/estimation/state/accessor_read.hpp"
#include "cyclops/details/estimation/state/state_internal.hpp"

#include "cyclops_tests/default.hpp"

#include <range/v3/all.hpp>
#include <memory>

#include <doctest/doctest.h>

namespace cyclops::measurement {
  namespace views = ranges::views;

  template <typename container_t, typename key_t>
  static bool contains(container_t const& container, key_t const& key) {
    return container.find(key) != container.end();
  }

  static auto make_empty_feature_point() {
    return feature_point_t {
      .point = Eigen::Vector2d::Zero(),
      .weight = Eigen::Matrix2d::Identity(),
    };
  }

  TEST_CASE("Measurement provider data update and marginalization") {
    auto config = make_default_config();
    auto state = std::make_shared<estimation::StateVariableInternal>();
    auto state_accessor =
      std::make_shared<estimation::StateVariableReadAccessor>(state, nullptr);
    auto mprovider = MeasurementDataProvider::create(config, state_accessor);

    GIVEN("Landmark observation of ID 1~5 at t = 0 and frame_id = 0") {
      auto l0 = image_data_t {0.0};
      l0.features.emplace(1, make_empty_feature_point());
      l0.features.emplace(2, make_empty_feature_point());
      l0.features.emplace(3, make_empty_feature_point());
      l0.features.emplace(4, make_empty_feature_point());
      l0.features.emplace(5, make_empty_feature_point());
      mprovider->updateFrame(0, l0);

      GIVEN("Landmark observation of ID 1~3 at t = 0.1 and frame_id = 1") {
        auto l1 = image_data_t {0.1};
        l1.features.emplace(1, make_empty_feature_point());
        l1.features.emplace(2, make_empty_feature_point());
        l1.features.emplace(3, make_empty_feature_point());
        mprovider->updateFrame(0, 1, l1, nullptr);

        GIVEN("Landmark observation of ID 1~3 at t = 0.2 and frame_id = 2") {
          auto l2 = image_data_t {0.2};
          l2.features.emplace(1, make_empty_feature_point());
          l2.features.emplace(2, make_empty_feature_point());
          l2.features.emplace(3, make_empty_feature_point());
          mprovider->updateFrame(1, 2, l2, nullptr);

          THEN("Five landmark tracks exist") {
            auto const& landmarks = mprovider->tracks();
            REQUIRE(landmarks.size() == 5);
          }

          THEN("Two IMU motion frames come out") {
            auto const& imu_frames = mprovider->imu();
            REQUIRE(imu_frames.size() == 2);

            AND_THEN(
              "The from/to IDs of each IMU motion frames are consistent") {
              for (auto const& frame : imu_frames)
                CHECK(frame.from < frame.to);
              CHECK(imu_frames.at(0).to == imu_frames.at(1).from);
            }
          }

          WHEN("dropped frame 0 and landmark 2, 4.") {
            mprovider->marginalize(0, {2, 4});

            THEN("measurements are correctly removed") {
              auto const& imu = mprovider->imu();
              REQUIRE(imu.size() == 1);
              CHECK(imu.at(0).from == 1);
              CHECK(imu.at(0).to == 2);

              auto const& tracks = mprovider->tracks();
              REQUIRE(tracks.size() == 2);
              REQUIRE(contains(tracks, 1));
              REQUIRE(contains(tracks, 3));
              CHECK(tracks.at(1).size() == 2);
              CHECK(tracks.at(3).size() == 2);
              CHECK(contains(tracks.at(1), 1));
              CHECK(contains(tracks.at(1), 2));
              CHECK(contains(tracks.at(3), 1));
              CHECK(contains(tracks.at(3), 2));
            }
          }
        }
      }
    }
  }
}  // namespace cyclops::measurement
