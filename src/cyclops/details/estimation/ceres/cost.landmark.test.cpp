#include "cyclops/details/estimation/ceres/cost.landmark.hpp"
#include "cyclops/details/estimation/state/state_block.hpp"

#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/default.hpp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops::estimation {
  namespace views = ranges::views;

  using Eigen::AngleAxisd;
  using Eigen::Matrix3d;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  TEST_CASE("Landmark projection ceres factor") {
    std::mt19937 rgen(20240516);

    GIVEN("Constant pose signal") {
      auto pose_signal = cyclops::pose_signal_t {
        .position = [](auto _) { return Vector3d(3, 0, 0); },
        .orientation =
          [](auto _) {
            return Quaterniond(AngleAxisd(M_PI, Vector3d::UnitZ()));
          },
      };

      auto extrinsic = cyclops::make_default_imu_camera_extrinsic();
      auto landmarks = generate_landmarks(
        rgen, {200, Vector3d(0, 0, 0), 0.5 * Matrix3d::Identity()});

      GIVEN("Perfect landmark observations at t = 0") {
        auto timestamps = std::map<frame_id_t, timestamp_t> {{0, 0.0}};
        auto tracks =
          make_landmark_tracks(pose_signal, extrinsic, landmarks, timestamps);

        THEN("Feature track is not empty") {
          REQUIRE_FALSE(tracks.empty());
          for (auto const& [id, track] : tracks)
            REQUIRE_FALSE(track.empty());
        }

        WHEN("Evaluated the landmark projection cost at the true state") {
          auto residuals =  //
            tracks | views::transform([&](auto const& id_track) {
              auto const& [id, track] = id_track;

              return  //
                track |
                views::transform([&, id = id](auto const& feature_view) {
                  auto const& [frame_id, feature] = feature_view;
                  auto [p, q] = pose_signal.evaluate(timestamps.at(frame_id));
                  auto cost =
                    LandmarkProjectionCostEvaluator(feature, extrinsic);

                  motion_frame_parameter_block_t x;
                  Eigen::Map<Quaterniond>(x.data()) = q;
                  Eigen::Map<Vector3d>(x.data() + 4) = p;
                  Eigen::Map<Vector3d>(x.data() + 7) = Vector3d::Zero();

                  landmark_parameter_block_t f;
                  Eigen::Map<Vector3d>(f.data()) = landmarks.at(id);

                  std::array<double, 2> r = {1., 1.};
                  REQUIRE(cost(x.data(), f.data(), r.data()));

                  auto [r_x, r_y] = r;
                  return Eigen::Vector2d(r_x, r_y);
                }) |
                ranges::to_vector;
            }) |
            ranges::to_vector;

          THEN("Residuals are all effectively zero") {
            for (auto const& track : residuals) {
              for (auto const& r : track)
                CHECK(r.norm() < 1e-6);
            }
          }
        }
      }
    }
  }
}  // namespace cyclops::estimation
