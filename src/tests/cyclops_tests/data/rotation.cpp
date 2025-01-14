#include "cyclops_tests/data/rotation.hpp"
#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/type.hpp"

#include <range/v3/all.hpp>

namespace cyclops {
  namespace views = ranges::views;

  std::map<frame_id_t, initializer::two_view_imu_rotation_constraint_t>
  make_multiview_rotation_prior(
    pose_signal_t const& pose_signal, se3_transform_t const& camera_extrinsic,
    std::map<frame_id_t, timestamp_t> const& frame_timestamps) {
    auto multiview_frame_pair = views::zip(
      views::drop_last(frame_timestamps, 1), views::drop(frame_timestamps, 1));

    auto make_camera_relative_rotation_prior =
      [&](auto const& frametime_1, auto const& frametime_2) {
        auto const& [frame1, time1] = frametime_1;
        auto const& [frame2, time2] = frametime_2;
        auto q1 = pose_signal.orientation(time1) * camera_extrinsic.rotation;
        auto q2 = pose_signal.orientation(time2) * camera_extrinsic.rotation;

        auto prior = initializer::two_view_imu_rotation_constraint_t {
          .init_frame_id = frame1,
          .term_frame_id = frame2,
          .rotation =
            {
              .value = q1.conjugate() * q2,
              .covariance = 1e-6 * Eigen::Matrix3d::Identity(),
            },
        };
        return std::make_pair(frame1, prior);
      };
    auto camera_relative_rotation_prior_transform = [&](auto const& _) {
      return std::apply(make_camera_relative_rotation_prior, _);
    };

    return multiview_frame_pair |
      views::transform(camera_relative_rotation_prior_transform) |
      ranges::to<std::map<
        frame_id_t, initializer::two_view_imu_rotation_constraint_t>>;
  }
}  // namespace cyclops
