#include "cyclops_tests/default.hpp"

#include "cyclops_tests/data/typefwd.hpp"
#include "cyclops_tests/data/landmark.hpp"

#include "cyclops/details/config.hpp"

namespace cyclops {
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  Quaterniond make_default_camera_rotation() {
    Eigen::Matrix3d result;

    // clang-format off
    result <<
      +0, +0, +1,
      -1, +0, +0,
      +0, -1, +0;
    // clang-format on

    return Eigen::Quaterniond(result);
  }

  se3_transform_t make_default_imu_camera_extrinsic() {
    return {
      .translation = Vector3d(0.1, 0, 0),
      .rotation = make_default_camera_rotation(),
    };
  }

  std::shared_ptr<cyclops_global_config_t> make_default_config() {
    return make_default_cyclops_global_config(
      sensor_statistics_t {
        .acc_white_noise = 0.01,
        .gyr_white_noise = 0.003,
        .acc_random_walk = 0.00001,
        .gyr_random_walk = 0.00001,
        .acc_bias_prior_stddev = 0.1,
        .gyr_bias_prior_stddev = 0.1,
      },
      sensor_extrinsics_t {
        .imu_camera_time_delay = 0,
        .imu_camera_transform = make_default_imu_camera_extrinsic(),
      });
  }

  landmark_generation_arguments_t make_default_landmark_set() {
    return landmark_generation_arguments_t {
      landmark_generation_argument_t {
        .count = 30,
        .center = Vector3d(1.5, 0.5, 0.8),
        .concentration = Vector3d(0.5, 1.0, 0.8).asDiagonal(),
      },
      landmark_generation_argument_t {
        .count = 30,
        .center = Vector3d(2.0, 2.8, 0.6),
        .concentration = Vector3d(0.5, 0.5, 0.6).asDiagonal(),
      },
      landmark_generation_argument_t {
        .count = 30,
        .center = Vector3d(4.5, 0.75, 0.5),
        .concentration = Vector3d(0.5, 0.75, 0.5).asDiagonal(),
      },
      landmark_generation_argument_t {
        .count = 30,
        .center = Vector3d(3.5, 3.25, 0.75),
        .concentration = Vector3d(0.5, 0.75, 0.75).asDiagonal(),
      },
      landmark_generation_argument_t {
        .count = 30,
        .center = Vector3d(0.0, 5.0, 0.5),
        // clang-format off
        .concentration = (Eigen::Matrix3d() <<
          -0.5, -2.0, 0,
          +0.5, -2.0, 0,
          +0.0, +0.0, 1
        ).finished(),
        // clang-format on
      },
    };
  }
}  // namespace cyclops
