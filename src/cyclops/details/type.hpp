#pragma once

#include <Eigen/Dense>

#include <cstdlib>
#include <map>

namespace cyclops {
  using timestamp_t = double;

  using frame_id_t = std::size_t;
  using landmark_id_t = std::size_t;

  struct imu_data_t {
    timestamp_t timestamp;
    Eigen::Vector3d accel;
    Eigen::Vector3d rotat;
  };

  struct feature_point_t {
    Eigen::Vector2d point;
    Eigen::Matrix2d weight;
  };

  struct image_data_t {
    timestamp_t timestamp;
    std::map<landmark_id_t, feature_point_t> features;
  };

  using landmark_positions_t = std::map<landmark_id_t, Eigen::Vector3d>;

  struct se3_transform_t {
    Eigen::Vector3d translation;
    Eigen::Quaterniond rotation;

    static se3_transform_t Identity();
  };

  struct imu_motion_state_t {
    Eigen::Quaterniond orientation;
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
  };
}  // namespace cyclops
