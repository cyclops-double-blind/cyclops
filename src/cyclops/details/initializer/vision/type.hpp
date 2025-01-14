#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <tuple>

namespace cyclops::initializer {
  struct two_view_imu_rotation_data_t {
    Eigen::Quaterniond value;
    Eigen::Matrix3d covariance;
  };

  struct two_view_imu_rotation_constraint_t {
    frame_id_t init_frame_id;
    frame_id_t term_frame_id;

    two_view_imu_rotation_data_t rotation;
  };

  using two_view_feature_pair_t = std::tuple<Eigen::Vector2d, Eigen::Vector2d>;

  struct two_view_correspondence_data_t {
    two_view_imu_rotation_data_t rotation_prior;
    std::map<landmark_id_t, two_view_feature_pair_t> features;
  };

  struct multiview_correspondences_t {
    frame_id_t reference_frame;
    std::map<frame_id_t, two_view_correspondence_data_t> view_frames;
  };

  struct two_view_geometry_t {
    se3_transform_t camera_motion;
    landmark_positions_t landmarks;
  };

  struct multiview_geometry_t {
    std::map<frame_id_t, se3_transform_t> camera_motions;
    landmark_positions_t landmarks;
  };

  struct vision_bootstrap_solution_t {
    bool acceptable;

    double solution_significant_probability;
    double measurement_inlier_ratio;

    multiview_geometry_t geometry;
    Eigen::MatrixXd motion_information_weight;
  };
}  // namespace cyclops::initializer
