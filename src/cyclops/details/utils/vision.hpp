#pragma once

#include "cyclops/details/measurement/type.hpp"

namespace cyclops {
  struct image_frame_motion_statistics_t {
    int new_features;
    int common_features;

    double average_parallax;
  };

  struct rotation_translation_matrix_pair_t {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
  };

  image_frame_motion_statistics_t compute_image_frame_motion_statistics(
    measurement::feature_tracks_t const& tracks, frame_id_t frame1,
    frame_id_t frame2);

  image_frame_motion_statistics_t compute_image_frame_motion_statistics(
    std::map<landmark_id_t, feature_point_t> const& frame1,
    std::map<landmark_id_t, feature_point_t> const& frame2);

  std::optional<Eigen::Vector3d> triangulate_point(
    measurement::feature_track_t const& features,
    std::map<frame_id_t, rotation_translation_matrix_pair_t> const&
      camera_pose_lookup);

  std::map<landmark_id_t, std::tuple<Eigen::Vector2d, Eigen::Vector2d>>
  make_two_view_feature_pairs(
    std::map<landmark_id_t, feature_point_t> const& frame1,
    std::map<landmark_id_t, feature_point_t> const& frame2);
}  // namespace cyclops
