#pragma once

#include "cyclops_tests/signal.hpp"

#include "cyclops/details/measurement/type.hpp"
#include "cyclops/details/type.hpp"

#include <functional>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace cyclops {
  struct landmark_generation_argument_t {
    int count;
    Eigen::Vector3d center;
    Eigen::Matrix3d concentration;
  };

  landmark_positions_t generate_landmarks(
    std::mt19937& rgen, landmark_generation_argument_t const& arg);
  landmark_positions_t generate_landmarks(
    std::mt19937& rgen,
    std::vector<landmark_generation_argument_t> const& args);
  landmark_positions_t generate_landmarks(
    std::set<landmark_id_t> ids,
    std::function<Eigen::Vector3d(landmark_id_t)> gen);

  std::map<landmark_id_t, feature_point_t> generate_landmark_observations(
    Eigen::Matrix3d const& R, Eigen::Vector3d const& p,
    landmark_positions_t const& landmarks);
  std::map<landmark_id_t, feature_point_t> generate_landmark_observations(
    std::mt19937& rgen, Eigen::Matrix2d const& cov, Eigen::Matrix3d const& R,
    Eigen::Vector3d const& p, landmark_positions_t const& landmarks);

  std::vector<image_data_t> make_landmark_frames(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const&, std::vector<timestamp_t> const&);
  std::vector<image_data_t> make_landmark_frames(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const&, std::vector<timestamp_t> const&,
    std::mt19937& rgen, Eigen::Matrix2d const& cov);

  measurement::feature_tracks_t make_landmark_tracks(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const&, std::map<frame_id_t, timestamp_t> const&);
  measurement::feature_tracks_t make_landmark_tracks(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const&, std::map<frame_id_t, timestamp_t> const&,
    std::mt19937& rgen, Eigen::Matrix2d const& cov);

  std::map<frame_id_t, std::map<landmark_id_t, feature_point_t>>
  make_landmark_multiview_observation(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const&, std::map<frame_id_t, timestamp_t> const&);
}  // namespace cyclops
