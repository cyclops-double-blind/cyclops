#pragma once

#include "cyclops/details/utils/block_meta.hpp"
#include "cyclops/details/type.hpp"

#include <map>

namespace cyclops::estimation {
  using motion_frame_parameter_block_t = block_meta::block_cascade<
    block_meta::orientation, block_meta::position, block_meta::velocity,
    block_meta::bias_acc, block_meta::bias_gyr>;
  using landmark_parameter_block_t =
    block_meta::block_cascade<block_meta::landmark_position>;

  using motion_frame_parameter_blocks_t =
    std::map<frame_id_t, motion_frame_parameter_block_t>;
  using landmark_parameter_blocks_t =
    std::map<landmark_id_t, landmark_parameter_block_t>;

  Eigen::Quaterniond orientation_of_motion_frame_ptr(double const* frame_ptr);
  Eigen::Vector3d position_of_motion_frame_ptr(double const* frame_ptr);
  Eigen::Vector3d velocity_of_motion_frame_ptr(double const* frame_ptr);
  Eigen::Vector3d acc_bias_of_motion_frame_ptr(double const* frame_ptr);
  Eigen::Vector3d gyr_bias_of_motion_frame_ptr(double const* frame_ptr);
  se3_transform_t se3_of_motion_frame_ptr(double const* frame_ptr);

  se3_transform_t se3_of_motion_frame_block(
    motion_frame_parameter_block_t const& frame);

  imu_motion_state_t motion_state_of_motion_frame_block(
    motion_frame_parameter_block_t const& frame);
  Eigen::Vector3d acc_bias_of_motion_frame_block(
    motion_frame_parameter_block_t const& frame);
  Eigen::Vector3d gyr_bias_of_motion_frame_block(
    motion_frame_parameter_block_t const& frame);

  Eigen::Vector3d position_of_landmark_block(
    landmark_parameter_block_t const& block);
}  // namespace cyclops::estimation
