#include "cyclops/details/estimation/state/state_block.hpp"

namespace cyclops::estimation {
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  Quaterniond orientation_of_motion_frame_ptr(double const* frame_ptr) {
    return decltype(orientation_of_motion_frame_ptr(frame_ptr))(frame_ptr);
  }

  Vector3d position_of_motion_frame_ptr(double const* frame_ptr) {
    return decltype(position_of_motion_frame_ptr(frame_ptr))(frame_ptr + 4);
  }

  Vector3d velocity_of_motion_frame_ptr(double const* frame_ptr) {
    return decltype(velocity_of_motion_frame_ptr(frame_ptr))(frame_ptr + 7);
  }

  Vector3d acc_bias_of_motion_frame_ptr(double const* frame_ptr) {
    return decltype(acc_bias_of_motion_frame_ptr(frame_ptr))(frame_ptr + 10);
  }

  Vector3d gyr_bias_of_motion_frame_ptr(double const* frame_ptr) {
    return decltype(gyr_bias_of_motion_frame_ptr(frame_ptr))(frame_ptr + 13);
  }

  se3_transform_t se3_of_motion_frame_ptr(double const* frame_ptr) {
    return {
      .translation = position_of_motion_frame_ptr(frame_ptr),
      .rotation = orientation_of_motion_frame_ptr(frame_ptr),
    };
  }

  se3_transform_t se3_of_motion_frame_block(
    motion_frame_parameter_block_t const& block) {
    return se3_of_motion_frame_ptr(block.data());
  }

  Vector3d position_of_landmark_block(landmark_parameter_block_t const& block) {
    return Vector3d(block.data());
  }

  imu_motion_state_t motion_state_of_motion_frame_block(
    motion_frame_parameter_block_t const& frame) {
    auto frame_ptr = frame.data();
    return imu_motion_state_t {
      .orientation = orientation_of_motion_frame_ptr(frame_ptr),
      .position = position_of_motion_frame_ptr(frame_ptr),
      .velocity = velocity_of_motion_frame_ptr(frame_ptr),
    };
  }

  Vector3d acc_bias_of_motion_frame_block(
    motion_frame_parameter_block_t const& block) {
    return acc_bias_of_motion_frame_ptr(block.data());
  }

  Vector3d gyr_bias_of_motion_frame_block(
    motion_frame_parameter_block_t const& block) {
    return gyr_bias_of_motion_frame_ptr(block.data());
  }
}  // namespace cyclops::estimation
