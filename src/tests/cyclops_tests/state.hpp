#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

namespace cyclops {
  static estimation::motion_frame_parameter_block_t make_motion_frame_parameter(
    Eigen::Quaterniond const& q, Eigen::Vector3d const& p,
    Eigen::Vector3d const& v) {
    estimation::motion_frame_parameter_block_t x;
    (Eigen::Map<Eigen::Quaterniond>(x.data())) = q;
    (Eigen::Map<Eigen::Vector3d>(x.data() + 4)) = p;
    (Eigen::Map<Eigen::Vector3d>(x.data() + 7)) = v;
    (Eigen::Map<Eigen::Vector3d>(x.data() + 10)).setZero();
    (Eigen::Map<Eigen::Vector3d>(x.data() + 13)).setZero();
    return x;
  }

  static estimation::motion_frame_parameter_block_t make_motion_frame_parameter(
    imu_motion_state_t const& x) {
    return make_motion_frame_parameter(x.orientation, x.position, x.velocity);
  }

  static estimation::landmark_parameter_block_t make_landmark_parameter(
    Eigen::Vector3d const& f) {
    estimation::landmark_parameter_block_t f_;
    Eigen::Map<Eigen::Vector3d>(f_.data()) = f;
    return f_;
  }
}  // namespace cyclops
