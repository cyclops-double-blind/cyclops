#pragma once

#include "cyclops_tests/signal.hpp"

#include <map>

namespace cyclops::initializer {
  struct two_view_imu_rotation_constraint_t;
}

namespace cyclops {
  struct se3_transform_t;

  std::map<frame_id_t, initializer::two_view_imu_rotation_constraint_t>
  make_multiview_rotation_prior(
    pose_signal_t const& pose_signal, se3_transform_t const& camera_extrinsic,
    std::map<frame_id_t, timestamp_t> const& frame_timestamps);
}  // namespace cyclops
