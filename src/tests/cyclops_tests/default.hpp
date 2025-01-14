#pragma once

#include "cyclops/details/type.hpp"
#include <memory>
#include <vector>

namespace cyclops {
  struct cyclops_global_config_t;
  struct landmark_generation_argument_t;

  Eigen::Quaterniond make_default_camera_rotation();
  se3_transform_t make_default_imu_camera_extrinsic();

  std::shared_ptr<cyclops_global_config_t> make_default_config();
  std::vector<landmark_generation_argument_t> make_default_landmark_set();
}  // namespace cyclops
