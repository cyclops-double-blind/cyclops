#pragma once

#include "cyclops/details/type.hpp"

#include <Eigen/Dense>
#include <functional>

namespace cyclops {
  using scalar_signal_t = std::function<double(timestamp_t)>;
  using vector3_signal_t = std::function<Eigen::Vector3d(timestamp_t)>;
  using quaternion_signal_t = std::function<Eigen::Quaterniond(timestamp_t)>;

  struct pose_signal_t {
    vector3_signal_t position;
    quaternion_signal_t orientation;

    se3_transform_t evaluate(timestamp_t t) const;
  };
}  // namespace cyclops
