#include "cyclops/details/type.hpp"

namespace cyclops {
  se3_transform_t se3_transform_t::Identity() {
    return se3_transform_t {
      .translation = Eigen::Vector3d::Zero(),
      .rotation = Eigen::Quaterniond::Identity(),
    };
  }
}  // namespace cyclops
