#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <vector>

namespace cyclops {
  struct imu_mockup_t;
  struct landmark_generation_argument_t;

  using imu_mockup_sequence_t = std::map<timestamp_t, imu_mockup_t>;
  using landmark_generation_arguments_t =
    std::vector<landmark_generation_argument_t>;
}  // namespace cyclops
