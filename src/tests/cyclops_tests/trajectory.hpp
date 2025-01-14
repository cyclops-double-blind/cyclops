#pragma once

#include "cyclops_tests/signal.hpp"
#include <vector>

namespace cyclops {
  vector3_signal_t bezier(double T, std::vector<Eigen::Vector3d> const& points);
}  // namespace cyclops
