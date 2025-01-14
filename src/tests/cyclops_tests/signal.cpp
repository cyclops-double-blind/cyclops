#include "cyclops_tests/signal.hpp"

namespace cyclops {
  se3_transform_t pose_signal_t::evaluate(timestamp_t t) const {
    return {position(t), orientation(t)};
  }
}  // namespace cyclops
