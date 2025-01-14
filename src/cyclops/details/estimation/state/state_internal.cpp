#include "cyclops/details/estimation/state/state_internal.hpp"

namespace cyclops::estimation {
  motion_frame_parameter_blocks_t const& StateVariableInternal::motionFrames()
    const {
    return _motion_frames;
  }

  motion_frame_parameter_blocks_t& StateVariableInternal::motionFrames() {
    return _motion_frames;
  }

  landmark_parameter_blocks_t const& StateVariableInternal::landmarks() const {
    return _landmarks;
  }

  landmark_parameter_blocks_t& StateVariableInternal::landmarks() {
    return _landmarks;
  }

  landmark_positions_t const& StateVariableInternal::mappedLandmarks() const {
    return _mapped_landmarks;
  }

  landmark_positions_t& StateVariableInternal::mappedLandmarks() {
    return _mapped_landmarks;
  }
}  // namespace cyclops::estimation
