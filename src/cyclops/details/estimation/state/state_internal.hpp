#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

namespace cyclops::estimation {
  class StateVariableInternal {
  private:
    motion_frame_parameter_blocks_t _motion_frames;
    landmark_parameter_blocks_t _landmarks;
    landmark_positions_t _mapped_landmarks;

  public:
    motion_frame_parameter_blocks_t const& motionFrames() const;
    motion_frame_parameter_blocks_t& motionFrames();

    landmark_parameter_blocks_t const& landmarks() const;
    landmark_parameter_blocks_t& landmarks();

    landmark_positions_t const& mappedLandmarks() const;
    landmark_positions_t& mappedLandmarks();
  };
}  // namespace cyclops::estimation
