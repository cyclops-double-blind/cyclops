#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

#include <memory>
#include <set>

namespace cyclops::estimation {
  class StateVariableInternal;

  class StateVariableWriteAccessor {
  private:
    template <typename value_t>
    using maybe_ref_t = std::optional<std::reference_wrapper<value_t>>;

  private:
    std::shared_ptr<StateVariableInternal> _state;

  public:
    explicit StateVariableWriteAccessor(
      std::shared_ptr<StateVariableInternal> state);
    ~StateVariableWriteAccessor();

    std::tuple<std::set<frame_id_t>, std::set<landmark_id_t>> prune(
      std::set<frame_id_t> const& current_frames,
      std::set<landmark_id_t> const& current_landmarks);
    void reset();

    void updateMotionFrameGuess(motion_frame_parameter_blocks_t const& frames);
    void updateLandmarkGuess(landmark_parameter_blocks_t const& landmarks);
    void updateMappedLandmarks(landmark_positions_t const& positions);

    maybe_ref_t<motion_frame_parameter_block_t> motionFrame(frame_id_t id);
    maybe_ref_t<landmark_parameter_block_t> landmark(landmark_id_t id);
  };
}  // namespace cyclops::estimation
