#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"
#include <memory>

namespace cyclops::estimation {
  class IMUPropagationUpdateHandler;
  class StateVariableInternal;

  class StateVariableReadAccessor {
  private:
    std::shared_ptr<StateVariableInternal const> _state;
    std::shared_ptr<IMUPropagationUpdateHandler const> _propagator;

  private:
    template <typename value_t>
    using maybe_cref_t = std::optional<std::reference_wrapper<value_t const>>;

  public:
    StateVariableReadAccessor(
      std::shared_ptr<StateVariableInternal const> state_internal,
      std::shared_ptr<IMUPropagationUpdateHandler const> propagation_accessor);
    ~StateVariableReadAccessor();

    maybe_cref_t<motion_frame_parameter_block_t> motionFrame(
      frame_id_t id) const;
    maybe_cref_t<landmark_parameter_block_t> landmark(landmark_id_t id) const;

    frame_id_t lastMotionFrameId() const;
    motion_frame_parameter_block_t const& lastMotionFrameBlock() const;

    motion_frame_parameter_blocks_t const& motionFrames() const;
    landmark_parameter_blocks_t const& landmarks() const;
    landmark_positions_t const& mappedLandmarks() const;

    std::optional<std::tuple<timestamp_t, imu_motion_state_t>> propagatedState()
      const;
  };
}  // namespace cyclops::estimation
