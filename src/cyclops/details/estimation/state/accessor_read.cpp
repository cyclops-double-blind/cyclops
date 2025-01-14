#include "cyclops/details/estimation/state/accessor_read.hpp"
#include "cyclops/details/estimation/state/state_internal.hpp"
#include "cyclops/details/estimation/propagation.hpp"

namespace cyclops::estimation {
  template <typename value_t>
  using maybe_cref_t = StateVariableReadAccessor::maybe_cref_t<value_t>;

  template <typename key_t, typename container_t>
  static auto maybe_find(container_t&& container, key_t key)
    -> maybe_cref_t<std::remove_reference_t<decltype(container.at(key))>> {
    auto i = container.find(key);
    if (i == container.end())
      return std::nullopt;

    auto& [_, result] = *i;
    return result;
  }

  StateVariableReadAccessor::StateVariableReadAccessor(
    std::shared_ptr<StateVariableInternal const> state,
    std::shared_ptr<IMUPropagationUpdateHandler const> propagator)
      : _state(state), _propagator(propagator) {
  }

  StateVariableReadAccessor::~StateVariableReadAccessor() = default;

  maybe_cref_t<motion_frame_parameter_block_t>
  StateVariableReadAccessor::motionFrame(frame_id_t id) const {
    return maybe_find(_state->motionFrames(), id);
  }

  maybe_cref_t<landmark_parameter_block_t> StateVariableReadAccessor::landmark(
    landmark_id_t id) const {
    return maybe_find(_state->landmarks(), id);
  }

  frame_id_t StateVariableReadAccessor::lastMotionFrameId() const {
    return _state->motionFrames().rbegin()->first;
  }

  motion_frame_parameter_block_t const&
  StateVariableReadAccessor::lastMotionFrameBlock() const {
    return _state->motionFrames().rbegin()->second;
  }

  motion_frame_parameter_blocks_t const&
  StateVariableReadAccessor::motionFrames() const {
    return _state->motionFrames();
  }

  landmark_parameter_blocks_t const& StateVariableReadAccessor::landmarks()
    const {
    return _state->landmarks();
  }

  landmark_positions_t const& StateVariableReadAccessor::mappedLandmarks()
    const {
    return _state->mappedLandmarks();
  }

  std::optional<std::tuple<timestamp_t, imu_motion_state_t>>
  StateVariableReadAccessor::propagatedState() const {
    return _propagator->get();
  }
}  // namespace cyclops::estimation
