#include "cyclops/details/estimation/state/accessor_write.hpp"
#include "cyclops/details/estimation/state/state_internal.hpp"

#include <range/v3/all.hpp>

namespace cyclops::estimation {
  namespace views = ranges::views;

  template <typename container_t, typename key_t>
  static bool contains(container_t const& container, key_t const& key) {
    return container.find(key) != container.end();
  }

  template <typename container_t, typename pred_t>
  static void erase_if(container_t& container, pred_t const& predicate) {
    for (auto i = container.begin(); i != container.end();) {
      if (predicate(*i)) {
        i = container.erase(i);
      } else {
        ++i;
      }
    }
  }

  template <typename value_t>
  using maybe_ref_t = StateVariableWriteAccessor::maybe_ref_t<value_t>;

  template <typename key_t, typename container_t>
  static auto maybe_find(container_t&& container, key_t key)
    -> maybe_ref_t<std::remove_reference_t<decltype(container.at(key))>> {
    auto i = container.find(key);
    if (i == container.end())
      return std::nullopt;

    auto& [_, result] = *i;
    return result;
  }

  StateVariableWriteAccessor::StateVariableWriteAccessor(
    std::shared_ptr<StateVariableInternal> state)
      : _state(state) {
  }

  StateVariableWriteAccessor::~StateVariableWriteAccessor() = default;

  std::tuple<std::set<frame_id_t>, std::set<landmark_id_t>>
  StateVariableWriteAccessor::prune(
    std::set<frame_id_t> const& data_frames,
    std::set<landmark_id_t> const& data_landmarks) {
    erase_if(_state->motionFrames(), [&](auto const& id_frame) {
      return !contains(data_frames, id_frame.first);
    });
    erase_if(_state->landmarks(), [&](auto const& id_landmark) {
      return !contains(data_landmarks, id_landmark.first);
    });

    auto active_frames = views::set_intersection(
      data_frames, _state->motionFrames() | views::keys);
    auto active_landmarks = views::set_intersection(
      data_landmarks, _state->landmarks() | views::keys);

    return std::make_tuple(
      active_frames | ranges::to<std::set>,
      active_landmarks | ranges::to<std::set>);
  }

  void StateVariableWriteAccessor::reset() {
    _state->motionFrames().clear();
    _state->landmarks().clear();
    _state->mappedLandmarks().clear();
  }

  void StateVariableWriteAccessor::updateMotionFrameGuess(
    motion_frame_parameter_blocks_t const& new_frames) {
    _state->motionFrames().insert(new_frames.begin(), new_frames.end());
  }

  void StateVariableWriteAccessor::updateLandmarkGuess(
    landmark_parameter_blocks_t const& new_landmarks) {
    _state->landmarks().insert(new_landmarks.begin(), new_landmarks.end());
  }

  void StateVariableWriteAccessor::updateMappedLandmarks(
    landmark_positions_t const& positions) {
    for (auto const& [id, position] : positions)
      _state->mappedLandmarks()[id] = position;
  }

  maybe_ref_t<motion_frame_parameter_block_t>
  StateVariableWriteAccessor::motionFrame(frame_id_t id) {
    return maybe_find(_state->motionFrames(), id);
  }

  maybe_ref_t<landmark_parameter_block_t> StateVariableWriteAccessor::landmark(
    landmark_id_t id) {
    return maybe_find(_state->landmarks(), id);
  }
}  // namespace cyclops::estimation
