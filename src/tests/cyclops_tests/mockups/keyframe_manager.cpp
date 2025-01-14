#include "cyclops_tests/mockups/keyframe_manager.hpp"

namespace cyclops::measurement {
  void KeyframeManagerMock::reset() {
    _keyframes.clear();
    _pending_frames.clear();
  }

  frame_id_t KeyframeManagerMock::createNewFrame(timestamp_t timestamp) {
    _last_frame_id++;
    _pending_frames.emplace(_last_frame_id, timestamp);
    return _last_frame_id;
  }

  void KeyframeManagerMock::setKeyframe(frame_id_t id) {
    auto i = _pending_frames.find(id);
    if (i == _pending_frames.end())
      return;

    _keyframes.emplace(id, i->second);
    _pending_frames.erase(i);
  }

  void KeyframeManagerMock::removeFrame(frame_id_t frame) {
    _pending_frames.erase(frame);
    _keyframes.erase(frame);
  }

  KeyframeManager::frame_sequence_t const& KeyframeManagerMock::keyframes()
    const {
    return _keyframes;
  }

  KeyframeManager::frame_sequence_t const& KeyframeManagerMock::pendingFrames()
    const {
    return _pending_frames;
  }
}  // namespace cyclops::measurement
