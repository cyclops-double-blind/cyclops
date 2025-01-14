#pragma once

#include "cyclops/details/measurement/keyframe.hpp"

namespace cyclops::measurement {
  struct KeyframeManagerMock: public KeyframeManager {
    frame_id_t _last_frame_id = 0;
    frame_sequence_t _keyframes;
    frame_sequence_t _pending_frames;

    void reset() override;

    frame_id_t createNewFrame(timestamp_t timestamp) override;
    void setKeyframe(frame_id_t id) override;
    void removeFrame(frame_id_t frame) override;

    frame_sequence_t const& keyframes() const override;
    frame_sequence_t const& pendingFrames() const override;
  };
}  // namespace cyclops::measurement
