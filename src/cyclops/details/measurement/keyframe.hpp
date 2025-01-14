#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>

namespace cyclops::telemetry {
  class KeyframeTelemetry;
}

namespace cyclops::measurement {
  class KeyframeManager {
  public:
    virtual ~KeyframeManager() = default;
    virtual void reset() = 0;

    virtual frame_id_t createNewFrame(timestamp_t timestamp) = 0;
    virtual void setKeyframe(frame_id_t id) = 0;
    virtual void removeFrame(frame_id_t frame) = 0;

    using frame_sequence_t = std::map<frame_id_t, timestamp_t>;
    virtual frame_sequence_t const& keyframes() const = 0;
    virtual frame_sequence_t const& pendingFrames() const = 0;

    static std::unique_ptr<KeyframeManager> create(
      std::shared_ptr<telemetry::KeyframeTelemetry> telemetry);
  };
}  // namespace cyclops::measurement
