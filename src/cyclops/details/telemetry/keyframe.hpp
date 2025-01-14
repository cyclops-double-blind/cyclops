#pragma once

#include "cyclops/details/type.hpp"
#include <memory>

namespace cyclops::telemetry {
  class KeyframeTelemetry {
  public:
    virtual ~KeyframeTelemetry() = default;
    virtual void reset();

    struct on_new_motion_frame_argument_t {
      frame_id_t frame_id;
      timestamp_t timestamp;
    };
    virtual void onNewMotionFrame(
      on_new_motion_frame_argument_t const& argument);

    static std::unique_ptr<KeyframeTelemetry> createDefault();
  };
}  // namespace cyclops::telemetry
