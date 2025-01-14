#include "cyclops/details/telemetry/keyframe.hpp"

namespace cyclops::telemetry {
  void KeyframeTelemetry::reset() {
    // Nothing
  }

  void KeyframeTelemetry::onNewMotionFrame(
    on_new_motion_frame_argument_t const& argument) {
    // nothing.
  }

  std::unique_ptr<KeyframeTelemetry> KeyframeTelemetry::createDefault() {
    return std::make_unique<KeyframeTelemetry>();
  }
}  // namespace cyclops::telemetry
