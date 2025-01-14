#include "cyclops/details/measurement/keyframe.hpp"
#include "cyclops/details/telemetry/keyframe.hpp"

namespace cyclops::measurement {
  class KeyframeManagerImpl: public KeyframeManager {
  private:
    std::shared_ptr<telemetry::KeyframeTelemetry> _telemetry;

    frame_sequence_t _keyframes;
    frame_sequence_t _pending_frames;
    frame_id_t _frame_id_ctr = 0;

  public:
    explicit KeyframeManagerImpl(
      std::shared_ptr<telemetry::KeyframeTelemetry> telemetry)
        : _telemetry(telemetry) {
    }

    void reset() override;

    frame_id_t createNewFrame(timestamp_t timestamp) override;
    void setKeyframe(frame_id_t id) override;
    void removeFrame(frame_id_t frame) override;

    frame_sequence_t const& keyframes() const override;
    frame_sequence_t const& pendingFrames() const override;
  };

  void KeyframeManagerImpl::reset() {
    _keyframes.clear();
    _pending_frames.clear();
    _telemetry->reset();
  }

  frame_id_t KeyframeManagerImpl::createNewFrame(timestamp_t timestamp) {
    _pending_frames.emplace(_frame_id_ctr, timestamp);
    _telemetry->onNewMotionFrame({
      .frame_id = _frame_id_ctr,
      .timestamp = timestamp,
    });

    return _frame_id_ctr++;
  }

  void KeyframeManagerImpl::setKeyframe(frame_id_t id) {
    auto i = _pending_frames.find(id);
    if (i == _pending_frames.end())
      return;
    auto timestamp = i->second;

    _pending_frames.erase(i);
    _keyframes.emplace(id, timestamp);
  }

  void KeyframeManagerImpl::removeFrame(frame_id_t frame) {
    _pending_frames.erase(frame);
    _keyframes.erase(frame);
  }

  KeyframeManagerImpl::frame_sequence_t const& KeyframeManagerImpl::keyframes()
    const {
    return _keyframes;
  }

  KeyframeManagerImpl::frame_sequence_t const&
  KeyframeManagerImpl::pendingFrames() const {
    return _pending_frames;
  }

  std::unique_ptr<KeyframeManager> KeyframeManager::create(
    std::shared_ptr<telemetry::KeyframeTelemetry> telemetry) {
    return std::make_unique<KeyframeManagerImpl>(telemetry);
  }
}  // namespace cyclops::measurement
