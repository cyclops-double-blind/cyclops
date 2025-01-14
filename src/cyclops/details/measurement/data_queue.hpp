#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>
#include <optional>
#include <set>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::estimation {
  struct StateVariableReadAccessor;
}

namespace cyclops::measurement {
  struct MeasurementDataProvider;
  struct KeyframeManager;

  class MeasurementDataQueue {
  public:
    virtual ~MeasurementDataQueue() = default;
    virtual void reset() = 0;

    virtual void updateImu(imu_data_t const&) = 0;
    virtual std::optional<frame_id_t> updateLandmark(image_data_t const&) = 0;

    virtual bool detectKeyframe(frame_id_t candidate_frame) const = 0;
    virtual void acceptCurrentPendingKeyframe() = 0;

    virtual void marginalize(frame_id_t drop_frame) = 0;
    virtual void marginalizeKeyframe(
      frame_id_t drop_frame, std::set<landmark_id_t> const& drop_landmarks,
      frame_id_t inserted_frame) = 0;
    virtual void marginalizePendingFrame(
      frame_id_t drop_frame, std::set<landmark_id_t> const& drop_landmarks) = 0;

    virtual std::map<frame_id_t, timestamp_t> const& keyframes() const = 0;
    virtual std::map<frame_id_t, timestamp_t> const& pendingFrames() const = 0;

    static std::unique_ptr<MeasurementDataQueue> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<MeasurementDataProvider> measurements,
      std::shared_ptr<KeyframeManager> keyframe_manager,
      std::shared_ptr<estimation::StateVariableReadAccessor const> state);
  };
}  // namespace cyclops::measurement
