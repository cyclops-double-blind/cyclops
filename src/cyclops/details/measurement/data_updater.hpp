#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>
#include <optional>

namespace cyclops {
  struct cyclops_global_config_t;
  struct imu_data_t;
  struct image_data_t;
}  // namespace cyclops

namespace cyclops::estimation {
  class IMUPropagationUpdateHandler;
  class StateVariableReadAccessor;
}  // namespace cyclops::estimation

namespace cyclops::measurement {
  class MeasurementDataQueue;

  class MeasurementDataUpdater {
  public:
    virtual ~MeasurementDataUpdater() = default;
    virtual void reset() = 0;

    virtual void updateImu(imu_data_t const& data) = 0;
    virtual std::optional<frame_id_t> updateLandmark(
      image_data_t const& data) = 0;
    virtual void repropagate(frame_id_t last_frame, timestamp_t timestamp) = 0;

    virtual std::map<frame_id_t, timestamp_t> frames() const = 0;

    static std::unique_ptr<MeasurementDataUpdater> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<MeasurementDataQueue> measurement_queue,
      std::shared_ptr<estimation::IMUPropagationUpdateHandler> propagator,
      std::shared_ptr<estimation::StateVariableReadAccessor const>
        state_reader);
  };
}  // namespace cyclops::measurement
