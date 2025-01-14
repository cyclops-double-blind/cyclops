#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

#include <memory>
#include <optional>
#include <tuple>

namespace cyclops {
  struct cyclops_global_config_t;
}  // namespace cyclops

namespace cyclops::estimation {
  class IMUPropagationUpdateHandler {
  public:
    virtual ~IMUPropagationUpdateHandler() = default;
    virtual void reset() = 0;

    virtual void updateOptimization(
      timestamp_t last_timestamp,
      motion_frame_parameter_block_t const& last_state) = 0;
    virtual void updateIMUData(imu_data_t const& data) = 0;

    using timestamped_motion_state_t =
      std::tuple<timestamp_t, imu_motion_state_t>;
    virtual std::optional<timestamped_motion_state_t> get() const = 0;

    static std::unique_ptr<IMUPropagationUpdateHandler> create(
      std::shared_ptr<cyclops_global_config_t const> config);
  };
}  // namespace cyclops::estimation
