#pragma once

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"
#include "cyclops/details/type.hpp"

#include "cyclops/details/telemetry/initializer.hpp"
#include "cyclops/details/telemetry/keyframe.hpp"
#include "cyclops/details/telemetry/optimizer.hpp"

#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace cyclops {
  using telemetry::InitializerTelemetry;
  using telemetry::KeyframeTelemetry;
  using telemetry::OptimizerTelemetry;

  struct cyclops_image_update_handle_t {
    frame_id_t frame_id;
    timestamp_t timestamp;
  };

  struct cyclops_keyframe_state_t {
    timestamp_t timestamp;

    Eigen::Vector3d acc_bias;
    Eigen::Vector3d gyr_bias;
    imu_motion_state_t motion_state;
  };

  struct cyclops_propagation_state_t {
    timestamp_t timestamp;
    imu_motion_state_t motion_state;
  };

  struct cyclops_main_argument_t {
    std::shared_ptr<cyclops_global_config_t const> config;
    std::optional<uint32_t> seed = std::nullopt;

    std::shared_ptr<OptimizerTelemetry> optimizer_telemetry = nullptr;
    std::shared_ptr<KeyframeTelemetry> keyframe_telemetry = nullptr;
    std::shared_ptr<InitializerTelemetry> initializer_telemetry = nullptr;
  };

  struct cyclops_estimation_update_result_t {
    bool reset;
    std::vector<cyclops_image_update_handle_t> update_handles;
  };

  class CyclopsMain {
  public:
    virtual ~CyclopsMain() = default;

    /*
     * ========================= Data thread methods ========================
     *
     * The following four methods are intended to be invoked in a "data thread",
     * a thread that spins in parallel to the optimizer thread.
     *
     * In addition, each IMU and landmark data is assumed to be updated in
     * time-aligned order. That is, if t1 and t2 are timestamps of two
     * consecutive IMU data updates, then t1 <= t2 must hold, and if s1, s2 are
     * timestamps of two consecutive landmark updates, then s1 <= s2.
     */
    virtual void enqueueLandmarkData(image_data_t const& data) = 0;
    virtual void enqueueIMUData(imu_data_t const& data) = 0;

    /*
     * Enqueue external reset request. This reset request is handled in the next
     * `updateEstimation()` call.
     */
    virtual void enqueueResetRequest() = 0;

    /*
     * Get the current IMU-rate propagated motion states.
     */
    virtual std::optional<cyclops_propagation_state_t> propagation() const = 0;
    /* ========================= Data thread methods ======================== */

    /*
     * ====================== Optimizer thread methods ======================
     *
     * The following three methods are intended to be invoked in the optimizer
     * thread.
     */
    virtual cyclops_estimation_update_result_t updateEstimation() = 0;
    virtual landmark_positions_t mappedLandmarks() const = 0;
    virtual std::map<frame_id_t, cyclops_keyframe_state_t> motions() const = 0;
    /* ====================== Optimizer thread methods ====================== */

    static std::unique_ptr<CyclopsMain> create(cyclops_main_argument_t args);
  };
}  // namespace cyclops
