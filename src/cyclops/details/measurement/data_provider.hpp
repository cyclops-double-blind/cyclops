#pragma once

#include "cyclops/details/measurement/type.hpp"

#include <memory>
#include <set>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::estimation {
  struct StateVariableReadAccessor;
}  // namespace cyclops::estimation

namespace cyclops::measurement {
  class MeasurementDataProvider {
  public:
    virtual ~MeasurementDataProvider() = default;
    virtual void reset() = 0;

    virtual void updateFrame(
      frame_id_t frame_id, image_data_t const& image_data) = 0;
    virtual void updateFrame(
      frame_id_t prev_frame_id, frame_id_t curr_frame_id,
      image_data_t const& image, std::unique_ptr<IMUPreintegration> imu) = 0;

    virtual void updateImuBias() = 0;
    virtual void updateImuBias(
      Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr) = 0;

    virtual void marginalize(
      frame_id_t drop_frame, std::set<landmark_id_t> const& drop_landmarks) = 0;

    virtual imu_motions_t const& imu() const = 0;
    virtual feature_tracks_t const& tracks() const = 0;

    static std::unique_ptr<MeasurementDataProvider> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<estimation::StateVariableReadAccessor const> state);
  };
}  // namespace cyclops::measurement
