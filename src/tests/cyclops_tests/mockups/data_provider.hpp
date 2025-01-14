#pragma once

#include "cyclops/details/measurement/data_provider.hpp"
#include "cyclops/details/measurement/type.hpp"

#include "cyclops_tests/signal.hpp"

#include <map>
#include <memory>
#include <random>
#include <set>
#include <tuple>

namespace cyclops {
  struct sensor_statistics_t;
  struct se3_transform_t;
}  // namespace cyclops

namespace cyclops::measurement {
  struct IMUPreintegration;

  struct MeasurementDataProviderMockup: public MeasurementDataProvider {
    imu_motions_t _imu;
    feature_tracks_t _tracks;

    std::set<frame_id_t> _updated_frames;
    std::set<std::tuple<frame_id_t, frame_id_t>> _updated_imu_frames;
    size_t _bias_update_count = 0;
    size_t _marginalization_count = 0;

    std::set<frame_id_t> _dropped_frames;
    std::set<landmark_id_t> _dropped_landmarks;

    void reset() override;

    void updateFrame(frame_id_t id, image_data_t const&) override;
    void updateFrame(
      frame_id_t prev_frame, frame_id_t curr_frame,
      image_data_t const& landmark_measurement,
      std::unique_ptr<IMUPreintegration> imu_motion) override;

    void updateImuBias() override;
    void updateImuBias(
      Eigen::Vector3d const& bias_acc,
      Eigen::Vector3d const& bias_gyr) override;

    void marginalize(
      frame_id_t drop_frame,
      std::set<landmark_id_t> const& drop_landmarks) override;

    imu_motions_t const& imu() const override;
    feature_tracks_t const& tracks() const override;
  };

  std::unique_ptr<MeasurementDataProviderMockup>
  make_measurement_provider_mockup(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const&, std::map<frame_id_t, timestamp_t> const&);

  std::unique_ptr<MeasurementDataProviderMockup>
  make_measurement_provider_mockup(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
    landmark_positions_t const&, std::map<frame_id_t, timestamp_t> const&);

  std::unique_ptr<MeasurementDataProviderMockup>
  make_measurement_provider_mockup(
    std::mt19937& rgen, sensor_statistics_t const& imu_noise,
    Eigen::Matrix2d const& landmark_cov, pose_signal_t pose_signal,
    se3_transform_t const& extrinsic, Eigen::Vector3d const& bias_acc,
    Eigen::Vector3d const& bias_gyr, landmark_positions_t const&,
    std::map<frame_id_t, timestamp_t> const&);
}  // namespace cyclops::measurement
