#include "cyclops_tests/mockups/data_provider.hpp"
#include "cyclops_tests/data/imu.hpp"
#include "cyclops_tests/data/landmark.hpp"

#include "cyclops/details/measurement/preintegration.hpp"

namespace cyclops::measurement {
  using Mockup = MeasurementDataProviderMockup;

  using Eigen::Matrix2d;
  using Eigen::Vector3d;

  void Mockup::reset() {
    _imu.clear();
    _tracks.clear();
  }

  void Mockup::updateFrame(frame_id_t id, image_data_t const&) {
    _updated_frames.insert(id);
  }

  void Mockup::updateFrame(
    frame_id_t prev_frame, frame_id_t curr_frame, image_data_t const&,
    IMUPreintegration::UniquePtr) {
    _updated_frames.insert(curr_frame);
    _updated_imu_frames.emplace(std::make_tuple(prev_frame, curr_frame));
  }

  void Mockup::updateImuBias() {
    _bias_update_count++;
  }

  void Mockup::updateImuBias(
    Vector3d const& bias_acc, Vector3d const& bias_gyr) {
    _bias_update_count++;
  }

  void Mockup::marginalize(
    frame_id_t drop_frame, std::set<landmark_id_t> const& drop_landmarks) {
    _dropped_frames.insert(drop_frame);
    _dropped_landmarks.insert(drop_landmarks.begin(), drop_landmarks.end());
    _marginalization_count++;
  }

  imu_motions_t const& Mockup::imu() const {
    return _imu;
  }

  feature_tracks_t const& Mockup::tracks() const {
    return _tracks;
  }

  std::unique_ptr<MeasurementDataProviderMockup>
  make_measurement_provider_mockup(
    std::mt19937& rgen, sensor_statistics_t const& imu_noise,
    Matrix2d const& landmark_cov, pose_signal_t pose_signal,
    se3_transform_t const& extrinsic, Vector3d const& bias_acc,
    Vector3d const& bias_gyr, landmark_positions_t const& landmarks,
    std::map<frame_id_t, timestamp_t> const& frames) {
    auto result = std::make_unique<Mockup>();
    result->_imu = make_imu_motions(
      rgen, imu_noise, bias_acc, bias_gyr, pose_signal, frames);
    result->_tracks = make_landmark_tracks(
      pose_signal, extrinsic, landmarks, frames, rgen, landmark_cov);
    return result;
  }

  std::unique_ptr<MeasurementDataProviderMockup>
  make_measurement_provider_mockup(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    Vector3d const& bias_acc, Vector3d const& bias_gyr,
    landmark_positions_t const& landmarks,
    std::map<frame_id_t, timestamp_t> const& frames) {
    auto result = std::make_unique<Mockup>();
    result->_imu = make_imu_motions(bias_acc, bias_gyr, pose_signal, frames);
    result->_tracks =
      make_landmark_tracks(pose_signal, extrinsic, landmarks, frames);
    return result;
  }

  std::unique_ptr<MeasurementDataProviderMockup>
  make_measurement_provider_mockup(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const& landmarks,
    std::map<frame_id_t, timestamp_t> const& frames) {
    return make_measurement_provider_mockup(
      pose_signal, extrinsic, Vector3d::Zero(), Vector3d::Zero(), landmarks,
      frames);
  }
}  // namespace cyclops::measurement
