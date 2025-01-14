#include "cyclops/details/measurement/data_provider.hpp"
#include "cyclops/details/measurement/preintegration.hpp"

#include "cyclops/details/estimation/state/accessor_read.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

namespace cyclops::measurement {
  using std::set;

  using Eigen::Vector3d;
  using estimation::StateVariableReadAccessor;

  class MeasurementDataProviderImpl: public MeasurementDataProvider {
  private:
    std::shared_ptr<cyclops_global_config_t const> _config;
    std::shared_ptr<StateVariableReadAccessor const> _state;

    imu_motions_t _imu_motions;
    feature_tracks_t _feature_tracks;

    void updateLandmark(frame_id_t frame_id, image_data_t const& image_data);

  public:
    MeasurementDataProviderImpl(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<StateVariableReadAccessor const> state);
    ~MeasurementDataProviderImpl();
    void reset() override;

    void updateFrame(frame_id_t frame_id, image_data_t const& image) override;
    void updateFrame(
      frame_id_t prev_frame_id, frame_id_t curr_frame_id,
      image_data_t const& image, IMUPreintegration::UniquePtr imu) override;

    void updateImuBias() override;
    void updateImuBias(
      Vector3d const& bias_acc, Vector3d const& bias_gyr) override;

    void marginalize(
      frame_id_t drop_frame, set<landmark_id_t> const& drop_landmarks) override;

    imu_motions_t const& imu() const override;
    feature_tracks_t const& tracks() const override;
  };

  MeasurementDataProviderImpl::~MeasurementDataProviderImpl() = default;

  MeasurementDataProviderImpl::MeasurementDataProviderImpl(
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<StateVariableReadAccessor const> state)
      : _config(config), _state(state) {
  }

  void MeasurementDataProviderImpl::reset() {
    _imu_motions.clear();
    _feature_tracks.clear();
  }

  void MeasurementDataProviderImpl::updateLandmark(
    frame_id_t frame_id, image_data_t const& image) {
    for (auto const& [feature_id, feature] : image.features)
      _feature_tracks[feature_id].emplace(frame_id, feature);
  }

  void MeasurementDataProviderImpl::updateFrame(
    frame_id_t frame_id, image_data_t const& image) {
    updateLandmark(frame_id, image);
  }

  void MeasurementDataProviderImpl::updateFrame(
    frame_id_t prev_frame_id, frame_id_t curr_frame_id,
    image_data_t const& image_data, IMUPreintegration::UniquePtr imu_data) {
    updateLandmark(curr_frame_id, image_data);
    _imu_motions.push_back({
      .from = prev_frame_id,
      .to = curr_frame_id,
      .data = std::move(imu_data),
    });
  }

  void MeasurementDataProviderImpl::updateImuBias() {
    auto const& motion_frames = _state->motionFrames();

    if (motion_frames.empty())
      return;
    for (auto& imu_motion : _imu_motions) {
      auto i = motion_frames.lower_bound(imu_motion.from);
      if (i == motion_frames.end()) {
        __logger__->warn(
          "Unknown frame ({}) queried during IMU bias update", imu_motion.from);
        __logger__->warn(
          "IMU motion: {} -> {}", imu_motion.from, imu_motion.to);
        continue;
      }
      auto const& [_, x] = *i;

      auto b_a = estimation::acc_bias_of_motion_frame_block(x);
      auto b_w = estimation::gyr_bias_of_motion_frame_block(x);
      __logger__->trace(
        "Updating IMU bias for frame {} -> {}: {}, {}", imu_motion.from,
        imu_motion.to, b_a.transpose(), b_w.transpose());

      imu_motion.data->updateBias(b_a, b_w);
    }
  }

  void MeasurementDataProviderImpl::updateImuBias(
    Vector3d const& bias_acc, Vector3d const& bias_gyr) {
    for (auto& imu_motion : _imu_motions)
      imu_motion.data->updateBias(bias_acc, bias_gyr);
  }

  void MeasurementDataProviderImpl::marginalize(
    frame_id_t drop_frame, set<landmark_id_t> const& drop_landmarks) {
    ranges::actions::remove_if(_imu_motions, [&](auto const& _) {
      return _.from == drop_frame || _.to == drop_frame;
    });

    for (auto const& landmark : drop_landmarks)
      _feature_tracks.erase(landmark);
    for (auto& [_, track] : _feature_tracks)
      track.erase(drop_frame);  // XXX: remove only when the track has been
                                // participated to the optimization

    for (auto i = _feature_tracks.begin(); i != _feature_tracks.end();) {
      if (i->second.empty()) {
        i = _feature_tracks.erase(i);
      } else {
        ++i;
      }
    }
  }

  imu_motions_t const& MeasurementDataProviderImpl::imu() const {
    return _imu_motions;
  }

  feature_tracks_t const& MeasurementDataProviderImpl::tracks() const {
    return _feature_tracks;
  }

  std::unique_ptr<MeasurementDataProvider> MeasurementDataProvider::create(
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<StateVariableReadAccessor const> state) {
    return std::make_unique<MeasurementDataProviderImpl>(config, state);
  }
}  // namespace cyclops::measurement
