#pragma once

#include "cyclops_tests/data/typefwd.hpp"
#include "cyclops_tests/signal.hpp"
#include "cyclops/details/measurement/type.hpp"

#include <map>
#include <memory>
#include <random>
#include <vector>

namespace cyclops::measurement {
  struct IMUPreintegration;
}

namespace cyclops {
  struct imu_data_t;
  struct sensor_statistics_t;

  struct imu_mockup_t {
    Eigen::Vector3d bias_acc;
    Eigen::Vector3d bias_gyr;
    imu_data_t measurement;
  };

  imu_mockup_sequence_t generate_imu_data(
    pose_signal_t pose_signal, std::vector<timestamp_t> const& timestamps);
  imu_mockup_sequence_t generate_imu_data(
    pose_signal_t pose_signal, std::vector<timestamp_t> const& timestamps,
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr);
  imu_mockup_sequence_t generate_imu_data(
    pose_signal_t pose_signal, std::vector<timestamp_t> const& timestamps,
    std::mt19937& rgen, sensor_statistics_t const& noise);
  imu_mockup_sequence_t generate_imu_data(
    pose_signal_t pose_signal, std::vector<timestamp_t> const& timestamps,
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
    std::mt19937& rgen, sensor_statistics_t const& noise);

  std::unique_ptr<measurement::IMUPreintegration> make_imu_preintegration(
    pose_signal_t pose_signal, timestamp_t t_s, timestamp_t t_e);
  std::unique_ptr<measurement::IMUPreintegration> make_imu_preintegration(
    sensor_statistics_t const& noise, pose_signal_t pose_signal,
    timestamp_t t_s, timestamp_t t_e);
  std::unique_ptr<measurement::IMUPreintegration> make_imu_preintegration(
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
    pose_signal_t pose_signal, timestamp_t t_s, timestamp_t t_e);
  std::unique_ptr<measurement::IMUPreintegration> make_imu_preintegration(
    std::mt19937& rgen, sensor_statistics_t const& noise,
    pose_signal_t pose_signal, timestamp_t t_s, timestamp_t t_e);
  std::unique_ptr<measurement::IMUPreintegration> make_imu_preintegration(
    std::mt19937& rgen, sensor_statistics_t const& noise,
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
    pose_signal_t pose_signal, timestamp_t t_s, timestamp_t t_e);

  measurement::imu_motions_t make_imu_motions(
    pose_signal_t pose_signal, std::map<frame_id_t, timestamp_t> const& frames);
  measurement::imu_motions_t make_imu_motions(
    sensor_statistics_t const& noise, pose_signal_t pose_signal,
    std::map<frame_id_t, timestamp_t> const& frames);
  measurement::imu_motions_t make_imu_motions(
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
    pose_signal_t pose_signal, std::map<frame_id_t, timestamp_t> const& frames);
  measurement::imu_motions_t make_imu_motions(
    std::mt19937& rgen, sensor_statistics_t const& noise,
    pose_signal_t pose_signal, std::map<frame_id_t, timestamp_t> const& frames);
  measurement::imu_motions_t make_imu_motions(
    std::mt19937& rgen, sensor_statistics_t const& noise,
    Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr,
    pose_signal_t pose_signal, std::map<frame_id_t, timestamp_t> const& frames);
}  // namespace cyclops
