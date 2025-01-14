#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

#include <Eigen/Dense>
#include <random>

namespace cyclops {
  struct imu_motion_state_t;

  double perturbate(double const& x, double const& s, std::mt19937& rgen);

  Eigen::Vector2d perturbate(
    Eigen::Vector2d const& x, double const& S, std::mt19937& rgen);
  Eigen::Vector2d perturbate(
    Eigen::Vector2d const& x, Eigen::Matrix2d const& S, std::mt19937& rgen);

  Eigen::Vector3d perturbate(
    Eigen::Vector3d const& x, double const& S, std::mt19937& rgen);
  Eigen::Vector3d perturbate(
    Eigen::Vector3d const& x, Eigen::Matrix3d const& S, std::mt19937& rgen);
  Eigen::Quaterniond perturbate(
    Eigen::Quaterniond const& q, double const& S, std::mt19937& rgen);
  Eigen::Quaterniond perturbate(
    Eigen::Quaterniond const& q, Eigen::Matrix3d const& S, std::mt19937& rgen);

  imu_motion_state_t perturbate(
    imu_motion_state_t const& x, double S, std::mt19937& rgen);

  estimation::motion_frame_parameter_block_t make_perturbated_frame_state(
    imu_motion_state_t const& x, double perturbation, std::mt19937& rgen);
  estimation::landmark_parameter_block_t make_perturbated_landmark_state(
    Eigen::Vector3d const& landmark, double perturbation, std::mt19937& rgen);
}  // namespace cyclops
