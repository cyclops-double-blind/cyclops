#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <tuple>

namespace cyclops::initializer {
  struct vision_bootstrap_solution_t;

  struct imu_match_camera_rotation_prior_t {
    std::map<frame_id_t, Eigen::Quaterniond> rotations;
    Eigen::MatrixXd weight;
  };

  struct imu_match_camera_translation_prior_t {
    std::map<frame_id_t, Eigen::Vector3d> translations;
    Eigen::MatrixXd weight;
  };

  std::tuple<
    imu_match_camera_rotation_prior_t, imu_match_camera_translation_prior_t>
  make_imu_match_camera_motion_prior(vision_bootstrap_solution_t const& sfm);
}  // namespace cyclops::initializer
