#pragma once

#include "cyclops/details/type.hpp"

#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace cyclops::estimation {
  struct node_t;
}  // namespace cyclops::estimation

namespace cyclops::measurement {
  struct IMUPreintegration;

  struct imu_motion_t {
    frame_id_t from;
    frame_id_t to;
    std::unique_ptr<IMUPreintegration> data;
  };

  using imu_motion_ref_t = std::reference_wrapper<imu_motion_t const>;
  using imu_motions_t = std::vector<imu_motion_t>;
  using imu_motion_refs_t = std::vector<imu_motion_ref_t>;

  using feature_track_t = std::map<frame_id_t, feature_point_t>;
  using feature_tracks_t = std::map<landmark_id_t, feature_track_t>;
}  // namespace cyclops::measurement
