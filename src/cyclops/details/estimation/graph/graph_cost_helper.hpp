#pragma once

#include "cyclops/details/estimation/graph/node.hpp"
#include "cyclops/details/measurement/type.hpp"

#include <set>
#include <variant>

namespace ceres {
  struct Problem;
}

namespace cyclops::measurement {
  struct IMUPreintegration;
}  // namespace cyclops::measurement

namespace cyclops {
  struct cyclops_global_config_t;
}  // namespace cyclops

namespace cyclops::estimation {
  class FactorGraphStateNodeMap;

  struct landmark_acceptance_t {
    struct accepted {
      int observation_count;
      size_t accepted_count;
    };

    struct rejected__uninitialized_landmark_state {};

    struct rejected__no_inlier_observation {
      int observation_count;
      int depth_threshold_failure_count;
      int mahalanobis_norm_test_failure_count;
    };

    struct rejected__deficient_information_weight {
      int observation_count;
      double information_index;
    };

    // clang-format off
    std::variant<
      accepted,
      rejected__uninitialized_landmark_state,
      rejected__no_inlier_observation,
      rejected__deficient_information_weight>
      variant;
    // clang-format on
  };

  class FactorGraphCostUpdater {
  private:
    struct Impl;
    std::unique_ptr<Impl> _pimpl;

  public:
    FactorGraphCostUpdater(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<FactorGraphStateNodeMap> node_map);
    ~FactorGraphCostUpdater();

    bool addImuCost(
      ceres::Problem& problem, measurement::imu_motion_t const& imu_motion);
    bool addBiasPriorCost(ceres::Problem& problem, frame_id_t frame_id);

    landmark_acceptance_t addLandmarkCostBatch(
      ceres::Problem& problem, std::set<frame_id_t> const& solvable_motions,
      landmark_id_t feature_id, measurement::feature_track_t const& track);
  };
}  // namespace cyclops::estimation
