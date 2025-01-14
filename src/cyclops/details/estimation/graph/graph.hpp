#pragma once

#include "cyclops/details/estimation/type.hpp"

#include "cyclops/details/measurement/type.hpp"
#include "cyclops/details/utils/type.hpp"
#include "cyclops/details/type.hpp"

#include <ceres/ceres.h>

#include <memory>
#include <vector>
#include <tuple>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::estimation {
  struct prior_node_t;
  struct neighbor_query_result_t;
  struct landmark_acceptance_t;
  struct gaussian_prior_t;

  struct FactorGraphCostUpdater;
  struct FactorGraphStateNodeMap;

  class FactorGraphInstance {
  private:
    struct Impl;
    std::unique_ptr<Impl> _impl;

  public:
    FactorGraphInstance(
      std::unique_ptr<FactorGraphCostUpdater> cost_helper,
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<FactorGraphStateNodeMap> node_map);
    ~FactorGraphInstance();

    void fixGauge(frame_id_t frame_id);
    bool addFrameStateBlock(frame_id_t frame_id);
    bool addLandmarkStateBlock(landmark_id_t landmark_id);

    void addImuCost(measurement::imu_motion_t const& imu_motion);
    void addBiasPriorCost(frame_id_t frame_id);

    landmark_acceptance_t addLandmarkCost(
      std::set<frame_id_t> const& solvable_motions, landmark_id_t feature_id,
      measurement::feature_track_t const& track);

    void setPriorCost(gaussian_prior_t const& priors);

    std::optional<node_set_cref_t> queryNeighbors(node_t const& node) const;
    neighbor_query_result_t queryNeighbors(node_set_t const& nodes) const;

    std::optional<prior_node_t> const& prior() const;

    std::string report() const;
    ceres::Solver::Summary solve();
    std::tuple<EigenCRSMatrix, Eigen::VectorXd> evaluate(
      std::vector<node_t> const& nodes,
      std::vector<factor_ptr_t> const& factors);
  };
}  // namespace cyclops::estimation
