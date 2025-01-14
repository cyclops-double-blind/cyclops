#pragma once

#include "cyclops/details/initializer/vision/type.hpp"

#include <set>
#include <map>
#include <memory>
#include <vector>

namespace cyclops {
  struct cyclops_global_config_t;
  struct rotation_translation_matrix_pair_t;
}  // namespace cyclops

namespace cyclops::initializer {
  struct two_view_geometry_t;
  struct two_view_imu_rotation_data_t;

  class TwoViewMotionHypothesisSelector {
  public:
    using motion_hypotheses_t = std::vector<rotation_translation_matrix_pair_t>;

    using two_view_feature_set_t =
      std::map<landmark_id_t, two_view_feature_pair_t>;
    using inlier_set_t = std::set<landmark_id_t>;

  public:
    virtual ~TwoViewMotionHypothesisSelector() = default;
    virtual void reset() = 0;

    virtual std::vector<two_view_geometry_t> selectPossibleMotions(
      motion_hypotheses_t const& motions,
      two_view_feature_set_t const& image_data, inlier_set_t const& inliers,
      two_view_imu_rotation_data_t const& rotation_prior) = 0;

    static std::unique_ptr<TwoViewMotionHypothesisSelector> create(
      std::shared_ptr<cyclops_global_config_t const> config);
  };
}  // namespace cyclops::initializer
