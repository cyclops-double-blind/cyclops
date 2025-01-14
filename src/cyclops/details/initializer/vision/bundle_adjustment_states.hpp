#pragma once

#include "cyclops/details/utils/block_meta.hpp"
#include "cyclops/details/type.hpp"

#include <Eigen/Dense>
#include <functional>

namespace cyclops {
  struct se3_transform_t;
}  // namespace cyclops

namespace cyclops::initializer {
  struct multiview_geometry_t;

  class BundleAdjustmentCameraMotionStateBlock {
  private:
    block_meta::block_cascade<block_meta::orientation, block_meta::position>
      _data_block;

  public:
    explicit BundleAdjustmentCameraMotionStateBlock(
      se3_transform_t const& guess);

    Eigen::Map<Eigen::Quaterniond> orientation();
    Eigen::Map<Eigen::Quaterniond const> orientation() const;

    Eigen::Map<Eigen::Vector3d> position();
    Eigen::Map<Eigen::Vector3d const> position() const;

    double* data();
    double const* data() const;

    static Eigen::Map<Eigen::Quaterniond> orientation(double* data);
    static Eigen::Map<Eigen::Quaterniond const> orientation(double const* data);
    static Eigen::Map<Eigen::Vector3d> position(double* data);
    static Eigen::Map<Eigen::Vector3d const> position(double const* data);

    se3_transform_t asSE3Transform() const;
  };

  class BundleAdjustmentLandmarkPositionStateBlock {
  private:
    std::array<double, 3> _data_block;

  public:
    explicit BundleAdjustmentLandmarkPositionStateBlock(
      Eigen::Vector3d const& guess);

    Eigen::Map<Eigen::Vector3d> position();
    Eigen::Map<Eigen::Vector3d const> position() const;
    double* data();
    double const* data() const;

    static Eigen::Map<Eigen::Vector3d> position(double* data);
    static Eigen::Map<Eigen::Vector3d const> position(double const* data);

    Eigen::Vector3d asVector3() const;
  };

  struct BundleAdjustmentOptimizationState {
    using MotionBlock = BundleAdjustmentCameraMotionStateBlock;
    using LandmarkBlock = BundleAdjustmentLandmarkPositionStateBlock;

    explicit BundleAdjustmentOptimizationState(
      multiview_geometry_t const& initial_guess);

    std::map<frame_id_t, MotionBlock> camera_motions;
    std::map<landmark_id_t, LandmarkBlock> landmark_positions;

    using MotionBlockRef = std::reference_wrapper<MotionBlock>;
    using MotionBlockRefPair = std::tuple<MotionBlockRef, MotionBlockRef>;

    std::optional<MotionBlockRefPair> normalize();

    multiview_geometry_t as_multi_view_geometry() const;
  };
}  // namespace cyclops::initializer
