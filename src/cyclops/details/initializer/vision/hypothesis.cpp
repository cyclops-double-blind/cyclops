#include "cyclops/details/initializer/vision/hypothesis.hpp"
#include "cyclops/details/initializer/vision/triangulation.hpp"
#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/utils/vision.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  namespace views = ranges::views;

  class TwoViewMotionHypothesisSelectorImpl:
      public TwoViewMotionHypothesisSelector {
  private:
    std::shared_ptr<cyclops_global_config_t const> _config;

    bool testMotionIMURotationPrior(
      rotation_translation_matrix_pair_t const& motion,
      two_view_imu_rotation_data_t const& imu_prior) const;
    bool testTriangulationSuccess(
      two_view_triangulation_t const& triangulation, int inliers) const;

  public:
    explicit TwoViewMotionHypothesisSelectorImpl(
      std::shared_ptr<cyclops_global_config_t const> config);
    void reset() override;

    std::vector<two_view_geometry_t> selectPossibleMotions(
      motion_hypotheses_t const& motions,
      two_view_feature_set_t const& image_data, inlier_set_t const& inliers,
      two_view_imu_rotation_data_t const& prior) override;
  };

  TwoViewMotionHypothesisSelectorImpl::TwoViewMotionHypothesisSelectorImpl(
    std::shared_ptr<cyclops_global_config_t const> config)
      : _config(config) {
  }

  void TwoViewMotionHypothesisSelectorImpl::reset() {
    // Does nothing
  }

  static Eigen::Vector3d log_so3(Eigen::Matrix3d const& R) {
    auto w = Eigen::AngleAxisd(R);
    return w.angle() * w.axis();
  }

  bool TwoViewMotionHypothesisSelectorImpl::testMotionIMURotationPrior(
    rotation_translation_matrix_pair_t const& motion,
    two_view_imu_rotation_data_t const& imu_prior) const {
    auto const& vision_config = _config->initialization.vision;
    auto const& hypothesis_config = vision_config.two_view.motion_hypothesis;
    auto min_p_value = hypothesis_config.min_imu_rotation_consistency_p_value;

    auto R_hat = imu_prior.value.matrix().eval();
    auto v = log_so3(R_hat.transpose() * motion.rotation);

    auto llt = Eigen::LDLT<Eigen::Matrix3d>(imu_prior.covariance);
    auto error = v.dot(llt.solve(v));

    auto p_value = 1.0 - chi_squared_cdf(3, error);

    return p_value > min_p_value;
  }

  bool TwoViewMotionHypothesisSelectorImpl::testTriangulationSuccess(
    two_view_triangulation_t const& triangulation, int inliers) const {
    auto const& config =
      _config->initialization.vision.two_view.motion_hypothesis;

    auto const success = triangulation.landmarks.size();
    auto const min_success = config.min_triangulation_success;

    __logger__->trace("Success: {}, min success: {}", success, min_success);
    __logger__->trace("Expected inliers: {}", triangulation.expected_inliers);

    return success >= min_success;
  }

  std::vector<two_view_geometry_t>
  TwoViewMotionHypothesisSelectorImpl::selectPossibleMotions(
    motion_hypotheses_t const& motions, two_view_feature_set_t const& features,
    inlier_set_t const& inliers, two_view_imu_rotation_data_t const& prior) {
    __logger__->debug("Selecting best two-view motion hypothesis");
    __logger__->debug("Motion candidates: {}", motions.size());

    auto motions_filtered =  //
      motions | views::filter([&](auto const& motion) {
        return testMotionIMURotationPrior(motion, prior);
      }) |
      ranges::to_vector;

    if (motions_filtered.empty()) {
      __logger__->warn("Visual motion does not align with the IMU rotation");
      __logger__->info(
        "Suggestion: ensure that the IMU-camera extrinsic is correct.");
    }
    __logger__->debug(
      "Candidates that match IMU rotation: {}", motions_filtered.size());

    auto motion_triangulations =
      motions_filtered | views::transform([&](auto const& motion) {
        auto triangulation = triangulate_two_view_feature_pairs(
          _config->initialization.vision, features, inliers, motion);
        return std::make_tuple(motion, triangulation);
      }) |
      ranges::to_vector;

    auto successed_motion_triangulations =
      motion_triangulations | views::filter([&](auto const& pair) {
        auto const& [motion, triangulation] = pair;
        return testTriangulationSuccess(triangulation, inliers.size());
      }) |
      ranges::to_vector;
    __logger__->debug(
      "Triangulation successes: {}", successed_motion_triangulations.size());

    return  //
      successed_motion_triangulations | views::transform([](auto const& pair) {
        auto const& [motion, triangulation] = pair;
        auto const& [R, p] = motion;
        return two_view_geometry_t {
          .camera_motion = se3_transform_t {p, Eigen::Quaterniond(R)},
          .landmarks = triangulation.landmarks,
        };
      }) |
      ranges::to_vector;
  }

  std::unique_ptr<TwoViewMotionHypothesisSelector>
  TwoViewMotionHypothesisSelector::create(
    std::shared_ptr<cyclops_global_config_t const> config) {
    return std::make_unique<TwoViewMotionHypothesisSelectorImpl>(config);
  }
}  // namespace cyclops::initializer
