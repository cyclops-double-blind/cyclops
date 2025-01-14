#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace cyclops::telemetry {
  class InitializerTelemetry {
  public:
    virtual ~InitializerTelemetry() = default;
    virtual void reset();

    /* Vision initialization telemetry methods */

    struct image_observability_statistics_t {
      int common_features;
      double motion_parallax;
    };
    struct image_observability_pretest_t {
      std::map<frame_id_t, image_observability_statistics_t> frames;
      std::set<frame_id_t> connected_frames;
    };
    virtual void onImageObservabilityPretest(
      image_observability_pretest_t const& test);

    enum vision_initialization_failure_reason_t {
      NOT_ENOUGH_CONNECTED_IMAGE_FRAMES,
      NOT_ENOUGH_MOTION_PARALLAX,
      BEST_TWO_VIEW_SELECTION_FAILED,
      TWO_VIEW_GEOMETRY_FAILED,
      MULTI_VIEW_GEOMETRY_FAILED,
      BUNDLE_ADJUSTMENT_FAILED,
    };
    struct vision_initialization_failure_t {
      std::set<frame_id_t> frames;
      vision_initialization_failure_reason_t reason;
    };
    virtual void onVisionFailure(
      vision_initialization_failure_t const& failure);

    struct best_two_view_selection_t {
      std::set<frame_id_t> frames;

      frame_id_t frame_id_1;
      frame_id_t frame_id_2;
    };
    virtual void onBestTwoViewSelection(
      best_two_view_selection_t const& selection);

    struct two_view_motion_hypothesis_test_t {
      bool rotation_prior_test_passed;
      int triangulation_success_count;

      se3_transform_t motion;
    };
    struct two_view_motion_hypothesis_t {
      std::set<frame_id_t> frames;

      frame_id_t frame_id_1;
      frame_id_t frame_id_2;

      std::vector<two_view_motion_hypothesis_test_t> candidates;
    };
    virtual void onTwoViewMotionHypothesis(
      two_view_motion_hypothesis_t const& hypothesis);

    enum two_view_geometry_model_t { EPIPOLAR, HOMOGRAPHY };

    struct two_view_geometry_t {
      se3_transform_t motion;
      int triangulation_successes;
    };

    struct two_view_solver_success_t {
      std::set<frame_id_t> frames;

      two_view_geometry_model_t initial_selected_model;
      two_view_geometry_model_t final_selected_model;

      int landmarks_count;
      double homography_expected_inliers;
      double epipolar_expected_inliers;

      std::vector<two_view_geometry_t> motion_hypothesis;
    };
    virtual void onTwoViewSolverSuccess(
      two_view_solver_success_t const& success);

    struct bundle_adjustment_solution_t {
      std::map<frame_id_t, se3_transform_t> camera_motions;
      landmark_positions_t landmarks;
    };
    virtual void onBundleAdjustmentSuccess(
      bundle_adjustment_solution_t const& solution);

    struct bundle_adjustment_sanity_t {
      bool acceptable;

      double inlier_ratio;
      double final_cost_significant_probability;
    };
    struct bundle_adjustment_candidates_sanity_t {
      std::set<frame_id_t> frames;

      std::vector<bundle_adjustment_sanity_t> candidates_sanity;
    };
    virtual void onBundleAdjustmentSanity(
      bundle_adjustment_candidates_sanity_t const& sanity);

    /* IMU initialization telemetry methods */

    struct imu_match_attempt_t {
      int degrees_of_freedom;
      std::set<frame_id_t> frames;

      std::vector<std::tuple<double, double>> landscape;
      std::vector<std::tuple<double, double>> minima;
    };
    virtual void onIMUMatchAttempt(imu_match_attempt_t const& argument);

    struct imu_match_solution_point_t {
      double scale;
      double cost;

      Eigen::Vector3d gravity;
      Eigen::Vector3d acc_bias;
      Eigen::Vector3d gyr_bias;
      std::map<frame_id_t, Eigen::Quaterniond> imu_orientations;
      std::map<frame_id_t, Eigen::Vector3d> imu_body_velocities;
      std::map<frame_id_t, Eigen::Vector3d> sfm_positions;
    };

    struct imu_match_uncertainty_t {
      double final_cost_significant_probability;
      double scale_log_deviation;
      double gravity_max_deviation;
      double bias_max_deviation;
      double body_velocity_max_deviation;
      double scale_symmetric_translation_error_max_deviation;
    };

    struct imu_match_ambiguity_t {
      std::vector<imu_match_solution_point_t> solutions;
      std::vector<imu_match_uncertainty_t> uncertainties;
    };
    virtual void onIMUMatchAmbiguity(imu_match_ambiguity_t const& argument);

    enum imu_match_candidate_reject_reason_t {
      UNCERTAINTY_EVALUATION_FAILED,
      COST_PROBABILITY_INSIGNIFICANT,
      UNDERINFORMATIVE_PARAMETER,
      SCALE_LESS_THAN_ZERO,
    };

    struct imu_match_reject_t {
      imu_match_candidate_reject_reason_t reason;
      imu_match_solution_point_t solution;
      std::optional<imu_match_uncertainty_t> uncertainty;
    };
    virtual void onIMUMatchReject(imu_match_reject_t const& argument);
    virtual void onIMUMatchCandidateReject(imu_match_reject_t const& argument);

    struct imu_match_accept_t {
      imu_match_solution_point_t solution;
      imu_match_uncertainty_t uncertainty;
    };
    virtual void onIMUMatchAccept(imu_match_accept_t const& argument);

    struct vision_solution_candidate_digest_t {
      bool acceptable;
      std::set<frame_id_t> keyframes;
    };

    struct imu_solution_candidate_digest_t {
      int vision_solution_index;
      bool acceptable;

      double scale;
      std::set<frame_id_t> keyframes;
    };

    struct onfailure_argument_t {
      std::vector<vision_solution_candidate_digest_t> vision_solutions;
      std::vector<imu_solution_candidate_digest_t> imu_solutions;
    };
    virtual void onFailure(onfailure_argument_t const& argument);

    struct onsuccess_argument_t {
      frame_id_t initial_motion_frame_id;
      timestamp_t initial_motion_frame_timestamp;
      std::map<frame_id_t, se3_transform_t> sfm_camera_pose;

      double cost;
      double scale;
      Eigen::Vector3d gravity;
      std::map<frame_id_t, imu_motion_state_t> motions;
    };
    virtual void onSuccess(onsuccess_argument_t const& argument);

    static std::unique_ptr<InitializerTelemetry> createDefault();
  };
}  // namespace cyclops::telemetry
