#pragma once

#include "cyclops/details/type.hpp"
#include <memory>

namespace cyclops::config::measurement {
  struct keyframe_detection_threshold_t {
    int min_novel_landmarks;
    double min_average_parallax;
  };

  struct keyframe_window_config_t {
    int optimization_phase_max_keyframes;
    int initialization_phase_max_keyframes;
  };

  struct image_update_throttling_config_t {
    double update_rate_target;
    double update_rate_smoothing_window_size;
  };
}  // namespace cyclops::config::measurement

namespace cyclops::config::initializer::vision {
  struct two_view_geometry_model_selection_config_t {
    int ransac_batch_size;
    double homography_selection_score_threshold;
  };

  struct two_view_triangulation_success_threshold_t {
    double min_p_value;
    double max_normalized_deviation;
  };

  struct two_view_motion_hypothesis_test_threshold_t {
    int min_triangulation_success;
    double min_imu_rotation_consistency_p_value;
  };

  struct two_view_config_t {
    two_view_geometry_model_selection_config_t model_selection;
    two_view_triangulation_success_threshold_t triangulation_acceptance;
    two_view_motion_hypothesis_test_threshold_t motion_hypothesis;
  };

  struct multiview_config_t {
    int bundle_adjustment_max_iterations;
    double bundle_adjustment_max_solver_time;
    double scale_gauge_soft_constraint_deviation;
  };

  struct solution_acceptance_threshold_t {
    double min_significant_probability;
    double min_inlier_ratio;
  };
}  // namespace cyclops::config::initializer::vision

namespace cyclops::config::initializer::imu {
  struct rotation_match_config_t {
    double vision_imu_rotation_consistency_angle_threshold;
  };

  struct scale_sampling_config_t {
    double sampling_domain_lowerbound;
    double sampling_domain_upperbound;
    int samples_count;

    double min_evaluation_success_rate;
  };

  struct solution_refinement_config_t {
    int max_iteration;
    double stepsize_tolerance;
    double gradient_tolerance;

    double duplicate_tolerance;
  };

  struct solution_candidate_threshold_t {
    double cost_significance;
  };

  struct solution_acceptance_threshold_t {
    double max_rotation_deviation;
    double max_scale_log_deviation;
    double max_normalized_gravity_deviation;
    double max_normalized_velocity_deviation;
    double max_sfm_perturbation;

    double rotation_match_min_p_value;
    double translation_match_min_p_value;
  };
}  // namespace cyclops::config::initializer::imu

namespace cyclops::config::initializer {
  struct observability_pretest_threshold_t {
    int min_landmark_overlap;
    int min_keyframes;

    double min_average_parallax;
  };

  struct vision_solver_config_t {
    double feature_point_isotropic_noise;
    double bundle_adjustment_robust_kernel_radius;

    vision::two_view_config_t two_view;
    vision::multiview_config_t multiview;
    vision::solution_acceptance_threshold_t acceptance_test;

    static vision_solver_config_t createDefault();
  };

  struct imu_solver_config_t {
    imu::rotation_match_config_t rotation_match;
    imu::scale_sampling_config_t sampling;
    imu::solution_refinement_config_t refinement;
    imu::solution_candidate_threshold_t candidate_test;
    imu::solution_acceptance_threshold_t acceptance_test;

    static imu_solver_config_t createDefault();
  };

  struct initialization_config_t {
    observability_pretest_threshold_t observability_pretest;

    vision_solver_config_t vision;
    imu_solver_config_t imu;

    static initialization_config_t createDefault();
  };
}  // namespace cyclops::config::initializer

namespace cyclops::config::estimation {
  struct optimizer_config_t {
    int max_num_iterations;
    double max_solver_time_in_seconds;
  };

  struct landmark_acceptance_threshold_t {
    double inlier_min_information_index;
    double inlier_min_depth;
    double inlier_mahalanobis_error;
    double mapping_acceptance_min_eigenvalue;
  };

  struct fault_detection_threshold_t {
    double min_landmark_accept_rate;
    double min_final_cost_p_value;

    int max_landmark_update_failures;
    int max_final_cost_sanity_failures;
  };

  struct estimator_config_t {
    optimizer_config_t optimizer;
    landmark_acceptance_threshold_t landmark_acceptance;
    fault_detection_threshold_t fault_detection;

    static estimator_config_t createDefault();
  };
}  // namespace cyclops::config::estimation

namespace cyclops {
  struct sensor_statistics_t {
    double acc_white_noise;
    double gyr_white_noise;
    double acc_random_walk;
    double gyr_random_walk;
    double acc_bias_prior_stddev;
    double gyr_bias_prior_stddev;
  };

  struct sensor_extrinsics_t {
    double imu_camera_time_delay;
    se3_transform_t imu_camera_transform;
  };

  struct cyclops_global_config_t {
    double gravity_norm;

    sensor_statistics_t noise;
    sensor_extrinsics_t extrinsics;

    config::measurement::keyframe_detection_threshold_t keyframe_detection;
    config::measurement::keyframe_window_config_t keyframe_window;
    config::measurement::image_update_throttling_config_t update_throttling;

    config::initializer::initialization_config_t initialization;
    config::estimation::estimator_config_t estimation;
  };

  std::unique_ptr<cyclops_global_config_t> make_default_cyclops_global_config(
    sensor_statistics_t const& sensor_noise,
    sensor_extrinsics_t const& sensor_extrinsics);
}  // namespace cyclops
