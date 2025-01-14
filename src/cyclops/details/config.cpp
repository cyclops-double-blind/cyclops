#include "cyclops/details/config.hpp"

namespace cyclops::config::initializer {
  vision_solver_config_t vision_solver_config_t::createDefault() {
    return {
      .feature_point_isotropic_noise = 0.0075,
      .bundle_adjustment_robust_kernel_radius = 1.0,
      .two_view =
        {
          .model_selection =
            {
              .ransac_batch_size = 200,
              .homography_selection_score_threshold = 0.45,
            },
          .triangulation_acceptance =
            {
              .min_p_value = 0.05,
              .max_normalized_deviation = 0.5,
            },
          .motion_hypothesis =
            {
              .min_triangulation_success = 25,
              .min_imu_rotation_consistency_p_value = 0.05,
            },
        },
      .multiview =
        {
          .bundle_adjustment_max_iterations = 20,
          .bundle_adjustment_max_solver_time = 0.06,
          .scale_gauge_soft_constraint_deviation = 0.001,
        },
      .acceptance_test =
        {
          .min_significant_probability = 0.05,
          .min_inlier_ratio = 0.2,
        },
    };
  }

  imu_solver_config_t imu_solver_config_t::createDefault() {
    return {
      .rotation_match =
        {
          .vision_imu_rotation_consistency_angle_threshold = 0.05,
        },
      .sampling =
        {
          .sampling_domain_lowerbound = 0.001,
          .sampling_domain_upperbound = 50.0,
          .samples_count = 200,
          .min_evaluation_success_rate = 0.9,
        },
      .refinement =
        {
          .max_iteration = 100,
          .stepsize_tolerance = 1e-4,
          .gradient_tolerance = 1e-6,
          .duplicate_tolerance = 1e-4,
        },
      .candidate_test =
        {
          .cost_significance = 0.003,
        },
      .acceptance_test =
        {
          .max_rotation_deviation = 0.01,
          .max_scale_log_deviation = 0.7,
          .max_normalized_gravity_deviation = 0.05,
          .max_normalized_velocity_deviation = 0.05,
          .max_sfm_perturbation = 0.10,
          .rotation_match_min_p_value = 0.05,
          .translation_match_min_p_value = 0.003,
        },
    };
  }

  initialization_config_t initialization_config_t::createDefault() {
    return {
      .observability_pretest =
        {
          .min_landmark_overlap = 10,
          .min_keyframes = 4,
          .min_average_parallax = 0.005,
        },
      .vision = vision_solver_config_t::createDefault(),
      .imu = imu_solver_config_t::createDefault(),
    };
  }
}  // namespace cyclops::config::initializer

namespace cyclops::config::estimation {
  estimator_config_t estimator_config_t::createDefault() {
    return {
      .optimizer =
        {
          .max_num_iterations = 100,
          .max_solver_time_in_seconds = 0.06,
        },
      .landmark_acceptance =
        {
          .inlier_min_information_index = 30.0,
          .inlier_min_depth = 0.05,
          .inlier_mahalanobis_error = 25.0,
          .mapping_acceptance_min_eigenvalue = 10.0,
        },
      .fault_detection =
        {
          .min_landmark_accept_rate = 0.2,
          .min_final_cost_p_value = 0.01,
          .max_landmark_update_failures = 3,
          .max_final_cost_sanity_failures = -1,
        },
    };
  }
}  // namespace cyclops::config::estimation

namespace cyclops {
  std::unique_ptr<cyclops_global_config_t> make_default_cyclops_global_config(
    sensor_statistics_t const& sensor_noise,
    sensor_extrinsics_t const& sensor_extrinsics) {
    return std::make_unique<cyclops_global_config_t>(cyclops_global_config_t {
      .gravity_norm = 9.81,
      .noise = sensor_noise,
      .extrinsics = sensor_extrinsics,
      .keyframe_detection =
        {
          .min_novel_landmarks = 30,
          .min_average_parallax = 0.05,
        },
      .keyframe_window =
        {
          .optimization_phase_max_keyframes = 10,
          .initialization_phase_max_keyframes = 8,
        },
      .update_throttling =
        {
          .update_rate_target = 8,
          .update_rate_smoothing_window_size = 1.0,
        },
      .initialization =
        config::initializer::initialization_config_t::createDefault(),
      .estimation = config::estimation::estimator_config_t::createDefault(),
    });
  }
}  // namespace cyclops
