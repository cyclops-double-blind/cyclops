#include "cyclops/details/estimation/graph/graph_cost_helper.hpp"
#include "cyclops/details/estimation/graph/graph_node_map.hpp"
#include "cyclops/details/estimation/graph/factor.hpp"
#include "cyclops/details/estimation/graph/node.hpp"

#include "cyclops/details/estimation/ceres/cost.imu_preintegration.hpp"
#include "cyclops/details/estimation/ceres/cost.imu_bias_walk.hpp"
#include "cyclops/details/estimation/ceres/cost.imu_bias_prior.hpp"
#include "cyclops/details/estimation/ceres/cost.landmark.hpp"

#include "cyclops/details/measurement/type.hpp"
#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/estimation/state/state_block.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <ceres/ceres.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

namespace cyclops::estimation {
  using Eigen::Matrix2d;
  using Eigen::Matrix3d;
  using Eigen::Vector2d;
  using Eigen::Vector3d;
  using Matrix2x3d = Eigen::Matrix<double, 2, 3>;

  using ceres::AutoDiffCostFunction;

  using measurement::feature_track_t;
  using measurement::imu_motion_t;
  using measurement::IMUPreintegration;

  namespace views = ranges::views;

  using node_context_id_t = std::pair<node_t, graph_node_context_ref_t>;
  using node_context_id_set_t = std::vector<node_context_id_t>;

  using ACCEPTED = landmark_acceptance_t::accepted;
  using UNINITIALIZED =
    landmark_acceptance_t::rejected__uninitialized_landmark_state;
  using NO_INLIER = landmark_acceptance_t::rejected__no_inlier_observation;
  using INFORMATION_DEFICIENT =
    landmark_acceptance_t::rejected__deficient_information_weight;

  enum feature_uncertainty_analysis_status_t {
    ANALYSIS_SUCCESS,
    ANALYSIS_FAIL_UNINITIALIZED_MOTION_STATE,
    ANALYSIS_FAIL_TOO_CLOSE,
    ANALYSIS_FAIL_LARGE_PROJECTION_ERROR,
  };

  struct feature_uncertainty_t {
    node_context_id_t input_frame;

    double depth;
    double mahalanobis_norm;
    Matrix3d information;
  };

  struct landmark_node_uncertainty_analysis_t {
    feature_uncertainty_analysis_status_t status;
    std::optional<feature_uncertainty_t> value = std::nullopt;
  };

  struct landmark_node_observation_skeleton_t {
    std::reference_wrapper<feature_point_t const> data;

    factor_t factor;
    node_context_id_t keyframe_node;
  };

  struct landmark_node_multiview_skeleton_t {
    node_context_id_t landmark_node;
    std::vector<landmark_node_observation_skeleton_t> observations;
  };

  struct landmark_node_multiview_skeleton_analysis_t {
    landmark_acceptance_t acceptance;
    std::optional<landmark_node_multiview_skeleton_t> skeleton = std::nullopt;
  };

  class FactorGraphCostUpdater::Impl {
  private:
    std::shared_ptr<cyclops_global_config_t const> _config;
    std::shared_ptr<FactorGraphStateNodeMap> _node_map;

    void addImuPreintegrationCost(
      ceres::Problem& problem, node_context_id_set_t const& inputs,
      frame_id_t from, frame_id_t to, IMUPreintegration const* data);
    void addImuRandomWalkCost(
      ceres::Problem& problem, node_context_id_set_t const& inputs,
      frame_id_t from, frame_id_t to, double dt);

    landmark_node_uncertainty_analysis_t analyzeLandmarkUncertainty(
      Vector3d const& f, feature_point_t const& feature, frame_id_t frame_id);
    landmark_node_multiview_skeleton_analysis_t makeLandmarkCostBatchArgument(
      std::set<frame_id_t> const& solvable_motions, landmark_id_t feature_id,
      feature_track_t const& track);

  public:
    Impl(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<FactorGraphStateNodeMap> node_map)
        : _config(config), _node_map(node_map) {
    }

    bool addImuCost(ceres::Problem& problem, imu_motion_t const& imu_motion);
    bool addBiasPriorCost(ceres::Problem& problem, frame_id_t frame_id);

    landmark_acceptance_t addLandmarkCostBatch(
      ceres::Problem& problem, std::set<frame_id_t> const& solvable_motions,
      landmark_id_t landmark_id, feature_track_t const& track);
  };

  template <typename evaluator_t, int... dimensions>
  static void create_autodiff_factor(
    ceres::Problem& problem, FactorGraphStateNodeMap& node_map,
    evaluator_t* evaluator, factor_t factor, node_context_id_set_t const& nodes,
    ceres::LossFunction* loss = nullptr) {
    auto parameters =  //
      nodes | views::transform([](auto const& input) {
        auto const& [_, ctxt_ref] = input;
        return ctxt_ref.get().parameter;
      }) |
      ranges::to_vector;

    auto residual = problem.AddResidualBlock(
      new AutoDiffCostFunction<evaluator_t, dimensions...>(evaluator), loss,
      parameters);
    node_map.createFactor(std::make_tuple(residual, factor), nodes);
  }

  static std::optional<node_context_id_set_t> query_node_contexts(
    FactorGraphStateNodeMap& node_map, std::vector<node_t> const& nodes) {
    node_context_id_set_t result;
    for (auto const& node : nodes) {
      auto maybe_ctxt = node_map.findContext(node);
      if (!maybe_ctxt.has_value()) {
        __logger__->error("Node {} is not found in the graph.", node);
        return std::nullopt;
      }
      result.emplace_back(std::make_pair(node, *maybe_ctxt));
    }
    return result;
  }

  void FactorGraphCostUpdater::Impl::addImuPreintegrationCost(
    ceres::Problem& problem, node_context_id_set_t const& input_nodes,
    frame_id_t from, frame_id_t to, IMUPreintegration const* data) {
    create_autodiff_factor<IMUPreintegrationCostEvaluator, 9, 10, 10, 6>(
      problem, *_node_map,
      new IMUPreintegrationCostEvaluator(data, _config->gravity_norm),
      factor::imu(from, to), input_nodes);
  }

  void FactorGraphCostUpdater::Impl::addImuRandomWalkCost(
    ceres::Problem& problem, node_context_id_set_t const& input_nodes,
    frame_id_t from, frame_id_t to, double dt) {
    create_autodiff_factor<IMUBiasRandomWalkCostEvaluator, 6, 6, 6>(
      problem, *_node_map,
      new IMUBiasRandomWalkCostEvaluator(
        dt, _config->noise.acc_random_walk, _config->noise.gyr_random_walk),
      factor::bias_walk(from, to), input_nodes);
  }

  bool FactorGraphCostUpdater::Impl::addImuCost(
    ceres::Problem& problem, imu_motion_t const& imu_motion) {
    auto const& [from, to, data] = imu_motion;
    if (data->time_delta <= 0) {
      __logger__->error("IMU dt <= 0 (frame: {} -> {})", from, to);
      return false;
    }
    auto maybe_preintegration_inputs = query_node_contexts(
      *_node_map, {node::frame(from), node::frame(to), node::bias(from)});
    if (!maybe_preintegration_inputs)
      return false;

    auto maybe_bias_walk_inputs =
      query_node_contexts(*_node_map, {node::bias(from), node::bias(to)});
    if (!maybe_bias_walk_inputs)
      return false;
    __logger__->trace("Adding IMU cost {} -> {}", from, to);

    addImuPreintegrationCost(
      problem, *maybe_preintegration_inputs, from, to, data.get());
    addImuRandomWalkCost(
      problem, *maybe_bias_walk_inputs, from, to, data->time_delta);
    return true;
  }

  bool FactorGraphCostUpdater::Impl::addBiasPriorCost(
    ceres::Problem& problem, frame_id_t frame_id) {
    auto maybe_inputs = query_node_contexts(*_node_map, {node::bias(frame_id)});
    if (!maybe_inputs)
      return false;
    __logger__->trace("Adding bias prior cost {}", frame_id);

    create_autodiff_factor<IMUBiasPriorCostEvaluator, 6, 6>(
      problem, *_node_map,
      new IMUBiasPriorCostEvaluator(
        _config->noise.acc_bias_prior_stddev,
        _config->noise.gyr_bias_prior_stddev),
      factor::bias_prior(frame_id), *maybe_inputs);
    return true;
  }

  static Matrix2x3d hstack(Matrix2d const& A, Vector2d const& b) {
    Matrix2x3d result;
    result << A, b;
    return result;
  }

  landmark_node_uncertainty_analysis_t
  FactorGraphCostUpdater::Impl::analyzeLandmarkUncertainty(
    Vector3d const& f, feature_point_t const& feature, frame_id_t frame_id) {
    auto frame_node = node::frame(frame_id);
    auto maybe_x_ctxt = _node_map->findContext(frame_node);
    if (!maybe_x_ctxt) {
      __logger__->error(
        "Uninitialized keyfrane node {} while adding landmark observation "
        "cost.",
        frame_id);
      return {ANALYSIS_FAIL_UNINITIALIZED_MOTION_STATE};
    }
    auto x = maybe_x_ctxt->get().parameter;
    auto [p_cam, q_cam] = compose(
      se3_of_motion_frame_ptr(x), _config->extrinsics.imu_camera_transform);
    auto z = (q_cam.conjugate() * (f - p_cam)).eval();

    auto d = z.z();
    if (d < _config->estimation.landmark_acceptance.inlier_min_depth)
      return {ANALYSIS_FAIL_TOO_CLOSE};

    Vector2d u = z.head<2>() / d;
    Vector2d r = u - feature.point;

    Matrix2d const& W = feature.weight;
    auto mnorm = std::sqrt(std::max(0., r.dot(W * r)));
    if (
      mnorm >
      _config->estimation.landmark_acceptance.inlier_mahalanobis_error) {
      return {ANALYSIS_FAIL_LARGE_PROJECTION_ERROR};
    }

    Matrix3d R_cam = q_cam.matrix();
    Matrix2x3d J = hstack(Matrix2d::Identity(), -u) * R_cam.transpose() / d;

    return {
      ANALYSIS_SUCCESS,
      feature_uncertainty_t {
        .input_frame = std::make_pair(frame_node, *maybe_x_ctxt),
        .depth = d,
        .mahalanobis_norm = mnorm,
        .information = J.transpose() * W * J,
      }};
  }

  landmark_node_multiview_skeleton_analysis_t
  FactorGraphCostUpdater::Impl::makeLandmarkCostBatchArgument(
    std::set<frame_id_t> const& solvable_motions, landmark_id_t landmark_id,
    feature_track_t const& track) {
    auto landmark_node = node::landmark(landmark_id);
    auto maybe_f_ctxt = _node_map->findContext(landmark_node);
    if (!maybe_f_ctxt)
      return {UNINITIALIZED {}};

    landmark_node_multiview_skeleton_t result = {
      .landmark_node = std::make_pair(landmark_node, *maybe_f_ctxt),
      .observations = {},
    };
    auto f = Vector3d(maybe_f_ctxt->get().parameter);

    int observations_count = 0;
    int depth_test_failure_count = 0;
    int mnorm_test_failure_count = 0;
    int valid_track_count = 0;
    double depth_sumsquare = 0.;
    Matrix3d total_information = Matrix3d::Zero();

    for (auto const& [frame_id, feature] : track) {
      if (solvable_motions.find(frame_id) == solvable_motions.end())
        continue;

      observations_count++;
      auto analysis = analyzeLandmarkUncertainty(f, feature, frame_id);
      switch (analysis.status) {
      case ANALYSIS_SUCCESS: {
        auto const& [input_frame, d, mnorm, H] = *analysis.value;

        valid_track_count++;
        total_information += H;
        depth_sumsquare += d * d;
        result.observations.emplace_back(landmark_node_observation_skeleton_t {
          .data = feature,
          .factor = factor::feature(frame_id, landmark_id),
          .keyframe_node = input_frame,
        });
        break;
      }
      case ANALYSIS_FAIL_UNINITIALIZED_MOTION_STATE:
        break;

      case ANALYSIS_FAIL_TOO_CLOSE:
        depth_test_failure_count++;
        break;

      case ANALYSIS_FAIL_LARGE_PROJECTION_ERROR:
        mnorm_test_failure_count++;
        break;
      }
    }
    if (valid_track_count == 0) {
      return {NO_INLIER {
        .observation_count = observations_count,
        .depth_threshold_failure_count = depth_test_failure_count,
        .mahalanobis_norm_test_failure_count = mnorm_test_failure_count,
      }};
    }

    auto depth_meansquare = depth_sumsquare / valid_track_count;
    auto min_information_eigenvalue =
      total_information.selfadjointView<Eigen::Lower>().eigenvalues().x();
    auto information_index = min_information_eigenvalue * depth_meansquare;
    if (
      information_index >
      _config->estimation.landmark_acceptance.inlier_min_information_index) {
      return {
        ACCEPTED {
          .observation_count = observations_count,
          .accepted_count = result.observations.size(),
        },
        result,
      };
    }
    return {INFORMATION_DEFICIENT {
      .observation_count = observations_count,
      .information_index = information_index,
    }};
  }

  landmark_acceptance_t FactorGraphCostUpdater::Impl::addLandmarkCostBatch(
    ceres::Problem& problem, std::set<frame_id_t> const& solvable_motions,
    landmark_id_t feature_id, feature_track_t const& track) {
    auto [acceptance, maybe_skeleton] =
      makeLandmarkCostBatchArgument(solvable_motions, feature_id, track);
    if (!maybe_skeleton)  // failed. just return.
      return acceptance;

    auto const& skeleton = *maybe_skeleton;

    for (auto const& [data, factor, frame_node] : skeleton.observations) {
      create_autodiff_factor<LandmarkProjectionCostEvaluator, 2, 10, 3>(
        problem, *_node_map,
        new LandmarkProjectionCostEvaluator(
          data.get(), _config->extrinsics.imu_camera_transform),
        factor, {frame_node, skeleton.landmark_node},
        new ceres::HuberLoss(2.0));
    }
    return acceptance;
  }

  FactorGraphCostUpdater::FactorGraphCostUpdater(
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<FactorGraphStateNodeMap> node_map)
      : _pimpl(std::make_unique<Impl>(config, node_map)) {
  }

  FactorGraphCostUpdater::~FactorGraphCostUpdater() = default;

  bool FactorGraphCostUpdater::addImuCost(
    ceres::Problem& problem, imu_motion_t const& imu_motion) {
    return _pimpl->addImuCost(problem, imu_motion);
  }

  bool FactorGraphCostUpdater::addBiasPriorCost(
    ceres::Problem& problem, frame_id_t frame_id) {
    return _pimpl->addBiasPriorCost(problem, frame_id);
  }

  landmark_acceptance_t FactorGraphCostUpdater::addLandmarkCostBatch(
    ceres::Problem& problem, std::set<frame_id_t> const& solvable_motions,
    landmark_id_t feature_id, feature_track_t const& track) {
    return _pimpl->addLandmarkCostBatch(
      problem, solvable_motions, feature_id, track);
  }
}  // namespace cyclops::estimation
