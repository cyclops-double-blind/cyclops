#include "cyclops/details/estimation/optimizer.hpp"
#include "cyclops/details/estimation/optimizer_guess.hpp"
#include "cyclops/details/estimation/graph/factor.hpp"
#include "cyclops/details/estimation/graph/node.hpp"
#include "cyclops/details/estimation/graph/graph.hpp"
#include "cyclops/details/estimation/graph/graph_cost_helper.hpp"
#include "cyclops/details/estimation/graph/graph_node_map.hpp"
#include "cyclops/details/estimation/state/accessor_write.hpp"

#include "cyclops/details/measurement/data_provider.hpp"
#include "cyclops/details/utils/debug.hpp"
#include "cyclops/details/utils/type.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <range/v3/all.hpp>

namespace cyclops::estimation {
  using std::set;
  using std::vector;

  namespace views = ranges::views;
  namespace actions = ranges::actions;

  class LikelihoodOptimizerImpl: public LikelihoodOptimizer {
  private:
    std::unique_ptr<OptimizerSolutionGuessPredictor> _predictor;

    std::shared_ptr<cyclops_global_config_t const> _config;
    std::shared_ptr<StateVariableWriteAccessor> _state_accessor;
    std::shared_ptr<measurement::MeasurementDataProvider> _data_provider;

    std::optional<frame_id_t> _initial_frame_id;

    bool initialize();

    set<frame_id_t> collectAllMotionFrames(gaussian_prior_t const& prior) const;
    set<landmark_id_t> collectAllLandmarks(gaussian_prior_t const& prior) const;

    set<landmark_id_t> addStateBlocks(
      FactorGraphInstance& graph, set<frame_id_t> const& frames,
      set<landmark_id_t> const& landmarks);

    landmark_sanity_statistics_t addAndStatLandmarkCosts(
      set<frame_id_t> const& motions, FactorGraphInstance& graph);
    optimization_result_t solve(gaussian_prior_t const& prior);

    set<landmark_id_t> collectMappedLandmarks(
      FactorGraphInstance& graph, vector<landmark_id_t> const& landmarks) const;
    void updateOptimizationResult(
      FactorGraphInstance& graph, set<landmark_id_t> const& mapped_landmarks);

  public:
    LikelihoodOptimizerImpl(
      std::unique_ptr<OptimizerSolutionGuessPredictor> predictor,
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<StateVariableWriteAccessor> state_accessor,
      std::shared_ptr<measurement::MeasurementDataProvider> measurement);
    void reset() override;

    std::optional<optimization_result_t> optimize(
      gaussian_prior_t const& prior) override;
  };

  LikelihoodOptimizerImpl::LikelihoodOptimizerImpl(
    std::unique_ptr<OptimizerSolutionGuessPredictor> predictor,
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<StateVariableWriteAccessor> state_accessor,
    std::shared_ptr<measurement::MeasurementDataProvider> data_provider)
      : _predictor(std::move(predictor)),
        _config(config),
        _state_accessor(state_accessor),
        _data_provider(data_provider) {
  }

  void LikelihoodOptimizerImpl::reset() {
    _initial_frame_id = std::nullopt;

    _predictor->reset();
    _state_accessor->reset();
    _data_provider->reset();
  }

  bool LikelihoodOptimizerImpl::initialize() {
    auto maybe_prediction = _predictor->solve();
    if (!maybe_prediction.has_value()) {
      __logger__->info("Initialization failed");
      return false;
    }

    auto const& [new_frames, new_landmarks] = maybe_prediction.value();
    _state_accessor->updateMotionFrameGuess(new_frames);
    _state_accessor->updateLandmarkGuess(new_landmarks);

    return true;
  }

  set<landmark_id_t> LikelihoodOptimizerImpl::addStateBlocks(
    FactorGraphInstance& graph, set<frame_id_t> const& frames,
    set<landmark_id_t> const& landmarks) {
    for (auto const frame_id : frames)
      graph.addFrameStateBlock(frame_id);

    set<landmark_id_t> map_landmark_candidates;
    for (auto const landmark_id : landmarks) {
      if (graph.addLandmarkStateBlock(landmark_id))
        map_landmark_candidates.insert(landmark_id);
    }
    graph.fixGauge(*frames.begin());
    __logger__->debug("Successed to construct state blocks");

    return map_landmark_candidates;
  }

  landmark_sanity_statistics_t LikelihoodOptimizerImpl::addAndStatLandmarkCosts(
    set<frame_id_t> const& motions, FactorGraphInstance& graph) {
    using accept_t = landmark_acceptance_t;

    landmark_sanity_statistics_t result = {
      .landmark_observations = 0,
      .landmark_accepts = 0,
      .uninitialized_landmarks = 0,
      .depth_threshold_failures = 0,
      .mnorm_threshold_failures = 0,
    };

    for (auto const& [feature_id, track] : _data_provider->tracks()) {
      auto acceptance = graph.addLandmarkCost(motions, feature_id, track);
      auto visitor = overloaded {
        [&](accept_t::accepted accept) {
          result.landmark_observations += accept.observation_count;
          result.landmark_accepts += accept.accepted_count;
        },
        [&](accept_t::rejected__uninitialized_landmark_state _) {
          result.uninitialized_landmarks++;
        },
        [&](accept_t::rejected__no_inlier_observation reject) {
          result.landmark_observations += reject.observation_count;
          result.depth_threshold_failures +=
            reject.depth_threshold_failure_count;
          result.mnorm_threshold_failures +=
            reject.mahalanobis_norm_test_failure_count;
        },
        [&](accept_t::rejected__deficient_information_weight reject) {
          result.landmark_observations += reject.observation_count;
        },
      };
      std::visit(visitor, acceptance.variant);
    }
    return result;
  }

  set<frame_id_t> LikelihoodOptimizerImpl::collectAllMotionFrames(
    gaussian_prior_t const& prior) const {
    auto const& imu = _data_provider->imu();
    auto const& tracks = _data_provider->tracks();

    set<frame_id_t> result;
    actions::insert(
      result, imu | views::transform([](auto const& _) { return _.from; }));
    actions::insert(
      result, imu | views::transform([](auto const& _) { return _.to; }));
    for (auto const& track : tracks | views::values)
      actions::insert(result, track | views::keys);

    for (auto const& node : prior.input_nodes) {
      std::visit(
        overloaded {
          [&](node_t::frame_t const& frame) { result.emplace(frame.id); },
          [](auto const&) {},
        },
        node.variant);
    }
    return result;
  }

  set<landmark_id_t> LikelihoodOptimizerImpl::collectAllLandmarks(
    gaussian_prior_t const& prior) const {
    set<landmark_id_t> result;
    actions::insert(result, _data_provider->tracks() | views::keys);
    for (auto const& node : prior.input_nodes) {
      std::visit(
        overloaded {
          [&](node_t::landmark_t const& _) { result.emplace(_.id); },
          [](auto const&) {},
        },
        node.variant);
    }
    return result;
  }

  set<landmark_id_t> LikelihoodOptimizerImpl::collectMappedLandmarks(
    FactorGraphInstance& graph, vector<landmark_id_t> const& candidates) const {
    auto landmark_nodes = candidates |
      views::transform([](auto const& _) { return node::landmark(_); }) |
      ranges::to<set>;
    auto [_, landmark_factors] = graph.queryNeighbors(landmark_nodes);
    auto [jacobian, residuals] = graph.evaluate(
      landmark_nodes | ranges::to_vector,
      landmark_factors | views::values |
        views::transform([](auto const& factor) {
          auto const& [ptr, _] = factor;
          return ptr;
        }) |
        ranges::to_vector);

    EigenCRSMatrix H = jacobian.transpose() * jacobian;
    cyclops_assert(
      "Landmark information matrix size mismatch",
      (H.cols() == 3 * candidates.size()) &&
        (H.rows() == 3 * candidates.size()));

    set<landmark_id_t> result;
    for (int i = 0; i < candidates.size(); i++) {
      Eigen::Matrix3d H_i = H.block(3 * i, 3 * i, 3, 3);
      auto lambda = H_i.selfadjointView<Eigen::Upper>().eigenvalues().x();

      auto const& threshold = _config->estimation.landmark_acceptance;
      auto min_lambda = threshold.mapping_acceptance_min_eigenvalue;

      if (lambda >= min_lambda)
        result.emplace(candidates.at(i));
    }
    return result;
  }

  void LikelihoodOptimizerImpl::updateOptimizationResult(
    FactorGraphInstance& graph, set<landmark_id_t> const& map_landmarks) {
    _data_provider->updateImuBias();

    landmark_positions_t mapping_completed_landmark_positions;
    for (auto landmark_id : map_landmarks) {
      auto maybe_f = _state_accessor->landmark(landmark_id);
      if (!maybe_f)
        continue;
      mapping_completed_landmark_positions.emplace(
        landmark_id, Eigen::Vector3d(maybe_f->get().data()));
    }
    _state_accessor->updateMappedLandmarks(
      mapping_completed_landmark_positions);
  }

  template <typename container_t, typename key_t>
  static bool contains(container_t const& container, key_t const& key) {
    return container.find(key) != container.end();
  }

  LikelihoodOptimizer::optimization_result_t LikelihoodOptimizerImpl::solve(
    gaussian_prior_t const& prior) {
    auto tic = ::cyclops::tic();

    auto [motion_frames, landmarks] = _state_accessor->prune(
      collectAllMotionFrames(prior), collectAllLandmarks(prior));

    if (!_initial_frame_id)
      _initial_frame_id = *motion_frames.begin();

    auto node_map = std::make_shared<FactorGraphStateNodeMap>(_state_accessor);
    auto graph = std::make_unique<FactorGraphInstance>(
      std::make_unique<FactorGraphCostUpdater>(_config, node_map), _config,
      node_map);
    auto active_landmarks = addStateBlocks(*graph, motion_frames, landmarks);

    graph->setPriorCost(prior);
    for (auto const& frame : _data_provider->imu()) {
      if (!contains(motion_frames, frame.from))
        continue;
      if (!contains(motion_frames, frame.to))
        continue;

      graph->addImuCost(frame);
    }

    if (contains(motion_frames, _initial_frame_id.value()))
      graph->addBiasPriorCost(_initial_frame_id.value());
    auto landmark_sanity = addAndStatLandmarkCosts(motion_frames, *graph);

    __logger__->debug("Graph construction time: {}[s]", toc(tic));
    __logger__->debug("Graph stat before optimization:\n{}", graph->report());

    auto summary = graph->solve();
    auto optimizer_sanity = optimizer_sanity_statistics_t {
      .final_cost = summary.final_cost,
      .num_residuals = summary.num_residuals_reduced,
      .num_parameters = summary.num_effective_parameters_reduced,
    };

    auto mapped_landmarks =
      collectMappedLandmarks(*graph, active_landmarks | ranges::to_vector);
    updateOptimizationResult(*graph, mapped_landmarks);

    return {
      .graph = std::move(graph),
      .landmark_sanity_statistics = landmark_sanity,
      .optimizer_sanity_statistics = optimizer_sanity,
      .solve_time = toc(tic),
      .optimizer_time = summary.total_time_in_seconds,
      .optimizer_report = summary.BriefReport(),

      .motion_frames = std::move(motion_frames),
      .active_landmarks = std::move(active_landmarks),
      .mapped_landmarks = std::move(mapped_landmarks),
    };
  }

  std::optional<LikelihoodOptimizer::optimization_result_t>
  LikelihoodOptimizerImpl::optimize(gaussian_prior_t const& prior) {
    if (initialize()) {
      __logger__->debug("Running local optimization.");
      return solve(prior);
    }
    return std::nullopt;
  }

  std::unique_ptr<LikelihoodOptimizer> LikelihoodOptimizer::create(
    std::unique_ptr<OptimizerSolutionGuessPredictor> predictor,
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<StateVariableWriteAccessor> state_accessor,
    std::shared_ptr<measurement::MeasurementDataProvider> measurement) {
    return std::make_unique<LikelihoodOptimizerImpl>(
      std::move(predictor), config, state_accessor, measurement);
  }
}  // namespace cyclops::estimation
