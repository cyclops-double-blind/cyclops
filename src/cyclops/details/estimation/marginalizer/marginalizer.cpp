#include "cyclops/details/estimation/marginalizer/marginalizer.hpp"
#include "cyclops/details/estimation/marginalizer/marginalizer_helper.hpp"

#include "cyclops/details/estimation/graph/factor.hpp"
#include "cyclops/details/estimation/graph/node.hpp"
#include "cyclops/details/estimation/graph/graph.hpp"
#include "cyclops/details/estimation/graph/graph_node_map.hpp"
#include "cyclops/details/estimation/state/accessor_read.hpp"

#include "cyclops/details/measurement/data_queue.hpp"
#include "cyclops/details/utils/debug.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

#include <set>

namespace cyclops::estimation {
  using std::set;

  namespace views = ranges::views;
  namespace actions = ranges::actions;

  using measurement::MeasurementDataQueue;

  class MarginalizationManagerImpl: public MarginalizationManager {
  private:
    std::shared_ptr<cyclops_global_config_t const> _config;
    std::shared_ptr<StateVariableReadAccessor const> _state;
    std::shared_ptr<MeasurementDataQueue> _data_queue;

    gaussian_prior_t _prior = {};

    gaussian_prior_t computePrior(
      node_set_t drop_nodes, node_set_t keep_nodes, factor_set_t factors,
      FactorGraphInstance& graph);
    set<node_t> collectLostLandmarks(FactorGraphInstance const& graph) const;

    void marginalizeKeyframe(
      FactorGraphInstance& graph_instance, frame_id_t drop_frame,
      frame_id_t new_frame);
    void marginalizePendingFrame(
      FactorGraphInstance& graph_instance, frame_id_t drop_frame,
      frame_id_t next_frame);
    void marginalizePastKeyframes();

  public:
    MarginalizationManagerImpl(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<StateVariableReadAccessor const> state,
      std::shared_ptr<MeasurementDataQueue> data_queue);
    void reset() override;

    void marginalize() override;
    void marginalize(FactorGraphInstance& graph_instance) override;
    gaussian_prior_t const& prior() const override;
  };

  MarginalizationManagerImpl::MarginalizationManagerImpl(
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<StateVariableReadAccessor const> state,
    std::shared_ptr<MeasurementDataQueue> data_queue)
      : _config(std::move(config)),
        _state(std::move(state)),
        _data_queue(std::move(data_queue)) {
  }

  void MarginalizationManagerImpl::reset() {
    _prior = {};
    _data_queue->reset();
  }

  gaussian_prior_t MarginalizationManagerImpl::computePrior(
    node_set_t drop_nodes, node_set_t keep_nodes, factor_set_t factors,
    FactorGraphInstance& graph) {
    auto subgraph = marginalization_subgraph_t {
      .drop_nodes = std::move(drop_nodes),
      .keep_nodes = std::move(keep_nodes),
      .factors = std::move(factors),
    };
    return evaluate_gaussian_prior(graph, *_state, subgraph);
  }

  template <typename value_t>
  static set<value_t> set_difference(
    set<value_t> const& a, set<value_t> const& b) {
    set<value_t> result;
    std::set_difference(
      a.begin(), a.end(), b.begin(), b.end(),
      std::inserter(result, result.end()));
    return result;
  }

  static void drop_landmarks(set<node_t>& drop_nodes, set<node_t>& keep_nodes) {
    for (auto i = keep_nodes.begin(); i != keep_nodes.end();) {
      if (node::is<node_t::landmark_t>(*i)) {
        drop_nodes.insert(*i);
        i = keep_nodes.erase(i);
      } else {
        ++i;
      }
    }
  }

  static set<landmark_id_t> as_landmark_set(node_set_t const& nodes) {
    // clang-format off
    return nodes
      | views::transform([](auto const& _) {
        return std::get<node_t::landmark_t>(_.variant).id;
      })
      | ranges::to<set>;
    // clang-format on
  }

  node_set_t MarginalizationManagerImpl::collectLostLandmarks(
    FactorGraphInstance const& graph) const {
    if (_data_queue->keyframes().empty())
      return {};
    if (_data_queue->pendingFrames().empty())
      return {};

    auto [last_keyframe_id, _] = *_data_queue->keyframes().rbegin();
    auto [pending_frame_id, __] = *_data_queue->pendingFrames().begin();

    auto maybe_kf_neighbors =
      graph.queryNeighbors(node::frame(last_keyframe_id));
    if (!maybe_kf_neighbors)
      return {};
    auto kf_landmarks = maybe_kf_neighbors->get() |
      views::filter(node::is<node_t::landmark_t>) | ranges::to<set>;

    auto maybe_pf_neighbors =
      graph.queryNeighbors(node::frame(pending_frame_id));
    if (!maybe_pf_neighbors)
      return kf_landmarks;
    auto pf_landmarks = maybe_pf_neighbors->get() |
      views::filter(node::is<node_t::landmark_t>) | ranges::to<set>;

    return set_difference(kf_landmarks, pf_landmarks);
  }

  void MarginalizationManagerImpl::marginalizePendingFrame(
    FactorGraphInstance& graph, frame_id_t drop_frame, frame_id_t next_frame) {
    __logger__->debug(
      "Marginalizing pending frame: {} <- {}", drop_frame, next_frame);

    set<node_t> drop_nodes = {node::frame(drop_frame), node::bias(drop_frame)};

    auto lost_landmark_nodes = collectLostLandmarks(graph);
    actions::insert(drop_nodes, lost_landmark_nodes);

    auto [neighbors, factors] = graph.queryNeighbors(drop_nodes);
    auto keep_nodes = set_difference(neighbors, drop_nodes);

    // compute prior weight assuming that all the landmark nodes are also being
    // marginalized. the resulting prior is only connected to frame nodes,
    // enforcing the sparsity structure assumed by the Schur complement trick.
    // even though the information to the landmark nodes contributed from the
    // observation of `drop_frame` is lost, we correctly preserve its
    // contribution to non-dropped frames up-to-linearization.
    drop_landmarks(drop_nodes, keep_nodes);

    auto maybe_previous_prior = graph.prior();
    if (maybe_previous_prior) {
      auto const& [prior_id, prior_ptr, prior_nodes] = *maybe_previous_prior;
      factors.emplace(prior_id, std::make_tuple(prior_ptr, factor::prior()));
      actions::insert(keep_nodes, set_difference(prior_nodes, drop_nodes));
    }

    _prior = computePrior(
      std::move(drop_nodes), std::move(keep_nodes), std::move(factors), graph);

    auto lost_landmarks = as_landmark_set(lost_landmark_nodes);
    _data_queue->marginalizePendingFrame(drop_frame, lost_landmarks);
  }

  void MarginalizationManagerImpl::marginalizeKeyframe(
    FactorGraphInstance& graph, frame_id_t drop_frame, frame_id_t new_frame) {
    __logger__->debug(
      "Marginalizing keyframe: {} <- {}", drop_frame, new_frame);

    set<node_t> drop_nodes = {node::frame(drop_frame), node::bias(drop_frame)};

    auto lost_landmark_nodes = collectLostLandmarks(graph);
    actions::insert(drop_nodes, lost_landmark_nodes);

    auto [neighbors, factors] = graph.queryNeighbors(drop_nodes);
    auto keep_nodes = set_difference(neighbors, drop_nodes);

    // same to the above. we compute prior weight assuming the marginalization
    // of all landmark nodes.
    drop_landmarks(drop_nodes, keep_nodes);

    auto maybe_previous_prior = graph.prior();
    if (maybe_previous_prior) {
      auto const& [prior_id, prior_ptr, prior_nodes] = *maybe_previous_prior;
      factors.emplace(prior_id, std::make_tuple(prior_ptr, factor::prior()));
      actions::insert(keep_nodes, set_difference(prior_nodes, drop_nodes));
    }

    _prior = computePrior(
      std::move(drop_nodes), std::move(keep_nodes), std::move(factors), graph);

    auto lost_landmarks = as_landmark_set(lost_landmark_nodes);
    _data_queue->marginalizeKeyframe(drop_frame, lost_landmarks, new_frame);
  }

  void MarginalizationManagerImpl::marginalize() {
    auto const& window = _config->keyframe_window;
    auto max_keyframes = std::max(0, window.initialization_phase_max_keyframes);

    while (_data_queue->keyframes().size() > max_keyframes) {
      auto [drop_frame, _] = *_data_queue->keyframes().begin();
      _data_queue->marginalize(drop_frame);
    }
  }

  void MarginalizationManagerImpl::marginalizePastKeyframes() {
    auto const& keyframes = _data_queue->keyframes();
    auto const& state_frames = _state->motionFrames();

    if (state_frames.empty()) {
      __logger__->error("Empty motion state while marginalizing past keyframe");
      return;
    }

    auto drop_frames =  //
      _data_queue->keyframes() | views::keys |
      views::filter([&](auto frame_id) {
        auto const& [oldest_state_frame_id, _] = *state_frames.begin();
        return frame_id < oldest_state_frame_id;
      }) |
      ranges::to_vector;

    for (auto drop_frame : drop_frames)
      _data_queue->marginalize(drop_frame);
  }

  void MarginalizationManagerImpl::marginalize(FactorGraphInstance& graph) {
    auto __tic__ = tic();
    marginalizePastKeyframes();

    auto const& keyframes = _data_queue->keyframes();
    auto const& pending_frames = _data_queue->pendingFrames();

    if (pending_frames.size() < 2) {
      __logger__->debug(
        "Number of pending frames ({}) < 2. Skipping marginalization...",
        pending_frames.size());
      return;
    }

    auto n_frames = keyframes.size() + pending_frames.size();
    auto max_frames = _config->keyframe_window.optimization_phase_max_keyframes;

    if (n_frames < max_frames) {
      __logger__->debug(
        "Number of frames ({}) < {}. Skipping marginalization...", n_frames,
        max_frames);
      _data_queue->acceptCurrentPendingKeyframe();
      return;
    }

    auto [pending_frame, _] = *pending_frames.begin();
    auto [next_pending_frame, __] = *std::next(pending_frames.begin());

    if (keyframes.empty()) {
      __logger__->debug("Empty keyframes. Skipping marginalization...");
      _data_queue->acceptCurrentPendingKeyframe();
      return;
    }
    auto [oldest_keyframe, ___] = *keyframes.begin();

    if (_data_queue->detectKeyframe(pending_frame)) {
      marginalizeKeyframe(graph, oldest_keyframe, pending_frame);
    } else {
      marginalizePendingFrame(graph, pending_frame, next_pending_frame);
    }
    __logger__->info("Marginalization total time: {}[s]", toc(__tic__));
  }

  gaussian_prior_t const& MarginalizationManagerImpl::prior() const {
    return _prior;
  }

  std::unique_ptr<MarginalizationManager> MarginalizationManager::create(
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<StateVariableReadAccessor const> state,
    std::shared_ptr<MeasurementDataQueue> data_queue) {
    return std::make_unique<MarginalizationManagerImpl>(
      config, state, data_queue);
  }
}  // namespace cyclops::estimation
