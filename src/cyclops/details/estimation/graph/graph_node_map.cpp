#include "cyclops/details/estimation/graph/graph_node_map.hpp"
#include "cyclops/details/estimation/graph/factor.hpp"
#include "cyclops/details/estimation/graph/node.hpp"

#include "cyclops/details/estimation/ceres/manifold.se3.hpp"
#include "cyclops/details/estimation/state/accessor_write.hpp"
#include "cyclops/details/logging.hpp"

#include <ceres/ceres.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

namespace cyclops::estimation {
  namespace views = ranges::views;

  FactorGraphStateNodeMap::FactorGraphStateNodeMap(
    std::shared_ptr<StateVariableWriteAccessor> state)
      : _state(state) {
  }

  FactorGraphStateNodeMap::~FactorGraphStateNodeMap() = default;

  bool FactorGraphStateNodeMap::createFrameNode(
    ceres::Problem& problem, frame_id_t frame_id) {
    auto maybe_x = _state->motionFrame(frame_id);
    if (!maybe_x) {
      __logger__->error(
        "Tried to add uninitialized motion state for frame {}.", frame_id);
      return false;
    }

    auto x = maybe_x->get().data();
    _node_contexts.emplace(
      node::frame(frame_id), graph_node_context_t {x, {}, {}});

    problem.AddParameterBlock(
      x, 10,
      new ceres::AutoDiffLocalParameterization<
        ExponentialSE3Plus<false>, 10, 9>);

    auto b = x + 10;
    _node_contexts.emplace(
      node::bias(frame_id), graph_node_context_t {b, {}, {}});

    problem.AddParameterBlock(b, 6);

    return true;
  }

  bool FactorGraphStateNodeMap::createLandmarkNode(
    ceres::Problem& problem, landmark_id_t landmark_id) {
    auto maybe_f = _state->landmark(landmark_id);
    if (!maybe_f)
      return false;

    auto f = maybe_f->get().data();
    _node_contexts.emplace(
      node::landmark(landmark_id), graph_node_context_t {f, {}, {}});
    problem.AddParameterBlock(f, 3);

    return true;
  }

  maybe_graph_node_context_ref_t FactorGraphStateNodeMap::findContext(
    node_t const& node) {
    auto maybe_context = _node_contexts.find(node);
    if (maybe_context == _node_contexts.end())
      return std::nullopt;

    auto& [_, context] = *maybe_context;
    return context;
  }

  maybe_graph_node_context_cref_t FactorGraphStateNodeMap::findContext(
    node_t const& node) const {
    auto maybe_context = _node_contexts.find(node);
    if (maybe_context == _node_contexts.end())
      return std::nullopt;

    auto const& [_, context] = *maybe_context;
    return context;
  }

  factor_id_t FactorGraphStateNodeMap::createPriorFactor(
    ceres::Problem& problem, factor_ptr_t ptr, node_set_t const& nodes) {
    if (_prior) {
      __logger__->warn("Warning: setting prior twice");
      problem.RemoveResidualBlock(_prior->ptr);
    }
    _last_factor_id++;
    _prior = {_last_factor_id, ptr, nodes};
    return _last_factor_id;
  }

  factor_id_t FactorGraphStateNodeMap::createFactor(
    factor_entry_t factor_entry,
    std::vector<std::pair<node_t, graph_node_context_ref_t>> const& nodes) {
    auto pairs =
      views::cartesian_product(
        views::iota(0, (int)nodes.size()), views::iota(0, (int)nodes.size())) |
      views::filter([](auto ab) {
        auto [a, b] = ab;
        return a != b;
      }) |
      ranges::to_vector;
    for (auto const& [i, j] : pairs) {
      auto const& [n1, ctxt1] = nodes.at(i);
      auto const& [n2, ctxt2] = nodes.at(j);
      ctxt1.get().neighbors.insert(n2);
      ctxt2.get().neighbors.insert(n1);
    }

    _last_factor_id++;
    for (auto const& ctxt : nodes | views::values)
      ctxt.get().factors.emplace(_last_factor_id, factor_entry);
    return _last_factor_id;
  }

  std::optional<node_set_cref_t> FactorGraphStateNodeMap::queryNeighbors(
    node_t const& node) const {
    auto maybe_context = findContext(node);
    if (!maybe_context)
      return std::nullopt;
    return maybe_context->get().neighbors;
  }

  neighbor_query_result_t FactorGraphStateNodeMap::queryNeighbors(
    node_set_t const& nodes) const {
    neighbor_query_result_t result;
    for (auto const& node : nodes) {
      auto maybe_context = findContext(node);
      if (!maybe_context) {
        __logger__->error("Queried non-existing node: {}", node);
        continue;
      }
      auto const& context = maybe_context->get();
      result.nodes.insert(context.neighbors.begin(), context.neighbors.end());
      result.factors.insert(context.factors.begin(), context.factors.end());
    }
    return result;
  }

  std::optional<prior_node_t> const& FactorGraphStateNodeMap::getPrior() const {
    return _prior;
  }

  std::vector<node_t> FactorGraphStateNodeMap::allNodes() const {
    return _node_contexts | views::keys | ranges::to_vector;
  }

  factor_set_t FactorGraphStateNodeMap::allFactors() const {
    factor_set_t result;
    for (auto const& ctxt : _node_contexts | views::values)
      result.insert(ctxt.factors.begin(), ctxt.factors.end());
    return result;
  }
}  // namespace cyclops::estimation
