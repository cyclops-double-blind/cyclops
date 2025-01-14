#pragma once

#include "cyclops/details/estimation/type.hpp"
#include "cyclops/details/type.hpp"

#include <map>
#include <memory>
#include <optional>

namespace cyclops::estimation {
  struct StateVariableWriteAccessor;

  struct graph_node_context_t {
    parameter_t parameter;
    node_set_t neighbors;
    factor_set_t factors;
  };

  using graph_node_context_ref_t = std::reference_wrapper<graph_node_context_t>;
  using graph_node_context_cref_t =
    std::reference_wrapper<graph_node_context_t const>;

  using maybe_graph_node_context_ref_t =
    std::optional<graph_node_context_ref_t>;
  using maybe_graph_node_context_cref_t =
    std::optional<graph_node_context_cref_t>;

  struct prior_node_t {
    factor_id_t id;
    factor_ptr_t ptr;
    node_set_t input_nodes;
  };

  struct neighbor_query_result_t {
    // neighbor nodes
    node_set_t nodes;

    // factors relating queried node set and their neighbors
    factor_set_t factors;
  };

  class FactorGraphStateNodeMap {
  private:
    std::shared_ptr<StateVariableWriteAccessor> _state;

    factor_id_t _last_factor_id = 0;
    std::map<node_t, graph_node_context_t> _node_contexts;
    std::optional<prior_node_t> _prior;

  public:
    explicit FactorGraphStateNodeMap(
      std::shared_ptr<StateVariableWriteAccessor> state);
    ~FactorGraphStateNodeMap();

    bool createFrameNode(ceres::Problem& problem, frame_id_t frame_id);
    bool createLandmarkNode(ceres::Problem& problem, landmark_id_t landmark_id);

    factor_id_t createPriorFactor(
      ceres::Problem& problem, factor_ptr_t ptr, node_set_t const& nodes);
    factor_id_t createFactor(
      factor_entry_t factor_entry,
      std::vector<std::pair<node_t, graph_node_context_ref_t>> const& nodes);

    maybe_graph_node_context_ref_t findContext(node_t const& node);
    maybe_graph_node_context_cref_t findContext(node_t const& node) const;

    std::optional<node_set_cref_t> queryNeighbors(node_t const& node) const;
    neighbor_query_result_t queryNeighbors(node_set_t const& nodes) const;

    std::optional<prior_node_t> const& getPrior() const;

    std::vector<node_t> allNodes() const;
    factor_set_t allFactors() const;
  };
}  // namespace cyclops::estimation
