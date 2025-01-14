#pragma once

#include "cyclops/details/estimation/type.hpp"

#include <Eigen/Dense>

#include <vector>
#include <set>

namespace cyclops::estimation {
  struct gaussian_prior_t;

  struct FactorGraphInstance;
  struct StateVariableReadAccessor;

  struct marginalization_subgraph_t {
    node_set_t drop_nodes;
    node_set_t keep_nodes;
    factor_set_t factors;
  };

  gaussian_prior_t evaluate_gaussian_prior(
    FactorGraphInstance& graph, StateVariableReadAccessor const& state_accessor,
    marginalization_subgraph_t const& drop_subgraph);
}  // namespace cyclops::estimation
