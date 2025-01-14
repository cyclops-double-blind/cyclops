#pragma once

#include <Eigen/Dense>

#include <functional>
#include <map>
#include <set>

namespace ceres {
  struct Problem;

  namespace internal {
    struct ResidualBlock;
  }
}  // namespace ceres

namespace cyclops::estimation {
  struct node_t;
  using node_set_t = std::set<node_t>;
  using node_set_cref_t = std::reference_wrapper<node_set_t const>;

  struct factor_t;

  using factor_id_t = uint64_t;
  using factor_ptr_t = ceres::internal::ResidualBlock*;
  using factor_entry_t = std::tuple<factor_ptr_t, factor_t>;

  using factor_set_t = std::map<factor_id_t, factor_entry_t>;

  using parameter_t = double*;

  struct gaussian_prior_t {
    Eigen::MatrixXd jacobian;
    Eigen::VectorXd residual;

    node_set_t input_nodes;
    std::vector<double> nominal_parameters;
  };
}  // namespace cyclops::estimation
