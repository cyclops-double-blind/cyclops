#include "cyclops/details/estimation/graph/graph.hpp"

#include "cyclops/details/estimation/graph/factor.hpp"
#include "cyclops/details/estimation/graph/node.hpp"

#include "cyclops/details/estimation/graph/graph_node_map.hpp"
#include "cyclops/details/estimation/graph/graph_cost_helper.hpp"
#include "cyclops/details/estimation/ceres/cost.gaussian_prior.hpp"
#include "cyclops/details/estimation/ceres/manifold.se3.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

#include <iomanip>

namespace cyclops::estimation {
  using std::optional;
  using std::set;
  using std::string;
  using std::vector;

  using ceres::AutoDiffLocalParameterization;

  using measurement::feature_track_t;
  using measurement::imu_motion_t;

  using Eigen::VectorXd;

  namespace views = ranges::views;

  class FactorGraphInstance::Impl {
  private:
    std::unique_ptr<FactorGraphCostUpdater> _cost_helper;

    std::shared_ptr<cyclops_global_config_t const> _config;
    std::shared_ptr<FactorGraphStateNodeMap> _node_map;

    ceres::Problem _problem;
    optional<frame_id_t> _gauge_constraint_frame;

    void setGaugeConstraint(frame_id_t frame_id, bool set_or_not);
    void evaluate(
      vector<node_t> const& nodes, vector<factor_ptr_t> const& factors,
      vector<double>* residual, ceres::CRSMatrix* jacobian);

  public:
    Impl(
      std::unique_ptr<FactorGraphCostUpdater> cost_helper,
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<FactorGraphStateNodeMap> node_map)
        : _cost_helper(std::move(cost_helper)),
          _config(config),
          _node_map(node_map) {
    }

    void fixGauge(frame_id_t frame_id);
    bool addFrameStateBlock(frame_id_t frame_id);
    bool addLandmarkStateBlock(landmark_id_t landmark_id);

    void addImuCost(imu_motion_t const& imu_motion);
    void addBiasPriorCost(frame_id_t frame_id);

    landmark_acceptance_t addLandmarkCost(
      set<frame_id_t> const& solvable_motions, landmark_id_t feature_id,
      feature_track_t const& track);

    void setPriorCost(gaussian_prior_t const& priors);

    optional<node_set_cref_t> queryNeighbors(node_t const& node) const;
    neighbor_query_result_t queryNeighbors(node_set_t const& nodes) const;

    optional<prior_node_t> const& prior() const;

    string report();
    ceres::Solver::Summary solve();
    std::tuple<EigenCRSMatrix, VectorXd> evaluate(
      vector<node_t> const& nodes, vector<factor_ptr_t> const& factors);
  };

  static ceres::LocalParameterization* make_frame_manifold(bool gauge_fix) {
    if (gauge_fix) {
      return new AutoDiffLocalParameterization<ExponentialSE3Plus<true>, 10, 9>;
    } else {
      return new AutoDiffLocalParameterization<
        ExponentialSE3Plus<false>, 10, 9>;
    }
  }

  void FactorGraphInstance::Impl::setGaugeConstraint(
    frame_id_t frame_id, bool set_or_not) {
    auto maybe_context = _node_map->findContext(node::frame(frame_id));
    if (!maybe_context) {
      __logger__->error(
        "Tried to {} gauge constraint for frame {}, but the parameter is not "
        "found.",
        set_or_not ? "set" : "unset", frame_id);
      return;
    }
    auto const& context = maybe_context->get();
    _problem.SetParameterization(
      context.parameter, make_frame_manifold(set_or_not));
  }

  void FactorGraphInstance::Impl::fixGauge(frame_id_t frame_id) {
    if (_gauge_constraint_frame)
      setGaugeConstraint(*_gauge_constraint_frame, false);

    setGaugeConstraint(frame_id, true);
    _gauge_constraint_frame = frame_id;
  }

  bool FactorGraphInstance::Impl::addFrameStateBlock(frame_id_t frame_id) {
    return _node_map->createFrameNode(_problem, frame_id);
  }

  bool FactorGraphInstance::Impl::addLandmarkStateBlock(
    landmark_id_t landmark_id) {
    return _node_map->createLandmarkNode(_problem, landmark_id);
  }

  void FactorGraphInstance::Impl::addImuCost(imu_motion_t const& imu_motion) {
    _cost_helper->addImuCost(_problem, imu_motion);
  }

  landmark_acceptance_t FactorGraphInstance::Impl::addLandmarkCost(
    set<frame_id_t> const& solvable_motions, landmark_id_t landmark_id,
    feature_track_t const& track) {
    return _cost_helper->addLandmarkCostBatch(
      _problem, solvable_motions, landmark_id, track);
  }

  void FactorGraphInstance::Impl::addBiasPriorCost(frame_id_t frame_id) {
    _cost_helper->addBiasPriorCost(_problem, frame_id);
  }

  void FactorGraphInstance::Impl::setPriorCost(gaussian_prior_t const& prior) {
    vector<std::reference_wrapper<graph_node_context_t>> node_contexts;
    for (auto const& node : prior.input_nodes) {
      auto maybe_ctxt = _node_map->findContext(node);
      if (!maybe_ctxt) {
        __logger__->error(
          "Adding prior cost, but node {} is not initialized", node);
        return;
      }
      node_contexts.push_back(*maybe_ctxt);
    }
    auto parameters = node_contexts |
      views::transform([](auto const& _) { return _.get().parameter; }) |
      ranges::to_vector;

    auto factor = new GaussianPriorCost(prior);
    _node_map->createPriorFactor(
      _problem, _problem.AddResidualBlock(factor, nullptr, parameters),
      node_set_t(prior.input_nodes.begin(), prior.input_nodes.end()));
  }

  static double norm(vector<double> const& v) {
    return (Eigen::Map<VectorXd const>(v.data(), v.size())).norm();
  }

  template <int residual_dimension>
  static void print_factor_stats(
    std::ostringstream& ss, vector<double> const& residual,
    std::vector<int> const& start_indices) {
    using vector_t = Eigen::Matrix<double, residual_dimension, 1>;

    double max_cost = 0;
    double mean_cost = 0;
    vector_t average_residual_squared = vector_t::Zero();

    for (auto i : start_indices) {
      auto r = Eigen::Map<vector_t const>(residual.data() + i);

      auto cost = r.dot(r);
      if (cost > max_cost)
        max_cost = cost;
      mean_cost += cost;
      average_residual_squared += r.array().square().matrix();
    }
    if (start_indices.size() != 0) {
      mean_cost /= start_indices.size();
      average_residual_squared /= start_indices.size();
    }

    ss << "count = " << start_indices.size() << "; ";
    ss << "max = " << max_cost << "; ";
    ss << "average = " << mean_cost << std::endl;
    ss << "average squared residuals = " << average_residual_squared.transpose()
       << std::endl;
  }

  string FactorGraphInstance::Impl::report() {
    std::ostringstream ss;
    auto const& maybe_prior = _node_map->getPrior();
    if (maybe_prior) {
      vector<double> residual;
      evaluate(
        maybe_prior->input_nodes | ranges::to_vector, {maybe_prior->ptr},
        &residual, nullptr);
      ss << "Prior: " << std::setprecision(16) << norm(residual) << std::endl;
    }

    auto nodes = _node_map->allNodes();
    auto factors = _node_map->allFactors();
    vector<double> residual;
    evaluate(
      nodes,
      factors | views::values | views::transform([](auto const& entry) {
        auto const& [ptr, skeleton] = entry;
        return ptr;
      }) |
        ranges::to_vector,
      &residual, nullptr);

    std::vector<int> imu_indices;
    std::vector<int> bias_walk_indices;
    std::vector<int> landmark_indices;
    int i = 0;
    for (auto const& [ptr, skeleton] : factors | views::values) {
      std::visit(
        overloaded {
          [&](factor_t::imu_t const& _) {
            imu_indices.emplace_back(i);
            i += 9;
          },
          [&](factor_t::bias_walk_t const& _) {
            bias_walk_indices.emplace_back(i);
            i += 6;
          },
          [&](factor_t::bias_prior_t const& _) { i += 6; },
          [&](factor_t::feature_t const& _) {
            landmark_indices.emplace_back(i);
            i += 2;
          },
          [](factor_t::prior_t const& _) { throw _; },
        },
        skeleton.variant);
    }

    ss << "IMU: ";
    print_factor_stats<9>(ss, residual, imu_indices);
    ss << "Bias walk: ";
    print_factor_stats<6>(ss, residual, bias_walk_indices);
    ss << "Features: ";
    print_factor_stats<2>(ss, residual, landmark_indices);
    return ss.str();
  }

  optional<prior_node_t> const& FactorGraphInstance::Impl::prior() const {
    return _node_map->getPrior();
  }

  ceres::Solver::Summary FactorGraphInstance::Impl::solve() {
    ceres::Solver::Options options;
    options.dense_linear_algebra_library_type = ceres::EIGEN;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations =
      _config->estimation.optimizer.max_num_iterations;
    options.max_solver_time_in_seconds =
      _config->estimation.optimizer.max_solver_time_in_seconds;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &_problem, &summary);
    return summary;
  }

  void FactorGraphInstance::Impl::evaluate(
    vector<node_t> const& nodes, vector<factor_ptr_t> const& factors,
    vector<double>* residual, ceres::CRSMatrix* jacobian) {
    ceres::Problem::EvaluateOptions opt;

    opt.parameter_blocks.reserve(nodes.size());
    for (auto const& node : nodes) {
      auto maybe_context = _node_map->findContext(node);
      if (!maybe_context) {
        __logger__->error(
          "Could not find the node parameter {} while evaluating Jacobian",
          node);
        continue;
      }
      auto const& context = maybe_context->get();
      opt.parameter_blocks.emplace_back(context.parameter);
    }
    opt.residual_blocks = factors;

    // Unset gauge constraint before computing the Jacobian.
    if (_gauge_constraint_frame)
      setGaugeConstraint(*_gauge_constraint_frame, false);

    _problem.Evaluate(opt, nullptr, residual, nullptr, jacobian);

    // Restore the gauge constraint.
    if (_gauge_constraint_frame)
      setGaugeConstraint(*_gauge_constraint_frame, true);
  }

  std::tuple<EigenCRSMatrix, VectorXd> FactorGraphInstance::Impl::evaluate(
    vector<node_t> const& nodes, vector<factor_ptr_t> const& factors) {
    vector<double> residual_;
    ceres::CRSMatrix jacobian_;
    evaluate(nodes, factors, &residual_, &jacobian_);

    EigenCRSMatrix jacobian = Eigen::Map<EigenCRSMatrix>(
      jacobian_.num_rows, jacobian_.num_cols, jacobian_.values.size(),
      jacobian_.rows.data(), jacobian_.cols.data(), jacobian_.values.data());
    VectorXd residual =
      Eigen::Map<VectorXd>(residual_.data(), residual_.size());

    return std::make_tuple(jacobian, residual);
  }

  optional<node_set_cref_t> FactorGraphInstance::Impl::queryNeighbors(
    node_t const& node) const {
    return _node_map->queryNeighbors(node);
  }

  neighbor_query_result_t FactorGraphInstance::Impl::queryNeighbors(
    node_set_t const& nodes) const {
    return _node_map->queryNeighbors(nodes);
  }

  FactorGraphInstance::FactorGraphInstance(
    std::unique_ptr<FactorGraphCostUpdater> cost_helper,
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<FactorGraphStateNodeMap> node_map) {
    _impl = std::make_unique<Impl>(std::move(cost_helper), config, node_map);
  }

  FactorGraphInstance::~FactorGraphInstance() = default;

  void FactorGraphInstance::fixGauge(frame_id_t _) {
    return _impl->fixGauge(_);
  }

  bool FactorGraphInstance::addFrameStateBlock(frame_id_t _) {
    return _impl->addFrameStateBlock(_);
  }

  bool FactorGraphInstance::addLandmarkStateBlock(landmark_id_t _) {
    return _impl->addLandmarkStateBlock(_);
  }

  void FactorGraphInstance::addImuCost(imu_motion_t const& _) {
    return _impl->addImuCost(_);
  }

  landmark_acceptance_t FactorGraphInstance::addLandmarkCost(
    set<frame_id_t> const& solvable_motions, landmark_id_t feature_id,
    feature_track_t const& track) {
    return _impl->addLandmarkCost(solvable_motions, feature_id, track);
  }

  void FactorGraphInstance::addBiasPriorCost(frame_id_t frame_id) {
    return _impl->addBiasPriorCost(frame_id);
  }

  void FactorGraphInstance::setPriorCost(gaussian_prior_t const& _) {
    return _impl->setPriorCost(_);
  }

  optional<node_set_cref_t> FactorGraphInstance::queryNeighbors(
    node_t const& _) const {
    return _impl->queryNeighbors(_);
  }

  neighbor_query_result_t FactorGraphInstance::queryNeighbors(
    node_set_t const& _) const {
    return _impl->queryNeighbors(_);
  }

  optional<prior_node_t> const& FactorGraphInstance::prior() const {
    return _impl->prior();
  }

  string FactorGraphInstance::report() const {
    return _impl->report();
  }

  ceres::Solver::Summary FactorGraphInstance::solve() {
    return _impl->solve();
  }

  std::tuple<EigenCRSMatrix, VectorXd> FactorGraphInstance::evaluate(
    vector<node_t> const& nodes, vector<factor_ptr_t> const& factors) {
    return _impl->evaluate(nodes, factors);
  }
}  // namespace cyclops::estimation
