#include "cyclops/details/initializer/vision/bundle_adjustment.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment_factors.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment_states.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/estimation/ceres/manifold.se3.hpp"

#include "cyclops/details/utils/type.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

#include <map>
#include <memory>
#include <vector>
#include <optional>

namespace cyclops::initializer {
  using ceres::AutoDiffCostFunction;
  using ceres::AutoDiffLocalParameterization;

  using Eigen::MatrixXd;
  using Eigen::Quaterniond;
  using Eigen::Vector2d;
  using Eigen::Vector3d;

  namespace views = ranges::views;

  using multiview_image_frame_t = std::map<landmark_id_t, feature_point_t>;
  using multiview_image_data_t = std::map<frame_id_t, multiview_image_frame_t>;

  class BundleAdjustmentConstructor {
  private:
    config::initializer::vision_solver_config_t const& _config;
    std::shared_ptr<BundleAdjustmentOptimizationState> _state;

    ceres::Problem _problem;

    std::map<landmark_id_t, double*> _landmark_parameters;
    std::map<frame_id_t, double*> _frame_parameters;
    std::vector<ceres::ResidualBlockId> _residuals;

    bool constructCameraMotionStates();
    bool constructLandmarkFactors(multiview_image_data_t const& data);
    bool constructVirtualScaleGaugeFactor();

    struct solution_evaluation_t {
      std::vector<double> residuals;
      EigenCRSMatrix jacobian;
    };
    solution_evaluation_t evaluateFinalSolution();
    MatrixXd getCameraMotionFisherInformation(EigenCRSMatrix const& jacobian);

    int countOutliers(
      int n_measurements, std::vector<double> const& residuals) const;

  public:
    BundleAdjustmentConstructor(
      config::initializer::vision_solver_config_t const& config,
      std::shared_ptr<BundleAdjustmentOptimizationState> state);

    bool construct(multiview_image_data_t const& data);

    vision_bootstrap_solution_t solve();
  };

  bool BundleAdjustmentConstructor::constructCameraMotionStates() {
    for (auto& [frame_id, x] : _state->camera_motions) {
      auto parameterization = new AutoDiffLocalParameterization<
        estimation::ExponentialSE3Plus<false, false>, 7, 6>;
      _problem.AddParameterBlock(x.data(), 7, parameterization);
      _frame_parameters.emplace(frame_id, x.data());
    }
    return true;
  }

  bool BundleAdjustmentConstructor::constructLandmarkFactors(
    multiview_image_data_t const& data) {
    for (auto const& [frame_id, image_frame] : data) {
      auto maybe_x = _frame_parameters.find(frame_id);
      if (maybe_x == _frame_parameters.end()) {
        __logger__->error(
          "Uninitialized motion frame in vision-only bundle adjustment",
          frame_id);
        return false;
      }
      auto& [_, x] = *maybe_x;

      for (auto const& [landmark_id, feature] : image_frame) {
        auto maybe_f = _state->landmark_positions.find(landmark_id);
        if (maybe_f == _state->landmark_positions.end())
          continue;
        auto& [_, f] = *maybe_f;

        auto cost = new LandmarkProjectionCost(feature);
        auto loss =
          new ceres::HuberLoss(_config.bundle_adjustment_robust_kernel_radius);

        _residuals.emplace_back(
          _problem.AddResidualBlock(cost, loss, x, f.data()));
        _landmark_parameters.emplace(landmark_id, f.data());
      }
    }
    return true;
  }

  bool BundleAdjustmentConstructor::constructVirtualScaleGaugeFactor() {
    auto maybe_normalized_frame_pair = _state->normalize();
    if (!maybe_normalized_frame_pair)
      return false;
    auto& [x0, xn] = *maybe_normalized_frame_pair;

    auto x0_ptr = x0.get().data();
    auto xn_ptr = xn.get().data();

    auto stddev = _config.multiview.scale_gauge_soft_constraint_deviation;
    auto factor = new AutoDiffCostFunction<
      BundleAdjustmentScaleConstraintVirtualCost, 1, 7, 7>(
      new BundleAdjustmentScaleConstraintVirtualCost(1 / stddev));
    _residuals.emplace_back(
      _problem.AddResidualBlock(factor, nullptr, x0_ptr, xn_ptr));
    _problem.SetParameterBlockConstant(x0_ptr);

    return true;
  }

  BundleAdjustmentConstructor::solution_evaluation_t
  BundleAdjustmentConstructor::evaluateFinalSolution() {
    ceres::Problem::EvaluateOptions opt;

    auto fs = _landmark_parameters | views::values;
    auto xs = _frame_parameters | views::values;
    opt.parameter_blocks = views::concat(fs, xs) | ranges::to_vector;
    opt.residual_blocks = _residuals;

    std::vector<double> residuals;
    ceres::CRSMatrix jacobian;
    _problem.Evaluate(opt, nullptr, &residuals, nullptr, &jacobian);

    return solution_evaluation_t {
      .residuals = std::move(residuals),
      .jacobian = Eigen::Map<EigenCRSMatrix>(
        jacobian.num_rows, jacobian.num_cols, jacobian.values.size(),
        jacobian.rows.data(), jacobian.cols.data(), jacobian.values.data()),
    };
  }

  MatrixXd BundleAdjustmentConstructor::getCameraMotionFisherInformation(
    EigenCRSMatrix const& jacobian) {
    auto m = _landmark_parameters.size() * 3;
    auto k = _frame_parameters.size() * 6;

    EigenCRSMatrix J_m = jacobian.middleCols(0, m);
    EigenCRSMatrix J_k = jacobian.middleCols(m, k);
    MatrixXd const H_kk = J_k.transpose() * J_k;

    EigenCCSMatrix const H_mm = J_m.transpose() * J_m;
    MatrixXd const H_km = J_k.transpose() * J_m;
    MatrixXd const H_mk = H_km.transpose();

    Eigen::SimplicialLDLT<EigenCCSMatrix> H_mm__inv(H_mm);
    MatrixXd H_km__H_mm__inv__H_mk = H_km * H_mm__inv.solve(H_mk);
    return H_kk - H_km__H_mm__inv__H_mk;
  }

  bool BundleAdjustmentConstructor::construct(
    multiview_image_data_t const& data) {
    if (!constructCameraMotionStates()) {
      __logger__->error("BA camera motion state construction failed.");
      return false;
    }
    if (!constructLandmarkFactors(data)) {
      __logger__->error("BA landmark factor construction failed.");
      return false;
    }
    if (!constructVirtualScaleGaugeFactor()) {
      __logger__->error("BA virtual scale gauge factor construction failed.");
      return false;
    }
    return true;
  }

  int BundleAdjustmentConstructor::countOutliers(
    int n_measurements, std::vector<double> const& residuals) const {
    auto rho = _config.bundle_adjustment_robust_kernel_radius;
    auto rho_square = rho * rho;

    int n_outliers = 0;
    for (auto i = 0; i < n_measurements; i++) {
      auto r1 = residuals.at(2 * i);
      auto r2 = residuals.at(2 * i + 1);

      auto s = r1 * r1 + r2 * r2;
      if (s >= rho_square)
        n_outliers++;
    }
    return n_outliers;
  }

  vision_bootstrap_solution_t BundleAdjustmentConstructor::solve() {
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds =
      _config.multiview.bundle_adjustment_max_solver_time;
    options.max_num_iterations =
      _config.multiview.bundle_adjustment_max_iterations;
    ceres::Solve(options, &_problem, &summary);
    __logger__->info("Finished bundle adjustment: {}", summary.BriefReport());

    auto evaluation = evaluateFinalSolution();

    auto n_residuals = static_cast<int>(summary.num_residuals);
    auto n_parameters = static_cast<int>(summary.num_effective_parameters);
    auto degrees_of_freedom = n_residuals - n_parameters;
    auto failure_probability =
      chi_squared_cdf(degrees_of_freedom, summary.final_cost);
    auto success_probability = 1.0 - failure_probability;

    auto n_measurements = (n_residuals - 1) / 2;
    auto n_outliers = countOutliers(n_measurements, evaluation.residuals);
    auto outlier_ratio = static_cast<double>(n_outliers) / n_measurements;
    auto inlier_ratio = 1.0 - outlier_ratio;

    auto accept_min_success_probability =
      _config.acceptance_test.min_significant_probability;
    auto accept_min_inlier_ratio = _config.acceptance_test.min_inlier_ratio;

    auto acceptable = success_probability >= accept_min_success_probability &&
      inlier_ratio >= accept_min_inlier_ratio;

    return {
      .acceptable = acceptable,
      .solution_significant_probability = success_probability,
      .measurement_inlier_ratio = inlier_ratio,

      .geometry = _state->as_multi_view_geometry(),
      .motion_information_weight =
        getCameraMotionFisherInformation(evaluation.jacobian),
    };
  }

  BundleAdjustmentConstructor::BundleAdjustmentConstructor(
    config::initializer::vision_solver_config_t const& config,
    std::shared_ptr<BundleAdjustmentOptimizationState> state)
      : _config(config), _state(state) {
  }

  std::optional<vision_bootstrap_solution_t> solve_bundle_adjustment(
    config::initializer::vision_solver_config_t const& config,
    multiview_geometry_t const& guess, multiview_image_data_t const& data) {
    auto state = std::make_shared<BundleAdjustmentOptimizationState>(guess);
    auto context = BundleAdjustmentConstructor(config, state);

    if (!context.construct(data))
      return std::nullopt;
    return context.solve();
  }
}  // namespace cyclops::initializer
