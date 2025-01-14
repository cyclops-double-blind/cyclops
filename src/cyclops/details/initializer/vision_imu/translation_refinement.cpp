#include "cyclops/details/initializer/vision_imu/translation_refinement.hpp"
#include "cyclops/details/initializer/vision_imu/translation_analysis.hpp"
#include "cyclops/details/initializer/vision_imu/translation_cache.hpp"
#include "cyclops/details/initializer/vision_imu/translation_evaluation.hpp"
#include "cyclops/details/utils/debug.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace cyclops::initializer {
  using std::optional;

  struct primal_evaluation_t {
    double cost;
    double gradient;
    double multiplier;
  };

  static optional<primal_evaluation_t> evaluate_primal(
    IMUMatchScaleEvaluationContext const& evaluator, double s) {
    auto maybe_eval = evaluator.evaluate(s);
    if (!maybe_eval) {
      __logger__->debug("IMU match primal evaluation failed");
      return std::nullopt;
    }

    auto const& cost = maybe_eval->cost;
    auto const& mu = maybe_eval->multiplier;
    auto grad = evaluator.evaluateDerivative(*maybe_eval, s);

    return primal_evaluation_t {
      .cost = cost,
      .gradient = grad,
      .multiplier = mu,
    };
  }

  struct bfgs_trust_region_config_t {
    int max_iterations;
    double initial_hessian_approximation;
    double initial_trust_region_radius;

    double stepsize_tolerance;
    double gradient_tolerance;
    double hessian_approximation_safeguard_margin;

    double trust_region_shrink_rate;
    double trust_region_shrinkage_quality_threshold;

    double trust_region_expand_rate;
    double trust_region_expansion_quality_threshold;
    double trust_region_max_radius;

    double trust_region_step_accept_quality_threshold;
  };

  class ViMatchLocalOptimizationSolver {
  private:
    bfgs_trust_region_config_t _config;

    struct solver_state_t {
      double scale;
      double cost;
      double multiplier;
      double gradient;
      double approx_hessian;
    };

    struct solver_step_t {
      solver_state_t state;
      double trust_region_radius;
      double region_quality;
      double stepsize;
    };
    std::vector<solver_step_t> _history;

    std::string printHistory() const {
      std::ostringstream ss;
      ss << "initial local optimization failed." << std::endl;
      ss << "iterations: " << _history.size() << std::endl;
      ss << "history ";
      ss << "(scale, cost, multiplier, gradient, hessian, radius, quality, "
            "stepsize): [";
      if (!_history.empty()) {
        ss << std::endl;
        for (auto const& step : _history) {
          auto const& [state, radius, quality, stepsize] = step;
          auto const& [s, p, mu, g, H] = state;
          ss << "  ";
          ss << s << ", ";
          ss << p << ", ";
          ss << mu << ", ";
          ss << g << ", ";
          ss << H << ", ";
          ss << radius << ", ";
          ss << quality << ", ";
          ss << stepsize << std::endl;
        }
      }
      ss << "]";
      return ss.str();
    }

    struct trust_region_stepsize_t {
      double stepsize;
      bool constraint_hit;
    };

    trust_region_stepsize_t solveTrustRegionSubproblem(
      solver_state_t const& state, double radius) const {
      auto H = state.approx_hessian;
      auto g = state.gradient;

      auto ds_star = -g / H;
      auto abs_ds_star = std::abs(ds_star);
      if (abs_ds_star <= radius) {
        return {
          .stepsize = ds_star,
          .constraint_hit = abs_ds_star == radius,
        };
      }

      return {
        .stepsize = ds_star >= 0 ? radius : -radius,
        .constraint_hit = true,
      };
    }

    double evaluateTrustRegionQuality(
      solver_state_t const& state, double stepsize, double next_cost) const {
      auto [_, curr_cost, __, g, H] = state;
      auto reduction_expect = g * stepsize + 0.5 * H * std::pow(stepsize, 2);
      auto reduction_actual = next_cost - curr_cost;
      return reduction_actual / reduction_expect;
    }

    double updateTrustRegionRadius(
      double curr_radius, double quality, bool constraint_hit) const {
      if (quality < _config.trust_region_shrinkage_quality_threshold) {
        return _config.trust_region_shrink_rate * curr_radius;
      } else if (
        quality > _config.trust_region_expansion_quality_threshold &&
        constraint_hit) {
        return std::min(
          _config.trust_region_expand_rate * curr_radius,
          _config.trust_region_max_radius);
      }
      return curr_radius;
    }

    template <typename evaluator_t>
    optional<solver_step_t> updateStep(
      evaluator_t const& evaluator, solver_state_t const& state,
      trust_region_stepsize_t const& step) const {
      auto const& [stepsize, constraint_hit] = step;

      auto next_scale = state.scale + stepsize;
      auto maybe_eval = evaluator(next_scale);
      if (!maybe_eval) {
        __logger__->debug("IMU match cost evaluation failed.");
        __logger__->debug("History: {}", printHistory());
        return std::nullopt;
      }

      auto [next_cost, next_gradient, next_multiplier] = *maybe_eval;
      auto quality = evaluateTrustRegionQuality(state, stepsize, next_cost);
      auto next_radius = updateTrustRegionRadius(
        _history.back().trust_region_radius, quality, constraint_hit);

      if (quality > _config.trust_region_step_accept_quality_threshold) {
        auto next_hessian = (next_gradient - state.gradient) / stepsize;
        auto next_state = solver_state_t {
          .scale = next_scale,
          .cost = next_cost,
          .multiplier = next_multiplier,
          .gradient = next_gradient,
          .approx_hessian = std::max(
            _config.hessian_approximation_safeguard_margin, next_hessian),
        };
        return solver_step_t {
          .state = next_state,  // accept
          .trust_region_radius = next_radius,
          .region_quality = quality,
          .stepsize = stepsize,
        };
      } else {
        return solver_step_t {
          .state = state,  // reject
          .trust_region_radius = next_radius,
          .region_quality = quality,
          .stepsize = 0.,
        };
      }
    }

    template <typename evaluator_t>
    optional<solver_state_t> makeInitialState(
      evaluator_t const& evaluator, double scale) const {
      auto maybe_eval = evaluator(scale);
      if (!maybe_eval) {
        __logger__->debug("IMU match first evaluation is failed.");
        return std::nullopt;
      }

      auto [cost, gradient, multiplier] = *maybe_eval;
      return solver_state_t {
        .scale = scale,
        .cost = cost,
        .multiplier = multiplier,
        .gradient = gradient,
        .approx_hessian = _config.initial_hessian_approximation,
      };
    }

    imu_match_scale_refinement_t makeSuccess() const {
      auto const& [s, p, mu, _, __] = _history.back().state;

      __logger__->debug("Successed to find IMU match local optima.");
      __logger__->debug("Scale = {}, cost = {}, multiplier = {}", s, p, mu);
      return {s, p};
    }

    imu_match_scale_refinement_t makeFailure(std::string const& reason) const {
      auto const& [s, p, mu, _, __] = _history.back().state;

      __logger__->debug(
        "IMU match scale guess refinement failed. reason: {}.", reason);
      __logger__->debug("Returning at the best effort...");

      __logger__->debug("Scale = {}, cost = {}, multiplier = {}", s, p, mu);
      return {s, p};
    }

  public:
    explicit ViMatchLocalOptimizationSolver(
      bfgs_trust_region_config_t const& config)
        : _config(config) {
    }

    template <typename evaluator_t>
    std::optional<imu_match_scale_refinement_t> solve(
      evaluator_t const& evaluator, double scale_guess) {
      auto initial_state = makeInitialState(evaluator, scale_guess);
      if (!initial_state)
        return std::nullopt;

      _history.emplace_back(solver_step_t {
        .state = *initial_state,
        .trust_region_radius = _config.initial_trust_region_radius,
        .region_quality = 0.,
        .stepsize = 0,
      });

      for (auto _ = 0; _ < _config.max_iterations; _++) {
        auto const& [state, radius, __, ___] = _history.back();
        if (std::abs(state.gradient) < _config.gradient_tolerance) {
          __logger__->debug("Reached gradient tolerance");
          return makeSuccess();
        }
        auto stepsize = solveTrustRegionSubproblem(state, radius);
        if (
          std::abs(stepsize.stepsize) <
          std::abs(_config.stepsize_tolerance * state.scale)) {
          __logger__->debug("Stepsize tolerance reached.");
          return makeSuccess();
        }
        auto maybe_step = updateStep(evaluator, state, stepsize);
        if (!maybe_step)
          return makeFailure("step evaluation failed");

        _history.push_back(*maybe_step);
      }
      __logger__->debug("{}", printHistory());
      return makeFailure("exceeded max iterations");
    }
  };

  IMUMatchTranslationLocalOptimizer::IMUMatchTranslationLocalOptimizer(
    std::shared_ptr<cyclops_global_config_t const> config)
      : _config(config) {
  }

  IMUMatchTranslationLocalOptimizer::~IMUMatchTranslationLocalOptimizer() =
    default;

  void IMUMatchTranslationLocalOptimizer::reset() {
    // does nothing.
  }

  std::optional<imu_match_scale_refinement_t>
  IMUMatchTranslationLocalOptimizer::optimize(
    IMUMatchScaleEvaluationContext const& evaluator, double s0) {
    __logger__->debug(
      "Finding IMU match local optima around the initial guess...");

    auto tic = ::cyclops::tic();

    auto const& local_opt_config = _config->initialization.imu.refinement;
    auto gravity = _config->gravity_norm;

    auto solver = ViMatchLocalOptimizationSolver({
      .max_iterations = local_opt_config.max_iteration,
      .initial_hessian_approximation = 1,
      .initial_trust_region_radius = 0.1,
      .stepsize_tolerance = local_opt_config.stepsize_tolerance,
      .gradient_tolerance = local_opt_config.gradient_tolerance,
      .hessian_approximation_safeguard_margin = 1e-6,
      .trust_region_shrink_rate = 0.25,
      .trust_region_shrinkage_quality_threshold = 0.25,
      .trust_region_expand_rate = 2.,
      .trust_region_expansion_quality_threshold = 0.75,
      .trust_region_max_radius = 10.,
      .trust_region_step_accept_quality_threshold = 1e-4,
    });
    auto result =
      solver.solve([&](auto s) { return evaluate_primal(evaluator, s); }, s0);

    __logger__->debug("Duration: {}", ::cyclops::toc(tic));
    return result;
  }

  std::unique_ptr<IMUMatchTranslationLocalOptimizer>
  IMUMatchTranslationLocalOptimizer::create(
    std::shared_ptr<cyclops_global_config_t const> config) {
    return std::make_unique<IMUMatchTranslationLocalOptimizer>(config);
  }
}  // namespace cyclops::initializer
