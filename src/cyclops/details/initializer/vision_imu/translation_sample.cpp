#include "cyclops/details/initializer/vision_imu/translation_sample.hpp"

#include "cyclops/details/initializer/vision_imu/translation_analysis.hpp"
#include "cyclops/details/initializer/vision_imu/translation_cache.hpp"
#include "cyclops/details/initializer/vision_imu/translation_refinement.hpp"
#include "cyclops/details/initializer/vision_imu/translation_evaluation.hpp"
#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  using std::tuple;
  using std::vector;

  namespace views = ranges::views;

  class IMUMatchScaleSampleSolverContext {
  private:
    IMUMatchTranslationLocalOptimizer& local_optimizer;
    imu_match_translation_analysis_t const& analysis;

    IMUMatchScaleEvaluationContext evaluator;

    cyclops_global_config_t const& config;
    telemetry::InitializerTelemetry& telemetry;

    std::optional<double> evaluateCost(double s);
    std::optional<imu_match_scale_sample_solution_t> evaluateSample(double s);

    std::optional<vector<tuple<double, double>>> refineSolutionCandidates(
      vector<tuple<double, double>> const& candidates);
    std::optional<vector<tuple<double, double>>> sample();

    int degreesOfFreedom() const;

  public:
    IMUMatchScaleSampleSolverContext(
      IMUMatchTranslationLocalOptimizer& local_optimizer,
      imu_match_translation_analysis_t const& analysis,
      IMUMatchTranslationAnalysisCache const& cache,
      cyclops_global_config_t const& config,
      telemetry::InitializerTelemetry& telemetry);

    std::optional<vector<imu_match_scale_sample_solution_t>> solve(
      std::set<frame_id_t> const& motion_frames);
  };

  IMUMatchScaleSampleSolverContext::IMUMatchScaleSampleSolverContext(
    IMUMatchTranslationLocalOptimizer& local_optimizer,
    imu_match_translation_analysis_t const& analysis,
    IMUMatchTranslationAnalysisCache const& cache,
    cyclops_global_config_t const& config,
    telemetry::InitializerTelemetry& telemetry)
      : local_optimizer(local_optimizer),
        analysis(analysis),
        evaluator(config.gravity_norm, analysis, cache),
        config(config),
        telemetry(telemetry) {
  }

  using Context = IMUMatchScaleSampleSolverContext;

  int Context::degreesOfFreedom() const {
    return analysis.residual_dimension - analysis.parameter_dimension;
  }

  std::optional<double> Context::evaluateCost(double s) {
    auto maybe_primal = evaluator.evaluate(s);
    if (!maybe_primal)
      return std::nullopt;
    return maybe_primal->cost;
  }

  std::optional<imu_match_scale_sample_solution_t> Context::evaluateSample(
    double s) {
    auto maybe_primal = evaluator.evaluate(s);
    if (!maybe_primal)
      return std::nullopt;

    return imu_match_scale_sample_solution_t {
      .scale = s,
      .cost = maybe_primal->cost,
      .inertial_state = maybe_primal->inertial_solution,
      .visual_state = maybe_primal->visual_solution,
      .hessian = evaluator.evaluateHessian(*maybe_primal, s),
    };
  }

  std::optional<vector<tuple<double, double>>>
  Context::refineSolutionCandidates(
    vector<tuple<double, double>> const& candidates) {
    auto maybe_refined_candidates =
      candidates | views::transform([&](auto const& point) {
        auto const& [s0, _] = point;
        return local_optimizer.optimize(evaluator, s0);
      }) |
      ranges::to_vector;

    auto refined_candidates = maybe_refined_candidates |
      views::filter([](auto const& _) { return _.has_value(); }) |
      views::transform([](auto const& _) { return _.value(); }) |
      ranges::to_vector;

    if (refined_candidates.size() != candidates.size()) {
      __logger__->warn("IMU match local minima refinement failed");
      return std::nullopt;
    }

    auto duplicate_groups =  //
      refined_candidates |
      views::group_by([](auto const& cand1, auto const& cand2) {
        auto const& [s1, p1] = cand1;
        auto const& [s2, p2] = cand2;
        return std::abs((s1 - s2) / s1) < 1e-4;
      }) |
      views::filter([](auto const& _) { return ranges::distance(_) == 1; }) |
      ranges::to_vector;

    return  //
      duplicate_groups | views::transform([](auto const& _) {
        auto [s, p] = *_.begin();
        return std::make_tuple(s, p);
      }) |
      ranges::to<vector<tuple<double, double>>>;
  }

  static auto linspace(double a, double b, int n) {
    auto slice = [a, b, n](auto i) {
      if (n <= 1) {
        return b;
      }
      return a + i * (b - a) / (n - 1);
    };
    if (n <= 0)
      return views::iota(0, 0) | views::transform(slice);
    return views::iota(0, n) | views::transform(slice);
  }

  static vector<tuple<double, double>> detect_local_minimums_in_samples(
    vector<tuple<double, double>> const& samples) {
    int n_samples = samples.size();
    if (n_samples < 3)
      return {};

    vector<tuple<double, double>> result;

    auto sample_triplets = views::zip(
      samples | views::slice(0, n_samples - 2),
      samples | views::slice(1, n_samples - 1),
      samples | views::slice(2, n_samples));

    for (auto const& [prev, curr, next] : sample_triplets) {
      auto const& p_prev = std::get<1>(prev);
      auto const& p_next = std::get<1>(next);
      auto const& [s, p_curr] = curr;
      if (p_prev >= p_curr && p_curr <= p_next)
        result.emplace_back(std::make_tuple(s, p_curr));
    }
    return result;
  }

  std::optional<vector<tuple<double, double>>> Context::sample() {
    auto const& sampling_config = config.initialization.imu.sampling;
    auto sigma_min = std::log(sampling_config.sampling_domain_lowerbound);
    auto sigma_max = std::log(sampling_config.sampling_domain_upperbound);

    auto maybe_costs =  //
      linspace(sigma_min, sigma_max, sampling_config.samples_count) |
      views::transform([&](auto sigma) { return std::exp(sigma); }) |
      views::transform(
        [&](auto s) { return std::make_tuple(s, evaluateCost(s)); }) |
      ranges::to_vector;

    auto costs =  //
      maybe_costs | views::filter([](auto const& _) {
        auto const& [s, maybe_p] = _;
        return maybe_p.has_value();
      }) |
      views::transform([](auto const& _) {
        auto const& [s, p] = _;
        return std::make_tuple(s, p.value());
      }) |
      ranges::to_vector;

    if (
      costs.size() <
      maybe_costs.size() * sampling_config.min_evaluation_success_rate) {
      return std::nullopt;
    }
    return costs;
  }

  std::optional<vector<imu_match_scale_sample_solution_t>> Context::solve(
    std::set<frame_id_t> const& motion_frames) {
    auto maybe_costs = sample();
    if (!maybe_costs.has_value()) {
      __logger__->debug("IMU match cost sample evaluation failed.");
      return std::nullopt;
    }
    auto const& costs = maybe_costs.value();

    auto minima = detect_local_minimums_in_samples(costs);
    auto maybe_candidates = refineSolutionCandidates(minima);
    telemetry.onIMUMatchAttempt({
      .degrees_of_freedom = degreesOfFreedom(),
      .frames = motion_frames,
      .landscape = costs,
      .minima = maybe_candidates
        ? *maybe_candidates
        : std::remove_reference_t<decltype(*maybe_candidates)> {},
    });

    if (!maybe_candidates.has_value())
      return std::nullopt;

    vector<imu_match_scale_sample_solution_t> samples;
    for (auto const& [s, cost] : maybe_candidates.value()) {
      auto sample = evaluateSample(s);
      if (!sample)
        continue;
      samples.emplace_back(*sample);
    }
    return samples;
  }

  class IMUMatchScaleSampleSolverImpl: public IMUMatchScaleSampleSolver {
  private:
    std::unique_ptr<IMUMatchTranslationLocalOptimizer> _local_optimizer;

    std::shared_ptr<cyclops_global_config_t const> _config;
    std::shared_ptr<telemetry::InitializerTelemetry> _telemetry;

  public:
    IMUMatchScaleSampleSolverImpl(
      std::unique_ptr<IMUMatchTranslationLocalOptimizer> local_optimizer,
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
    void reset() override;

    std::optional<vector<imu_match_scale_sample_solution_t>> solve(
      std::set<frame_id_t> const& motion_frames,
      imu_match_translation_analysis_t const& analysis,
      IMUMatchTranslationAnalysisCache const& cache) const override;
  };

  IMUMatchScaleSampleSolverImpl::IMUMatchScaleSampleSolverImpl(
    std::unique_ptr<IMUMatchTranslationLocalOptimizer> local_optimizer,
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry)
      : _local_optimizer(std::move(local_optimizer)),
        _config(config),
        _telemetry(telemetry) {
  }

  void IMUMatchScaleSampleSolverImpl::reset() {
    _local_optimizer->reset();
    _telemetry->reset();
  }

  std::optional<vector<imu_match_scale_sample_solution_t>>
  IMUMatchScaleSampleSolverImpl::solve(
    std::set<frame_id_t> const& motion_frames,
    imu_match_translation_analysis_t const& analysis,
    IMUMatchTranslationAnalysisCache const& cache) const {
    auto context = IMUMatchScaleSampleSolverContext(
      *_local_optimizer, analysis, cache, *_config, *_telemetry);
    return context.solve(motion_frames);
  }

  std::unique_ptr<IMUMatchScaleSampleSolver> IMUMatchScaleSampleSolver::create(
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<IMUMatchScaleSampleSolverImpl>(
      IMUMatchTranslationLocalOptimizer::create(config), config, telemetry);
  }
}  // namespace cyclops::initializer
