#include "cyclops/details/initializer/vision.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment.hpp"
#include "cyclops/details/initializer/vision/multiview.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/utils/debug.hpp"
#include "cyclops/details/utils/vision.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using telemetry::InitializerTelemetry;
  using motion_statistics_lookup_t =
    std::map<frame_id_t, image_frame_motion_statistics_t>;

  template <typename key_t, typename value_t>
  static std::set<key_t> project_keys(std::map<key_t, value_t> const& map) {
    return map | views::keys | ranges::to<std::set<key_t>>;
  }

  class VisionBootstrapSolverImpl: public VisionBootstrapSolver {
  private:
    std::unique_ptr<MultiviewVisionGeometrySolver> _multiview_solver;
    std::shared_ptr<cyclops_global_config_t const> _config;
    std::shared_ptr<InitializerTelemetry> _telemetry;

    motion_statistics_lookup_t compileMotionStatistics(
      multiview_image_data_t const& image_data) const;

    motion_statistics_lookup_t filterConnectedImageTrackSequence(
      config::initializer::observability_pretest_threshold_t const& threshold,
      motion_statistics_lookup_t const& lookup) const;
    bool detectCameraMotion(
      config::initializer::observability_pretest_threshold_t const& threshold,
      motion_statistics_lookup_t const& lookup) const;

    void reportBundleAdjustmentTelemetry(
      std::set<frame_id_t> const& input_frames,
      std::vector<vision_bootstrap_solution_t> const& solutions);

  public:
    VisionBootstrapSolverImpl(
      std::unique_ptr<MultiviewVisionGeometrySolver> multiview_solver,
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<InitializerTelemetry> telemetry);
    ~VisionBootstrapSolverImpl();
    void reset() override;

    std::vector<vision_bootstrap_solution_t> solve(
      multiview_image_data_t const& image_data,
      camera_rotations_t const& camera_rotation_prior) override;
  };

  VisionBootstrapSolverImpl::~VisionBootstrapSolverImpl() = default;

  VisionBootstrapSolverImpl::VisionBootstrapSolverImpl(
    std::unique_ptr<MultiviewVisionGeometrySolver> multiview_solver,
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<InitializerTelemetry> telemetry)
      : _multiview_solver(std::move(multiview_solver)),
        _config(config),
        _telemetry(telemetry) {
  }

  void VisionBootstrapSolverImpl::reset() {
    _multiview_solver->reset();
  }

  motion_statistics_lookup_t VisionBootstrapSolverImpl::compileMotionStatistics(
    multiview_image_data_t const& image_data) const {
    return  //
      image_data | views::drop_last(1) |
      views::transform([&](auto const& id_frame) {
        auto const& [frame_id, frame] = id_frame;
        auto const& [_, last_frame] = *image_data.rbegin();

        return std::make_pair(
          frame_id, compute_image_frame_motion_statistics(frame, last_frame));
      }) |
      ranges::to<motion_statistics_lookup_t>;
  }

  static auto make_image_observability_statistics_telemetry(
    image_frame_motion_statistics_t const& motion_statistics) {
    return InitializerTelemetry::image_observability_statistics_t {
      .common_features = motion_statistics.common_features,
      .motion_parallax = motion_statistics.average_parallax,
    };
  }

  motion_statistics_lookup_t
  VisionBootstrapSolverImpl::filterConnectedImageTrackSequence(
    config::initializer::observability_pretest_threshold_t const& threshold,
    motion_statistics_lookup_t const& lookup) const {
    if (lookup.empty())
      return {};

    motion_statistics_lookup_t result;
    for (auto i = lookup.rbegin(); i != lookup.rend(); i++) {
      auto const& [frame_id, statistics] = *i;
      if (statistics.common_features < threshold.min_landmark_overlap)
        break;

      result.emplace(frame_id, statistics);
    }

    auto telemetry_frames =  //
      lookup | views::transform([](auto const& pair) {
        auto const& [frame_id, statistics] = pair;
        return std::make_pair(
          frame_id, make_image_observability_statistics_telemetry(statistics));
      }) |
      ranges::to<std::map<
        frame_id_t, InitializerTelemetry::image_observability_statistics_t>>;

    _telemetry->onImageObservabilityPretest({
      .frames = telemetry_frames,
      .connected_frames = project_keys(result),
    });
    return result;
  }

  bool VisionBootstrapSolverImpl::detectCameraMotion(
    config::initializer::observability_pretest_threshold_t const& threshold,
    motion_statistics_lookup_t const& lookup) const {
    auto average_parallax = ranges::max(  //
      lookup | views::values |
      views::transform([](auto const& _) { return _.average_parallax; }));
    auto min_average_parallax = threshold.min_average_parallax;

    if (average_parallax < min_average_parallax) {
      __logger__->debug(
        "Not enough image motion parallax ({} < {})",  //
        average_parallax, min_average_parallax);
      return false;
    }
    return true;
  }

  void VisionBootstrapSolverImpl::reportBundleAdjustmentTelemetry(
    std::set<frame_id_t> const& input_frames,
    std::vector<vision_bootstrap_solution_t> const& solutions) {
    if (solutions.empty()) {
      _telemetry->onVisionFailure({
        .frames = input_frames,
        .reason = InitializerTelemetry::BUNDLE_ADJUSTMENT_FAILED,
      });
      return;
    }

    for (auto const& solution : solutions) {
      _telemetry->onBundleAdjustmentSuccess({
        .camera_motions = solution.geometry.camera_motions,
        .landmarks = solution.geometry.landmarks,
      });
    }

    auto sanity_telemetry =  //
      solutions | views::transform([](auto const& solution) {
        return InitializerTelemetry::bundle_adjustment_sanity_t {
          .acceptable = solution.acceptable,
          .inlier_ratio = solution.measurement_inlier_ratio,
          .final_cost_significant_probability =
            solution.solution_significant_probability,
        };
      }) |
      ranges::to_vector;

    _telemetry->onBundleAdjustmentSanity({
      .frames = input_frames,
      .candidates_sanity = sanity_telemetry,
    });
  }

  std::vector<vision_bootstrap_solution_t> VisionBootstrapSolverImpl::solve(
    multiview_image_data_t const& image_data,
    camera_rotations_t const& camera_rotation_prior) {
    auto tic = ::cyclops::tic();

    auto const& threshold = _config->initialization.observability_pretest;
    auto connected_tracks = filterConnectedImageTrackSequence(
      threshold, compileMotionStatistics(image_data));
    if ((connected_tracks.size() + 1) < threshold.min_keyframes) {
      _telemetry->onVisionFailure({
        .frames = project_keys(image_data),
        .reason = InitializerTelemetry::NOT_ENOUGH_CONNECTED_IMAGE_FRAMES,
      });
      return {};
    }
    if (!detectCameraMotion(threshold, connected_tracks)) {
      _telemetry->onVisionFailure({
        .frames = project_keys(connected_tracks),
        .reason = InitializerTelemetry::NOT_ENOUGH_MOTION_PARALLAX,
      });
      return {};
    }

    auto const& [last_motion_frame, _] = *image_data.rbegin();
    auto image_data_filtered =
      views::concat(
        connected_tracks | views::keys, views::single(last_motion_frame)) |
      views::transform([&](auto frame_id) {
        return std::make_pair(frame_id, image_data.at(frame_id));
      }) |
      ranges::to<multiview_image_data_t>;

    auto multiview_solutions =
      _multiview_solver->solve(image_data_filtered, camera_rotation_prior);
    if (multiview_solutions.empty())
      return {};

    auto maybe_bundle_adjustments =
      multiview_solutions | views::transform([&](auto const& solution) {
        auto const& config = _config->initialization.vision;
        return solve_bundle_adjustment(config, solution, image_data_filtered);
      }) |
      ranges::to_vector;

    auto successful_bundle_adjustments =  //
      maybe_bundle_adjustments |
      views::filter([](auto const& maybe_x) { return maybe_x.has_value(); }) |
      views::transform([](auto const& maybe_x) { return maybe_x.value(); }) |
      ranges::to_vector;

    reportBundleAdjustmentTelemetry(
      project_keys(connected_tracks), successful_bundle_adjustments);

    __logger__->debug("Vision bootstrap time: {}[s]", toc(tic));
    return successful_bundle_adjustments;
  }

  std::unique_ptr<VisionBootstrapSolver> VisionBootstrapSolver::create(
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<std::mt19937> rgen,
    std::shared_ptr<InitializerTelemetry> telemetry) {
    return std::make_unique<VisionBootstrapSolverImpl>(
      MultiviewVisionGeometrySolver::create(config, rgen, telemetry),  //
      config, telemetry);
  }
}  // namespace cyclops::initializer
