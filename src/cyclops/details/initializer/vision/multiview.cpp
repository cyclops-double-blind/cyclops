#include "cyclops/details/initializer/vision/multiview.hpp"
#include "cyclops/details/initializer/vision/twoview.hpp"
#include "cyclops/details/initializer/vision/twoview_selection.hpp"
#include "cyclops/details/initializer/vision/triangulation.hpp"
#include "cyclops/details/initializer/vision/epnp.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/telemetry/initializer.hpp"
#include "cyclops/details/utils/vision.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

#include <functional>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using telemetry::InitializerTelemetry;

  using Eigen::Matrix3d;
  using Eigen::Quaterniond;

  class MultiviewVisionGeometrySolverImpl:
      public MultiviewVisionGeometrySolver {
  private:
    std::unique_ptr<TwoViewVisionGeometrySolver> _two_view_solver;
    std::shared_ptr<cyclops_global_config_t const> _config;
    std::shared_ptr<InitializerTelemetry> _telemetry;

    void logFailure(std::string const& reason);
    void logFatal(std::string const& reason);

    multiview_correspondences_t makeMultiViewCorrespondences(
      multiview_image_data_t const& multiview_data,
      camera_rotation_prior_lookup_t const& rotation_prior);

    template <typename landmark_range_t>
    landmark_positions_t triangulateUnknownLandmarks(
      landmark_range_t const& known_landmarks,
      std::map<landmark_id_t, two_view_feature_pair_t> const& features,
      rotation_translation_matrix_pair_t const& camera_pose);

    std::optional<multiview_geometry_t> solveMultiview(
      frame_id_t twoview_frame_id, two_view_geometry_t const& twoview_solution,
      multiview_correspondences_t const& multiview_correspondences);

  public:
    MultiviewVisionGeometrySolverImpl(
      std::unique_ptr<TwoViewVisionGeometrySolver> two_view_solver,
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<InitializerTelemetry> telemetry);
    ~MultiviewVisionGeometrySolverImpl();
    void reset() override;

    std::vector<multiview_geometry_t> solve(
      multiview_image_data_t const& multiview_image,
      camera_rotation_prior_lookup_t const& camera_rotations) override;
  };

  void MultiviewVisionGeometrySolverImpl::logFailure(
    std::string const& reason) {
    __logger__->info("Vision initialization failed. Reason: {}", reason);
  }

  void MultiviewVisionGeometrySolverImpl::logFatal(std::string const& reason) {
    __logger__->error(
      "Internal error during vision initialization: {}", reason);
  }

  static std::optional<two_view_imu_rotation_data_t>
  make_two_view_rotation_prior(
    std::map<frame_id_t, two_view_imu_rotation_constraint_t> const&
      camera_rotations,
    frame_id_t reference_view_frame, frame_id_t best_view_frame) {
    frame_id_t frame_id = best_view_frame;

    Quaterniond q_best_to_reference = Quaterniond::Identity();
    Matrix3d P_best_to_reference = Matrix3d::Identity();

    while (true) {
      if (frame_id == reference_view_frame)
        break;

      auto i = camera_rotations.find(frame_id);
      if (i == camera_rotations.end())
        return std::nullopt;

      auto const& [_, rotation] = *i;
      auto const& [q_delta, P_delta] = rotation.rotation;

      auto R_delta = q_delta.matrix().eval();

      q_best_to_reference = q_best_to_reference * q_delta;
      P_best_to_reference =
        R_delta.transpose() * P_best_to_reference * R_delta + P_delta;
      frame_id = rotation.term_frame_id;
    }

    Matrix3d R_br = q_best_to_reference.matrix();
    return two_view_imu_rotation_data_t {
      .value = q_best_to_reference.conjugate(),
      .covariance = R_br * P_best_to_reference * R_br.transpose(),
    };
  }

  multiview_correspondences_t
  MultiviewVisionGeometrySolverImpl::makeMultiViewCorrespondences(
    multiview_image_data_t const& multiview_data,
    camera_rotation_prior_lookup_t const& rotation_prior) {
    auto const& [frame1_id, frame1] = *multiview_data.rbegin();

    multiview_correspondences_t result;
    result.reference_frame = frame1_id;

    auto image_frames = multiview_data | views::drop_last(1);
    for (auto const& [frame2_id, frame2] : image_frames) {
      auto maybe_rotation_prior = make_two_view_rotation_prior(  //
        rotation_prior, frame1_id, frame2_id);
      if (!maybe_rotation_prior) {
        logFatal("Two view rotation prior generation failed");
        return {};
      }

      auto& view_frame = result.view_frames[frame2_id];
      view_frame.rotation_prior = *maybe_rotation_prior;

      for (auto const& [landmark_id, u2] : frame2) {
        auto i = frame1.find(landmark_id);
        if (i == frame1.end())
          continue;
        auto const& u1 = i->second;

        view_frame.features.emplace(
          landmark_id, std::make_tuple(u1.point, u2.point));
      }
    }
    return result;
  }

  template <typename landmark_range_t>
  landmark_positions_t
  MultiviewVisionGeometrySolverImpl::triangulateUnknownLandmarks(
    landmark_range_t const& known_landmarks,
    std::map<landmark_id_t, two_view_feature_pair_t> const& features,
    rotation_translation_matrix_pair_t const& camera_pose) {
    auto unknown_landmarks =
      views::set_difference(features | views::keys, known_landmarks) |
      ranges::to<std::set>;
    auto triangulation = triangulate_two_view_feature_pairs(
      _config->initialization.vision, features, unknown_landmarks, camera_pose);
    return triangulation.landmarks;
  }

  std::optional<multiview_geometry_t>
  MultiviewVisionGeometrySolverImpl::solveMultiview(
    frame_id_t twoview_frame_id, two_view_geometry_t const& twoview_solution,
    multiview_correspondences_t const& correspondences) {
    auto motions = std::map<frame_id_t, se3_transform_t> {
      {correspondences.reference_frame, se3_transform_t::Identity()},
      {twoview_frame_id, twoview_solution.camera_motion},
    };
    auto landmarks = twoview_solution.landmarks;

    auto solve_pnp = [&](auto const& view) {
      std::map<landmark_id_t, pnp_image_point_t> pnp_image_set;
      for (auto const& [feature_id, feature] : view.features) {
        auto i = landmarks.find(feature_id);
        if (i == landmarks.end())
          continue;
        auto const& [_, f] = *i;
        auto const& [u, v] = feature;

        pnp_image_set.emplace(feature_id, pnp_image_point_t {f, v});
      }
      return solve_pnp_camera_pose(pnp_image_set);
    };

    for (auto const& [frame_id, view_frame] : correspondences.view_frames) {
      auto maybe_camera_pose = solve_pnp(view_frame);
      if (!maybe_camera_pose) {
        logFailure("EPnP camera pose reconstruction failed.");
        __logger__->debug("Frame id: {}", frame_id);
        return std::nullopt;
      }

      auto const& camera_pose = *maybe_camera_pose;
      auto const& [R, p] = camera_pose;
      motions.emplace(frame_id, se3_transform_t {p, Quaterniond(R)});

      auto new_landmarks = triangulateUnknownLandmarks(
        landmarks | views::keys, view_frame.features, camera_pose);
      landmarks.insert(new_landmarks.begin(), new_landmarks.end());
    }
    return multiview_geometry_t {
      .camera_motions = motions,
      .landmarks = landmarks,
    };
  }

  MultiviewVisionGeometrySolverImpl::MultiviewVisionGeometrySolverImpl(
    std::unique_ptr<TwoViewVisionGeometrySolver> two_view_solver,
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<InitializerTelemetry> telemetry)
      : _two_view_solver(std::move(two_view_solver)),
        _config(config),
        _telemetry(telemetry) {
  }

  MultiviewVisionGeometrySolverImpl::~MultiviewVisionGeometrySolverImpl() =
    default;

  void MultiviewVisionGeometrySolverImpl::reset() {
    _two_view_solver->reset();
  }

  std::vector<multiview_geometry_t> MultiviewVisionGeometrySolverImpl::solve(
    multiview_image_data_t const& multiview_image,
    camera_rotation_prior_lookup_t const& rotation_prior) {
    if (multiview_image.empty())
      return {};
    auto frame_ids = multiview_image | views::keys | ranges::to<std::set>;

    auto correspondences =
      makeMultiViewCorrespondences(multiview_image, rotation_prior);

    auto maybe_best_view = select_best_two_view_pair(correspondences);
    if (!maybe_best_view) {
      _telemetry->onVisionFailure({
        .frames = frame_ids,
        .reason = InitializerTelemetry::BEST_TWO_VIEW_SELECTION_FAILED,
      });
      return {};
    }
    auto const& [best_frame_id, best_view] = maybe_best_view->get();

    _telemetry->onBestTwoViewSelection({
      .frames = frame_ids,
      .frame_id_1 = *frame_ids.rbegin(),
      .frame_id_2 = best_frame_id,
    });

    auto twoview_solutions = _two_view_solver->solve(best_view);
    if (twoview_solutions.empty()) {
      _telemetry->onVisionFailure({
        .frames = frame_ids,
        .reason = InitializerTelemetry::TWO_VIEW_GEOMETRY_FAILED,
      });
      return {};
    }

    auto multiview_solutions =  //
      twoview_solutions |
      views::transform(std::bind(
        &MultiviewVisionGeometrySolverImpl::solveMultiview, this, best_frame_id,
        std::placeholders::_1, correspondences)) |
      ranges::to_vector;

    auto multiview_solutions_successful = multiview_solutions |
      views::filter([](auto const& maybe) { return maybe.has_value(); }) |
      views::transform([](auto const& maybe) { return maybe.value(); }) |
      ranges::to_vector;
    if (multiview_solutions_successful.empty()) {
      _telemetry->onVisionFailure({
        .frames = frame_ids,
        .reason = InitializerTelemetry::MULTI_VIEW_GEOMETRY_FAILED,
      });
      return {};
    }

    return multiview_solutions_successful;
  }

  std::unique_ptr<MultiviewVisionGeometrySolver>
  MultiviewVisionGeometrySolver::create(
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<std::mt19937> rgen,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<MultiviewVisionGeometrySolverImpl>(
      TwoViewVisionGeometrySolver::create(config, rgen), config, telemetry);
  }
}  // namespace cyclops::initializer
