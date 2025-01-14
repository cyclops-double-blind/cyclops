#include "cyclops/details/initializer/solver.hpp"
#include "cyclops/details/initializer/vision.hpp"
#include "cyclops/details/initializer/vision_imu.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/measurement/data_provider.hpp"
#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/config.hpp"

#include <range/v3/all.hpp>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using Eigen::Matrix3d;

  using measurement::MeasurementDataProvider;
  using imu_motion_refs_t = std::vector<measurement::imu_motion_ref_t>;
  using frame_id_set_t = std::set<frame_id_t>;

  using multiview_image_observations_t =
    std::map<frame_id_t, std::map<landmark_id_t, feature_point_t>>;
  using camera_rotation_prior_lookup_t =
    std::map<frame_id_t, two_view_imu_rotation_constraint_t>;

  class InitializationSolverInternalImpl: public InitializationSolverInternal {
  private:
    std::unique_ptr<VisionBootstrapSolver> _vision_solver;
    std::unique_ptr<IMUBootstrapSolver> _imu_solver;

    std::shared_ptr<cyclops_global_config_t const> _config;
    std::shared_ptr<MeasurementDataProvider const> _data_provider;

    multiview_image_observations_t reorderMultiviewImageObservations() const;
    imu_motion_refs_t filterImageObservedIMU(frame_id_set_t image_frames) const;

    camera_rotation_prior_lookup_t makeCameraRotationPriorLookup(
      imu_motion_refs_t const& imu_motions) const;

  public:
    InitializationSolverInternalImpl(
      std::unique_ptr<initializer::VisionBootstrapSolver> vision_solver,
      std::unique_ptr<IMUBootstrapSolver> imu_solver,
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<MeasurementDataProvider const> data_provider);
    ~InitializationSolverInternalImpl();
    void reset() override;

    initializer_internal_solution_t solve() override;
  };

  InitializationSolverInternalImpl::InitializationSolverInternalImpl(
    std::unique_ptr<initializer::VisionBootstrapSolver> vision_solver,
    std::unique_ptr<IMUBootstrapSolver> imu_solver,
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<MeasurementDataProvider const> data_provider)
      : _vision_solver(std::move(vision_solver)),
        _imu_solver(std::move(imu_solver)),
        _config(config),
        _data_provider(data_provider) {
  }

  InitializationSolverInternalImpl::~InitializationSolverInternalImpl() =
    default;

  void InitializationSolverInternalImpl::reset() {
    _vision_solver->reset();
    _imu_solver->reset();
  }

  multiview_image_observations_t
  InitializationSolverInternalImpl::reorderMultiviewImageObservations() const {
    multiview_image_observations_t result;

    for (auto const& [landmark_id, track] : _data_provider->tracks()) {
      for (auto const& [frame_id, feature] : track)
        result[frame_id].emplace(landmark_id, feature);
    }
    return result;
  }

  imu_motion_refs_t InitializationSolverInternalImpl::filterImageObservedIMU(
    frame_id_set_t image_frames) const {
    return  //
      _data_provider->imu() | views::filter([&](auto const& motion) {
        auto init_frame_exists =
          image_frames.find(motion.from) != image_frames.end();
        auto term_frame_exists =
          image_frames.find(motion.to) != image_frames.end();

        return init_frame_exists && term_frame_exists;
      }) |
      views::transform(
        [](auto const& _) -> measurement::imu_motion_ref_t { return _; }) |
      ranges::to_vector;
  }

  camera_rotation_prior_lookup_t
  InitializationSolverInternalImpl::makeCameraRotationPriorLookup(
    imu_motion_refs_t const& imu_motions) const {
    return  //
      imu_motions | views::transform([&](auto const& ref) {
        auto const& data = ref.get();

        auto const& q = data.data->rotation_delta;
        auto const& q_ext = _config->extrinsics.imu_camera_transform.rotation;

        Matrix3d P = data.data->covariance.template topLeftCorner<3, 3>();
        Matrix3d R_ext = q_ext.matrix();

        auto rotation_data = two_view_imu_rotation_data_t {
          .value = q_ext.conjugate() * q * q_ext,
          .covariance = R_ext.transpose() * P * R_ext};

        return two_view_imu_rotation_constraint_t {
          .init_frame_id = data.from,
          .term_frame_id = data.to,
          .rotation = rotation_data};
      }) |
      views::transform([](auto const& rotation) {
        return std::make_pair(rotation.init_frame_id, rotation);
      }) |
      ranges::to<camera_rotation_prior_lookup_t>;
  }

  initializer_internal_solution_t InitializationSolverInternalImpl::solve() {
    auto image_data = reorderMultiviewImageObservations();
    auto image_motion_frames = image_data | views::keys | ranges::to<std::set>;
    auto imu_motions = filterImageObservedIMU(image_motion_frames);

    auto rotation_prior = makeCameraRotationPriorLookup(imu_motions);

    initializer_internal_solution_t result;
    result.vision_solutions = _vision_solver->solve(image_data, rotation_prior);
    result.imu_solutions.reserve(result.vision_solutions.size());

    for (int i = 0; i < result.vision_solutions.size(); i++) {
      auto const& vision_solution = result.vision_solutions.at(i);

      auto imu_solution = _imu_solver->solve(vision_solution, imu_motions);
      if (imu_solution.has_value())
        result.imu_solutions.emplace_back(std::make_tuple(i, *imu_solution));
    }
    return result;
  }

  std::unique_ptr<InitializationSolverInternal>
  InitializationSolverInternal::create(
    std::shared_ptr<std::mt19937> rgen,
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<MeasurementDataProvider const> data_provider,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<InitializationSolverInternalImpl>(
      initializer::VisionBootstrapSolver::create(config, rgen, telemetry),
      IMUBootstrapSolver::create(config, telemetry), config, data_provider);
  }
}  // namespace cyclops::initializer
