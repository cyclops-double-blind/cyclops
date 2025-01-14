#include "cyclops/details/initializer/vision_imu.hpp"
#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"
#include "cyclops/details/initializer/vision_imu/rotation.hpp"
#include "cyclops/details/initializer/vision_imu/translation.hpp"

#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  namespace views = ranges::views;

  class IMUBootstrapSolverImpl: public IMUBootstrapSolver {
  private:
    std::unique_ptr<IMUMatchRotationSolver> _rotation_solver;
    std::unique_ptr<IMUMatchTranslationSolver> _translation_solver;

    std::shared_ptr<cyclops_global_config_t const> _config;

    struct imu_match_t {
      imu_match_rotation_solution_t rotation_match;
      imu_translation_match_t translation_match;
    };
    std::optional<imu_match_t> solveMatch(
      vision_bootstrap_solution_t const& sfm_solution,
      measurement::imu_motion_refs_t const& imu_motions);

  public:
    IMUBootstrapSolverImpl(
      std::unique_ptr<IMUMatchRotationSolver> rotation_solver,
      std::unique_ptr<IMUMatchTranslationSolver> translation_solver,
      std::shared_ptr<cyclops_global_config_t const> config);
    ~IMUBootstrapSolverImpl();
    void reset() override;

    std::optional<imu_bootstrap_solution_t> solve(
      vision_bootstrap_solution_t const& sfm_solution,
      measurement::imu_motion_refs_t const& imu_motions) override;
  };

  IMUBootstrapSolverImpl::IMUBootstrapSolverImpl(
    std::unique_ptr<IMUMatchRotationSolver> rotation_solver,
    std::unique_ptr<IMUMatchTranslationSolver> translation_solver,
    std::shared_ptr<cyclops_global_config_t const> config)
      : _rotation_solver(std::move(rotation_solver)),
        _translation_solver(std::move(translation_solver)),
        _config(config) {
  }

  IMUBootstrapSolverImpl::~IMUBootstrapSolverImpl() = default;

  void IMUBootstrapSolverImpl::reset() {
    _rotation_solver->reset();
    _translation_solver->reset();
  }

  std::optional<IMUBootstrapSolverImpl::imu_match_t>
  IMUBootstrapSolverImpl::solveMatch(
    vision_bootstrap_solution_t const& sfm_solution,
    measurement::imu_motion_refs_t const& imu_motions) {
    auto [rotation_prior, translation_prior] =
      make_imu_match_camera_motion_prior(sfm_solution);

    auto rotation_match = _rotation_solver->solve(imu_motions, rotation_prior);
    if (!rotation_match.has_value()) {
      __logger__->info("IMU match rotation solver failed.");
      return std::nullopt;
    }

    auto translation_match = _translation_solver->solve(
      imu_motions, *rotation_match, translation_prior);
    if (!translation_match.has_value()) {
      __logger__->info("IMU match translation solver failed.");
      return std::nullopt;
    }
    return imu_match_t {*rotation_match, *translation_match};
  }

  std::optional<imu_bootstrap_solution_t> IMUBootstrapSolverImpl::solve(
    vision_bootstrap_solution_t const& sfm_solution,
    measurement::imu_motion_refs_t const& imu_motions) {
    auto const& camera_motions = sfm_solution.geometry.camera_motions;
    auto solvable_imu_motions =  //
      imu_motions | views::filter([&](auto const& motion_ref) {
        auto const& motion = motion_ref.get();
        return  //
          camera_motions.find(motion.from) != camera_motions.end() &&
          camera_motions.find(motion.to) != camera_motions.end();
      }) |
      ranges::to<measurement::imu_motion_refs_t>;

    auto n_imu = static_cast<int>(solvable_imu_motions.size());
    auto n_sfm = static_cast<int>(camera_motions.size());
    if (n_sfm != n_imu + 1) {
      __logger__->error("Unmatching number of motion frames (SfM vs IMU)");
      __logger__->error(
        "SfM motion frames ({}) != IMU motion frames + 1 ({} + 1)",  //
        n_sfm, n_imu);
      return std::nullopt;
    }

    auto match = solveMatch(sfm_solution, solvable_imu_motions);
    if (!match)
      return std::nullopt;

    auto const& rotation_match = match->rotation_match;
    auto const& translation_match = match->translation_match;
    auto const& translation_solution = translation_match.solution;

    auto s = translation_solution.scale;
    auto landmark_transform = views::transform([&](auto const& id_landmark) {
      auto [landmark_id, f] = id_landmark;
      return std::make_pair(landmark_id, (s * f).eval());
    });

    return imu_bootstrap_solution_t {
      .accept = translation_match.accept,
      .cost = translation_solution.cost,
      .scale = s,
      .gravity = translation_solution.gravity,

      .gyr_bias = rotation_match.gyro_bias,
      .acc_bias = translation_solution.acc_bias,

      .landmarks = sfm_solution.geometry.landmarks | landmark_transform |
        ranges::to<landmark_positions_t>,
      .motions =
        views::zip(
          translation_solution.imu_body_velocities | views::keys,
          rotation_match.body_orientations | views::values,
          translation_solution.imu_body_velocities | views::values,
          translation_solution.sfm_positions | views::values) |
        views::transform([&](auto const& pair) {
          auto const& [frame_id, q_b, v_body, p_c] = pair;
          auto v = (q_b * v_body).eval();

          auto const& [p_bc, _] = _config->extrinsics.imu_camera_transform;
          Eigen::Vector3d p = p_c * s - q_b * p_bc;
          return std::make_pair(frame_id, imu_motion_state_t {q_b, p, v});
        }) |
        ranges::to<std::map<frame_id_t, imu_motion_state_t>>,
    };
  }

  std::unique_ptr<IMUBootstrapSolver> IMUBootstrapSolver::create(
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<IMUBootstrapSolverImpl>(
      IMUMatchRotationSolver::create(config),
      IMUMatchTranslationSolver::create(config, telemetry), config);
  }
}  // namespace cyclops::initializer
