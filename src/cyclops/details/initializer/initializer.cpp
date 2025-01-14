#include "cyclops/details/initializer/initializer.hpp"
#include "cyclops/details/initializer/solver.hpp"

#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/initializer/vision_imu.hpp"

#include "cyclops/details/measurement/keyframe.hpp"
#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/logging.hpp"

#include <range/v3/all.hpp>
#include <spdlog/spdlog.h>

namespace cyclops::initializer {
  using Eigen::Matrix3d;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  namespace views = ranges::views;

  using vision_solution_telemetry_digest_t =
    telemetry::InitializerTelemetry::vision_solution_candidate_digest_t;
  using imu_solution_telemetry_digest_t =
    telemetry::InitializerTelemetry::imu_solution_candidate_digest_t;

  class InitializerMainImpl: public InitializerMain {
  private:
    std::unique_ptr<InitializationSolverInternal> _solver_internal;

    std::shared_ptr<measurement::KeyframeManager const> _keyframe_manager;
    std::shared_ptr<telemetry::InitializerTelemetry> _telemetry;

    void reportFailureTelemetry(
      initializer_internal_solution_t const& solution);
    std::optional<imu_bootstrap_solution_t> solveAndReportTelemetry();

  public:
    InitializerMainImpl(
      std::unique_ptr<InitializationSolverInternal> solver_internal,
      std::shared_ptr<measurement::KeyframeManager const> keyframe_manager,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
    ~InitializerMainImpl();
    void reset() override;

    std::optional<initialization_solution_t> solve() override;
  };

  InitializerMainImpl::InitializerMainImpl(
    std::unique_ptr<InitializationSolverInternal> solver_internal,
    std::shared_ptr<measurement::KeyframeManager const> keyframe_manager,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry)
      : _solver_internal(std::move(solver_internal)),
        _keyframe_manager(keyframe_manager),
        _telemetry(telemetry) {
  }

  InitializerMainImpl::~InitializerMainImpl() = default;

  void InitializerMainImpl::reset() {
    _solver_internal->reset();
    _telemetry->reset();
  }

  void InitializerMainImpl::reportFailureTelemetry(
    initializer_internal_solution_t const& solution) {
    auto vision_solutions =  //
      solution.vision_solutions | views::transform([](auto const& sol) {
        return vision_solution_telemetry_digest_t {
          .acceptable = sol.acceptable,
          .keyframes =
            sol.geometry.camera_motions | views::keys | ranges::to<std::set>,
        };
      }) |
      ranges::to_vector;

    auto imu_solutions =  //
      solution.imu_solutions | views::transform([](auto const& index_sol_pair) {
        auto const& [index, sol] = index_sol_pair;
        return imu_solution_telemetry_digest_t {
          .vision_solution_index = index,
          .acceptable = sol.accept,
          .scale = sol.scale,
          .keyframes = sol.motions | views::keys | ranges::to<std::set>,
        };
      }) |
      ranges::to_vector;

    __logger__->debug("Reporting failure...");
    _telemetry->onFailure({
      .vision_solutions = vision_solutions,
      .imu_solutions = imu_solutions,
    });
    __logger__->debug("Reported failure.");
  }

  std::optional<imu_bootstrap_solution_t>
  InitializerMainImpl::solveAndReportTelemetry() {
    auto solution = _solver_internal->solve();
    __logger__->debug("Initialization solution obtained.");

    if (solution.imu_solutions.empty()) {
      reportFailureTelemetry(solution);
      return std::nullopt;
    }
    if (solution.imu_solutions.size() > 1) {
      reportFailureTelemetry(solution);
      return std::nullopt;
    }

    auto const& [solution_index, imu_solution] = solution.imu_solutions.front();
    auto const& vision_solution = solution.vision_solutions.at(solution_index);

    if (!imu_solution.accept || !vision_solution.acceptable) {
      reportFailureTelemetry(solution);
      return std::nullopt;
    }

    if (imu_solution.motions.empty()) {
      __logger__->error("IMU bootstrap returned empty motion.");

      reportFailureTelemetry(solution);
      return std::nullopt;
    }

    auto initial_motion_frame_id = imu_solution.motions.begin()->first;
    auto initial_motion_frame_timestamp =
      _keyframe_manager->keyframes().at(initial_motion_frame_id);

    _telemetry->onSuccess({
      .initial_motion_frame_id = initial_motion_frame_id,
      .initial_motion_frame_timestamp = initial_motion_frame_timestamp,
      .sfm_camera_pose = vision_solution.geometry.camera_motions,
      .cost = imu_solution.cost,
      .scale = imu_solution.scale,
      .gravity = imu_solution.gravity,
      .motions = imu_solution.motions,
    });
    return imu_solution;
  }

  /**
   * Returns a rotation matrix such that the third column (the z direction)
   * matches the given argument `g`, and the first column (the x direction) lies
   * within the plane spanned by the given arguments `g` and `x`.
   */
  static Matrix3d solve_vision_origin_to_world_origin_rotation(
    Vector3d const& g, Vector3d const& x) {
    Vector3d r3 = g.normalized();
    Vector3d r2 = r3.cross(x).normalized();
    Vector3d r1 = r2.cross(r3);

    if (x.dot(r1) < 0)
      r1 = -r1;

    Matrix3d R;
    R << r1, r2, r3;
    if (R.determinant() < 0)
      R.col(1) = -R.col(1);

    return R;
  }

  struct initialization_gravity_rotation_t {
    landmark_positions_t landmarks;
    std::map<frame_id_t, imu_motion_state_t> motions;
  };

  static initialization_gravity_rotation_t rotate_gravity(
    imu_bootstrap_solution_t const& imu_matching) {
    auto const& g = imu_matching.gravity;

    auto R_vb0 = [&]() -> Matrix3d {
      if (imu_matching.motions.empty())
        return Matrix3d::Identity();

      auto const& [_, x_vb0] = *imu_matching.motions.begin();
      return x_vb0.orientation.matrix();
    }();

    auto R_vw = solve_vision_origin_to_world_origin_rotation(g, R_vb0.col(0));
    auto q_vw = Quaterniond(R_vw);
    auto q_wv = q_vw.conjugate();

    std::map<frame_id_t, imu_motion_state_t> motions =
      std::move(imu_matching.motions);
    landmark_positions_t landmarks = std::move(imu_matching.landmarks);

    for (auto& [_, motion] : motions) {
      motion.orientation = q_wv * motion.orientation;
      motion.position = q_wv * motion.position;
      motion.velocity = q_wv * motion.velocity;
    }
    for (auto& [_, landmark] : landmarks)
      landmark = q_wv * landmark;
    return {landmarks, motions};
  }

  std::optional<initialization_solution_t> InitializerMainImpl::solve() {
    auto maybe_imu_match = solveAndReportTelemetry();
    if (!maybe_imu_match)
      return std::nullopt;

    auto const& imu_match = *maybe_imu_match;
    auto [landmarks, motions] = rotate_gravity(imu_match);

    return initialization_solution_t {
      .acc_bias = imu_match.acc_bias,
      .gyr_bias = imu_match.gyr_bias,
      .landmarks = std::move(landmarks),
      .motions = std::move(motions),
    };
  }

  std::unique_ptr<InitializerMain> InitializerMain::create(
    std::unique_ptr<InitializationSolverInternal> solver_internal,
    std::shared_ptr<measurement::KeyframeManager const> keyframe_manager,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<InitializerMainImpl>(
      std::move(solver_internal), keyframe_manager, telemetry);
  }
}  // namespace cyclops::initializer
