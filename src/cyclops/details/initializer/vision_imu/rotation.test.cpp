#include "cyclops/details/initializer/vision_imu/rotation.cpp"
#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"

#include "cyclops/details/measurement/type.hpp"
#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/config.hpp"

#include <range/v3/all.hpp>
#include <random>

#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  using measurement::imu_motion_ref_t;
  using measurement::imu_motion_t;
  using measurement::imu_motions_t;
  using measurement::imu_noise_t;

  using measurement::IMUPreintegration;

  static auto make_preintegration(
    Quaterniond const& q_init, Quaterniond const& q_term,
    Vector3d const& gyro_bias, double duration, size_t samples) {
    auto result = std::make_unique<IMUPreintegration>(
      Vector3d::Zero(), Vector3d::Zero(), imu_noise_t {0.01, 0.01});

    auto delta_q = q_init.conjugate() * q_term;
    auto delta_r = Eigen::AngleAxisd(delta_q);

    auto dt = duration / samples;
    for (size_t _ = 0; _ < samples; _++) {
      auto w = (delta_r.axis() * delta_r.angle() / duration).eval();
      result->propagate(dt, Vector3d::Zero(), gyro_bias + w);
    }
    return result;
  }

  TEST_CASE("test rotation only visual-inertial matching") {
    std::mt19937 rgen(20220511);
    auto rand = [&rgen]() {
      return std::uniform_real_distribution<double>(-1, 1)(rgen);
    };
    auto rotation_axis = Vector3d(rand(), rand(), rand()).normalized().eval();
    auto q_extrinsic = Quaterniond(rand(), rand(), rand(), rand()).normalized();
    auto gyro_bias = (0.1 * Vector3d(rand(), rand(), rand())).eval();

    auto imu_orientation_signal = [&](double t) -> Quaterniond {
      return Quaterniond(Eigen::AngleAxisd(0.1 * t * t, rotation_axis));
    };
    auto imu_orientations = std::vector {
      imu_orientation_signal(0.0),
      imu_orientation_signal(0.1),
      imu_orientation_signal(0.2),
    };
    imu_motions_t imu_motions;
    imu_motions.emplace_back(imu_motion_t {
      .from = 0,
      .to = 1,
      .data = make_preintegration(
        imu_orientations.at(0), imu_orientations.at(1), gyro_bias, 0.1, 20),
    });
    imu_motions.emplace_back(imu_motion_t {
      .from = 1,
      .to = 2,
      .data = make_preintegration(
        imu_orientations.at(1), imu_orientations.at(2), gyro_bias, 0.1, 20),
    });

    auto camera_orientations = imu_orientations |
      views::transform([&](auto const& q) { return q * q_extrinsic; }) |
      ranges::to_vector;
    auto camera_rotation_prior = imu_match_camera_rotation_prior_t {
      .rotations =
        std::map<frame_id_t, Quaterniond> {
          {0, camera_orientations.at(0)},
          {1, camera_orientations.at(1)},
          {2, camera_orientations.at(2)},
        },
      .weight = 1e6 * Eigen::MatrixXd::Identity(6, 6),
    };

    auto config = std::make_shared<cyclops_global_config_t>();
    config->extrinsics.imu_camera_transform = {Vector3d::Zero(), q_extrinsic};
    config->noise.gyr_bias_prior_stddev = 1.0;
    config->initialization.imu.acceptance_test.max_rotation_deviation = 0.03;
    config->initialization.imu.rotation_match
      .vision_imu_rotation_consistency_angle_threshold = 0.05;

    auto solver = IMUMatchRotationSolverImpl(config);
    auto maybe_rotation_matching = solver.solve(
      imu_motions |
        views::transform([](auto& _) -> imu_motion_ref_t { return _; }) |
        ranges::to_vector,
      camera_rotation_prior);
    REQUIRE(maybe_rotation_matching.has_value());
    auto const& matching = *maybe_rotation_matching;

    {
      CAPTURE(gyro_bias.transpose());
      CAPTURE(matching.gyro_bias.transpose());
      CHECK(gyro_bias.isApprox(matching.gyro_bias, 1e-3));
    }

    for (auto const& [q_true, q_matched] : views::zip(
           imu_orientations, matching.body_orientations | views::values)) {
      CAPTURE(q_true.coeffs().transpose());
      CAPTURE(q_matched.coeffs().transpose());
      CHECK(q_true.isApprox(q_matched, 1e-6));
    }
  }
}  // namespace cyclops::initializer
