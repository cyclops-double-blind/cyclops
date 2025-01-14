#include "cyclops/details/estimation/ceres/cost.imu_preintegration.cpp"
#include "cyclops/details/estimation/state/state_block.hpp"

#include "cyclops_tests/data/imu.hpp"

#include "cyclops_tests/default.hpp"
#include "cyclops_tests/range.ipp"
#include "cyclops_tests/signal.ipp"

#include <doctest/doctest.h>

namespace cyclops::estimation {
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  static auto position_signal(timestamp_t t) {
    auto x = 3 * (1 - std::cos(t));
    auto y = 0.1 * t * t - 0.05 * std::cos(M_PI * t);
    auto z = -0.05 * t * exp(0.3 * t);

    return Vector3d(x, y, z);
  }

  static auto orientation_signal(timestamp_t t) -> Quaterniond {
    auto axis = Vector3d(1, 1, 0.5).normalized().eval();
    auto angle = atan2(1, cos(t));

    return static_cast<Quaterniond>(Eigen::AngleAxisd(angle, axis));
  }

  static auto evaluate_motion_state(pose_signal_t pose_signal, timestamp_t t) {
    auto const& [p, q] = pose_signal;
    auto v = numeric_derivative(p);

    return imu_motion_state_t {
      .orientation = q(t),
      .position = p(t),
      .velocity = v(t),
    };
  }

  static auto make_imu_motion_state_block(
    imu_motion_state_t const& x, Vector3d const& b_a, Vector3d const& b_w) {
    motion_frame_parameter_block_t buffer;

    auto buffer_q = Eigen::Map<Quaterniond>(buffer.data());
    auto buffer_p = Eigen::Map<Vector3d>(buffer.data() + 4);
    auto buffer_v = Eigen::Map<Vector3d>(buffer.data() + 7);
    auto buffer_b_a = Eigen::Map<Vector3d>(buffer.data() + 10);
    auto buffer_b_w = Eigen::Map<Vector3d>(buffer.data() + 13);

    buffer_q = x.orientation;
    buffer_p = x.position;
    buffer_v = x.velocity;
    buffer_b_a = b_a;
    buffer_b_w = b_w;

    return buffer;
  }

  static auto evaluate_residual(
    measurement::IMUPreintegration* integration,
    motion_frame_parameter_block_t const& x_s,
    motion_frame_parameter_block_t const& x_e) -> Eigen::VectorXd {
    auto cost = IMUPreintegrationCostEvaluator(integration, 9.81);

    std::array<double, 9> r;
    std::fill(r.begin(), r.end(), 1);

    auto success = cost(x_s.data(), x_e.data(), x_s.data() + 10, r.data());
    REQUIRE(success);

    return Eigen::Map<Eigen::VectorXd>(r.data(), 9);
  }

  TEST_CASE("IMU preintegration ceres factor") {
    auto pose_signal = pose_signal_t {position_signal, orientation_signal};
    timestamp_t t_s = 0.0;
    timestamp_t t_e = M_PI_4;

    auto const O3x1 = Vector3d::Zero().eval();

    GIVEN("Perfectly correct IMU preintegration at an arbitrary motion") {
      auto data = make_imu_preintegration(pose_signal, t_s, t_e);
      auto x_s = evaluate_motion_state(pose_signal, t_s);
      auto x_e = evaluate_motion_state(pose_signal, t_e);

      WHEN("Evaluated IMU preintegration cost with the correct state") {
        auto block_s = make_imu_motion_state_block(x_s, O3x1, O3x1);
        auto block_e = make_imu_motion_state_block(x_e, O3x1, O3x1);

        auto r = evaluate_residual(data.get(), block_s, block_e);
        CAPTURE(r.transpose());

        THEN("Norm of the result residual is effectively 0") {
          CHECK(r.norm() < 1e-6);
        }
      }

      WHEN("Evaluated IMU preintegration cost with the perturbated state") {
        auto cost = IMUPreintegrationCostEvaluator(data.get(), 9.81);

        auto b_a = Vector3d(0.1, 0.4, 0.3);
        auto b_w = Vector3d(-0.05, 0.03, 0.08);
        auto block_s = make_imu_motion_state_block(x_s, b_a, b_w);

        auto x_e_perturbed = imu_motion_state_t {
          .orientation = x_e.orientation * Quaterniond(0, 1, 0, 0),
          .position = x_e.position + Vector3d(0, 0, 1),
          .velocity = x_e.velocity + Vector3d(0, 1, 0),
        };
        auto block_e = make_imu_motion_state_block(x_e_perturbed, O3x1, O3x1);

        auto r1 = evaluate_residual(data.get(), block_s, block_e);
        auto r1_norm = r1.norm();
        CAPTURE(r1.transpose());

        THEN("Norm of the result residual is above 0 with sufficient margin") {
          CHECK(r1_norm > 0.1);
        }

        AND_WHEN(
          "Half-divided the information weight of the IMU data and "
          "re-evaluated the cost") {
          data->covariance /= 2;
          auto r2 = evaluate_residual(data.get(), block_s, block_e);
          auto r2_norm = r2.norm();

          THEN("`r2_norm` is sqrt(2) times of `r1_norm`") {
            CHECK(r2_norm == doctest::Approx(r1_norm * std::sqrt(2)));
          }
        }
      }
    }
  }
}  // namespace cyclops::estimation
