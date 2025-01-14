#include "cyclops/details/measurement/preintegration.cpp"
#include "cyclops/details/measurement/preintegration.ipp"

#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/config.hpp"

#include "cyclops_tests/data/imu.hpp"
#include "cyclops_tests/default.hpp"
#include "cyclops_tests/random.hpp"
#include "cyclops_tests/range.ipp"
#include "cyclops_tests/signal.ipp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops::measurement {
  namespace views = ranges::views;

  using Vector9d = Eigen::Matrix<double, 9, 1>;
  using Matrix9d = Eigen::Matrix<double, 9, 9>;

  static auto const O3x1 = Vector3d::Zero().eval();

  static auto make_position_signal(std::mt19937& rgen) {
    Vector3d direction =
      perturbate(Vector3d::Zero().eval(), 1, rgen).normalized();
    return [direction](timestamp_t t) -> Vector3d {
      return direction * 0.05 * t * t;
    };
  }

  static auto make_orientation_signal(std::mt19937& rgen) {
    Vector3d axis = perturbate(Vector3d::Zero().eval(), 1, rgen).normalized();
    return [axis](timestamp_t t) -> Quaterniond {
      auto const theta = 0.05 * t * t;
      return Quaterniond(Eigen::AngleAxisd(theta, axis));
    };
  }

  static auto evaluate_truth_integration_delta(
    pose_signal_t pose_signal, timestamp_t t_s, timestamp_t t_e) {
    auto integration_truth = make_imu_preintegration(pose_signal, t_s, t_e);
    auto y_q = integration_truth->rotation_delta;
    auto y_p = integration_truth->position_delta;
    auto y_v = integration_truth->velocity_delta;

    return std::make_tuple(y_q, y_p, y_v);
  }

  static auto sample_preintegration(
    imu_mockup_sequence_t const& sequence, sensor_statistics_t const& noise,
    Vector3d const& b_a, Vector3d const& b_w) {
    auto n_data = static_cast<int>(sequence.size());
    auto slice = [&](auto a, auto b) { return sequence | views::slice(a, b); };

    auto integration = std::make_unique<IMUPreintegration>(
      Vector3d::Zero(), Vector3d::Zero(),
      imu_noise_t {noise.acc_white_noise, noise.gyr_white_noise});

    for (auto const& [f0, f1] :
         views::zip(slice(0, n_data - 1), slice(1, n_data))) {
      auto const& [t0, d0] = f0;
      auto const& [t1, d1] = f1;
      auto dt = t1 - t0;
      Vector3d a_hat = (d0.measurement.accel + d1.measurement.accel) / 2;
      Vector3d w_hat = (d0.measurement.rotat + d1.measurement.rotat) / 2;
      integration->propagate(dt, a_hat - b_a, w_hat - b_w);
    }
    return integration;
  }

  static auto compute_imu_preintegration_covariance(
    sensor_statistics_t const& noise, pose_signal_t pose_signal) {
    auto imu_sequence =
      generate_imu_data(pose_signal, linspace(0., 1., 200) | ranges::to_vector);
    auto integration = sample_preintegration(imu_sequence, noise, O3x1, O3x1);
    return integration->covariance;
  }

  static auto sample_imu_preintegration_covariance(
    std::mt19937& rgen, sensor_statistics_t const& noise,
    pose_signal_t pose_signal) {
    auto covariance_sampled = Matrix9d::Zero().eval();

#ifdef NDEBUG
    auto constexpr samples = 1000;
#else
    auto constexpr samples = 50;
#endif
    for (auto _ = 0; _ < samples; _++) {
      auto timestamps = linspace(0., 1., 200) | ranges::to_vector;
      auto imu_raw_sequence =
        generate_imu_data(pose_signal, timestamps, rgen, noise);
      auto [y_q, y_p, y_v] =
        evaluate_truth_integration_delta(pose_signal, 0., 1.);

      auto integration =
        sample_preintegration(imu_raw_sequence, noise, O3x1, O3x1);
      auto r = integration->evaluateError(y_q, y_p, y_v, O3x1, O3x1);
      covariance_sampled += r * r.transpose();
    }
    covariance_sampled /= (samples - 1);

    return covariance_sampled;
  }

  TEST_CASE("IMU preintegration covariance computation") {
    auto rgen = std::mt19937(20210428);
    auto pose_signal = pose_signal_t {
      .position = make_position_signal(rgen),
      .orientation = make_orientation_signal(rgen),
    };
    auto noise = sensor_statistics_t {
      .acc_white_noise = 0.05,
      .gyr_white_noise = 0.05,
      .acc_random_walk = 0.,  // ignore bias drift
      .gyr_random_walk = 0.,
      .acc_bias_prior_stddev = 0.1,
      .gyr_bias_prior_stddev = 0.1,
    };

    auto covariance_computed =
      compute_imu_preintegration_covariance(noise, pose_signal);
    auto covariance_sampled =
      sample_imu_preintegration_covariance(rgen, noise, pose_signal);

    CAPTURE(covariance_computed);
    CAPTURE(covariance_sampled);
#ifdef NDEBUG
    CHECK(covariance_computed.isApprox(covariance_sampled, 0.1));
#else
    CHECK(covariance_computed.isApprox(covariance_sampled, 0.2));
#endif
  }

  template <typename evaluator_t>
  static auto differentiate_five_point_stencil(evaluator_t evaluator) {
    using result_t = std::remove_reference_t<decltype(evaluator(0.0))>;

    auto constexpr epsilon = 1e-2;

    auto r1 = evaluator(-2 * epsilon);
    auto r2 = evaluator(-epsilon);
    auto r3 = evaluator(+epsilon);
    auto r4 = evaluator(+2 * epsilon);

    result_t result = -(r1 - 8 * r2 + 8 * r3 - r4) / 12 / epsilon;
    return result;
  }

  static auto make_vector3_unit(int i) {
    auto x = Vector3d::Zero().eval();
    x(i) = 1;
    return x;
  }

  static auto sample_numeric_bias_jacobian(
    imu_mockup_sequence_t const& imu_sequence, sensor_statistics_t const& noise,
    Quaterniond const& y_q, Vector3d const& y_p, Vector3d const& y_v) {
    Eigen::Matrix<double, 9, 6> G_numeric;
    for (int i = 0; i < 3; i++) {
      G_numeric.col(i) = differentiate_five_point_stencil([&](auto h) {
        auto b_a = (h * make_vector3_unit(i)).eval();
        auto integration =
          sample_preintegration(imu_sequence, noise, b_a, O3x1);
        return integration->evaluateError(y_q, y_p, y_v, O3x1, O3x1);
      });
    }

    for (int i = 0; i < 3; i++) {
      G_numeric.col(i + 3) = differentiate_five_point_stencil([&](auto h) {
        auto b_w = (h * make_vector3_unit(i)).eval();
        auto integration =
          sample_preintegration(imu_sequence, noise, O3x1, b_w);
        return integration->evaluateError(y_q, y_p, y_v, O3x1, O3x1);
      });
    }
    return G_numeric;
  }

  TEST_CASE("IMU preintegration bias Jacobian computation") {
    auto rgen = std::mt19937(20210428);
    auto pose_signal = pose_signal_t {
      .position = make_position_signal(rgen),
      .orientation = make_orientation_signal(rgen),
    };

    auto noise = sensor_statistics_t {
      .acc_white_noise = 0.0,  // ignore all noises
      .gyr_white_noise = 0.0,
      .acc_random_walk = 0.0,
      .gyr_random_walk = 0.0,
      .acc_bias_prior_stddev = 0.1,
      .gyr_bias_prior_stddev = 0.1,
    };

    auto timestamps = linspace(0., 1., 1000) | ranges::to_vector;
    auto imu_raw_sequence =
      generate_imu_data(pose_signal, timestamps, rgen, noise);
    auto [y_q, y_p, y_v] =
      evaluate_truth_integration_delta(pose_signal, 0., 1.);

    auto evaluator = [&](auto const& b_a, auto const& b_w) {
      return sample_preintegration(imu_raw_sequence, noise, b_a, b_w);
    };

    auto G_algebraic = evaluator(O3x1, O3x1)->bias_jacobian;
    auto G_numeric =
      sample_numeric_bias_jacobian(imu_raw_sequence, noise, y_q, y_p, y_v);

    CAPTURE(G_numeric);
    CAPTURE(G_algebraic);
    CHECK(G_numeric.isApprox(G_algebraic, 1e-2));
  }
}  // namespace cyclops::measurement
