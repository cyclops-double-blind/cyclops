#include "cyclops_tests/data/imu.hpp"
#include "cyclops_tests/random.hpp"
#include "cyclops_tests/range.ipp"
#include "cyclops_tests/signal.ipp"

#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/config.hpp"

namespace cyclops {
  using std::map;
  using std::vector;

  using Eigen::Vector3d;

  using measurement::imu_motion_t;
  using measurement::imu_motions_t;
  using measurement::imu_noise_t;
  using measurement::IMUPreintegration;

  namespace views = ranges::views;

  static auto make_imu_signal(pose_signal_t const& pose_signal) {
    auto a = numeric_second_derivative(pose_signal.position);
    auto q = pose_signal.orientation;

    auto a_b = [a, q](timestamp_t t) {
      auto g = Vector3d(0, 0, 9.81);
      return (q(t).inverse() * (a(t) + g)).eval();
    };
    auto w_b = numeric_derivative(q);
    return std::make_tuple(a_b, w_b);
  }

  template <typename imu_data_gen_t>
  static imu_mockup_sequence_t make_imu_mockup(
    vector<timestamp_t> const& timestamps, imu_data_gen_t&& gen) {
    if (timestamps.empty())
      return {};

    imu_mockup_sequence_t result;

    auto const t_s = timestamps.front();
    auto const t_e = timestamps.back();
    auto const dt = (t_e - t_s) / timestamps.size();
    for (auto const t : timestamps)
      result.emplace(t, gen(t, dt));
    return result;
  }

  imu_mockup_sequence_t generate_imu_data(
    pose_signal_t pose_signal, std::vector<timestamp_t> const& timestamps,
    Vector3d const& bias_acc, Vector3d const& bias_gyr, std::mt19937& rgen,
    sensor_statistics_t const& noise) {
    auto signal = make_imu_signal(pose_signal);
    Vector3d b_a = bias_acc;
    Vector3d b_w = bias_gyr;

    return make_imu_mockup(timestamps, [&](timestamp_t t, timestamp_t dt) {
      auto const& [a, w] = signal;
      auto const a_m =
        perturbate((a(t) + b_a).eval(), noise.acc_white_noise, rgen);
      auto const w_m =
        perturbate((w(t) + b_w).eval(), noise.gyr_white_noise, rgen);

      auto const data_frame = imu_mockup_t {
        .bias_acc = b_a,
        .bias_gyr = b_w,
        .measurement = {t, a_m, w_m},
      };
      b_a = perturbate(b_a, dt * noise.acc_random_walk, rgen);
      b_w = perturbate(b_w, dt * noise.gyr_random_walk, rgen);
      return data_frame;
    });
  }

  imu_mockup_sequence_t generate_imu_data(
    pose_signal_t pose_signal, vector<timestamp_t> const& timestamps,
    std::mt19937& rgen, sensor_statistics_t const& noise) {
    return generate_imu_data(
      pose_signal, timestamps, Vector3d::Zero(), Vector3d::Zero(), rgen, noise);
  }

  imu_mockup_sequence_t generate_imu_data(
    pose_signal_t pose_signal, std::vector<timestamp_t> const& timestamps,
    Vector3d const& bias_acc, Vector3d const& bias_gyr) {
    auto signal = make_imu_signal(pose_signal);

    return make_imu_mockup(timestamps, [&](timestamp_t t, timestamp_t dt) {
      auto const& [a, w] = signal;
      auto const a_m = (a(t) + bias_acc).eval();
      auto const w_m = (w(t) + bias_gyr).eval();
      auto const data_frame = imu_mockup_t {
        .bias_acc = bias_acc,
        .bias_gyr = bias_gyr,
        .measurement = {t, a_m, w_m},
      };
      return data_frame;
    });
  }

  imu_mockup_sequence_t generate_imu_data(
    pose_signal_t pose_signal, vector<timestamp_t> const& timestamps) {
    auto signal = make_imu_signal(pose_signal);
    return make_imu_mockup(timestamps, [&](timestamp_t t, timestamp_t dt) {
      auto const& [a, w] = signal;
      return imu_mockup_t {
        .bias_acc = Vector3d::Zero(),
        .bias_gyr = Vector3d::Zero(),
        .measurement = {t, a(t), w(t)},
      };
    });
  }

  static std::unique_ptr<IMUPreintegration> make_imu_preintegration(
    imu_noise_t const& noise, imu_mockup_sequence_t const& imu_sequence) {
    auto samples = imu_sequence.size();
    if (samples <= 1) {
      return std::make_unique<IMUPreintegration>(
        Vector3d::Zero(), Vector3d::Zero(), noise);
    }

    auto const& [_, i0] = *imu_sequence.begin();
    auto result = std::make_unique<IMUPreintegration>(
      Vector3d::Zero(), Vector3d::Zero(), noise);
    for (auto const& [prev, curr] : views::zip(
           views::slice(imu_sequence, 0, samples - 1),
           views::slice(imu_sequence, 1, samples))) {
      auto const& [t_prev, data_prev] = prev;
      auto const& [t_curr, data_curr] = curr;
      auto const& m_prev = data_prev.measurement;
      auto const& m_curr = data_curr.measurement;

      auto dt = t_curr - t_prev;
      auto a = ((m_prev.accel + m_curr.accel) / 2).eval();
      auto w = ((m_prev.rotat + m_curr.rotat) / 2).eval();
      result->propagate(dt, a, w);
    }
    return result;
  }

  static std::unique_ptr<IMUPreintegration> make_imu_preintegration(
    imu_noise_t const& noise, imu_mockup_sequence_t const& imu_sequence,
    pose_signal_t pose_signal, timestamp_t t_s, timestamp_t t_e) {
    auto [p, q] = pose_signal;
    auto v = numeric_derivative(p);
    auto p_s = p(t_s);
    auto v_s = v(t_s);
    auto q_s = q(t_s);

    auto p_e = p(t_e);
    auto v_e = v(t_e);
    auto q_e = q(t_e);

    auto dt = t_e - t_s;
    auto g = Vector3d(0, 0, 9.81);

    auto y_q = q_s.conjugate() * q_e;
    auto y_p =
      (q_s.conjugate() * (p_e - p_s - v_s * dt + 0.5 * g * dt * dt)).eval();
    auto y_v = (q_s.conjugate() * (v_e - v_s + g * dt)).eval();

    auto data = make_imu_preintegration(noise, imu_sequence);
    data->rotation_delta = y_q;
    data->position_delta = y_p;
    data->velocity_delta = y_v;
    data->time_delta = dt;
    return data;
  }

  static auto make_imu_timestamps(timestamp_t t_s, timestamp_t t_e) {
    auto timedelta = t_e - t_s;
    auto samples = std::max(20, static_cast<int>(timedelta * 200));
    return linspace(t_s, t_e, samples) | ranges::to_vector;
  }

  std::unique_ptr<IMUPreintegration> make_imu_preintegration(
    std::mt19937& rgen, sensor_statistics_t const& noise,
    Vector3d const& bias_acc, Vector3d const& bias_gyr,
    pose_signal_t pose_signal, timestamp_t t_s, timestamp_t t_e) {
    auto imu_sequence = generate_imu_data(
      pose_signal, make_imu_timestamps(t_s, t_e), bias_acc, bias_gyr, rgen,
      noise);
    return make_imu_preintegration(
      imu_noise_t {
        .acc_white_noise = noise.acc_white_noise,
        .gyr_white_noise = noise.gyr_white_noise,
      },
      imu_sequence);
  }

  std::unique_ptr<IMUPreintegration> make_imu_preintegration(
    std::mt19937& rgen, sensor_statistics_t const& noise,
    pose_signal_t pose_signal, timestamp_t t_s, timestamp_t t_e) {
    auto imu_sequence = generate_imu_data(
      pose_signal, make_imu_timestamps(t_s, t_e), rgen, noise);
    return make_imu_preintegration(
      imu_noise_t {
        .acc_white_noise = noise.acc_white_noise,
        .gyr_white_noise = noise.gyr_white_noise,
      },
      imu_sequence);
  }

  std::unique_ptr<IMUPreintegration> make_imu_preintegration(
    Vector3d const& bias_acc, Vector3d const& bias_gyr,
    pose_signal_t pose_signal, timestamp_t t_s, timestamp_t t_e) {
    auto imu_sequence = generate_imu_data(
      pose_signal, make_imu_timestamps(t_s, t_e), bias_acc, bias_gyr);
    return make_imu_preintegration(imu_noise_t {1e-3, 1e-3}, imu_sequence);
  }

  std::unique_ptr<IMUPreintegration> make_imu_preintegration(
    sensor_statistics_t const& noise, pose_signal_t pose_signal,
    timestamp_t t_s, timestamp_t t_e) {
    auto imu_noise = imu_noise_t {
      .acc_white_noise = noise.acc_white_noise,
      .gyr_white_noise = noise.gyr_white_noise,
    };
    return make_imu_preintegration(
      imu_noise, generate_imu_data(pose_signal, make_imu_timestamps(t_s, t_e)),
      pose_signal, t_s, t_e);
  }

  std::unique_ptr<IMUPreintegration> make_imu_preintegration(
    pose_signal_t pose_signal, timestamp_t t_s, timestamp_t t_e) {
    auto imu_sequence =
      generate_imu_data(pose_signal, make_imu_timestamps(t_s, t_e));
    auto noise = imu_noise_t {1e-3, 1e-3};
    return make_imu_preintegration(noise, imu_sequence, pose_signal, t_s, t_e);
  }

  template <typename preintegrator_t>
  static auto make_imu_motions(
    map<frame_id_t, timestamp_t> const& frames,
    preintegrator_t const& preintegrator) {
    auto n = frames.size();
    auto prevs = views::slice(frames, 0, n - 1);
    auto currs = views::slice(frames, 1, n);

    auto transform = views::transform([&](auto const& frame_pair) {
      auto const& [s, e] = frame_pair;
      auto const& [id_s, t_s] = s;
      auto const& [id_e, t_e] = e;
      return imu_motion_t {
        .from = id_s,
        .to = id_e,
        .data = preintegrator(t_s, t_e),
      };
    });
    return views::zip(prevs, currs) | transform | ranges::to<imu_motions_t>;
  }

  imu_motions_t make_imu_motions(
    pose_signal_t pose_signal, map<frame_id_t, timestamp_t> const& frames) {
    return make_imu_motions(frames, [&](auto t_s, auto t_e) {
      return make_imu_preintegration(pose_signal, t_s, t_e);
    });
  }

  imu_motions_t make_imu_motions(
    sensor_statistics_t const& noise, pose_signal_t pose_signal,
    map<frame_id_t, timestamp_t> const& frames) {
    return make_imu_motions(frames, [&](auto t_s, auto t_e) {
      return make_imu_preintegration(noise, pose_signal, t_s, t_e);
    });
  }

  imu_motions_t make_imu_motions(
    Vector3d const& b_a, Vector3d const& b_w, pose_signal_t pose_signal,
    map<frame_id_t, timestamp_t> const& frames) {
    return make_imu_motions(frames, [&](auto t_s, auto t_e) {
      return make_imu_preintegration(b_a, b_w, pose_signal, t_s, t_e);
    });
  }

  imu_motions_t make_imu_motions(
    std::mt19937& rgen, sensor_statistics_t const& noise,
    pose_signal_t pose_signal, map<frame_id_t, timestamp_t> const& frames) {
    return make_imu_motions(frames, [&](auto t_s, auto t_e) {
      return make_imu_preintegration(rgen, noise, pose_signal, t_s, t_e);
    });
  }

  imu_motions_t make_imu_motions(
    std::mt19937& rgen, sensor_statistics_t const& noise,
    Vector3d const& bias_acc, Vector3d const& bias_gyr,
    pose_signal_t pose_signal,
    std::map<frame_id_t, timestamp_t> const& frames) {
    return make_imu_motions(frames, [&](auto t_s, auto t_e) {
      return make_imu_preintegration(
        rgen, noise, bias_acc, bias_gyr, pose_signal, t_s, t_e);
    });
  }
}  // namespace cyclops
