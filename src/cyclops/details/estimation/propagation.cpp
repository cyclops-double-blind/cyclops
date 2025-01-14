#include "cyclops/details/estimation/propagation.hpp"

#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/config.hpp"

#include <cmath>
#include <functional>
#include <iterator>

namespace cyclops::estimation {
  using Eigen::Vector3d;

  using measurement::imu_noise_t;
  using measurement::IMUPreintegration;

  class IMUPropagationUpdateHandlerImpl: public IMUPropagationUpdateHandler {
  private:
    std::shared_ptr<cyclops_global_config_t const> _config;

    struct propagation_state_t {
      timestamp_t timestamp;

      imu_motion_state_t motion_state;
      Vector3d bias_acc;
      Vector3d bias_gyr;

      IMUPreintegration integrator;

      propagation_state_t(
        timestamp_t timestamp, imu_motion_state_t const& motion_state,
        Vector3d const& b_a, Vector3d const& b_w, imu_noise_t const& noise);
    };

    std::unique_ptr<propagation_state_t> _propagation_state;
    std::map<timestamp_t, imu_data_t> _imu_queue;

    void propagateIMU(imu_data_t const& imu_prev, imu_data_t const& imu_next);

    std::map<timestamp_t, imu_data_t>::const_iterator
    findIMUQueuePointRightBefore(timestamp_t time) const;

  public:
    explicit IMUPropagationUpdateHandlerImpl(
      std::shared_ptr<cyclops_global_config_t const> config);
    ~IMUPropagationUpdateHandlerImpl();
    void reset() override;

    void updateOptimization(
      timestamp_t last_timestamp,
      motion_frame_parameter_block_t const& last_state) override;
    void updateIMUData(imu_data_t const& data) override;

    using timestamped_motion_state_t =
      std::tuple<timestamp_t, imu_motion_state_t>;
    std::optional<timestamped_motion_state_t> get() const override;
  };

  IMUPropagationUpdateHandlerImpl::propagation_state_t::propagation_state_t(
    timestamp_t timestamp, imu_motion_state_t const& motion_state,
    Vector3d const& b_a, Vector3d const& b_w, imu_noise_t const& noise)
      : timestamp(timestamp),
        motion_state(motion_state),
        bias_acc(b_a),
        bias_gyr(b_w),
        integrator(b_a, b_w, noise) {
  }

  template <typename value_t>
  using maybe_cref_t = std::optional<std::reference_wrapper<value_t const>>;

  std::map<timestamp_t, imu_data_t>::const_iterator
  IMUPropagationUpdateHandlerImpl::findIMUQueuePointRightBefore(
    timestamp_t time) const {
    auto i = _imu_queue.upper_bound(time);
    if (i == _imu_queue.begin())
      return _imu_queue.end();
    return std::prev(i);
  }

  static Vector3d interpolate(
    timestamp_t t_eval, timestamp_t t_init, Vector3d const& v_init,
    timestamp_t t_term, Vector3d const& v_term) {
    if (t_term - t_init < 1e-6) {
      if (std::abs(t_term - t_eval) < std::abs(t_init - t_eval))
        return v_term;
      return v_init;
    }

    Vector3d delta = v_term - v_init;
    auto n = t_eval - t_init;
    auto d = t_term - t_init;
    return v_init + std::min(1., std::max(0., n / d)) * delta;
  }

  void IMUPropagationUpdateHandlerImpl::propagateIMU(
    imu_data_t const& imu_prev, imu_data_t const& imu_next) {
    auto const& t_curr = _propagation_state->timestamp;
    auto const& [t_prev, a_prev, w_prev] = imu_prev;
    auto const& [t_next, a_next, w_next] = imu_next;

    auto dt = t_next - t_curr;
    if (dt <= 0)
      return;

    auto a_curr = interpolate(t_curr, t_prev, a_prev, t_next, a_next);
    auto w_curr = interpolate(t_curr, t_prev, w_prev, t_next, w_next);

    Vector3d a_hat = (a_curr + a_next) / 2;
    Vector3d w_hat = (w_curr + w_next) / 2;

    _propagation_state->integrator.propagate(dt, a_hat, w_hat);
    _propagation_state->timestamp = t_next;
  }

  IMUPropagationUpdateHandlerImpl::IMUPropagationUpdateHandlerImpl(
    std::shared_ptr<cyclops_global_config_t const> config)
      : _config(config) {
  }

  IMUPropagationUpdateHandlerImpl::~IMUPropagationUpdateHandlerImpl() = default;

  void IMUPropagationUpdateHandlerImpl::updateOptimization(
    timestamp_t timestamp, motion_frame_parameter_block_t const& state_block) {
    auto motion_state =
      estimation::motion_state_of_motion_frame_block(state_block);
    auto bias_acc = estimation::acc_bias_of_motion_frame_block(state_block);
    auto bias_gyr = estimation::gyr_bias_of_motion_frame_block(state_block);

    _propagation_state = std::make_unique<propagation_state_t>(
      timestamp, motion_state, bias_acc, bias_gyr,
      imu_noise_t {
        .acc_white_noise = _config->noise.acc_white_noise,
        .gyr_white_noise = _config->noise.gyr_white_noise,
      });

    auto i = _imu_queue.upper_bound(timestamp);
    if (i != _imu_queue.begin())
      _imu_queue.erase(_imu_queue.begin(), std::prev(i));
    if (i == _imu_queue.end())
      return;

    for (; std::next(i) != _imu_queue.end(); i++) {
      auto const& [_1, imu_prev] = *i;
      auto const& [_2, imu_next] = *std::next(i);
      propagateIMU(imu_prev, imu_next);
    }
  }

  void IMUPropagationUpdateHandlerImpl::updateIMUData(
    imu_data_t const& imu_next) {
    _imu_queue.emplace_hint(_imu_queue.end(), imu_next.timestamp, imu_next);
    if (_propagation_state == nullptr)
      return;
    auto const& t_curr = _propagation_state->timestamp;

    auto i_prev = findIMUQueuePointRightBefore(t_curr);
    if (i_prev == _imu_queue.end())
      return;
    auto const& imu_prev = i_prev->second;

    propagateIMU(imu_prev, imu_next);
  }

  std::optional<IMUPropagationUpdateHandlerImpl::timestamped_motion_state_t>
  IMUPropagationUpdateHandlerImpl::get() const {
    if (!_propagation_state)
      return std::nullopt;

    auto g = Vector3d(0, 0, _config->gravity_norm);
    auto const& y_q = _propagation_state->integrator.rotation_delta;
    auto const& y_p = _propagation_state->integrator.position_delta;
    auto const& y_v = _propagation_state->integrator.velocity_delta;
    auto const& dt = _propagation_state->integrator.time_delta;

    auto const& [q, p, v] = _propagation_state->motion_state;
    auto propagated_state = imu_motion_state_t {
      .orientation = q * y_q,
      .position = p + v * dt - 0.5 * g * dt * dt + q * y_p,
      .velocity = v - g * dt + q * y_v,
    };
    return std::make_tuple(_propagation_state->timestamp, propagated_state);
  }

  void IMUPropagationUpdateHandlerImpl::reset() {
    _propagation_state = nullptr;
    _imu_queue.clear();
  }

  std::unique_ptr<IMUPropagationUpdateHandler>
  IMUPropagationUpdateHandler::create(
    std::shared_ptr<cyclops_global_config_t const> config) {
    return std::make_unique<IMUPropagationUpdateHandlerImpl>(config);
  }
}  // namespace cyclops::estimation
