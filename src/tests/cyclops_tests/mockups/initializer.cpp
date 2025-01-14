#include "cyclops_tests/mockups/initializer.hpp"
#include "cyclops_tests/random.hpp"
#include "cyclops_tests/signal.ipp"

#include <range/v3/all.hpp>

namespace cyclops::estimation {
  namespace views = ranges::views;

  using estimation::landmark_parameter_blocks_t;
  using estimation::motion_frame_parameter_blocks_t;

  static auto make_local_landmarks(
    landmark_positions_t const& landmarks, pose_signal_t const& pose_signal,
    std::map<frame_id_t, timestamp_t> frame_timestamps) {
    if (frame_timestamps.empty())
      return landmark_positions_t {};

    auto [_, t0] = *frame_timestamps.begin();
    auto x0 = pose_signal.evaluate(t0);

    return  //
      landmarks | views::transform([&](auto const& pair) {
        auto const& [landmark_id, f] = pair;
        auto const& [p0, q0] = x0;

        auto f_bar = (q0.conjugate() * (f - p0)).eval();
        return std::make_pair(landmark_id, f_bar);
      }) |
      ranges::to<landmark_positions_t>;
  }

  static auto make_motion_state_lookup(
    pose_signal_t const& pose_signal,
    std::map<frame_id_t, timestamp_t> frame_timestamps) {
    if (frame_timestamps.empty())
      return std::map<frame_id_t, imu_motion_state_t> {};

    auto [_, t0] = *frame_timestamps.begin();
    auto x0 = pose_signal.evaluate(t0);
    auto const& p0 = x0.translation;
    auto const& q0 = x0.rotation;

    auto const& p = pose_signal.position;
    auto const& q = pose_signal.orientation;
    auto v = numeric_derivative(p);

    return  //
      frame_timestamps | views::transform([&](auto pair) {
        auto const& [frame_id, t] = pair;
        auto x = imu_motion_state_t {
          .orientation = q0.conjugate() * q(t),
          .position = q0.conjugate() * (p(t) - p0),
          .velocity = q0.conjugate() * v(t),
        };
        return std::make_pair(frame_id, x);
      }) |
      ranges::to<std::map<frame_id_t, imu_motion_state_t>>;
  }

  OptimizerSolutionGuessPredictorMock::OptimizerSolutionGuessPredictorMock(
    std::shared_ptr<std::mt19937> rgen, landmark_positions_t const& landmarks,
    pose_signal_t const& pose_signal,
    std::map<frame_id_t, timestamp_t> frame_timestamps)
      : _rgen(rgen),
        _landmarks(
          make_local_landmarks(landmarks, pose_signal, frame_timestamps)),
        _motions(make_motion_state_lookup(pose_signal, frame_timestamps)) {
  }

  void OptimizerSolutionGuessPredictorMock::reset() {
    // does nothing.
  }

  std::optional<OptimizerSolutionGuessPredictor::solution_t>
  OptimizerSolutionGuessPredictorMock::solve() {
    if (_motions.empty())
      return std::nullopt;

    auto initial_frame_id = _motions.begin()->first;
    auto motions =
      _motions | views::transform([&](auto const& id_motion) {
        auto const& [id, motion] = id_motion;
        double const perturbation = id == initial_frame_id ? 0. : 0.05;
        return std::make_pair(
          id, make_perturbated_frame_state(motion, perturbation, *_rgen));
      }) |
      ranges::to<motion_frame_parameter_blocks_t>;

    auto landmarks =
      _landmarks | views::transform([&](auto const& id_landmark) {
        auto const& [id, landmark] = id_landmark;
        return std::make_pair(
          id, make_perturbated_landmark_state(landmark, 0.05, *_rgen));
      }) |
      ranges::to<landmark_parameter_blocks_t>;

    return solution_t {motions, landmarks};
  }
}  // namespace cyclops::estimation
