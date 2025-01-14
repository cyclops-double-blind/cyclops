#pragma once

#include "cyclops/details/estimation/optimizer_guess.hpp"
#include "cyclops_tests/signal.hpp"

#include <random>
#include <memory>
#include <map>
#include <optional>

namespace cyclops::estimation {
  class OptimizerSolutionGuessPredictorMock:
      public OptimizerSolutionGuessPredictor {
  private:
    std::shared_ptr<std::mt19937> _rgen;
    landmark_positions_t _landmarks;
    std::map<frame_id_t, imu_motion_state_t> _motions;

  public:
    OptimizerSolutionGuessPredictorMock(
      std::shared_ptr<std::mt19937> rgen, landmark_positions_t const& landmarks,
      pose_signal_t const& pose_signal,
      std::map<frame_id_t, timestamp_t> frame_timestamps);

    void reset() override;

    std::optional<solution_t> solve() override;
  };
}  // namespace cyclops::estimation
