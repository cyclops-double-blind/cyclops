#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

#include <memory>
#include <tuple>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::estimation {
  struct IMUPropagationUpdateHandler;
  struct StateVariableReadAccessor;
  struct StateVariableWriteAccessor;

  class StateVariableAccessor {
  private:
    std::shared_ptr<StateVariableReadAccessor> _reader;
    std::shared_ptr<StateVariableWriteAccessor> _writer;

  private:
    template <typename value_t>
    using maybe_ref_t = std::optional<std::reference_wrapper<value_t>>;

  public:
    StateVariableAccessor(
      std::shared_ptr<StateVariableReadAccessor> reader,
      std::shared_ptr<StateVariableWriteAccessor> writer);
    ~StateVariableAccessor();

    void reset();
    std::shared_ptr<StateVariableReadAccessor> deriveReader();
    std::shared_ptr<StateVariableWriteAccessor> deriveWriter();

    maybe_ref_t<motion_frame_parameter_block_t> motionFrame(frame_id_t id);
    maybe_ref_t<landmark_parameter_block_t> landmark(landmark_id_t id);

    motion_frame_parameter_blocks_t const& motionFrames() const;
    landmark_parameter_blocks_t const& landmarks() const;
    landmark_positions_t const& mappedLandmarks() const;

    std::optional<std::tuple<timestamp_t, imu_motion_state_t>> propagatedState()
      const;

    static std::unique_ptr<StateVariableAccessor> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<IMUPropagationUpdateHandler> propagator);
  };
}  // namespace cyclops::estimation
