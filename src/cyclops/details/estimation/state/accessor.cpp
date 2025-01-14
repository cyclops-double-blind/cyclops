#include "cyclops/details/estimation/state/accessor.hpp"
#include "cyclops/details/estimation/state/accessor_read.hpp"
#include "cyclops/details/estimation/state/accessor_write.hpp"
#include "cyclops/details/estimation/state/state_internal.hpp"

namespace cyclops::estimation {
  using std::shared_ptr;

  StateVariableAccessor::StateVariableAccessor(
    std::shared_ptr<StateVariableReadAccessor> reader,
    std::shared_ptr<StateVariableWriteAccessor> writer)
      : _reader(reader), _writer(writer) {
  }

  StateVariableAccessor::~StateVariableAccessor() = default;

  std::shared_ptr<StateVariableReadAccessor>
  StateVariableAccessor::deriveReader() {
    return _reader;
  }

  std::shared_ptr<StateVariableWriteAccessor>
  StateVariableAccessor::deriveWriter() {
    return _writer;
  }

  void StateVariableAccessor::reset() {
    _writer->reset();
  }

  StateVariableAccessor::maybe_ref_t<motion_frame_parameter_block_t>
  StateVariableAccessor::motionFrame(frame_id_t id) {
    return _writer->motionFrame(id);
  }

  StateVariableAccessor::maybe_ref_t<landmark_parameter_block_t>
  StateVariableAccessor::landmark(landmark_id_t id) {
    return _writer->landmark(id);
  }

  motion_frame_parameter_blocks_t const& StateVariableAccessor::motionFrames()
    const {
    return _reader->motionFrames();
  }

  landmark_parameter_blocks_t const& StateVariableAccessor::landmarks() const {
    return _reader->landmarks();
  }

  landmark_positions_t const& StateVariableAccessor::mappedLandmarks() const {
    return _reader->mappedLandmarks();
  }

  std::optional<std::tuple<timestamp_t, imu_motion_state_t>>
  StateVariableAccessor::propagatedState() const {
    return _reader->propagatedState();
  }

  std::unique_ptr<StateVariableAccessor> StateVariableAccessor::create(
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<IMUPropagationUpdateHandler> propagator) {
    shared_ptr state_internal = std::make_shared<StateVariableInternal>();

    return std::make_unique<StateVariableAccessor>(
      std::make_shared<StateVariableReadAccessor>(state_internal, propagator),
      std::make_shared<StateVariableWriteAccessor>(state_internal));
  }
}  // namespace cyclops::estimation
