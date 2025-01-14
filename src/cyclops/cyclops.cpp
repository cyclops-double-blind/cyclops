#include "cyclops/cyclops.hpp"

#include "cyclops/details/utils/debug.hpp"

#include "cyclops/details/measurement/data_provider.hpp"
#include "cyclops/details/measurement/data_queue.hpp"
#include "cyclops/details/measurement/data_updater.hpp"
#include "cyclops/details/measurement/keyframe.hpp"
#include "cyclops/details/initializer/initializer.hpp"
#include "cyclops/details/initializer/solver.hpp"
#include "cyclops/details/initializer/vision_imu.hpp"
#include "cyclops/details/initializer/vision.hpp"
#include "cyclops/details/estimation/marginalizer/marginalizer.hpp"
#include "cyclops/details/estimation/state/accessor.hpp"
#include "cyclops/details/estimation/state/accessor_read.hpp"
#include "cyclops/details/estimation/state/accessor_write.hpp"
#include "cyclops/details/estimation/estimator.hpp"
#include "cyclops/details/estimation/optimizer.hpp"
#include "cyclops/details/estimation/optimizer_guess.hpp"
#include "cyclops/details/estimation/propagation.hpp"
#include "cyclops/details/estimation/sanity.hpp"

#include <spdlog/spdlog.h>

#include <deque>
#include <iterator>
#include <mutex>

namespace cyclops {
  using maybe_frame_id_t = std::optional<frame_id_t>;

  using measurement::KeyframeManager;
  using measurement::MeasurementDataProvider;
  using measurement::MeasurementDataQueue;
  using measurement::MeasurementDataUpdater;

  using estimation::EstimationFrameworkMain;
  using estimation::EstimationSanityDiscriminator;
  using estimation::IMUPropagationUpdateHandler;
  using estimation::LikelihoodOptimizer;
  using estimation::MarginalizationManager;
  using estimation::StateVariableAccessor;

  class MainImpl: public CyclopsMain {
  private:
    std::unique_ptr<StateVariableAccessor> _state_accessor;
    std::unique_ptr<MeasurementDataUpdater> _measurement_updater;
    std::unique_ptr<EstimationFrameworkMain> _estimation_core;

    std::shared_ptr<cyclops_global_config_t const> _config;

    std::atomic<bool> _reset_request = false;
    std::mutex mutable _imu_mutex;
    std::mutex mutable _landmark_mutex;

    std::deque<image_data_t> _landmark_buffer;
    timestamp_t _last_imu_timestamp = 0;

    maybe_frame_id_t consumeLandmarkData(image_data_t const& data);

    maybe_frame_id_t insertLandmarkFrame(image_data_t const& data);
    void invokeOptimization();
    void repropagate(frame_id_t frame, timestamp_t timestamp);
    bool checkOptimizerSanity();
    std::optional<image_data_t> popActiveLandmark();

    void reset();
    bool handleResetRequest();

  public:
    MainImpl(
      std::unique_ptr<StateVariableAccessor> state_accessor,
      std::unique_ptr<MeasurementDataUpdater> measurement_updater,
      std::unique_ptr<EstimationFrameworkMain> estimation_core,
      std::shared_ptr<cyclops_global_config_t const> config);

    cyclops_estimation_update_result_t updateEstimation() override;

    void enqueueIMUData(imu_data_t const& imu) override;
    void enqueueLandmarkData(image_data_t const& feature) override;
    void enqueueResetRequest() override;

    landmark_positions_t mappedLandmarks() const override;
    std::map<frame_id_t, cyclops_keyframe_state_t> motions() const override;
    std::optional<cyclops_propagation_state_t> propagation() const override;
  };

  MainImpl::MainImpl(
    std::unique_ptr<StateVariableAccessor> state_accessor,
    std::unique_ptr<MeasurementDataUpdater> measurement_updater,
    std::unique_ptr<EstimationFrameworkMain> estimation_core,
    std::shared_ptr<cyclops_global_config_t const> config)
      : _state_accessor(std::move(state_accessor)),
        _measurement_updater(std::move(measurement_updater)),
        _estimation_core(std::move(estimation_core)),
        _config(config) {
  }

  void MainImpl::enqueueIMUData(imu_data_t const& data) {
    std::scoped_lock _(_imu_mutex);
    _last_imu_timestamp = data.timestamp;
    _measurement_updater->updateImu(data);
  }

  void MainImpl::enqueueLandmarkData(image_data_t const& data) {
    if (data.features.empty())
      return;

    std::scoped_lock _(_landmark_mutex);
    _landmark_buffer.emplace_back(data);
  }

  void MainImpl::enqueueResetRequest() {
    _reset_request = true;
  }

  std::optional<image_data_t> MainImpl::popActiveLandmark() {
    std::scoped_lock _(_imu_mutex, _landmark_mutex);

    auto current = _landmark_buffer.begin();
    if (current == _landmark_buffer.end())
      return std::nullopt;

    auto next = std::next(current);
    if (next == _landmark_buffer.end())
      return std::nullopt;

    auto t_next = next->timestamp;
    auto t_imu_delay = _config->extrinsics.imu_camera_time_delay;
    auto t_imu = _last_imu_timestamp + t_imu_delay;

    if (t_next < t_imu) {
      auto data = std::move(*current);
      _landmark_buffer.pop_front();
      return data;
    }
    return std::nullopt;
  }

  maybe_frame_id_t MainImpl::insertLandmarkFrame(image_data_t const& data) {
    std::scoped_lock _(_imu_mutex, _landmark_mutex);
    return _measurement_updater->updateLandmark(data);
  }

  void MainImpl::invokeOptimization() {
    std::scoped_lock _(_landmark_mutex);
    _estimation_core->updateEstimation();
  }

  void MainImpl::repropagate(frame_id_t frame, timestamp_t timestamp) {
    std::scoped_lock _(_imu_mutex, _landmark_mutex);
    _measurement_updater->repropagate(frame, timestamp);
  }

  bool MainImpl::checkOptimizerSanity() {
    std::scoped_lock _(_landmark_mutex);
    return _estimation_core->sanity();
  }

  maybe_frame_id_t MainImpl::consumeLandmarkData(image_data_t const& data) {
    auto tic = ::cyclops::tic();

    auto maybe_inserted_frame = insertLandmarkFrame(data);
    if (!maybe_inserted_frame)
      return std::nullopt;
    invokeOptimization();
    repropagate(*maybe_inserted_frame, data.timestamp);

    __logger__->debug("Landmark update total time: {}[s]", toc(tic));

    return maybe_inserted_frame;
  }

  bool MainImpl::handleResetRequest() {
    auto initialized = !_state_accessor->motionFrames().empty();
    auto reset_request = _reset_request.load();

    if (!reset_request)
      return false;
    _reset_request = false;

    if (!initialized)
      return false;
    __logger__->error("Reset requested. Trying reset...");

    reset();
    return true;
  }

  cyclops_estimation_update_result_t MainImpl::updateEstimation() {
    cyclops_estimation_update_result_t result = {
      .reset = false, .update_handles = {}};

    while (true) {
      auto maybe_data = popActiveLandmark();
      if (!maybe_data.has_value())
        break;
      auto timestamp = maybe_data->timestamp;

      __logger__->debug("Processing landmark data. Timestamp: {}", timestamp);

      auto maybe_inserted_frame = consumeLandmarkData(*maybe_data);
      if (!maybe_inserted_frame.has_value())
        continue;

      result.update_handles.emplace_back(
        cyclops_image_update_handle_t {*maybe_inserted_frame, timestamp});

      if (handleResetRequest())
        return {.reset = true, .update_handles = {}};

      if (!checkOptimizerSanity()) {
        __logger__->error("Optimizer failure detected. Trying reset...");
        reset();
        return {.reset = true, .update_handles = {}};
      }
    }
    return result;
  }

  void MainImpl::reset() {
    std::scoped_lock _(_imu_mutex, _landmark_mutex);

    _landmark_buffer.clear();
    _last_imu_timestamp = 0;

    _state_accessor->reset();
    _measurement_updater->reset();
    _estimation_core->reset();
  }

  landmark_positions_t MainImpl::mappedLandmarks() const {
    std::scoped_lock _(_landmark_mutex);
    return _state_accessor->mappedLandmarks();
  }

  std::map<frame_id_t, cyclops_keyframe_state_t> MainImpl::motions() const {
    std::scoped_lock _(_landmark_mutex);
    auto frames = _measurement_updater->frames();

    std::map<frame_id_t, cyclops_keyframe_state_t> result;
    for (auto [frame_id, timestamp] : frames) {
      auto maybe_x = _state_accessor->motionFrame(frame_id);
      if (!maybe_x.has_value())
        continue;

      auto keyframe_state = cyclops_keyframe_state_t {
        .timestamp = timestamp,
        .acc_bias = estimation::acc_bias_of_motion_frame_block(*maybe_x),
        .gyr_bias = estimation::gyr_bias_of_motion_frame_block(*maybe_x),
        .motion_state =
          estimation::motion_state_of_motion_frame_block(*maybe_x),
      };
      result.emplace(frame_id, keyframe_state);
    }
    return result;
  }

  std::optional<cyclops_propagation_state_t> MainImpl::propagation() const {
    std::scoped_lock _(_imu_mutex);

    auto maybe_propagation = _state_accessor->propagatedState();
    if (!maybe_propagation.has_value())
      return std::nullopt;

    auto const& [timestamp, motion_state] = *maybe_propagation;
    return cyclops_propagation_state_t {timestamp, motion_state};
  }

  static void initialize_argument_defaults(cyclops_main_argument_t& args) {
    if (!args.seed.has_value())
      args.seed = std::random_device()();

    if (args.optimizer_telemetry == nullptr)
      args.optimizer_telemetry = OptimizerTelemetry::createDefault();

    if (args.keyframe_telemetry == nullptr)
      args.keyframe_telemetry = KeyframeTelemetry::createDefault();

    if (args.initializer_telemetry == nullptr)
      args.initializer_telemetry = InitializerTelemetry::createDefault();
  }

  std::unique_ptr<CyclopsMain> CyclopsMain::create(
    cyclops_main_argument_t args) {
    if (args.config == nullptr)
      return nullptr;
    initialize_argument_defaults(args);
    __logger__->debug("Running with seed: {}", args.seed.value());

    std::shared_ptr propagator =
      IMUPropagationUpdateHandler::create(args.config);

    auto state_accessor =
      StateVariableAccessor::create(args.config, propagator);
    std::shared_ptr state_reader = state_accessor->deriveReader();
    std::shared_ptr state_writer = state_accessor->deriveWriter();

    std::shared_ptr data_provider =
      MeasurementDataProvider::create(args.config, state_reader);
    std::shared_ptr frame_manager =
      KeyframeManager::create(args.keyframe_telemetry);

    std::shared_ptr data_queue = MeasurementDataQueue::create(
      args.config, data_provider, frame_manager, state_reader);

    auto rgen = std::make_shared<std::mt19937>(args.seed.value());
    auto bootstrap_internal = initializer::InitializationSolverInternal::create(
      rgen, args.config, data_provider, args.initializer_telemetry);
    auto bootstrap_solver = initializer::InitializerMain::create(
      std::move(bootstrap_internal), frame_manager, args.initializer_telemetry);

    auto optimizer = LikelihoodOptimizer::create(
      estimation::OptimizerSolutionGuessPredictor::create(
        std::move(bootstrap_solver), args.config, state_reader, data_provider),
      args.config, state_writer, data_provider);
    auto estimation_core = EstimationFrameworkMain::create(
      std::move(optimizer),
      MarginalizationManager::create(args.config, state_reader, data_queue),
      EstimationSanityDiscriminator::create(
        args.config, args.optimizer_telemetry));

    auto cyclops = std::make_unique<MainImpl>(
      std::move(state_accessor),
      MeasurementDataUpdater::create(
        args.config, data_queue, propagator, state_reader),
      std::move(estimation_core), args.config);
    return std::move(cyclops);
  }
}  // namespace cyclops
