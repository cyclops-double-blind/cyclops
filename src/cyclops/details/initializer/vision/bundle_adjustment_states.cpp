#include "cyclops/details/initializer/vision/bundle_adjustment_states.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include <range/v3/all.hpp>

namespace cyclops::initializer {
  namespace views = ranges::views;

  BundleAdjustmentCameraMotionStateBlock::
    BundleAdjustmentCameraMotionStateBlock(se3_transform_t const& guess) {
    orientation() = guess.rotation;
    position() = guess.translation;
  }

  Eigen::Map<Eigen::Quaterniond>
  BundleAdjustmentCameraMotionStateBlock::orientation(double* data) {
    return Eigen::Map<Eigen::Quaterniond>(data);
  }

  Eigen::Map<Eigen::Quaterniond const>
  BundleAdjustmentCameraMotionStateBlock::orientation(double const* data) {
    return Eigen::Map<Eigen::Quaterniond const>(data);
  }

  Eigen::Map<Eigen::Vector3d> BundleAdjustmentCameraMotionStateBlock::position(
    double* data) {
    return Eigen::Map<Eigen::Vector3d>(data + 4);
  }

  Eigen::Map<Eigen::Vector3d const>
  BundleAdjustmentCameraMotionStateBlock::position(double const* data) {
    return Eigen::Map<Eigen::Vector3d const>(data + 4);
  }

  Eigen::Map<Eigen::Quaterniond>
  BundleAdjustmentCameraMotionStateBlock::orientation() {
    return orientation(data());
  }

  Eigen::Map<Eigen::Quaterniond const>
  BundleAdjustmentCameraMotionStateBlock::orientation() const {
    return orientation(data());
  }

  Eigen::Map<Eigen::Vector3d>
  BundleAdjustmentCameraMotionStateBlock::position() {
    return position(data());
  }

  Eigen::Map<Eigen::Vector3d const>
  BundleAdjustmentCameraMotionStateBlock::position() const {
    return position(data());
  }

  double* BundleAdjustmentCameraMotionStateBlock::data() {
    return _data_block.data();
  }

  double const* BundleAdjustmentCameraMotionStateBlock::data() const {
    return _data_block.data();
  }

  se3_transform_t BundleAdjustmentCameraMotionStateBlock::asSE3Transform()
    const {
    return se3_transform_t {
      .translation = position(),
      .rotation = orientation(),
    };
  }

  BundleAdjustmentLandmarkPositionStateBlock::
    BundleAdjustmentLandmarkPositionStateBlock(Eigen::Vector3d const& guess) {
    position() = guess;
  }

  Eigen::Map<Eigen::Vector3d>
  BundleAdjustmentLandmarkPositionStateBlock::position(double* data) {
    return Eigen::Map<Eigen::Vector3d>(data);
  }

  Eigen::Map<Eigen::Vector3d const>
  BundleAdjustmentLandmarkPositionStateBlock::position(double const* data) {
    return Eigen::Map<Eigen::Vector3d const>(data);
  }

  Eigen::Map<Eigen::Vector3d>
  BundleAdjustmentLandmarkPositionStateBlock::position() {
    return position(data());
  }

  Eigen::Map<Eigen::Vector3d const>
  BundleAdjustmentLandmarkPositionStateBlock::position() const {
    return position(data());
  }

  double* BundleAdjustmentLandmarkPositionStateBlock::data() {
    return _data_block.data();
  }

  double const* BundleAdjustmentLandmarkPositionStateBlock::data() const {
    return _data_block.data();
  }

  Eigen::Vector3d BundleAdjustmentLandmarkPositionStateBlock::asVector3()
    const {
    return position();
  }

  template <typename transform_t>
  static auto value_transform(transform_t tf) {
    return views::transform([=](auto const& key_value) {
      auto const& [key, value] = key_value;
      return std::make_pair(key, tf(value));
    });
  }

  BundleAdjustmentOptimizationState::BundleAdjustmentOptimizationState(
    multiview_geometry_t const& guess)
      : camera_motions(
          guess.camera_motions |
          value_transform([](auto const& _) { return MotionBlock(_); }) |
          ranges::to<std::map<frame_id_t, MotionBlock>>),
        landmark_positions(
          guess.landmarks |
          value_transform([](auto const& _) { return LandmarkBlock(_); }) |
          ranges::to<std::map<landmark_id_t, LandmarkBlock>>) {
  }

  using MotionBlockRef = BundleAdjustmentOptimizationState::MotionBlockRef;
  using MotionBlock = BundleAdjustmentOptimizationState::MotionBlock;

  static std::optional<std::tuple<MotionBlockRef, MotionBlockRef, double>>
  find_farthest_from_initial_frame(
    std::map<frame_id_t, MotionBlock>& camera_motions) {
    if (camera_motions.size() < 2)
      return std::nullopt;

    auto i_0 = camera_motions.begin();
    auto& [x_0_id, x_0] = *i_0;

    auto s_max = std::numeric_limits<double>::lowest();
    auto i_max = i_0;

    for (auto i = std::next(i_0); i != camera_motions.end(); i++) {
      auto const& [_, x_i] = *i;
      auto s_i = (x_i.position() - x_0.position()).norm();
      if (s_i > s_max) {
        s_max = s_i;
        i_max = i;
      }
    }
    if (s_max < 1e-6)
      return std::nullopt;

    auto& [x_n_id, x_n] = *i_max;
    return std::make_tuple(MotionBlockRef(x_0), MotionBlockRef(x_n), s_max);
  }

  std::optional<BundleAdjustmentOptimizationState::MotionBlockRefPair>
  BundleAdjustmentOptimizationState::normalize() {
    auto maybe_pair = find_farthest_from_initial_frame(camera_motions);
    if (!maybe_pair)
      return std::nullopt;

    auto& [x_0, x_n, s] = *maybe_pair;
    auto p_0 = Eigen::Vector3d(x_0.get().position());
    auto q_0 = Eigen::Quaterniond(x_0.get().orientation());

    for (auto& [_, f] : landmark_positions) {
      auto f_ = (q_0.conjugate() * (f.position() - p_0) / s).eval();
      f.position() = f_;
    }
    for (auto& [_, motion] : camera_motions) {
      auto p_ = (q_0.conjugate() * (motion.position() - p_0) / s).eval();
      auto q_ = q_0.conjugate() * motion.orientation();

      motion.position() = p_;
      motion.orientation() = q_;
    }
    return std::make_tuple(x_0, x_n);
  }

  multiview_geometry_t
  BundleAdjustmentOptimizationState::as_multi_view_geometry() const {
    return {
      .camera_motions = camera_motions |
        value_transform([](auto const& _) { return _.asSE3Transform(); }) |
        ranges::to<std::map<frame_id_t, se3_transform_t>>,
      .landmarks = landmark_positions |
        value_transform([](auto const& _) { return _.asVector3(); }) |
        ranges::to<landmark_positions_t>,
    };
  }
}  // namespace cyclops::initializer
