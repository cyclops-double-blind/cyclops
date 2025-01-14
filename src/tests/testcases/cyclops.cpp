#include "cyclops_tests/data/imu.hpp"
#include "cyclops_tests/data/landmark.hpp"

#include "cyclops_tests/default.hpp"
#include "cyclops_tests/range.ipp"
#include "cyclops_tests/signal.ipp"

#include "cyclops/details/utils/type.hpp"
#include "cyclops/details/utils/math.hpp"
#include "cyclops/cyclops.hpp"

#include <variant>

#include <doctest/doctest.h>

namespace cyclops {
  using Eigen::Matrix3d;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  using sensor_data_t = std::variant<imu_mockup_t, image_data_t>;
  using timestamped_sensor_data_t = std::tuple<timestamp_t, sensor_data_t>;

  static auto make_sensor_data(
    std::mt19937& rgen, pose_signal_t const& pose_signal,
    se3_transform_t const& extrinsic, sensor_statistics_t const& noise,
    landmark_positions_t const& landmarks) {
    auto imu_timestamps = linspace(0, 10, 1000) | ranges::to_vector;
    auto imu_data = generate_imu_data(pose_signal, imu_timestamps, rgen, noise);

    auto landmark_timestamps = linspace(0, 10, 100) | ranges::to_vector;
    auto landmark_data = make_landmark_frames(
      pose_signal, extrinsic, landmarks, landmark_timestamps);

    std::vector<timestamped_sensor_data_t> result;
    for (auto const& [timestamp, data] : imu_data)
      result.emplace_back(std::make_tuple(timestamp, data));
    for (auto const& data : landmark_data)
      result.emplace_back(std::make_tuple(data.timestamp, data));

    std::sort(result.begin(), result.end(), [](auto const& a, auto const& b) {
      auto const& [t_a, _a] = a;
      auto const& [t_b, _b] = b;
      return t_a < t_b;
    });
    return result;
  }

  static auto make_se3_transform(imu_motion_state_t const& x) {
    return se3_transform_t {
      .translation = x.position,
      .rotation = x.orientation,
    };
  }

  static auto project_so3(Matrix3d const& C) -> Matrix3d {
    Eigen::JacobiSVD<Matrix3d> svd(
      C, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();
    Matrix3d S = U * V.transpose();

    return S.determinant() > 0
      ? S
      : U * Vector3d(1, 1, -1).asDiagonal() * V.transpose();
  }

  static auto determine_slam_origin_frame(
    pose_signal_t const& pose_signal,
    std::map<timestamp_t, imu_motion_state_t> const& slam_result) {
    Matrix3d rotation_sum = Matrix3d::Zero();
    Vector3d position_sum = Vector3d::Zero();

    for (auto const& [t, x_s] : slam_result) {
      auto x_w = se3_transform_t {
        .translation = pose_signal.position(t),
        .rotation = pose_signal.orientation(t),
      };
      auto x_ws = compose(x_w, inverse(make_se3_transform(x_s)));

      rotation_sum += x_ws.rotation.matrix();
      position_sum += x_ws.translation;
    }

    Vector3d position = position_sum / slam_result.size();
    Matrix3d rotation = project_so3(rotation_sum);

    return se3_transform_t {
      .translation = position,
      .rotation = Quaterniond(rotation),
    };
  }

  class CyclopsUpdateContext {
  private:
    std::shared_ptr<CyclopsMain> _cyclops;
    std::map<timestamp_t, imu_motion_state_t> _estimations;

  public:
    explicit CyclopsUpdateContext(std::shared_ptr<CyclopsMain> cyclops)
        : _cyclops(cyclops) {
    }

    void update(sensor_data_t const& data) {
      auto visitor = overloaded {
        [&](imu_mockup_t const& data) {
          _cyclops->enqueueIMUData(data.measurement);
        },
        [&](image_data_t const& data) {
          _cyclops->enqueueLandmarkData(data);

          auto update = _cyclops->updateEstimation();
          auto estimations = _cyclops->motions();

          for (auto [frame_id, timestamp] : update.update_handles) {
            auto maybe_x = estimations.find(frame_id);
            if (maybe_x == estimations.end())
              continue;

            auto const& [_, x] = *maybe_x;
            _estimations.emplace(timestamp, x.motion_state);
          }
        },
      };
      std::visit(visitor, data);
    }

    auto const& estimations() const {
      return _estimations;
    }
  };

  TEST_CASE("The whole pipeline") {
    auto pose_signal = pose_signal_t {
      .position = [](auto t) { return Vector3d(3 * (1 - std::cos(t)), 0, 0); },
      .orientation = yaw_rotation([](auto t) { return atan2(1, cos(t)); }),
    };
    std::mt19937 rgen(20230119001);

    auto config = make_default_config();
    auto const& extrinsic = config->extrinsics.imu_camera_transform;
    auto const& noise = config->noise;

    auto landmarks = generate_landmarks(
      rgen,
      landmark_generation_argument_t {
        200, Vector3d(3, 3, 0), 2 * Matrix3d::Identity()});
    auto sensor_data =
      make_sensor_data(rgen, pose_signal, extrinsic, noise, landmarks);

    auto context = CyclopsUpdateContext(CyclopsMain::create({config}));
    for (auto const& [_, data] : sensor_data)
      context.update(data);

    auto const& estimations = context.estimations();
    REQUIRE(estimations.size() >= 2);
    auto x_ws = determine_slam_origin_frame(pose_signal, estimations);

    THEN("The final state is effectively correct") {
      auto [t_f, x_s] = *estimations.rbegin();

      auto q_true = pose_signal.orientation(t_f);
      auto p_true = pose_signal.position(t_f);

      auto x_got = compose(x_ws, make_se3_transform(x_s));
      auto const& p_got = x_got.translation;
      auto const& q_got = x_got.rotation;

      CAPTURE(p_got.transpose());
      CAPTURE(p_true.transpose());
      CHECK(p_got.isApprox(p_true, 5e-2));

      CAPTURE(q_got.coeffs().transpose());
      CAPTURE(q_true.coeffs().transpose());
      auto error = Eigen::AngleAxisd(q_got.conjugate() * q_true);

      auto constexpr _1_degree = M_PI / 180;
      CHECK(error.angle() < _1_degree);
    }
  }
}  // namespace cyclops
