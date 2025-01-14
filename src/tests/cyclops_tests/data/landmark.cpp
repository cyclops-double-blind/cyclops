#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/data/typefwd.hpp"
#include "cyclops_tests/random.hpp"

#include "cyclops/details/utils/math.hpp"

#include <range/v3/all.hpp>

namespace cyclops {
  using std::function;
  using std::map;
  using std::set;
  using std::vector;

  using Eigen::Matrix2d;
  using Eigen::Matrix3d;
  using Eigen::Vector2d;
  using Eigen::Vector3d;

  namespace views = ranges::views;

  using measurement::feature_tracks_t;

  static Vector3d uniform_random_vector(std::mt19937& rgen) {
    std::uniform_real_distribution<> rand(-1, 1);
    return Vector3d(rand(rgen), rand(rgen), rand(rgen));
  }

  landmark_positions_t generate_landmarks(
    std::mt19937& rgen, landmark_generation_argument_t const& arg) {
    return generate_landmarks(rgen, landmark_generation_arguments_t {arg});
  }

  landmark_positions_t generate_landmarks(
    std::mt19937& rgen, landmark_generation_arguments_t const& args) {
    std::map<landmark_id_t, Vector3d> result;

    int _id = 0;
    auto id_generator = views::transform([&](auto _) {
      _id += std::uniform_int_distribution<>(1, 2)(rgen);
      return _id;
    });
    for (auto const& arg : args) {
      for (auto const id : views::iota(0, arg.count) | id_generator) {
        auto const& A = arg.concentration;
        auto const& C = arg.center;
        result.emplace(id, C + A * uniform_random_vector(rgen));
      }
    }
    return result;
  }

  landmark_positions_t generate_landmarks(
    set<landmark_id_t> ids, function<Vector3d(landmark_id_t)> gen) {
    return ids |
      views::transform([gen](auto id) { return std::make_pair(id, gen(id)); }) |
      ranges::to<std::map<landmark_id_t, Vector3d>>();
  }

  static auto generate_landmark_observation_range(
    Matrix3d const& R, Vector3d const& p,
    landmark_positions_t const& landmarks) {
    return views::for_each(landmarks, [&](auto const& id_landmark) {
      auto const& [id, landmark] = id_landmark;
      Vector3d const f = R.transpose() * (landmark - p);
      Vector2d const u = f.head<2>() / f.z();

      auto&& contained = [](auto x, auto a, auto b) { return x > a && x < b; };
      return ranges::yield_if(
        f.z() > 1e-3 && contained(u.x(), -1, 1) && contained(u.y(), -1, 1),
        std::make_pair(id, u));
    });
  }

  static Matrix2d make_default_landmark_weight() {
    return Vector2d(2.5e5, 2.5e5).asDiagonal();
  }

  std::map<landmark_id_t, feature_point_t> generate_landmark_observations(
    Matrix3d const& R, Vector3d const& p,
    landmark_positions_t const& landmarks) {
    // clang-format off
    return generate_landmark_observation_range(R, p, landmarks)
      | views::transform([](auto const& id_point) {
          auto const& [id, u] = id_point;
          return std::make_pair(
            id, feature_point_t {u, make_default_landmark_weight()});
        })
      | ranges::to<std::map<landmark_id_t, feature_point_t>>;
    // clang-format on
  }

  std::map<landmark_id_t, feature_point_t> generate_landmark_observations(
    std::mt19937& rgen, Matrix2d const& cov, Matrix3d const& R,
    Vector3d const& p, landmark_positions_t const& landmarks) {
    Matrix2d weight = cov.inverse();
    Matrix2d spread = cov.llt().matrixL();

    // clang-format off
    return generate_landmark_observation_range(R, p, landmarks)
      | views::transform([&](auto const& id_point) {
          auto const& [id, u] = id_point;
          return std::make_pair(
            id, feature_point_t {perturbate(u, spread, rgen), weight});
        })
      | ranges::to<std::map<landmark_id_t, feature_point_t>>;
    // clang-format on
  }

  static auto make_landmark_frame(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const& landmarks, timestamp_t t) {
    auto const [p, q] = pose_signal;
    auto const [p_c, q_c] = compose({p(t), q(t)}, extrinsic);
    return generate_landmark_observations(q_c.matrix(), p_c, landmarks);
  }

  static auto make_landmark_frame(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const& landmarks, timestamp_t t, std::mt19937& rgen,
    Matrix2d const& cov) {
    auto const [p, q] = pose_signal;
    auto const [p_c, q_c] = compose({p(t), q(t)}, extrinsic);
    return generate_landmark_observations(
      rgen, cov, q_c.matrix(), p_c, landmarks);
  }

  vector<image_data_t> make_landmark_frames(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const& landmarks, vector<timestamp_t> const& times) {
    auto transform = views::transform([&](auto timestamp) {
      return image_data_t {
        timestamp,
        make_landmark_frame(pose_signal, extrinsic, landmarks, timestamp)};
    });
    return times | transform | ranges::to_vector;
  }

  vector<image_data_t> make_landmark_frames(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const& landmarks, vector<timestamp_t> const& times,
    std::mt19937& rgen, Matrix2d const& cov) {
    auto transform = views::transform([&](auto timestamp) {
      return image_data_t {
        timestamp,
        make_landmark_frame(
          pose_signal, extrinsic, landmarks, timestamp, rgen, cov)};
    });
    return times | transform | ranges::to_vector;
  }

  static feature_tracks_t make_landmark_tracks(
    map<frame_id_t, timestamp_t> const& frames,
    vector<image_data_t> const& landmark_frames) {
    feature_tracks_t tracks;
    for (auto const& [frame_id, landmark_frame] :
         views::zip(frames | views::keys, landmark_frames)) {
      for (auto const& [feature_id, feature] : landmark_frame.features)
        tracks[feature_id].emplace(frame_id, feature);
    }
    return tracks;
  }

  feature_tracks_t make_landmark_tracks(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const& landmarks,
    map<frame_id_t, timestamp_t> const& frames) {
    auto landmark_frames = make_landmark_frames(
      pose_signal, extrinsic, landmarks,
      frames | views::values | ranges::to_vector);
    return make_landmark_tracks(frames, landmark_frames);
  }

  feature_tracks_t make_landmark_tracks(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const& landmarks,
    map<frame_id_t, timestamp_t> const& frames, std::mt19937& rgen,
    Matrix2d const& cov) {
    auto landmark_frames = make_landmark_frames(
      pose_signal, extrinsic, landmarks,
      frames | views::values | ranges::to_vector, rgen, cov);
    return make_landmark_tracks(frames, landmark_frames);
  }

  map<frame_id_t, map<landmark_id_t, feature_point_t>>
  make_landmark_multiview_observation(
    pose_signal_t pose_signal, se3_transform_t const& extrinsic,
    landmark_positions_t const& landmarks,
    std::map<frame_id_t, timestamp_t> const& frame_times) {
    auto feature_observation_transform = views::transform([&](auto const& _) {
      auto const& [frame_id, timestamp] = _;
      auto x = se3_transform_t {
        .translation = pose_signal.position(timestamp),
        .rotation = pose_signal.orientation(timestamp),
      };
      auto [p_c, q_c] = compose(x, extrinsic);
      auto R_c = q_c.matrix().eval();

      auto features = generate_landmark_observations(R_c, p_c, landmarks);
      return std::make_pair(frame_id, features);
    });

    return frame_times | feature_observation_transform |
      ranges::to<map<frame_id_t, map<landmark_id_t, feature_point_t>>>;
  }
}  // namespace cyclops
