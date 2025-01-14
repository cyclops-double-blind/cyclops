#include "cyclops/details/utils/vision.hpp"
#include <range/v3/all.hpp>

namespace cyclops {
  namespace views = ranges::views;

  using Eigen::Matrix3d;
  using Eigen::Vector2d;
  using Eigen::Vector3d;

  using Matrix34d = Eigen::Matrix<double, 3, 4>;
  using MatrixX4d = Eigen::Matrix<double, Eigen::Dynamic, 4>;

  using measurement::feature_track_t;
  using measurement::feature_tracks_t;

  image_frame_motion_statistics_t compute_image_frame_motion_statistics(
    feature_tracks_t const& tracks, frame_id_t frame1, frame_id_t frame2) {
    image_frame_motion_statistics_t result = {0, 0, 0};

    double parallax_sum = 0;
    for (auto const& [feature_id, track] : tracks) {
      auto i2 = track.find(frame2);
      if (i2 == track.end())
        continue;

      auto i1 = track.find(frame1);
      if (i1 == track.end()) {
        result.new_features++;
        continue;
      }

      result.common_features++;
      auto const& f1 = i1->second.point;
      auto const& f2 = i2->second.point;

      result.average_parallax += (f2 - f1).norm();
    }
    if (result.common_features != 0)
      result.average_parallax /= result.common_features;
    return result;
  }

  image_frame_motion_statistics_t compute_image_frame_motion_statistics(
    std::map<landmark_id_t, feature_point_t> const& frame1,
    std::map<landmark_id_t, feature_point_t> const& frame2) {
    image_frame_motion_statistics_t result = {0, 0, 0};

    int new_landmarks = 0;
    for (auto const& [landmark_id, f2] : frame2) {
      auto i = frame1.find(landmark_id);
      if (i == frame1.end()) {
        result.new_features++;
        continue;
      }
      result.common_features++;

      auto const& [_, f1] = *i;
      result.average_parallax += (f2.point - f1.point).norm();
    }

    if (result.common_features != 0)
      result.average_parallax /= result.common_features;
    return result;
  }

  static Matrix34d camera_projection(Matrix3d const& R, Vector3d const& p) {
    Matrix34d result;
    result << R.transpose(), -R.transpose() * p;
    return result;
  }

  using camera_pose_t = rotation_translation_matrix_pair_t;
  using camera_pose_lookup_t = std::map<frame_id_t, camera_pose_t>;

  std::optional<Vector3d> triangulate_point(
    feature_track_t const& track,
    camera_pose_lookup_t const& camera_pose_lookup) {
    if (track.size() < 2)
      return std::nullopt;

    int const n = track.size();

    MatrixX4d A(2 * n, 4);
    for (auto const& [i, id_feature] : views::enumerate(track)) {
      auto const& [id, feature] = id_feature;
      auto const& [R, p] = camera_pose_lookup.at(id);
      auto const& f = feature.point;

      auto const x = f.x();
      auto const y = f.y();
      auto const P = camera_projection(R, p);

      A.row(2 * i) = x * P.row(2) - P.row(0);
      A.row(2 * i + 1) = y * P.row(2) - P.row(1);
    }

    Eigen::JacobiSVD<MatrixX4d> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    auto const& X = svd.matrixV().col(3);
    if (std::abs(X.w()) < 1e-6)
      return std::nullopt;
    return X.head<3>() / X.w();
  }

  std::map<landmark_id_t, std::tuple<Vector2d, Vector2d>>
  make_two_view_feature_pairs(
    std::map<landmark_id_t, feature_point_t> const& frame1,
    std::map<landmark_id_t, feature_point_t> const& frame2) {
    std::map<landmark_id_t, std::tuple<Vector2d, Vector2d>> result;

    for (auto const& [landmark_id, f1] : frame1) {
      auto i = frame2.find(landmark_id);
      if (i == frame2.end())
        continue;

      auto const& f2 = i->second;
      result.emplace(landmark_id, std::make_tuple(f1.point, f2.point));
    }
    return result;
  }
}  // namespace cyclops
