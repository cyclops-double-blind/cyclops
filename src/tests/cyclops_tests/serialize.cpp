#include "cyclops_tests/serialize.hpp"

#include <nlohmann/json.hpp>

namespace cyclops {
  using nlohmann::json;

  using frame_estimations_t = std::map<frame_id_t, se3_transform_t>;

  using feature_track_t = std::map<frame_id_t, feature_point_t>;
  using feature_tracks_t = std::map<landmark_id_t, feature_track_t>;

  using Eigen::Quaterniond;
  using Eigen::Vector2d;
  using Eigen::Vector3d;

  static inline json parse(Vector2d const& v) {
    return json::object({
      {"x", v.x()},
      {"y", v.y()},
    });
  }

  static inline json parse(Vector3d const& v) {
    return json::object({
      {"x", v.x()},
      {"y", v.y()},
      {"z", v.z()},
    });
  }

  static inline json parse(Quaterniond const& q) {
    return json::object({
      {"w", q.w()},
      {"x", q.x()},
      {"y", q.y()},
      {"z", q.z()},
    });
  }

  static inline json parse(se3_transform_t const& tf) {
    return json::object({
      {"position", parse(tf.translation)},
      {"orientation", parse(tf.rotation)},
    });
  }

  static inline json parse(frame_estimations_t::value_type const& id_frame) {
    auto const& [frame_id, frame] = id_frame;
    return json::object({
      {"id", frame_id},
      {"pose", parse(frame)},
    });
  }

  static inline json parse(
    landmark_positions_t::value_type const& id_landmark) {
    auto const& [landmark_id, landmark] = id_landmark;
    return json::object({
      {"id", landmark_id},
      {"position", parse(landmark)},
    });
  }

  static json parse(feature_track_t::value_type const& id_feature) {
    auto const& [id, feature] = id_feature;
    return json::object({
      {"frame_id", id},
      {"point", parse(feature.point)},
    });
  }

  static json parse(feature_track_t const& track) {
    json result = json::array();
    for (auto const& id_feature : track)
      result.emplace_back(parse(id_feature));
    return result;
  }

  static json parse(feature_tracks_t::value_type const& id_track) {
    auto const& [id, track] = id_track;
    return json::object({
      {"id", id},
      {"track", parse(track)},
    });
  }

  static inline json parse(frame_estimations_t const& frames) {
    json result = json::array();
    for (auto const& frame : frames)
      result.emplace_back(parse(frame));
    return result;
  }

  static inline json parse(landmark_positions_t const& landmarks) {
    json result = json::array();
    for (auto const& landmark : landmarks)
      result.emplace_back(parse(landmark));
    return result;
  }

  static inline json parse(std::vector<Vector3d> const& vectors) {
    json result = json::array();
    for (auto const& vector : vectors)
      result.emplace_back(parse(vector));
    return result;
  }

  static inline json parse(std::vector<se3_transform_t> const& frames) {
    json result = json::array();
    for (auto const& frame : frames)
      result.emplace_back(parse(frame));
    return result;
  }

  static json parse(feature_tracks_t const& tracks) {
    json result = json::array();
    for (auto const& track : tracks)
      result.emplace_back(parse(track));
    return result;
  }

  std::string serialize(std::vector<Vector3d> const& vectors) {
    return parse(vectors).dump();
  }

  std::string serialize(std::vector<se3_transform_t> const& frames) {
    return parse(frames).dump();
  }

  std::string serialize(feature_tracks_t const& tracks) {
    return parse(tracks).dump();
  }
}  // namespace cyclops
