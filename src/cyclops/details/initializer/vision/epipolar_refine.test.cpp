#include "cyclops/details/initializer/vision/epipolar_refine.cpp"
#include "cyclops/details/utils/math.hpp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using Eigen::AngleAxisd;
  using Eigen::Matrix2d;
  using Eigen::Matrix3d;
  using Eigen::Quaterniond;

  using Eigen::Vector2d;
  using Eigen::Vector3d;

  using Matrix2x3d = Eigen::Matrix<double, 2, 3>;

  static auto make_random_unit_vector(std::mt19937& rgen) {
    auto d = std::uniform_real_distribution<double>(-1, 1);
    return Vector3d(d(rgen), d(rgen), d(rgen)).normalized().eval();
  }

  static auto make_landmarks(std::mt19937& rgen) {
    auto landmark_point_transform = [&](landmark_id_t id) {
      auto d = std::uniform_real_distribution<double>(-0.5, 0.5);
      return std::make_pair(id, Vector3d(d(rgen), d(rgen), 1.0 + d(rgen)));
    };
    return views::ints(0, 1000) | views::transform(landmark_point_transform) |
      ranges::to<std::map<landmark_id_t, Vector3d>>;
  }

  static Vector2d project(Vector3d const& z) {
    return z.head<2>() / z.z();
  }

  template <typename keys_range_t, typename values_range_t>
  static auto make_zipdict(
    keys_range_t const& keys, values_range_t const& values) {
    auto pair_transform = [](auto _) {
      return std::make_pair(std::get<0>(_), std::get<1>(_));
    };
    return views::zip(keys, values) | views::transform(pair_transform);
  }

  static auto make_two_view_feature_pairs(
    std::map<landmark_id_t, Vector3d> const& landmarks,
    Quaterniond const& rotation, Vector3d const& translation,
    std::mt19937& rgen, double noise) {
    auto perturbation = [&](auto const& u) -> Vector2d {
      auto d = std::normal_distribution<double>(0, 1);
      return u + Vector2d(noise * d(rgen), noise * d(rgen));
    };

    auto fst_view_features = landmarks | views::values |
      views::transform(project) | views::transform(perturbation);
    auto snd_view_features =
      landmarks | views::values | views::transform([&](auto const& f) {
        return project(rotation.conjugate() * (f - translation));
      }) |
      views::transform(perturbation);
    auto two_view_features = views::zip(fst_view_features, snd_view_features);

    return make_zipdict(landmarks | views::keys, two_view_features) |
      ranges::to<std::map<landmark_id_t, two_view_feature_pair_t>>;
  }

  TEST_CASE("Test epipolar geometry refinement") {
    auto rgen = std::make_shared<std::mt19937>(20230804005);
    auto landmarks = make_landmarks(*rgen);

    auto sigma = 0.0075;

    auto rotation = Quaterniond(AngleAxisd(0.2, Vector3d::UnitX()));
    auto translation = (0.2 * make_random_unit_vector(*rgen)).eval();

    auto features = make_two_view_feature_pairs(
      landmarks, rotation, translation, *rgen, sigma);
    auto ids = features | views::keys | ranges::to<std::set<landmark_id_t>>;

    auto E_truth =
      (-rotation.conjugate().matrix() * skew3d(translation)).eval();
    E_truth /= E_truth.norm();

    auto E_perturbed = E_truth;

    auto features_flatten =  //
      ids | views::transform([&](auto id) { return features.at(id); }) |
      views::transform([](auto const& feature_pair) -> Vector4d {
        auto const& [u, v] = feature_pair;
        return (Vector4d() << u, v).finished();
      }) |
      ranges::to_vector;

    auto context =
      EpipolarGeometrySQPRefinementContext(E_truth, features_flatten);
    auto const& E_refined = context.solve(8);

    CAPTURE(E_truth);
    CAPTURE(E_refined);
    CHECK(E_truth.isApprox(E_refined, 0.01));
  }
}  // namespace cyclops::initializer
