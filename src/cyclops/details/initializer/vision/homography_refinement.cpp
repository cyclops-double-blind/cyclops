#include "cyclops/details/initializer/vision/homography_refinement.hpp"

#include "cyclops/details/utils/debug.hpp"
#include "cyclops/details/logging.hpp"

#include <range/v3/all.hpp>
#include <spdlog/spdlog.h>

#include <ceres/ceres.h>

namespace cyclops::initializer {
  using Eigen::Map;
  using Eigen::Matrix;
  using Eigen::Matrix3d;
  using Eigen::Vector2d;

  namespace views = ranges::views;

  template <typename scalar_t, int dim>
  using Vector = Matrix<scalar_t, dim, 1>;

  template <typename scalar_t>
  static Vector<scalar_t, 2> project(Vector<scalar_t, 3> const& x) {
    return x.template head<2>() / x.z();
  }

  struct HomographyFeaturePriorCost {
    double const sigma;
    Vector2d const u_hat;

    HomographyFeaturePriorCost(double sigma, Vector2d const& u_hat)
        : sigma(sigma), u_hat(u_hat) {
    }

    template <typename scalar_t>
    bool operator()(scalar_t const* const u_ptr, scalar_t* const r_ptr) const {
      auto u = Map<Vector<scalar_t, 2> const>(u_ptr);
      auto r = Map<Vector<scalar_t, 2>>(r_ptr);
      r = (u - u_hat.cast<scalar_t>()) / scalar_t(sigma);
      return true;
    }
  };

  struct HomographySecondViewProjectionCost {
    double const sigma;
    Vector2d const v_hat;

    HomographySecondViewProjectionCost(double sigma, Vector2d const& v_hat)
        : sigma(sigma), v_hat(v_hat) {
    }

    template <typename scalar_t>
    bool operator()(
      scalar_t const* const H_ptr, scalar_t const* const u_ptr,
      scalar_t* const r_ptr) const {
      auto H = Map<Matrix<scalar_t, 3, 3> const>(H_ptr);
      auto u = Map<Vector<scalar_t, 2> const>(u_ptr);
      auto v = project((H * u.homogeneous()).eval());

      auto r = Map<Vector<scalar_t, 2>>(r_ptr);
      r = (v - v_hat.cast<scalar_t>()) / scalar_t(sigma);

      return true;
    }
  };

  static std::array<double, 9> make_homography_parameter_block(
    Matrix3d const& guess) {
    std::array<double, 9> data;
    (Map<Matrix3d>(data.data())) = guess;
    return data;
  }

  static std::vector<two_view_feature_pair_t> flatten_features(
    std::set<landmark_id_t> const& ids,
    std::map<landmark_id_t, two_view_feature_pair_t> const& features) {
    return ids | views::transform([&](auto id) { return features.at(id); }) |
      ranges::to_vector;
  }

  static std::vector<std::array<double, 2>> make_feature_parameter_blocks(
    std::vector<two_view_feature_pair_t> const& features_flatten) {
    return  //
      features_flatten | views::transform([](auto const& x) {
        auto const& [u, _] = x;
        return std::array<double, 2> {u.x(), u.y()};
      }) |
      ranges::to_vector;
  }

  Matrix3d refine_homography_geometry(
    double sigma,  //
    Matrix3d const& H_initial, std::set<landmark_id_t> const& ids,
    std::map<landmark_id_t, two_view_feature_pair_t> const& features) {
    auto tic = ::cyclops::tic();
    ceres::Problem problem;

    auto features_flatten = flatten_features(ids, features);
    auto u_blocks = make_feature_parameter_blocks(features_flatten);
    auto H_block = make_homography_parameter_block(H_initial);

    for (auto& u : u_blocks)
      problem.AddParameterBlock(u.data(), u.size());
    problem.AddParameterBlock(H_block.data(), H_block.size());

    auto u_ptrs = u_blocks | views::transform([](auto& _) { return _.data(); });
    for (auto const& [feature, u_ptr] : views::zip(features_flatten, u_ptrs)) {
      auto const& [u_hat, v_hat] = feature;
      auto cost1 =
        new ceres::AutoDiffCostFunction<HomographyFeaturePriorCost, 2, 2>(
          new HomographyFeaturePriorCost(sigma, u_hat));
      auto cost2 = new ceres::AutoDiffCostFunction<
        HomographySecondViewProjectionCost, 2, 9, 2>(
        new HomographySecondViewProjectionCost(sigma, v_hat));

      problem.AddResidualBlock(cost1, nullptr, u_ptr);
      problem.AddResidualBlock(
        cost2, new ceres::CauchyLoss(1), H_block.data(), u_ptr);
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 4;
    ceres::Solve(options, &problem, &summary);

    __logger__->debug("Homography refinement time: {}", ::cyclops::toc(tic));
    __logger__->debug("Summary: {}", summary.BriefReport());

    return Matrix3d(H_block.data());
  }
}  // namespace cyclops::initializer
