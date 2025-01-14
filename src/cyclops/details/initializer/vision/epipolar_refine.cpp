#include "cyclops/details/initializer/vision/epipolar_refine.hpp"
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  using std::optional;
  using std::vector;

  using Eigen::Matrix3d;
  using Eigen::Vector4d;

  using Vector8d = Eigen::Matrix<double, 8, 1>;
  using Vector9d = Eigen::Matrix<double, 9, 1>;
  using Matrix8d = Eigen::Matrix<double, 8, 8>;
  using Matrix9d = Eigen::Matrix<double, 9, 9>;
  using Matrix9x8d = Eigen::Matrix<double, 9, 8>;

  using flatten_feature_pairs_t = vector<Vector4d>;

  namespace views = ranges::views;

  class EpipolarGeometrySQPRefinementContext {
  private:
    flatten_feature_pairs_t const _features_hat;

    Matrix3d _E;
    flatten_feature_pairs_t _features;

    vector<Vector4d> solveFeatureErrors() const;
    vector<Vector4d> solveConstraintFeaturePerturbationJacobians() const;
    vector<Vector9d> solveConstraintEssentialMatrixJacobians() const;
    vector<double> solveEpipolarConstraintViolationComplements(
      vector<Vector4d> const& feature_jacobians,
      vector<Vector4d> const& feature_errors) const;

    optional<Vector9d> solveEssentialMatrixUpdate(
      vector<double> const& feature_weights,
      vector<double> const& constraint_violation_complements,
      vector<Vector9d> const& constraint_essential_matrix_jacobians) const;

    vector<double> solveMultipliers(
      vector<double> const& feature_weights,
      vector<double> const& constraint_violation_complements,
      vector<Vector9d> const& constraint_essential_matrix_jacobians,
      Vector9d const& essential_matrix_update) const;

    vector<Vector4d> solveFeatureUpdate(
      vector<Vector4d> const& constraint_feature_perturbation_jacobians,
      vector<Vector4d> const& feature_errors,
      vector<double> const& multipliers) const;
    bool iterate();

  public:
    EpipolarGeometrySQPRefinementContext(
      Matrix3d const& E, flatten_feature_pairs_t const& features_hat);
    Matrix3d const& solve(int max_iterations);
  };

  vector<Vector4d> EpipolarGeometrySQPRefinementContext::solveFeatureErrors()
    const {
    return  //
      views::zip(_features, _features_hat) |
      views::transform([](auto const& pair) -> Vector4d {
        auto const& [x, x_hat] = pair;
        return x - x_hat;
      }) |
      ranges::to_vector;
  }

  static auto solve_constraint_feature_perturbation_jacobians(
    Matrix3d const& E, flatten_feature_pairs_t const& features) {
    return  //
      features | views::transform([&](Vector4d const& pair) -> Vector4d {
        auto E1 = [&]() { return E.leftCols<2>(); };
        auto E2 = [&]() { return E.topRows<2>(); };

        auto u_h = [&]() { return pair.head<2>().homogeneous(); };
        auto v_h = [&]() { return pair.tail<2>().homogeneous(); };

        Vector4d gx;
        gx << E1().transpose() * v_h(), E2() * u_h();

        return gx;
      });
  }

  vector<Vector4d> EpipolarGeometrySQPRefinementContext::
    solveConstraintFeaturePerturbationJacobians() const {
    return solve_constraint_feature_perturbation_jacobians(_E, _features) |
      ranges::to_vector;
  }

  vector<Vector9d> EpipolarGeometrySQPRefinementContext::
    solveConstraintEssentialMatrixJacobians() const {
    return  //
      _features |
      views::transform([](Vector4d const& feature_pair) -> Vector9d {
        auto u_h = [&]() { return feature_pair.head<2>().homogeneous(); };
        auto v_h = [&]() { return feature_pair.tail<2>().homogeneous(); };

        Vector9d ge;
        ge << u_h().x() * v_h(), u_h().y() * v_h(), u_h().z() * v_h();
        return ge;
      }) |
      ranges::to_vector;
  }

  vector<double> EpipolarGeometrySQPRefinementContext::
    solveEpipolarConstraintViolationComplements(
      vector<Vector4d> const& feature_jacobians,
      vector<Vector4d> const& feature_errors) const {
    return  //
      views::zip(_features, feature_jacobians, feature_errors) |
      views::transform([&](auto const& tuple) -> double {
        auto const& [x, gx, r] = tuple;
        auto u_h = [&x = x]() { return x.template head<2>().homogeneous(); };
        auto v_h = [&x = x]() { return x.template tail<2>().homogeneous(); };
        return v_h().dot(_E * u_h()) - gx.dot(r);
      }) |
      ranges::to_vector;
  }

  optional<Vector9d>
  EpipolarGeometrySQPRefinementContext::solveEssentialMatrixUpdate(
    vector<double> const& feature_weights,
    vector<double> const& constraint_violation_complements,
    vector<Vector9d> const& constraint_essential_matrix_jacobians) const {
    auto const& w_range = feature_weights;
    auto const& h_range = constraint_violation_complements;
    auto const& ge_range = constraint_essential_matrix_jacobians;

    Matrix9d A = Matrix9d::Zero();
    Eigen::SelfAdjointView<Matrix9d, Eigen::Lower> A_symmetric(A);

    Vector9d b = Vector9d::Zero();
    for (auto const& [w, h, ge] : views::zip(w_range, h_range, ge_range)) {
      b += ge * h / w;
      A_symmetric.rankUpdate(ge, 1 / w);
    }

    Vector9d e = (Vector9d() << _E.col(0), _E.col(1), _E.col(2)).finished();
    // Tangent of the scale gauge transform of the essential matrix e.
    Matrix9x8d Te = e.jacobiSvd(Eigen::ComputeFullU).matrixU().rightCols<8>();

    Matrix8d A_bar = Te.transpose() * A_symmetric * Te;
    Vector8d b_bar = Te.transpose() * b;

    Eigen::LDLT<Matrix8d> A_bar_inv(A_bar);
    return -Te * A_bar_inv.solve(b_bar);
  }

  vector<double> EpipolarGeometrySQPRefinementContext::solveMultipliers(
    vector<double> const& feature_weights,
    vector<double> const& constraint_violation_complements,
    vector<Vector9d> const& constraint_essential_matrix_jacobians,
    Vector9d const& essential_matrix_update) const {
    auto const& w_range = feature_weights;
    auto const& h_range = constraint_violation_complements;
    auto const& ge_range = constraint_essential_matrix_jacobians;
    auto const& delta_e = essential_matrix_update;

    return  //
      views::zip(w_range, h_range, ge_range) |
      views::transform([&](auto const& tuple) {
        auto const& [w, h, ge] = tuple;
        return (ge.dot(delta_e) + h) / w;
      }) |
      ranges::to_vector;
  }

  vector<Vector4d> EpipolarGeometrySQPRefinementContext::solveFeatureUpdate(
    vector<Vector4d> const& constraint_feature_perturbation_jacobians,
    vector<Vector4d> const& feature_errors,
    vector<double> const& multipliers) const {
    auto const& gx_range = constraint_feature_perturbation_jacobians;
    auto const& r_range = feature_errors;

    return  //
      views::zip(gx_range, r_range, multipliers) |
      views::transform([](auto const& tuple) -> Vector4d {
        auto const& [gx, r, lambda] = tuple;
        return -gx * lambda - r;
      }) |
      ranges::to_vector;
  }

  bool EpipolarGeometrySQPRefinementContext::iterate() {
    auto r_range = solveFeatureErrors();
    auto ge_range = solveConstraintEssentialMatrixJacobians();
    auto gx_range = solveConstraintFeaturePerturbationJacobians();

    auto w_range = gx_range |
      views::transform([](Vector4d const& gx) { return gx.dot(gx); }) |
      ranges::to_vector;
    auto h_range =
      solveEpipolarConstraintViolationComplements(gx_range, r_range);

    auto maybe_delta_e = solveEssentialMatrixUpdate(w_range, h_range, ge_range);
    if (!maybe_delta_e.has_value())
      return false;
    auto const& delta_e = *maybe_delta_e;

    auto multipliers = solveMultipliers(w_range, h_range, ge_range, delta_e);
    auto delta_x_range = solveFeatureUpdate(gx_range, r_range, multipliers);

    _E.col(0) += delta_e.segment<3>(0);
    _E.col(1) += delta_e.segment<3>(3);
    _E.col(2) += delta_e.segment<3>(6);
    _E /= _E.norm();

    for (int i = 0; i < _features.size(); i++)
      _features.at(i) += delta_x_range.at(i);

    return true;
  }

  EpipolarGeometrySQPRefinementContext::EpipolarGeometrySQPRefinementContext(
    Matrix3d const& E, flatten_feature_pairs_t const& features_hat)
      : _E(E), _features_hat(features_hat), _features(features_hat) {
  }

  Matrix3d const& EpipolarGeometrySQPRefinementContext::solve(
    int max_iterations) {
    for (int i = 0; i < max_iterations; i++) {
      if (!iterate())
        break;
    }
    return _E;
  }

  static auto flatten_features(
    Eigen::Matrix3d const& E, std::set<landmark_id_t> const& ids,
    std::map<landmark_id_t, two_view_feature_pair_t> const& features) {
    auto flat_transform = [](auto const& feature_pair) {
      auto const& [u, v] = feature_pair;
      return (Vector4d() << u, v).finished().eval();
    };

    auto features_flatten = ids |
      views::transform([&](auto id) { return features.at(id); }) |
      views::transform(flat_transform) | ranges::to_vector;
    auto feature_sanities =
      solve_constraint_feature_perturbation_jacobians(E, features_flatten) |
      views::transform([](auto const& Ju) { return Ju.dot(Ju); }) |
      views::transform([](auto w) { return std::abs(w) > 1e-6; }) |
      ranges::to_vector;

    auto ids_filtered = views::zip(ids, feature_sanities) |
      views::filter([](auto const& tuple) { return std::get<1>(tuple); }) |
      views::transform([](auto const& tuple) { return std::get<0>(tuple); }) |
      ranges::to_vector;

    auto features_flatten_filtered =  //
      views::zip(features_flatten, feature_sanities) |
      views::filter([](auto const& tuple) { return std::get<1>(tuple); }) |
      views::transform([](auto const& tuple) { return std::get<0>(tuple); }) |
      ranges::to_vector;

    return std::make_tuple(ids_filtered, features_flatten_filtered);
  }

  Matrix3d refine_epipolar_geometry(
    Matrix3d const& E_initial, std::set<landmark_id_t> const& ids,
    std::map<landmark_id_t, two_view_feature_pair_t> const& features) {
    auto E = (E_initial / E_initial.norm()).eval();
    auto [ids_filtered, features_flatten] = flatten_features(E, ids, features);
    auto context = EpipolarGeometrySQPRefinementContext(E, features_flatten);
    return context.solve(8);
  }
}  // namespace cyclops::initializer
