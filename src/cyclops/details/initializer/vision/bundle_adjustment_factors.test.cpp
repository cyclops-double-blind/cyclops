#include "cyclops/details/initializer/vision/bundle_adjustment_factors.cpp"

#include <random>
#include <doctest/doctest.h>

namespace cyclops::initializer {
  using Eigen::Matrix2d;
  using Eigen::Vector2d;
  using Matrix2x7d = Eigen::Matrix<double, 2, 7, Eigen::RowMajor>;
  using Matrix2x3d = Eigen::Matrix<double, 2, 3, Eigen::RowMajor>;

  struct AutoDiffCost {
    Vector2d _u;
    Matrix2d _W;

    AutoDiffCost(Vector2d const& u, Matrix2d const& W): _u(u), _W(W) {
    }

    template <typename scalar_t>
    bool operator()(
      scalar_t const* const x, scalar_t const* const f,
      scalar_t* const r) const {
      using quaternion_t = Eigen::Quaternion<scalar_t>;
      using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
      using vector2_t = Eigen::Matrix<scalar_t, 2, 1>;
      using vector2_map_t = Eigen::Map<vector2_t>;

      auto v = Eigen::Map<vector3_t const>(x);
      auto w = x[3];
      auto p = Eigen::Map<vector3_t const>(x + 4);

      auto d = (Eigen::Map<vector3_t const>(f) - p).eval();
      auto _2 = scalar_t(2);

      // clang-format off
      auto z =
        (
          + w * w * d
          - _2 * w * v.cross(d)
          + _2 * v * v.transpose() * d
          - v.dot(v) * d
        ).eval();
      // clang-format on

      auto u_hat = (z.template head<2>() / z.z()).eval();
      auto r_map = vector2_map_t(r);
      r_map = _W.cast<scalar_t>() * (u_hat - _u.cast<scalar_t>());

      return true;
    }
  };

  static auto evaluate(
    ceres::CostFunction const& cost,  //
    std::array<double, 7> const& x, std::array<double, 3> const& f) {
    std::array<double, 14> J_x;
    std::array<double, 6> J_f;

    std::array<double const*, 2> parameters = {x.data(), f.data()};
    std::array<double*, 2> jacobians = {J_x.data(), J_f.data()};

    std::array<double, 2> r;
    auto success = cost.Evaluate(parameters.data(), r.data(), jacobians.data());
    REQUIRE(success);

    return std::make_tuple(
      Vector2d(r.data()), Matrix2x7d(J_x.data()), Matrix2x3d(J_f.data()));
  }

  TEST_CASE("Test analytic landmark cost function") {
    std::mt19937 rgen(20220511);
    auto rand = [&rgen]() {
      return std::uniform_real_distribution<double>(-1, 1)(rgen);
    };

    GIVEN("Random-generated feature position and weight") {
      Vector2d u = Vector2d(rand(), rand());
      // Matrix2d W = (Matrix2d() << rand(), rand(), rand(),
      // rand()).finished();
      Matrix2d W = Matrix2d::Identity();

      CAPTURE(u.transpose());

      auto cost_autodiff = ceres::AutoDiffCostFunction<AutoDiffCost, 2, 7, 3>(
        new AutoDiffCost(u, W));
      auto cost_analytic = LandmarkProjectionCost(
        feature_point_t {.point = u, .weight = W.transpose() * W});

      WHEN("Evaluated analytic cost function at random-given parameter") {
        auto q =
          Eigen::Quaterniond(rand(), rand(), rand(), rand()).normalized();
        auto p = Eigen::Vector3d(rand(), rand(), rand());

        std::array<double, 7> x = {
          q.x(), q.y(), q.z(), q.w(), p.x(), p.y(), p.z(),
        };
        std::array<double, 3> f = {rand(), rand(), rand()};

        auto [r1, J_x_1, J_f_1] = evaluate(cost_autodiff, x, f);
        auto [r2, J_x_2, J_f_2] = evaluate(cost_analytic, x, f);

        THEN("The evaluated residual is equivalent up to numerical accuracy") {
          CAPTURE(r1.transpose());
          CAPTURE(r2.transpose());
          CHECK(r1.isApprox(r2));
        }

        THEN(
          "The evaluated pose Jacobian is equivalent up to numerical "
          "accuracy") {
          CAPTURE(J_x_1);
          CAPTURE(J_x_2);
          CHECK(J_x_1.isApprox(J_x_2));
        }

        THEN(
          "The evaluated landmark Jacobian is equivalent up to numerical "
          "accuracy") {
          CAPTURE(J_f_1);
          CAPTURE(J_f_2);
          CHECK(J_f_1.isApprox(J_f_2));
        }
      }
    }
  }
}  // namespace cyclops::initializer
