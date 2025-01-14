#include "cyclops/details/initializer/vision/bundle_adjustment_factors.hpp"
#include "cyclops/details/utils/math.hpp"

namespace cyclops::initializer {
  LandmarkProjectionCost::LandmarkProjectionCost(feature_point_t const& feature)
      : u(feature.point),
        weight_sqrt(Eigen::LLT<Eigen::Matrix2d>(feature.weight).matrixU()) {
  }

  bool LandmarkProjectionCost::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
    using Eigen::Matrix2d;
    using Eigen::Matrix3d;
    using Eigen::Quaterniond;
    using Eigen::Vector2d;
    using Eigen::Vector3d;

    using Matrix2x3d = Eigen::Matrix<double, 2, 3, Eigen::RowMajor>;
    using Matrix2x7d = Eigen::Matrix<double, 2, 7, Eigen::RowMajor>;

    auto x_data = parameters[0];
    auto f_data = parameters[1];
    auto r = Eigen::Map<Vector2d>(residuals);

    auto q = Quaterniond(x_data);
    auto p = Eigen::Map<Vector3d const>(x_data + 4);
    auto f = Eigen::Map<Vector3d const>(f_data);

    Matrix3d R_T = q.conjugate().matrix();
    Vector3d d = f - p;
    Vector3d z = R_T * d;

    Vector2d u_hat = z.head<2>() / z.z();
    r = weight_sqrt * (u_hat - u);

    if (jacobians != nullptr) {
      Matrix2d W = weight_sqrt / z.z();
      Matrix2x3d S = (Matrix2x3d() << W, -W * u_hat).finished();
      Matrix2x3d S__R_T = S * R_T;

      if (jacobians[0] != nullptr) {
        auto J_x = Eigen::Map<Matrix2x7d>(jacobians[0]);

        auto w = q.w();
        auto v = q.vec().eval();

        auto d_T = [&]() { return d.transpose(); };
        auto v_T = [&]() { return v.transpose(); };
        auto I3 = []() { return Matrix3d::Identity(); };

        J_x.leftCols<3>() =
          S * 2 * (w * skew3d(d) + v.dot(d) * I3() + v * d_T() - d * v_T());
        J_x.col(3) = S * 2 * (w * d - v.cross(d));

        J_x.rightCols<3>() = -S__R_T;
      }

      if (jacobians[1] != nullptr) {
        auto J_f = Eigen::Map<Matrix2x3d>(jacobians[1]);
        J_f = S__R_T;
      }
    }

    return true;
  }
}  // namespace cyclops::initializer
