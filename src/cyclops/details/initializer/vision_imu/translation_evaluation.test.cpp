#include "cyclops/details/initializer/vision_imu/translation_evaluation.hpp"
#include "cyclops/details/initializer/vision_imu/translation_analysis.hpp"
#include "cyclops/details/initializer/vision_imu/translation_cache.hpp"

#include <random>
#include <cmath>

#include <doctest/doctest.h>

namespace cyclops::initializer {
  using doctest::Approx;

  template <int dim>
  using vector_t = Eigen::Matrix<double, dim, 1>;

  template <int rows, int cols>
  using matrix_t = Eigen::Matrix<double, rows, cols>;

  template <int dim>
  static vector_t<dim> randvec(std::mt19937& rgen) {
    auto rand = std::normal_distribution<double>(0, 1);

    vector_t<dim> result;
    for (auto i = 0; i < dim; i++)
      result(i) = rand(rgen);
    return result;
  }

  template <int rows, int cols>
  static matrix_t<rows, cols> randmat(std::mt19937& rgen) {
    auto rand = std::normal_distribution<double>(0, 1);

    matrix_t<rows, cols> result;
    for (auto i = 0; i < rows; i++) {
      for (auto j = 0; j < cols; j++)
        result(i, j) = rand(rgen);
    }
    return result;
  }

  template <int rows, int cols>
  static matrix_t<rows, cols> randchol(std::mt19937& rgen) {
    auto A = randmat<rows, cols>(rgen);
    auto H = (A.transpose() * A).eval();

    Eigen::LLT<matrix_t<rows, cols>> _(H);
    return _.matrixU();
  }

  TEST_CASE("test scale-fixed visual-inertial translation matching") {
    std::mt19937 rgen(20220518);
    auto g = (9.81 * randvec<3>(rgen).normalized()).eval();
    auto b = (0.1 * randvec<3>(rgen)).eval();
    auto v = (10 * randvec<15>(rgen)).eval();
    auto p = vector_t<12>::Zero().eval();
    auto s = std::exp(std::uniform_real_distribution<double>(-2, 2)(rgen));

    auto x_I = Eigen::VectorXd(21);
    x_I << g, b, v;
    auto x_V = p;

    auto A_I = randmat<24, 21>(rgen);
    auto B_I = randmat<24, 12>(rgen);
    auto A_V = randchol<12, 12>(rgen);

    auto alpha = randvec<24>(rgen);
    auto beta = (-(A_I * x_I + alpha) / s).eval();

    auto analysis = imu_match_translation_analysis_t {
      .frames_count = 5,
      .residual_dimension = 36,
      .parameter_dimension = 33,
      .inertial_weight = A_I,
      .translational_weight = B_I,
      .visual_weight = A_V,
      .inertial_perturbation = alpha,
      .translation_perturbation = beta,
    };
    auto cache = IMUMatchTranslationAnalysisCache(analysis);

    auto evaluator = IMUMatchScaleEvaluationContext(9.81, analysis, cache);
    auto maybe_solution = evaluator.evaluate(s);

    REQUIRE(static_cast<bool>(maybe_solution));

    auto const& solution = *maybe_solution;
    CHECK(std::abs(solution.cost) < 1e-6);

    CHECK((solution.inertial_solution - x_I).norm() < 1e-6);
    CHECK((solution.visual_solution - x_V).norm() < 1e-6);
  }

  TEST_CASE(
    "test scale-fixed visual-inertial translation matching cost derivative") {
    std::mt19937 rgen(20220518);
    auto g = (9.81 * randvec<3>(rgen).normalized()).eval();
    auto b = (0.1 * randvec<3>(rgen)).eval();
    auto v = (10 * randvec<15>(rgen)).eval();
    auto p = vector_t<12>::Zero().eval();
    auto s0 = std::exp(std::uniform_real_distribution<double>(-2, 2)(rgen));

    auto x_I = Eigen::VectorXd(21);
    x_I << g, b, v;
    auto x_V = p;

    auto A_I = randmat<24, 21>(rgen);
    auto B_I = randmat<24, 12>(rgen);
    auto A_V = randchol<12, 12>(rgen);

    auto alpha = randvec<24>(rgen);
    auto beta = (-(A_I * x_I + alpha) / s0).eval();

    auto analysis = imu_match_translation_analysis_t {
      .frames_count = 5,
      .residual_dimension = 36,
      .parameter_dimension = 33,
      .inertial_weight = A_I,
      .translational_weight = B_I,
      .visual_weight = A_V,
      .inertial_perturbation = alpha,
      .translation_perturbation = beta,
    };
    auto cache = IMUMatchTranslationAnalysisCache(analysis);
    auto evaluator = IMUMatchScaleEvaluationContext(9.81, analysis, cache);

    auto constexpr ds = 1e-6;
    for (auto _ = 0; _ < 10; _++) {
      auto s = std::exp(std::uniform_real_distribution<double>(-2, 2)(rgen));
      auto maybe_p1 = evaluator.evaluate(s - ds);
      auto maybe_p2 = evaluator.evaluate(s + ds);
      REQUIRE(static_cast<bool>(maybe_p1));
      REQUIRE(static_cast<bool>(maybe_p2));

      auto p__dot_numeric = (maybe_p2->cost - maybe_p1->cost) / 2 / ds;

      auto maybe_p0 = evaluator.evaluate(s);
      REQUIRE(static_cast<bool>(maybe_p0));

      auto p__dot_analytic = evaluator.evaluateDerivative(*maybe_p0, s);
      CHECK(p__dot_numeric == Approx(p__dot_analytic).epsilon(1e-6));
    }
  }
}  // namespace cyclops::initializer
