#include "cyclops/details/utils/math.cpp"
#include "cyclops_tests/random.hpp"

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <range/v3/all.hpp>

#include <cmath>
#include <random>

#include <doctest/doctest.h>

namespace cyclops {
  namespace views = ranges::views;

  TEST_CASE("std::log1p(x) - x is accurate for x << 1") {
    auto x = 1e-6;
    auto y = std::log1p(x) - x;
    CHECK(y == doctest::Approx(-x * x / 2));
  }

  static std::vector<double> make_rpoly_from_roots(
    std::vector<double> const& roots, double leading_coefficient = 1) {
    auto step = [](auto const& p, double r) -> std::vector<double> {
      if (p.empty())
        return {-r, 1};

      std::vector<double> q(p.size() + 1);

      q[0] = -r * p[0];
      for (size_t i = 0; i < p.size() - 1; i++)
        q[i + 1] = p[i] - r * p[i + 1];
      q.back() = 1.;
      return q;
    };

    std::vector<double> p;
    for (auto r : roots) {
      auto q = step(p, r);
      p = std::move(q);
    }
    for (auto& c : p)
      c *= leading_coefficient;

    return p;
  }

  TEST_CASE("Polynomial generation") {
    auto p = make_rpoly_from_roots({1, 2, 3}, 2);
    REQUIRE(p.size() == 4);
    CHECK(p.at(0) == doctest::Approx(-12));
    CHECK(p.at(1) == doctest::Approx(+22));
    CHECK(p.at(2) == doctest::Approx(-12));
    CHECK(p.at(3) == doctest::Approx(+2));
  }

  TEST_CASE("Polynomial evaluation") {
    std::mt19937 rgen(20230420);
    auto rand = std::uniform_real_distribution<double>(-10, 10);

    auto s = std::vector<double>(6);
    std::generate(s.begin(), s.end(), [&]() { return rand(rgen); });

    auto c = rand(rgen);
    auto p = make_rpoly_from_roots(s, c * c);
    REQUIRE(p.size() == 7);

    auto x = rand(rgen);

    auto p_x__expected = ranges::accumulate(
      views::enumerate(p) | views::transform([x](auto const& _) {
        auto const& [n, c] = _;
        return c * std::pow(x, n);
      }),
      0.0);
    auto p_x__got = evaluate_polynomial<7>(p.data(), x);

    CHECK(p_x__expected == doctest::Approx(p_x__got));
  }

  TEST_CASE("Test `gamma_lower_incomplete_normal`") {
    auto const r_max_seq = std::vector<double> {3, 1, 0.5, 0.01, 1e-6};

    SUBCASE("a ~ [0.5, 30], x / a ~ [0, r_max]") {
      std::mt19937 rgen(20230420);
      for (auto r_max : r_max_seq) {
        auto arand = std::uniform_int_distribution<int>(1, 60);
        auto rrand = std::uniform_real_distribution<double>(0, r_max);

        for (auto _ = 0; _ < 200; _++) {
          auto a = static_cast<double>(arand(rgen)) / 2;
          auto x = rrand(rgen) * a;
          CAPTURE(r_max);
          CAPTURE(a);
          CAPTURE(x);
          CHECK(
            gamma_lower_incomplete_normal(a, x) ==
            doctest::Approx(boost::math::gamma_p(a, x)));
        }
      }
    }

    SUBCASE("a ~ [30, 1000], x / a ~ [0, r_max]") {
      std::mt19937 rgen(20230420);
      for (auto r_max : r_max_seq) {
        auto arand = std::uniform_int_distribution<int>(60, 2000);
        auto rrand = std::uniform_real_distribution<double>(0, r_max);

        for (auto _ = 0; _ < 200; _++) {
          auto a = static_cast<double>(arand(rgen)) / 2;
          auto x = rrand(rgen) * a;
          CAPTURE(r_max);
          CAPTURE(a);
          CAPTURE(x);
          CHECK(
            gamma_lower_incomplete_normal(a, x) ==
            doctest::Approx(boost::math::gamma_p(a, x)));
        }
      }
    }
  }

  TEST_CASE("Test `chi_squared_cdf`") {
    std::mt19937 rgen(20230420);
    auto dof_rand = std::uniform_int_distribution<int>(1, 1000);

    auto const r_max_seq = std::vector<double> {3, 1, 0.5, 0.01, 1e-6};
    for (auto r_max : r_max_seq) {
      auto rrand = std::uniform_real_distribution<double>(0, r_max);

      for (auto _ = 0; _ < 1000; _++) {
        auto dof = dof_rand(rgen);
        auto x = rrand(rgen) * dof;

        auto boost_chi_squared = boost::math::chi_squared(dof);
        auto boost_cdf = boost::math::cdf(boost_chi_squared, x);
        auto my_cdf = ::cyclops::chi_squared_cdf(dof, x);

        CAPTURE(r_max);
        CAPTURE(dof);
        CAPTURE(x);
        CHECK(boost_cdf == doctest::Approx(my_cdf));
      }
    }
  }

  TEST_CASE("SO(3) logarithm map") {
    using Eigen::Quaterniond;
    using Eigen::Vector3d;

    auto rgen = std::mt19937(20210428);

    for (auto _ = 0; _ < 1000; _++) {
      auto angle = perturbate(Vector3d::Zero().eval(), 1, rgen);
      auto rotation =
        Quaterniond(Eigen::AngleAxisd(angle.norm(), angle.normalized()));

      auto logangle = so3_logmap(rotation);
      CAPTURE(logangle.transpose());
      CAPTURE(angle.transpose());
      CHECK(logangle.isApprox(angle));
    }

    CHECK(so3_logmap(Quaterniond::Identity()).norm() == 0);

    for (auto _ = 0; _ < 1000; _++) {
      auto angle = perturbate(Vector3d::Zero().eval(), 1e-6, rgen);
      auto rotation =
        Quaterniond(Eigen::AngleAxisd(angle.norm(), angle.normalized()));

      auto logangle = so3_logmap(rotation);
      CAPTURE(logangle.transpose());
      CAPTURE(angle.transpose());
      CHECK(logangle.isApprox(angle));
    }

    for (auto _ = 0; _ < 1000; _++) {
      auto angle = perturbate(Vector3d::Zero().eval(), 1e-10, rgen);
      auto rotation =
        Quaterniond(Eigen::AngleAxisd(angle.norm(), angle.normalized()));

      auto logangle = so3_logmap(rotation);
      CAPTURE(logangle.transpose());
      CAPTURE(angle.transpose());
      CHECK(logangle.isApprox(angle));
    }
  }
}  // namespace cyclops
