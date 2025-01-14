#include "cyclops/details/utils/math.hpp"

namespace cyclops {
  static double constexpr make_pi() {
    return std::atan2(0, -1);
  }

  static double log1pmx(double x) {
    auto a = std::abs(x);

    if (a > 0.95)
      return std::log(1 + x) - x;
    if (a < 1e-6)
      return -x * x / 2;
    return std::log1p(x) - x;
  }

  template <int N>
  static double evaluate_polynomial(double const* coeffs, double x) {
    if (N <= 0)
      return 0;

    // use Horner's method to evaluate polynomial.
    double r = 0;
    for (int i = N; i > 0; i--) {
      auto a = coeffs[i - 1];
      r = a + r * x;
    }
    return r;
  }

  /*
   * copied from boost/math/special_functions/detail/igamma_large.hpp
   */
  static double igamma_temme_large(double a, double x) {
    auto sigma = (x - a) / a;
    auto phi = -log1pmx(sigma);
    auto y = a * phi;
    auto z = sqrt(2 * phi);

    if (x < a)
      z = -z;

    double workspace[10];

    // clang-format off
    double constexpr C0[15] = {
      -0.33333333333333333,
      +0.083333333333333333,
      -0.014814814814814815,
      +0.0011574074074074074,
      +0.0003527336860670194,
      -0.00017875514403292181,
      +0.39192631785224378e-4,
      -0.21854485106799922e-5,
      -0.185406221071516e-5,
      +0.8296711340953086e-6,
      -0.17665952736826079e-6,
      +0.67078535434014986e-8,
      +0.10261809784240308e-7,
      -0.43820360184533532e-8,
      +0.91476995822367902e-9,
    };
    // clang-format on
    workspace[0] = evaluate_polynomial<15>(C0, z);

    // clang-format off
    double constexpr C1[13] = {
      -0.0018518518518518519,
      -0.0034722222222222222,
      +0.0026455026455026455,
      -0.00099022633744855967,
      +0.00020576131687242798,
      -0.40187757201646091e-6,
      -0.18098550334489978e-4,
      +0.76491609160811101e-5,
      -0.16120900894563446e-5,
      +0.46471278028074343e-8,
      +0.1378633446915721e-6,
      -0.5752545603517705e-7,
      +0.11951628599778147e-7,
    };
    // clang-format on
    workspace[1] = evaluate_polynomial<13>(C1, z);

    // clang-format off
    double constexpr C2[11] = {
      +0.0041335978835978836,
      -0.0026813271604938272,
      +0.00077160493827160494,
      +0.20093878600823045e-5,
      -0.00010736653226365161,
      +0.52923448829120125e-4,
      -0.12760635188618728e-4,
      +0.34235787340961381e-7,
      +0.13721957309062933e-5,
      -0.6298992138380055e-6,
      +0.14280614206064242e-6,
    };
    // clang-format on
    workspace[2] = evaluate_polynomial<11>(C2, z);

    // clang-format off
    double constexpr C3[9] = {
      +0.00064943415637860082,
      +0.00022947209362139918,
      -0.00046918949439525571,
      +0.00026772063206283885,
      -0.75618016718839764e-4,
      -0.23965051138672967e-6,
      +0.11082654115347302e-4,
      -0.56749528269915966e-5,
      +0.14230900732435884e-5,
    };
    // clang-format on
    workspace[3] = evaluate_polynomial<9>(C3, z);

    // clang-format off
    double constexpr C4[7] = {
      -0.0008618882909167117,
      +0.00078403922172006663,
      -0.00029907248030319018,
      -0.14638452578843418e-5,
      +0.66414982154651222e-4,
      -0.39683650471794347e-4,
      +0.11375726970678419e-4,
    };
    // clang-format on
    workspace[4] = evaluate_polynomial<7>(C4, z);

    // clang-format off
    double constexpr C5[9] = {
      -0.00033679855336635815,
      -0.69728137583658578e-4,
      +0.00027727532449593921,
      -0.00019932570516188848,
      +0.67977804779372078e-4,
      +0.1419062920643967e-6,
      -0.13594048189768693e-4,
      +0.80184702563342015e-5,
      -0.22914811765080952e-5,
    };
    // clang-format on
    workspace[5] = evaluate_polynomial<9>(C5, z);

    // clang-format off
    double constexpr C6[7] = {
      +0.00053130793646399222,
      -0.00059216643735369388,
      +0.00027087820967180448,
      +0.79023532326603279e-6,
      -0.81539693675619688e-4,
      +0.56116827531062497e-4,
      -0.18329116582843376e-4,
    };
    // clang-format on
    workspace[6] = evaluate_polynomial<7>(C6, z);

    // clang-format off
    double constexpr C7[5] = {
      +0.00034436760689237767,
      +0.51717909082605922e-4,
      -0.00033493161081142236,
      +0.0002812695154763237,
      -0.00010976582244684731,
    };
    // clang-format on
    workspace[7] = evaluate_polynomial<5>(C7, z);

    // clang-format off
    double constexpr C8[3] = {
      -0.00065262391859530942,
      +0.00083949872067208728,
      -0.00043829709854172101,
    };
    // clang-format on
    workspace[8] = evaluate_polynomial<3>(C8, z);
    workspace[9] = -0.00059676129019274625;

    double result = evaluate_polynomial<10>(workspace, 1 / a);
    result *= std::exp(-y) / std::sqrt(2 * make_pi() * a);
    if (x < a)
      result = -result;
    result += std::erfc(sqrt(y)) / 2;

    if (x < a)
      return result;
    return 1 - result;
  }

  // copied from boost/math/special_functions/gamma.hpp
  static double finite_gamma_q(double a, double x) {
    double e = std::exp(-x);
    double sum = e;
    if (sum != 0) {
      double term = sum;
      for (unsigned n = 1; n < a; ++n) {
        term /= n;
        term *= x;
        sum += term;
      }
    }
    return sum;
  }

  // copied from boost/math/special_functions/gamma.hpp
  static double finite_half_gamma_q(double a, double x) {
    auto e = std::erfc(std::sqrt(x));
    if ((e != 0) && (a > 1)) {
      auto term = std::exp(-x) / std::sqrt(make_pi() * x);
      term *= x;
      term /= 0.5;

      auto sum = term;
      for (int n = 2; n < a; ++n) {
        term /= n - 0.5;
        term *= x;
        sum += term;
      }
      e += sum;
    }
    return e;
  }

  static auto safeguard(double x, double lb = 0, double ub = 1) {
    return std::min(ub, std::max(lb, x));
  }

  static double gamma_lower_incomplete_normal(double a, double x) {
    // See: https://en.wikipedia.org/wiki/Incomplete_gamma_function
    //
    // Computes regularized lower incomplete gamma function
    //
    // \[
    //            P(a, x) = \frac {\gamma(a, x)}
    //                            {\Gamma(a)}   .
    // \]
    //
    // Here, $\gamma(a, x)$ and $\Gamma(a)$ are
    // \[
    //      \gamma(a, x) = \int_{0}^{x} t^{a-1}e^{-t} dt,
    //      \Gamma(a) = \int_{0}^{\infty} t^{a-1}e^{-t} dt.
    // \]
    if (a <= 5.0) {
      auto is_a_int = std::floor(a) == a;
      auto is_a_half_int = !is_a_int;

      if (is_a_int)
        return safeguard(1 - finite_gamma_q(a, x));
      if (is_a_half_int)
        return safeguard(1 - finite_half_gamma_q(a, x));
    }
    return safeguard(igamma_temme_large(a, x));
  }

  double chi_squared_cdf(int degrees_of_freedom, double z) {
    auto a = degrees_of_freedom / 2.0;
    auto x = z / 2.0;
    return gamma_lower_incomplete_normal(a, x);
  }
}  // namespace cyclops
