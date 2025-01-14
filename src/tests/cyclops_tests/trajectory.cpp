#include "cyclops_tests/trajectory.hpp"

#include <range/v3/all.hpp>

namespace cyclops {
  using Eigen::Vector3d;

  static int nCk(int n, int k) {
    if (k > n)
      return 0;
    if (2 * k > n)
      k = n - k;
    if (k == 0)
      return 1;

    int result = n;
    for (int i = 2; i <= k; ++i) {
      result *= (n - i + 1);
      result /= i;
    }
    return result;
  }

  vector3_signal_t bezier(double T, std::vector<Vector3d> const& points) {
    if (points.size() < 2)
      throw points;

    return [=](double time) {
      auto t = std::min(1., std::max(0., time / T));
      auto x = t - sin(4 * M_PI * t) / (8 * M_PI);
      auto n = points.size() - 1;
      auto result = Vector3d::Zero().eval();
      for (auto const& [i, p] : ranges::views::enumerate(points))
        result += nCk(n, i) * std::pow(1 - x, n - i) * std::pow(x, i) * p;
      return result;
    };
  }
}  // namespace cyclops
