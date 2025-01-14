#pragma once

#include <range/v3/view.hpp>

namespace cyclops {
  template <typename key_t, typename value_t, typename range_t>
  static auto make_dictionary(range_t const& range) {
    return  //
      ranges::views::all(range) | ranges::views::transform([](auto const& _) {
        auto const& [key, value] = _;
        return std::make_pair(key, value);
      }) |
      ranges::to<std::map<key_t, value_t>>;
  }

  static auto linspace(double a, double b, int n) {
    auto&& slice = [a, b, n](auto i) {
      if (n <= 1) {
        return b;
      }
      return a + i * (b - a) / (n - 1);
    };

    namespace views = ranges::views;
    if (n <= 0)
      return views::iota(0, 0) | views::transform(slice);
    return views::iota(0, n) | views::transform(slice);
  }

  template <typename fullorder_range_t>
  static auto combinations(
    fullorder_range_t const& a, fullorder_range_t const& b) {
    namespace views = ranges::views;
    auto lessthan = [](auto const& ab_pair) {
      auto const& [a, b] = ab_pair;
      return a < b;
    };
    return views::cartesian_product(a, b) | views::filter(lessthan);
  }
}  // namespace cyclops
