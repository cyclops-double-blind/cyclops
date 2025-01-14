#pragma once

#include <set>
#include <string>

#include <chrono>

namespace cyclops {
  void cyclops_assert(std::string const& reason, bool requirement);

  std::chrono::time_point<std::chrono::steady_clock> tic();
  double toc(std::chrono::time_point<std::chrono::steady_clock> const& tic);

  template <typename value_t>
  static std::string set_to_string(std::set<value_t> const& s);

  template <typename vectorxd_t>
  static std::string vector_to_string(vectorxd_t const& v);

  template <typename matrixxd_t>
  static std::string matrix_to_string(matrixxd_t const& M);
}  // namespace cyclops
