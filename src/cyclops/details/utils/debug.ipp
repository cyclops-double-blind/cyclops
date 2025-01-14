#pragma once

#include "cyclops/details/utils/debug.hpp"

#include <set>
#include <sstream>
#include <string>

namespace cyclops {
  template <typename value_t>
  static std::string set_to_string(std::set<value_t> const& s) {
    if (s.empty())
      return "{}";

    int ssize = s.size();

    std::ostringstream stream;
    stream << "{";
    for (int i = 0; i < ssize - 1; i++)
      stream << std::to_string(s.at(i)) << ", ";
    stream << *s.rbegin() << "}";

    return stream.str();
  }

  template <typename vectorxd_t>
  static std::string vector_to_string(vectorxd_t const& v) {
    std::ostringstream ss;
    ss << v.transpose();
    return ss.str();
  }

  template <typename matrixxd_t>
  static std::string matrix_to_string(matrixxd_t const& M) {
    std::ostringstream ss;
    ss << M;
    return ss.str();
  }
}  // namespace cyclops
