#include "cyclops/details/utils/debug.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>

namespace cyclops {
  void cyclops_assert(std::string const& reason, bool requirement) {
    if (!requirement) {
      __logger__->critical("Assertion failed: {}", reason);
      throw std::logic_error(reason);
    }
  }

  std::chrono::time_point<std::chrono::steady_clock> tic() {
    return std::chrono::steady_clock::now();
  }

  double toc(std::chrono::time_point<std::chrono::steady_clock> const& tic) {
    std::chrono::duration<double> dt = (std::chrono::steady_clock::now() - tic);
    return dt.count();
  }
}  // namespace cyclops
