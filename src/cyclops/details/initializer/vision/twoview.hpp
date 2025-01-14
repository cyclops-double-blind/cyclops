#pragma once

#include <random>
#include <vector>
#include <memory>

namespace cyclops {
  struct cyclops_global_config_t;
}

namespace cyclops::initializer {
  struct two_view_geometry_t;
  struct two_view_correspondence_data_t;

  class TwoViewVisionGeometrySolver {
  public:
    virtual ~TwoViewVisionGeometrySolver() = default;
    virtual void reset() = 0;

    // returns a sequence of possible solutions.
    virtual std::vector<two_view_geometry_t> solve(
      two_view_correspondence_data_t const& two_view_correspondence) = 0;

    static std::unique_ptr<TwoViewVisionGeometrySolver> create(
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<std::mt19937> rgen);
  };
}  // namespace cyclops::initializer
