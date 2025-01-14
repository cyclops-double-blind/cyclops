#pragma once

#include "cyclops/details/type.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace cyclops::initializer {
  struct two_view_correspondence_data_t;
  struct multiview_correspondences_t;

  std::optional<std::reference_wrapper<
    std::pair<frame_id_t const, two_view_correspondence_data_t> const>>
  select_best_two_view_pair(multiview_correspondences_t const& multiviews);
}  // namespace cyclops::initializer
