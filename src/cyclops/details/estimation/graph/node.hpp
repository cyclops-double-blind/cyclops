#pragma once

#include "cyclops/details/type.hpp"

#include <ostream>
#include <variant>

namespace cyclops::estimation {
  struct node_t {
    struct frame_t {
      frame_id_t id;

      int dimension() const;
      int manifold_dimension() const;
    };

    struct bias_t {
      frame_id_t id;

      int dimension() const;
      int manifold_dimension() const;
    };

    struct landmark_t {
      landmark_id_t id;

      int dimension() const;
      int manifold_dimension() const;
    };

    std::variant<frame_t, bias_t, landmark_t> variant;

    bool operator<(node_t const& other) const;
    bool operator==(node_t const& other) const;
    int dimension() const;
    int manifold_dimension() const;

    template <size_t i>
    using variant_t_at =
      std::variant_alternative_t<i, decltype(node_t::variant)>;
  };

  namespace node {
    static inline node_t frame(frame_id_t id) {
      return {node_t::frame_t {id}};
    }

    static inline node_t bias(frame_id_t id) {
      return {node_t::bias_t {id}};
    }

    static inline node_t landmark(landmark_id_t id) {
      return {node_t::landmark_t {id}};
    }

    template <typename type>
    static bool is(node_t const& node) {
      return std::holds_alternative<type>(node.variant);
    }
  }  // namespace node

  bool operator<(node_t::frame_t const& a, node_t::frame_t const& b);
  bool operator<(node_t::bias_t const& a, node_t::bias_t const& b);
  bool operator<(node_t::landmark_t const& a, node_t::landmark_t const& b);
  bool operator==(node_t::frame_t const& a, node_t::frame_t const& b);
  bool operator==(node_t::bias_t const& a, node_t::bias_t const& b);
  bool operator==(node_t::landmark_t const& a, node_t::landmark_t const& b);

  std::ostream& operator<<(std::ostream&, node_t const&);
}  // namespace cyclops::estimation
