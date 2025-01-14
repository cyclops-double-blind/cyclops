#include "cyclops/details/estimation/graph/node.hpp"
#include "cyclops/details/utils/type.hpp"

namespace cyclops::estimation {
  bool operator<(node_t::frame_t const& a, node_t::frame_t const& b) {
    return a.id < b.id;
  }

  bool operator<(node_t::bias_t const& a, node_t::bias_t const& b) {
    return a.id < b.id;
  }

  bool operator<(node_t::landmark_t const& a, node_t::landmark_t const& b) {
    return a.id < b.id;
  }

  bool node_t::operator<(node_t const& other) const {
    return this->variant < other.variant;
  }

  bool operator==(node_t::frame_t const& a, node_t::frame_t const& b) {
    return a.id == b.id;
  }

  bool operator==(node_t::bias_t const& a, node_t::bias_t const& b) {
    return a.id == b.id;
  }

  bool operator==(node_t::landmark_t const& a, node_t::landmark_t const& b) {
    return a.id == b.id;
  }

  bool node_t::operator==(node_t const& other) const {
    return this->variant == other.variant;
  }

  int node_t::frame_t::dimension() const {
    return 10;
  }

  int node_t::frame_t::manifold_dimension() const {
    return 9;
  }

  int node_t::bias_t::dimension() const {
    return 6;
  }

  int node_t::bias_t::manifold_dimension() const {
    return 6;
  }

  int node_t::landmark_t::dimension() const {
    return 3;
  }

  int node_t::landmark_t::manifold_dimension() const {
    return 3;
  }

  int node_t::dimension() const {
    return std::visit([](auto const& _) { return _.dimension(); }, variant);
  }

  int node_t::manifold_dimension() const {
    return std::visit(
      [](auto const& _) { return _.manifold_dimension(); }, variant);
  }

  std::ostream& operator<<(std::ostream& o, node_t const& node) {
    std::visit(
      overloaded {
        [&o](node_t::frame_t frame) { o << "frame(" << frame.id << ")"; },
        [&o](node_t::bias_t frame) { o << "bias(" << frame.id << ")"; },
        [&o](node_t::landmark_t node) { o << "landmark(" << node.id << ")"; },
      },
      node.variant);
    return o;
  }
}  // namespace cyclops::estimation
