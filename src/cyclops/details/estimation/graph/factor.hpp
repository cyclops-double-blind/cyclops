#pragma once

#include "cyclops/details/type.hpp"
#include <variant>

namespace cyclops::estimation {
  struct factor_t {
    struct imu_t {
      frame_id_t from;
      frame_id_t to;
    };

    struct bias_walk_t {
      frame_id_t from;
      frame_id_t to;
    };

    struct bias_prior_t {
      frame_id_t frame;
    };

    struct feature_t {
      frame_id_t frame;
      landmark_id_t landmark;
    };

    struct prior_t {};

    std::variant<imu_t, bias_walk_t, bias_prior_t, feature_t, prior_t> variant;

    bool operator<(factor_t const& other) const;
    bool operator==(factor_t const& other) const;
  };

  namespace factor {
    static inline factor_t imu(frame_id_t from, frame_id_t to) {
      return {factor_t::imu_t {from, to}};
    }

    static inline factor_t bias_walk(frame_id_t from, frame_id_t to) {
      return {factor_t::bias_walk_t {from, to}};
    }

    static inline factor_t bias_prior(frame_id_t frame) {
      return {factor_t::bias_prior_t {frame}};
    }

    static inline factor_t feature(frame_id_t frame, landmark_id_t landmark) {
      return {factor_t::feature_t {frame, landmark}};
    }

    static inline factor_t prior() {
      return {factor_t::prior_t {}};
    }

    template <typename type>
    static bool is(factor_t const& factor) {
      return std::holds_alternative<type>(factor.variant);
    }
  }  // namespace factor

  bool operator<(factor_t::imu_t const& a, factor_t::imu_t const& b);
  bool operator<(
    factor_t::bias_walk_t const& a, factor_t::bias_walk_t const& b);
  bool operator<(
    factor_t::bias_prior_t const& a, factor_t::bias_prior_t const& b);
  bool operator<(factor_t::feature_t const& a, factor_t::feature_t const& b);
  bool operator<(factor_t::prior_t const& a, factor_t::prior_t const& b);

  bool operator==(factor_t::imu_t const& a, factor_t::imu_t const& b);
  bool operator==(
    factor_t::bias_walk_t const& a, factor_t::bias_walk_t const& b);
  bool operator==(
    factor_t::bias_prior_t const& a, factor_t::bias_prior_t const& b);
  bool operator==(factor_t::feature_t const& a, factor_t::feature_t const& b);
  bool operator==(factor_t::prior_t const& a, factor_t::prior_t const& b);

  std::ostream& operator<<(std::ostream&, factor_t const&);
}  // namespace cyclops::estimation
