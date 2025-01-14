#include "cyclops/details/estimation/graph/factor.hpp"
#include "cyclops/details/utils/type.hpp"

namespace cyclops::estimation {
  bool operator<(factor_t::imu_t const& a, factor_t::imu_t const& b) {
    if (a.from < b.from)
      return true;
    if (a.from > b.from)
      return false;
    return a.to < b.to;
  }

  bool operator<(
    factor_t::bias_walk_t const& a, factor_t::bias_walk_t const& b) {
    if (a.from < b.from)
      return true;
    if (a.from > b.from)
      return false;
    return a.to < b.to;
  }

  bool operator<(
    factor_t::bias_prior_t const& a, factor_t::bias_prior_t const& b) {
    return a.frame < b.frame;
  }

  bool operator<(factor_t::feature_t const& a, factor_t::feature_t const& b) {
    if (a.frame < b.frame)
      return true;
    if (a.frame > b.frame)
      return false;
    return a.landmark < b.landmark;
  }

  bool operator<(factor_t::prior_t const& a, factor_t::prior_t const& b) {
    return false;
  }

  bool operator==(factor_t::imu_t const& a, factor_t::imu_t const& b) {
    return a.from == b.from && a.to == b.to;
  }

  bool operator==(
    factor_t::bias_walk_t const& a, factor_t::bias_walk_t const& b) {
    return a.from == b.from && a.to == b.to;
  }

  bool operator==(
    factor_t::bias_prior_t const& a, factor_t::bias_prior_t const& b) {
    return a.frame == b.frame;
  }

  bool operator==(factor_t::feature_t const& a, factor_t::feature_t const& b) {
    return a.frame == b.frame && a.landmark == b.landmark;
  }

  bool operator==(factor_t::prior_t const& a, factor_t::prior_t const& b) {
    return true;
  }

  bool factor_t::operator<(factor_t const& other) const {
    return this->variant < other.variant;
  }

  bool factor_t::operator==(factor_t const& other) const {
    return this->variant == other.variant;
  }

  std::ostream& operator<<(std::ostream& ostr, factor_t const& factor) {
    auto visitor = overloaded {
      [&ostr](factor_t::imu_t const& _) {
        ostr << "IMU [" << _.from << ", " << _.to << "]";
      },
      [&ostr](factor_t::bias_walk_t const& _) {
        ostr << "IMU walk [" << _.from << ", " << _.to << "]";
      },
      [&ostr](factor_t::bias_prior_t const& _) {
        ostr << "IMU bias prior [" << _.frame << "]";
      },
      [&ostr](factor_t::feature_t const& _) {
        ostr << "Feature [" << _.frame << ", " << _.landmark << "]";
      },
      [&ostr](factor_t::prior_t const& _) { ostr << "Prior"; },
    };
    std::visit(visitor, factor.variant);
    return ostr;
  }
}  // namespace cyclops::estimation
