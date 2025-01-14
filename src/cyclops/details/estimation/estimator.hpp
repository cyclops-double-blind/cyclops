#pragma once

#include <memory>

namespace cyclops::estimation {
  class EstimationSanityDiscriminator;
  class LikelihoodOptimizer;
  class MarginalizationManager;

  class EstimationFrameworkMain {
  public:
    virtual ~EstimationFrameworkMain() = default;
    virtual void reset() = 0;

    virtual bool updateEstimation() = 0;
    virtual bool sanity() const = 0;

    static std::unique_ptr<EstimationFrameworkMain> create(
      std::unique_ptr<LikelihoodOptimizer> optimizer,
      std::unique_ptr<MarginalizationManager> marginalizer,
      std::unique_ptr<EstimationSanityDiscriminator> sanity_discriminator);
  };
}  // namespace cyclops::estimation
