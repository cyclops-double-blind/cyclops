#include "cyclops/details/estimation/estimator.hpp"
#include "cyclops/details/estimation/optimizer.hpp"
#include "cyclops/details/estimation/sanity.hpp"
#include "cyclops/details/estimation/graph/graph.hpp"

#include "cyclops/details/estimation/marginalizer/marginalizer.hpp"

namespace cyclops::estimation {
  class EstimationFrameworkMainImpl: public EstimationFrameworkMain {
  private:
    std::unique_ptr<LikelihoodOptimizer> _optimizer;
    std::unique_ptr<MarginalizationManager> _marginalizer;
    std::unique_ptr<EstimationSanityDiscriminator> _sanity_discriminator;

  public:
    EstimationFrameworkMainImpl(
      std::unique_ptr<LikelihoodOptimizer> optimizer,
      std::unique_ptr<MarginalizationManager> marginalizer,
      std::unique_ptr<EstimationSanityDiscriminator> sanity_discriminator);
    void reset() override;

    bool updateEstimation() override;
    bool sanity() const override;
  };

  EstimationFrameworkMainImpl::EstimationFrameworkMainImpl(
    std::unique_ptr<LikelihoodOptimizer> optimizer,
    std::unique_ptr<MarginalizationManager> marginalizer,
    std::unique_ptr<EstimationSanityDiscriminator> sanity_discriminator)
      : _optimizer(std::move(optimizer)),
        _marginalizer(std::move(marginalizer)),
        _sanity_discriminator(std::move(sanity_discriminator)) {
  }

  void EstimationFrameworkMainImpl::reset() {
    _optimizer->reset();
    _marginalizer->reset();
    _sanity_discriminator->reset();
  }

  bool EstimationFrameworkMainImpl::updateEstimation() {
    auto maybe_optimization = _optimizer->optimize(_marginalizer->prior());
    if (!maybe_optimization.has_value()) {
      _marginalizer->marginalize();
      return false;
    }

    _sanity_discriminator->update(
      maybe_optimization->landmark_sanity_statistics,
      maybe_optimization->optimizer_sanity_statistics);
    _marginalizer->marginalize(*maybe_optimization->graph);

    return true;
  }

  bool EstimationFrameworkMainImpl::sanity() const {
    return _sanity_discriminator->sanity();
  }

  std::unique_ptr<EstimationFrameworkMain> EstimationFrameworkMain::create(
    std::unique_ptr<LikelihoodOptimizer> optimizer,
    std::unique_ptr<MarginalizationManager> marginalizer,
    std::unique_ptr<EstimationSanityDiscriminator> sanity_discriminator) {
    return std::make_unique<EstimationFrameworkMainImpl>(
      std::move(optimizer), std::move(marginalizer),
      std::move(sanity_discriminator));
  }
}  // namespace cyclops::estimation
