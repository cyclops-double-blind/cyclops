#include "cyclops/details/initializer/vision/twoview.hpp"

#include "cyclops/details/initializer/vision/epipolar.hpp"
#include "cyclops/details/initializer/vision/homography.hpp"
#include "cyclops/details/initializer/vision/hypothesis.hpp"
#include "cyclops/details/utils/vision.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using std::set;
  using std::vector;

  using Eigen::Matrix2d;
  using Eigen::Matrix3d;

  using two_view_data_t = std::map<landmark_id_t, two_view_feature_pair_t>;

  static vector<set<landmark_id_t>> make_ransac_batch(
    int size, set<landmark_id_t> const& features, std::mt19937& rgen) {
    if (features.size() < 8)
      return {};

    return  //
      views::ints(0, size) | views::transform([&](auto _) {
        return features | views::sample(8, rgen) | ranges::to<set>;
      }) |
      ranges::to_vector;
  }

  enum geometry_model_selection_t {
    FAILURE,
    EPIPOLAR,
    HOMOGRAPHY,
  };

  class TwoViewVisionGeometrySolverImpl: public TwoViewVisionGeometrySolver {
  private:
    std::unique_ptr<TwoViewMotionHypothesisSelector> _motion_selector;

    std::shared_ptr<cyclops_global_config_t const> _config;
    std::shared_ptr<std::mt19937> _rgen;

    geometry_model_selection_t selectGeometryModel(
      epipolar_analysis_t const& epipolar,
      homography_analysis_t const& homography) const;

    vector<two_view_geometry_t> solveHomography(
      homography_analysis_t const& homography, two_view_data_t const& features,
      two_view_imu_rotation_data_t const& rotation_prior);
    vector<two_view_geometry_t> solveEpipolar(
      epipolar_analysis_t const& epipolar, two_view_data_t const& features,
      two_view_imu_rotation_data_t const& rotation_prior);

    vector<two_view_geometry_t> solveGeometry(
      two_view_correspondence_data_t const& correspondence);

  public:
    TwoViewVisionGeometrySolverImpl(
      std::unique_ptr<TwoViewMotionHypothesisSelector> motion_selector,
      std::shared_ptr<cyclops_global_config_t const> config,
      std::shared_ptr<std::mt19937> rgen);
    ~TwoViewVisionGeometrySolverImpl();
    void reset() override;

    vector<two_view_geometry_t> solve(
      two_view_correspondence_data_t const& correspondence) override;
  };

  TwoViewVisionGeometrySolverImpl::TwoViewVisionGeometrySolverImpl(
    std::unique_ptr<TwoViewMotionHypothesisSelector> motion_selector,
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<std::mt19937> rgen)
      : _motion_selector(std::move(motion_selector)),
        _config(config),
        _rgen(rgen) {
  }

  TwoViewVisionGeometrySolverImpl::~TwoViewVisionGeometrySolverImpl() = default;

  void TwoViewVisionGeometrySolverImpl::reset() {
    _motion_selector->reset();
  }

  geometry_model_selection_t
  TwoViewVisionGeometrySolverImpl::selectGeometryModel(
    epipolar_analysis_t const& epipolar,
    homography_analysis_t const& homography) const {
    auto S_H = homography.expected_inliers;
    auto S_E = epipolar.expected_inliers;

    __logger__->trace("Selecting two-view vision initialization model.");
    __logger__->trace("Homography expected number of inliers: {}", S_H);
    __logger__->trace("Epipolar expected number of inliers: {}", S_E);

    if (S_H <= 0 && S_E <= 0) {
      // both of them failed. abort
      __logger__->debug(
        "Initialization model selection failed; S_H: {}, S_E: {}", S_H, S_E);
      return FAILURE;
    }

    auto R_H = S_H / (S_H + S_E);
    auto const& model_selection_config =
      _config->initialization.vision.two_view.model_selection;

    if (R_H > model_selection_config.homography_selection_score_threshold) {
      __logger__->debug("Selecting homography model");
      return HOMOGRAPHY;
    } else {
      __logger__->debug("Selecting epipolar model");
      return EPIPOLAR;
    }
  }

  vector<two_view_geometry_t> TwoViewVisionGeometrySolverImpl::solveHomography(
    homography_analysis_t const& homography, two_view_data_t const& data,
    two_view_imu_rotation_data_t const& rotation_prior) {
    auto const& [expected_inliers, H, inliers] = homography;
    if (expected_inliers <= 0)
      return {};

    auto hypotheses = solve_homography_motion_hypothesis(H);
    if (hypotheses.size() != 8)
      return {};

    return _motion_selector->selectPossibleMotions(
      hypotheses, data, inliers, rotation_prior);
  }

  vector<two_view_geometry_t> TwoViewVisionGeometrySolverImpl::solveEpipolar(
    epipolar_analysis_t const& epipolar, two_view_data_t const& data,
    two_view_imu_rotation_data_t const& rotation_prior) {
    auto const& [expected_inliers, E, inliers] = epipolar;
    if (expected_inliers <= 0)
      return {};

    auto hypotheses = solve_epipolar_motion_hypothesis(E);
    if (hypotheses.size() != 4)
      return {};

    return _motion_selector->selectPossibleMotions(
      hypotheses, data, inliers, rotation_prior);
  }

  vector<two_view_geometry_t> TwoViewVisionGeometrySolverImpl::solveGeometry(
    two_view_correspondence_data_t const& correspondence) {
    auto tic = std::chrono::steady_clock::now();
    auto const& vision_config = _config->initialization.vision;
    auto const& features = correspondence.features;
    auto const& rotation_prior = correspondence.rotation_prior;

    __logger__->trace("Solving two-view structure for vision initialization.");
    __logger__->trace("Number of common features: {}", features.size());

    auto rotation_vector = so3_logmap(rotation_prior.value);
    __logger__->debug(
      "Two-view rotation prior: {}", rotation_vector.transpose());

    auto landmark_ids = features | views::keys | ranges::to<set>;
    auto ransac_batch = make_ransac_batch(
      vision_config.two_view.model_selection.ransac_batch_size, landmark_ids,
      *_rgen);
    auto homography = analyze_two_view_homography(
      vision_config.feature_point_isotropic_noise, ransac_batch, features);
    auto epipolar = analyze_two_view_epipolar(
      vision_config.feature_point_isotropic_noise, ransac_batch, features);

    auto model_selection = selectGeometryModel(epipolar, homography);
    auto toc = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration<double>(toc - tic).count();

    __logger__->debug(
      "Two-view geometry model selection complete. duration: {}", dt);

    switch (model_selection) {
    case FAILURE:
      return {};

    case EPIPOLAR: {
      auto solutions = solveEpipolar(epipolar, features, rotation_prior);
      if (!solutions.empty())
        return solutions;

      __logger__->debug(
        "Epipolar reconstruction failed. Fallback to homography model");
      return solveHomography(homography, features, rotation_prior);
    }

    case HOMOGRAPHY: {
      auto solutions = solveHomography(homography, features, rotation_prior);
      if (!solutions.empty())
        return solutions;

      __logger__->debug(
        "Homography reconstruction failed. Fallback to epipolar model");
      return solveEpipolar(epipolar, features, rotation_prior);
    }
    }
  }

  vector<two_view_geometry_t> TwoViewVisionGeometrySolverImpl::solve(
    two_view_correspondence_data_t const& correspondence) {
    return solveGeometry(correspondence);
  }

  std::unique_ptr<TwoViewVisionGeometrySolver>
  TwoViewVisionGeometrySolver::create(
    std::shared_ptr<cyclops_global_config_t const> config,
    std::shared_ptr<std::mt19937> rgen) {
    return std::make_unique<TwoViewVisionGeometrySolverImpl>(
      TwoViewMotionHypothesisSelector::create(config), config, rgen);
  }
}  // namespace cyclops::initializer
