#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include <range/v3/all.hpp>

namespace cyclops::initializer {
  using Eigen::MatrixXd;
  using Vector6i = Eigen::Matrix<int, 6, 1>;

  namespace views = ranges::views;

  std::tuple<
    imu_match_camera_rotation_prior_t, imu_match_camera_translation_prior_t>
  make_imu_match_camera_motion_prior(vision_bootstrap_solution_t const& sfm) {
    auto const& geometry = sfm.geometry;

    if (geometry.camera_motions.size() == 0)
      return {};

    // discount by one to handle symmetry
    auto n = static_cast<int>(geometry.camera_motions.size() - 1);

    Eigen::VectorXi orientation_indices(3 * n);
    Eigen::VectorXi translation_indices(3 * n);
    for (int i = 0; i < n; i++) {
      auto m = 6 * i + 6;
      orientation_indices.segment(3 * i, 3) << m + 0, m + 1, m + 2;
      translation_indices.segment(3 * i, 3) << m + 3, m + 4, m + 5;
    }
    Vector6i drop_indices;
    drop_indices << 0, 1, 2, 3, 4, 5;

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(6 * (n + 1));
    perm.indices() << translation_indices, orientation_indices, drop_indices;

    auto const& weight = sfm.motion_information_weight;
    MatrixXd H = (perm.transpose() * weight * perm).topLeftCorner(6 * n, 6 * n);

    MatrixXd const H_pp = H.block(0, 0, 3 * n, 3 * n);
    MatrixXd const H_pr = H.block(0, 3 * n, 3 * n, 3 * n);
    MatrixXd const H_rp = H.block(3 * n, 0, 3 * n, 3 * n);
    MatrixXd const H_rr = H.block(3 * n, 3 * n, 3 * n, 3 * n);

    Eigen::LDLT<MatrixXd> H_pp__inv(H_pp);
    Eigen::LDLT<MatrixXd> H_rr__inv(H_rr);
    auto orientation_prior = imu_match_camera_rotation_prior_t {
      // clang-format off
      .rotations = geometry.camera_motions
        | views::transform([](auto const& id_frame) {
          auto const& [id, frame] = id_frame;
          return std::make_pair(id, frame.rotation);
        })
        | ranges::to<std::map<frame_id_t, Eigen::Quaterniond>>,
      // clang-format on
      .weight = H_rr - H_rp * H_pp__inv.solve(H_pr),
    };
    auto translation_prior = imu_match_camera_translation_prior_t {
      // clang-format off
      .translations = geometry.camera_motions
        | views::transform([](auto const& id_frame) {
          auto const& [id, frame] = id_frame;
          return std::make_pair(id, frame.translation);
        })
        | ranges::to<std::map<frame_id_t, Eigen::Vector3d>>,
      // clang-format on
      .weight = H_pp - H_pr * H_rr__inv.solve(H_rp),
    };
    return std::make_tuple(orientation_prior, translation_prior);
  }
}  // namespace cyclops::initializer
