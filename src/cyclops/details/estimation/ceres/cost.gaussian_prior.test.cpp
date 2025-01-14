#include "cyclops/details/estimation/ceres/cost.gaussian_prior.cpp"
#include "cyclops/details/estimation/ceres/manifold.se3.hpp"
#include "cyclops/details/estimation/graph/node.hpp"
#include "cyclops/details/estimation/state/state_block.hpp"

#include "cyclops_tests/random.ipp"
#include "cyclops_tests/state.hpp"

#include <range/v3/all.hpp>

#include <doctest/doctest.h>

namespace cyclops::estimation {
  using se3_parameterization_t =
    ceres::AutoDiffLocalParameterization<ExponentialSE3Plus<false>, 10, 9>;

  TEST_CASE("Gaussian prior cost evaluation") {
    std::mt19937 rgen(20211104);

    GIVEN("Arbitrary nominal state") {
      auto state = imu_motion_state_t {
        .orientation = perturbate(Quaterniond::Identity(), 0.1, rgen),
        .position = perturbate(Vector3d::Zero().eval(), 0.1, rgen),
        .velocity = perturbate(Vector3d::Zero().eval(), 0.1, rgen),
      };
      auto block_nominal = make_motion_frame_parameter(state);

      GIVEN("A prior information of random weight") {
        auto prior = gaussian_prior_t {
          .jacobian = make_random_matrix(rgen, 15, 15, 100),
          .residual = VectorXd::Zero(15),
          .input_nodes = {node::frame(0), node::bias(0)},
          .nominal_parameters =
            std::vector<double>(block_nominal.begin(), block_nominal.end()),
        };

        WHEN("Optimized the prior cost") {
          auto block_optimized = make_perturbated_frame_state(state, 0.1, rgen);
          auto x = block_optimized.data();
          auto b = block_optimized.data() + 10;

          ceres::Problem problem;
          problem.AddParameterBlock(x, 10, new se3_parameterization_t);
          problem.AddParameterBlock(b, 6);

          problem.AddResidualBlock(
            new GaussianPriorCost(prior), nullptr, {x, b});

          ceres::Solver::Options options;
          options.dense_linear_algebra_library_type = ceres::EIGEN;
          options.linear_solver_type = ceres::DENSE_SCHUR;
          options.max_num_iterations = 10;
          options.max_solver_time_in_seconds = 1.0;

          ceres::Solver::Summary summary;
          ceres::Solve(options, &problem, &summary);

          THEN("The solution is converged") {
            CAPTURE(summary.FullReport());
            CHECK(std::abs(summary.final_cost) < 1e-6);

            auto x_got = motion_state_of_motion_frame_block(block_optimized);
            auto b_a_got = acc_bias_of_motion_frame_block(block_optimized);
            auto b_w_got = gyr_bias_of_motion_frame_block(block_optimized);

            AND_THEN("The optimized orientation is correct") {
              auto const& q_got = x_got.orientation;
              auto const& q_sol = state.orientation;
              CAPTURE(q_got.coeffs().transpose());
              CAPTURE(q_sol.coeffs().transpose());

              CHECK(q_got.isApprox(q_sol, 1e-6));
            }

            AND_THEN("The optimized position is correct") {
              auto const& p_got = x_got.position;
              auto const& p_sol = state.position;
              CAPTURE(p_got.transpose());
              CAPTURE(p_sol.transpose());

              CHECK(p_got.isApprox(p_sol, 1e-6));
            }

            AND_THEN("The optimized velocity is correct") {
              auto const& v_got = x_got.velocity;
              auto const& v_sol = state.velocity;
              CAPTURE(v_got.transpose());
              CAPTURE(v_sol.transpose());

              CHECK(v_got.isApprox(v_sol, 1e-6));
            }

            AND_THEN("The optimized accelerometer bias is correct") {
              CAPTURE(b_a_got.transpose());
              CHECK(b_a_got.norm() < 1e-6);
            }

            AND_THEN("The optimized gyrometer bias is correct") {
              CAPTURE(b_w_got.transpose());
              CHECK(b_w_got.norm() < 1e-6);
            }
          }
        }
      }
    }
  }
}  // namespace cyclops::estimation
