/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_UPDATER_HOMOGRAPHY_H
#define OV_MSCKF_UPDATER_HOMOGRAPHY_H

#include <map>
#include <memory>

#include "UpdaterOptions.h"

namespace ov_core {
class FeatureDatabase;
} // namespace ov_core

namespace ov_msckf {

class State;

/**
 * @brief Homography-based rotation constraint updater for near-pure-rotation scenarios.
 *
 * Designed for high-altitude UAVs with a downward-facing monocular camera.
 * When the drone undergoes near-pure rotation (hovering+yaw, banking), standard MSCKF
 * triangulation fails because the baseline is too short relative to depth to recover
 * consistent depth. However, the homography H = K R K^{-1} is well-conditioned and
 * provides a valid rotation-only EKF constraint.
 *
 * ## Detection criterion
 *
 * Pure rotation is detected by a HIGH homography inlier ratio, NOT by parallax magnitude.
 * This distinction is critical:
 *
 *   - Under pure rotation ALL features — near and far — obey a SINGLE homography.
 *     A 10° yaw at 50 m altitude produces hundreds of pixels of optical flow, yet the
 *     homography inlier ratio remains ~1.0 because every feature follows the same R.
 *
 *   - Under translation the homography only fits features on one plane; features at
 *     different depths produce large residuals → the inlier ratio drops.
 *
 * Gates applied (in order):
 *   1. Minimum parallax > min_parallax_px  — skip if scene is stationary (just noise).
 *   2. Homography RANSAC succeeds.
 *   3. Inlier ratio > pure_rotation_ratio  — high ratio signals pure/near-pure rotation.
 *   4. Chi2 gate on the EKF residual.
 *
 * ## What is and is NOT constrained
 *
 * This updater provides a **rotation-only** measurement — it constrains the relative
 * orientation between the two latest clones. Velocity and position are NOT directly
 * constrained here; they remain governed by IMU propagation.
 *
 * Recommended complementary strategies:
 *   - Enable ZUPT (`try_zupt = true`) for hovering: ZUPT constrains velocity and
 *     position while this updater constrains attitude.
 *   - For translating+rotating flight: accept that position drifts on IMU alone during
 *     the pure-rotation interval; MSCKF resumes constraining position once translation
 *     is sufficient to lower the H inlier ratio below `pure_rotation_ratio`.
 *
 * ## EKF Update math
 *
 * Measured rotation in IMU frame: R_meas_IMU = R_ItoC^T * R_meas_cam * R_ItoC
 *
 * Residual (clone t1 = second-latest, clone t2 = latest):
 *   r = log_so3(R_meas_IMU * R_GtoI1 * R_GtoI2^T)   [3×1, want = 0]
 *
 * Jacobians (left JPL perturbation, H is 3×12):
 *   H(0:3, 0:3) = R_meas_IMU              (clone t1 rotation block)
 *   H(0:3, 6:9) = -R_meas_IMU * R_GtoI1  (clone t2 rotation block, FEJ if enabled)
 *   All position blocks = 0
 */
class UpdaterHomography {

public:
  /**
   * @brief Constructor
   * @param options Chi2 multiplier
   * @param sigma_rotation Rotation measurement noise in radians (default 1e-2 ≈ 0.57°)
   * @param pure_rotation_ratio H inlier ratio threshold to declare pure rotation (default 0.85)
   * @param min_parallax_px Minimum average pixel displacement to bother computing H (default 0.5 px)
   * @param sigma_trans Noise sigma for the altitude-scaled translation measurement (meters, default 0.1)
   * @param camera_id Camera ID to use (should be the downward-facing camera, default 0)
   */
  UpdaterHomography(UpdaterOptions &options, double sigma_rotation, double pure_rotation_ratio, double min_parallax_px,
                    double sigma_trans, int camera_id);

  /**
   * @brief Try to perform a homography-based rotation + translation update.
   *
   * Queries the feature database for correspondences between the two most recent clones,
   * estimates a homography via RANSAC, gates on inlier ratio to detect pure rotation,
   * decomposes H, and applies two sequential EKF updates:
   *   1. Rotation constraint (always, when pure rotation is detected).
   *   2. Translation constraint scaled by altitude (only when altitude > 0).
   * Silently returns (no update) when any gate fails.
   *
   * @param state    Current VIO state
   * @param db       Feature database from the active tracker
   * @param altitude Height above ground in meters from barometer (≤ 0 = unavailable).
   *                 When > 0: t_actual = altitude * (t/d) from H decomposition is used
   *                 as a metric translation measurement to constrain clone positions,
   *                 preventing position/velocity drift during the pure-rotation interval.
   */
  void update(std::shared_ptr<State> state, std::shared_ptr<ov_core::FeatureDatabase> db, double altitude = -1.0);

private:
  /// Updater options (chi2 multiplier)
  UpdaterOptions _options;

  /// Rotation measurement noise sigma (radians)
  double _sigma_r;

  /// Homography inlier ratio threshold to declare pure/near-pure rotation
  double _pure_rotation_ratio;

  /// Minimum average parallax in pixels — below this the scene is stationary, skip
  double _min_parallax_px;

  /// Noise sigma for zero-translation constraint between clones (meters)
  double _sigma_trans;

  /// Camera ID used for homography estimation
  int _cam_id;

  /// Precomputed chi-squared 95th-percentile table (index = DOF)
  std::map<int, double> chi_squared_table;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_HOMOGRAPHY_H
