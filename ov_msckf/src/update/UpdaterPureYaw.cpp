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

#include "UpdaterPureYaw.h"

#include "UpdaterHelper.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterPureYaw::UpdaterPureYaw(const NoiseManager &noises, double min_altitude, double gyro_min_rate, double yaw_dom_ratio,
                               double sigma_depth)
    : _noises(noises), _min_altitude(min_altitude), _gyro_min_rate(gyro_min_rate), _yaw_dom_ratio(yaw_dom_ratio),
      _sigma_depth(sigma_depth) {
  _noises.sigma_w_2 = std::pow(_noises.sigma_w, 2);
  _noises.sigma_a_2 = std::pow(_noises.sigma_a, 2);
  _noises.sigma_wb_2 = std::pow(_noises.sigma_wb, 2);
  _noises.sigma_ab_2 = std::pow(_noises.sigma_ab, 2);
}

bool UpdaterPureYaw::try_update(std::shared_ptr<State> state, double timestamp, double altitude) {

  // Return if we don't have any imu data yet
  if (imu_data.empty())
    return false;

  // Return if the state is already at the desired time
  if (state->_timestamp == timestamp)
    return false;

  // Return if altitude is below the activation threshold or unknown
  if (altitude < _min_altitude)
    return false;

  // Set the last time offset value if we have just started the system up
  if (!have_last_prop_time_offset) {
    last_prop_time_offset = state->_calib_dt_CAMtoIMU->value()(0);
    have_last_prop_time_offset = true;
  }

  // Get what our IMU-camera offset should be (t_imu = t_cam + calib_dt)
  double t_off_new = state->_calib_dt_CAMtoIMU->value()(0);

  // Construct IMU bounding interval
  double time0 = state->_timestamp + last_prop_time_offset;
  double time1 = timestamp + t_off_new;

  // Select bounding inertial measurements
  std::vector<ImuData> imu_recent = Propagator::select_imu_readings(imu_data, time0, time1, false);

  // Move forward in time
  last_prop_time_offset = t_off_new;

  // Check that we have at least one measurement to process
  if (imu_recent.size() < 2)
    return false;

  // IMU intrinsic calibration estimates (static)
  Eigen::Matrix3d Dw = State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  Eigen::Matrix3d Da = State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  Eigen::Matrix3d Tg = State::Tg(state->_calib_imu_tg->value());

  // -------------------------------------------------------------------------
  // Phase 1: compute mean bias-corrected angular velocity in IMU frame
  // -------------------------------------------------------------------------
  Eigen::Vector3d w_sum = Eigen::Vector3d::Zero();
  double dt_summed = 0.0;
  for (size_t i = 0; i < imu_recent.size() - 1; i++) {
    double dt = imu_recent.at(i + 1).timestamp - imu_recent.at(i).timestamp;
    Eigen::Vector3d a_hat = state->_calib_imu_ACCtoIMU->Rot() * Da * (imu_recent.at(i).am - state->_imu->bias_a());
    Eigen::Vector3d w_hat = state->_calib_imu_GYROtoIMU->Rot() * Dw * (imu_recent.at(i).wm - state->_imu->bias_g() - Tg * a_hat);
    w_sum += w_hat * dt;
    dt_summed += dt;
  }
  if (dt_summed <= 0.0)
    return false;

  Eigen::Vector3d w_mean_I = w_sum / dt_summed;
  double w_norm = w_mean_I.norm();

  // Guard: must be rotating fast enough to be meaningful
  if (w_norm < _gyro_min_rate)
    return false;

  // Guard: rotation must be dominated by the yaw component in global frame
  // R_GtoI^T transforms from global to IMU → R_GtoI.transpose() rotates IMU vector to global
  Eigen::Vector3d w_mean_G = state->_imu->Rot().transpose() * w_mean_I;
  if (std::abs(w_mean_G(2)) / w_norm < _yaw_dom_ratio)
    return false;

  PRINT_INFO(CYAN "[PURE-YAW]: detected, w_norm=%.3f rad/s, |w_G_z|/|w_G|=%.2f, altitude=%.1f m\n" RESET, w_norm,
             std::abs(w_mean_G(2)) / w_norm, altitude);

  //===================================================================================
  // EKF Step 1: Rotation update using gyro constraint
  //
  // State variables updated: [q_GtoI (3-dof), bg (3-dof)]
  // Measurement model: w_meas = w_true + bg + noise  →  residual = -(w_hat)  (whitened)
  // Jacobian: H_q = 0 (no direct orientation coupling), H_bg = -w_omega * I
  //===================================================================================

  std::vector<std::shared_ptr<Type>> Hx_order_rot;
  Hx_order_rot.push_back(state->_imu->q());
  Hx_order_rot.push_back(state->_imu->bg());

  int m_size_rot = 3 * (int)(imu_recent.size() - 1);
  Eigen::MatrixXd H_rot = Eigen::MatrixXd::Zero(m_size_rot, 6);
  Eigen::VectorXd res_rot = Eigen::VectorXd::Zero(m_size_rot);

  double dt_summed2 = 0.0;
  for (size_t i = 0; i < imu_recent.size() - 1; i++) {
    double dt = imu_recent.at(i + 1).timestamp - imu_recent.at(i).timestamp;
    Eigen::Vector3d a_hat = state->_calib_imu_ACCtoIMU->Rot() * Da * (imu_recent.at(i).am - state->_imu->bias_a());
    Eigen::Vector3d w_hat = state->_calib_imu_GYROtoIMU->Rot() * Dw * (imu_recent.at(i).wm - state->_imu->bias_g() - Tg * a_hat);

    // Whitening weight: 1 / (sigma_w / sqrt(dt)) = sqrt(dt) / sigma_w
    double w_omega = std::sqrt(dt) / _noises.sigma_w;

    // Residual: true angular velocity treated as zero (whitened)
    res_rot.block(3 * i, 0, 3, 1) = -w_omega * w_hat;

    // Jacobian w.r.t. bg (columns 3..5); Jacobian w.r.t. q_GtoI (columns 0..2) stays zero
    H_rot.block(3 * i, 3, 3, 3) = -w_omega * Eigen::Matrix3d::Identity();

    dt_summed2 += dt;
  }

  // Compress the over-determined system
  UpdaterHelper::measurement_compress_inplace(H_rot, res_rot);

  if (H_rot.rows() >= 1) {
    // Inflate gyro bias covariance to model random walk over the elapsed interval
    // G*Qd*G^T = dt * Qc
    Eigen::MatrixXd Q_bg = dt_summed2 * _noises.sigma_wb_2 * Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd Phi_bg = Eigen::MatrixXd::Identity(3, 3);
    std::vector<std::shared_ptr<Type>> bg_order;
    bg_order.push_back(state->_imu->bg());
    StateHelper::EKFPropagation(state, bg_order, bg_order, Phi_bg, Q_bg);

    // Identity noise after whitening
    Eigen::MatrixXd R_rot = Eigen::MatrixXd::Identity(H_rot.rows(), H_rot.rows());
    StateHelper::EKFUpdate(state, Hx_order_rot, H_rot, res_rot, R_rot);
  }

  //===================================================================================
  // EKF Step 2: Depth-pin pseudo-measurement per SLAM feature
  //
  // Zero-residual pseudo-measurement with uncertainty proportional to sigma_depth.
  // This caps the depth covariance growth so features survive the chi2 check on
  // re-entry into the normal update pipeline.
  //===================================================================================
  using Repr = LandmarkRepresentation::Representation;

  for (auto &kv : state->_features_SLAM) {
    auto &lm = kv.second;

    // Skip ARUCO features (id range is [1, 4*max_aruco_features])
    if ((int)lm->_featid <= 4 * state->_options.max_aruco_features)
      continue;

    if (lm->_feat_representation == Repr::ANCHORED_INVERSE_DEPTH_SINGLE) {
      // State is rho = 1/z (scalar).  Depth-pin constrains rho directly.
      double rho = lm->value()(0);
      if (rho <= 0.0)
        continue;
      double z = 1.0 / rho;
      // Plausibility guard: feature depth should be within [0.5, 3.0] * altitude
      if (z < 0.5 * altitude || z > 3.0 * altitude)
        continue;

      // sigma_rho propagated from sigma_depth via rho = 1/z → drho/dz = -1/z² ≈ -rho²
      double sigma_rho = _sigma_depth * rho * rho;

      Eigen::MatrixXd H_pin = Eigen::MatrixXd::Identity(1, 1);
      Eigen::VectorXd res_pin = Eigen::VectorXd::Zero(1);
      Eigen::MatrixXd R_pin = (sigma_rho * sigma_rho) * Eigen::MatrixXd::Identity(1, 1);

      std::vector<std::shared_ptr<Type>> pin_order;
      pin_order.push_back(lm);
      StateHelper::EKFUpdate(state, pin_order, H_pin, res_pin, R_pin);

    } else if (lm->_feat_representation == Repr::ANCHORED_3D || lm->_feat_representation == Repr::GLOBAL_3D) {
      // State is [x, y, z] (3D).  Depth-pin constrains the z component.
      Eigen::Vector3d p = lm->get_xyz(false);
      double z = p(2);
      // Plausibility guard
      if (z < 0.5 * altitude || z > 3.0 * altitude)
        continue;

      // H selects the z-component of the 3-DOF landmark state
      Eigen::MatrixXd H_pin = Eigen::MatrixXd::Zero(1, 3);
      H_pin(0, 2) = 1.0;
      Eigen::VectorXd res_pin = Eigen::VectorXd::Zero(1);
      Eigen::MatrixXd R_pin = (_sigma_depth * _sigma_depth) * Eigen::MatrixXd::Identity(1, 1);

      std::vector<std::shared_ptr<Type>> pin_order;
      pin_order.push_back(lm);
      StateHelper::EKFUpdate(state, pin_order, H_pin, res_pin, R_pin);
    }
    // Other representations (ANCHORED_FULL_INVERSE_DEPTH etc.) are left untouched.
  }

  //===================================================================================
  // Step 3: Reset SLAM feature update fail counts
  //
  // Clearing update_fail_count prevents the SLAM updater from marking features for
  // marginalization on the next normal frame (threshold is > 1 consecutive failures).
  //===================================================================================
  for (auto &kv : state->_features_SLAM) {
    if (kv.second->update_fail_count > 0) {
      kv.second->update_fail_count = 0;
    }
  }

  return true;
}
