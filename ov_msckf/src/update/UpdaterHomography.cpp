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

#include "UpdaterHomography.h"

#include "state/State.h"
#include "state/StateHelper.h"

#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterHomography::UpdaterHomography(UpdaterOptions &options, double sigma_rotation, double pure_rotation_ratio, double min_parallax_px,
                                     double sigma_trans, int camera_id)
    : _options(options), _sigma_r(sigma_rotation), _pure_rotation_ratio(pure_rotation_ratio), _min_parallax_px(min_parallax_px),
      _sigma_trans(sigma_trans), _cam_id(camera_id) {

  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

void UpdaterHomography::update(std::shared_ptr<State> state, std::shared_ptr<ov_core::FeatureDatabase> db, double altitude) {

  // ================================================================
  // 1. Get two most recent clone timestamps
  // ================================================================
  if ((int)state->_clones_IMU.size() < 2)
    return;

  auto it_end = state->_clones_IMU.end();
  --it_end;
  double t2 = it_end->first;
  --it_end;
  double t1 = it_end->first;

  // ================================================================
  // 2. Collect feature correspondences at t1 and t2
  // ================================================================
  std::vector<cv::Point2f> pts1_px, pts2_px;

  for (const auto &feat_pair : db->get_internal_data()) {
    auto &feat = feat_pair.second;
    if (feat->to_delete)
      continue;

    auto ts_it = feat->timestamps.find(_cam_id);
    if (ts_it == feat->timestamps.end())
      continue;
    const auto &times = ts_it->second;

    auto t1_it = std::find(times.begin(), times.end(), t1);
    if (t1_it == times.end())
      continue;
    auto t2_it = std::find(times.begin(), times.end(), t2);
    if (t2_it == times.end())
      continue;

    int idx1 = (int)std::distance(times.begin(), t1_it);
    int idx2 = (int)std::distance(times.begin(), t2_it);

    const auto &uvs = feat->uvs.at(_cam_id);
    pts1_px.push_back(cv::Point2f(uvs[idx1](0), uvs[idx1](1)));
    pts2_px.push_back(cv::Point2f(uvs[idx2](0), uvs[idx2](1)));
  }

  if ((int)pts1_px.size() < 8) {
    PRINT_DEBUG(YELLOW "[HOMO]: insufficient correspondences (%d < 8)\n" RESET, (int)pts1_px.size());
    return;
  }

  // ================================================================
  // 3. Minimum parallax gate (filter stationary / zero-motion case)
  //
  // NOTE: Under pure rotation parallax can be LARGE. We only check a
  // lower bound to skip the case where the drone is completely still
  // (tracking noise only). There is NO upper bound — large parallax
  // is normal and expected during fast yaw at altitude.
  // ================================================================
  double avg_parallax = 0.0;
  for (size_t i = 0; i < pts1_px.size(); i++) {
    double dx = pts2_px[i].x - pts1_px[i].x;
    double dy = pts2_px[i].y - pts1_px[i].y;
    avg_parallax += std::sqrt(dx * dx + dy * dy);
  }
  avg_parallax /= (double)pts1_px.size();

  if (avg_parallax < _min_parallax_px) {
    PRINT_DEBUG("[HOMO]: avg parallax %.2f px below min %.2f — scene stationary, skipping\n", avg_parallax, _min_parallax_px);
    return;
  }

  // ================================================================
  // 4. Estimate homography via RANSAC
  // ================================================================
  cv::Mat inlier_mask;
  cv::Mat H_cv = cv::findHomography(pts1_px, pts2_px, cv::RANSAC, 1.5, inlier_mask);

  if (H_cv.empty()) {
    PRINT_DEBUG(YELLOW "[HOMO]: findHomography failed\n" RESET);
    return;
  }

  int num_inliers = cv::countNonZero(inlier_mask);
  double inlier_ratio = (double)num_inliers / (double)pts1_px.size();

  // ================================================================
  // 5. Pure-rotation detection via inlier ratio
  //
  // Under pure rotation every feature — near or far — satisfies the
  // same H = K R K^{-1}, giving inlier ratio ≈ 1.0 regardless of
  // how large the pixel displacement is.
  //
  // Under translation features at different depths have different
  // flow, breaking the homography for off-plane points → ratio drops.
  // ================================================================
  if (inlier_ratio < _pure_rotation_ratio) {
    PRINT_DEBUG("[HOMO]: inlier ratio %.2f < %.2f — translation present, deferring to MSCKF\n", inlier_ratio, _pure_rotation_ratio);
    return;
  }
  PRINT_DEBUG("[HOMO]: pure rotation detected — parallax=%.1f px, inliers=%d/%d (%.0f%%)\n", avg_parallax, num_inliers,
              (int)pts1_px.size(), 100.0 * inlier_ratio);

  // ================================================================
  // 6. Decompose H into rotation + translation candidates
  // ================================================================
  cv::Matx33d K = state->_cam_intrinsics_cameras.at(_cam_id)->get_K();
  std::vector<cv::Mat> Rs, Ts, Ns;
  int num_solutions = cv::decomposeHomographyMat(H_cv, cv::Mat(K), Rs, Ts, Ns);

  if (num_solutions == 0) {
    PRINT_DEBUG(YELLOW "[HOMO]: decomposeHomographyMat returned no solutions\n" RESET);
    return;
  }

  // ================================================================
  // 7. Select best candidate: maximum N_z (camera Z component)
  //
  // For a downward-looking camera the ground plane normal in camera
  // frame points along +Z (optical axis). Among the (up to 4)
  // decomposition solutions, the physically correct one consistently
  // has the largest N_z — no IMU reference required.
  // ================================================================
  Eigen::Matrix3d R_GtoI1 = state->_clones_IMU.at(t1)->Rot();
  Eigen::Matrix3d R_GtoI2 = state->_clones_IMU.at(t2)->Rot();
  Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(_cam_id)->Rot();

  int best_idx = 0;
  for (int i = 1; i < num_solutions; i++) {
    std::stringstream ss;
    ss << "\nR[" << i << "] =\n" << Rs[i] << "\n";
    ss << "T[" << i << "] =\n" << Ts[i] << "\n";
    ss << "N[" << i << "] =\n" << Ns[i] << "\n";
    PRINT_DEBUG("%s", ss.str().c_str());
    Eigen::Vector3d N_cand, N_best;
    cv::cv2eigen(Ns[i], N_cand);
    cv::cv2eigen(Ns[best_idx], N_best);
    if (N_cand(2) > N_best(2))
      best_idx = i;
  }

  Eigen::Vector3d N_selected;
  cv::cv2eigen(Ns[best_idx], N_selected);

  if (altitude <= 35.0) {
    return;
  }
  // T_normalized from decomposeHomographyMat is noise-dominated for pure rotation.
  // Feature tracking pixel noise → T_normalized ≈ 0.003–0.006 → t_actual = altitude × noise ≈ 0.1–0.3m
  // in a random direction each frame. Trusting this drives a velocity random walk.
  // Since we have already confirmed pure rotation (inlier_ratio > threshold), the physical
  // translation is KNOWN to be zero — always force T_normalized = 0.
  Eigen::Matrix3d R_meas_cam;
  Eigen::Vector3d T_normalized;// = Eigen::Vector3d::Zero();

  cv::cv2eigen(Rs[best_idx], R_meas_cam);
  cv::cv2eigen(Ts[best_idx], T_normalized);


  // ================================================================
  // 8. EKF Combined Update: rotation (3-DOF) + translation (3-DOF)
  //
  // Rotation measurement: R_meas_cam (camera-frame C1→C2)
  //   R_meas_IMU = R_ItoC^T * R_meas_cam * R_ItoC
  //   res_rot = log_so3(R_meas_IMU * R_GtoI1 * R_GtoI2^T)
  //
  // Translation measurement: t_actual = altitude * T_normalized
  //   z_pred = R_ItoC * R_GtoI2 * (p_I1inG - p_I2inG)
  //   res_trans = t_actual - z_pred
  //
  // When n_is_zero: R_meas_cam=I, T_normalized=0 → zero-motion constraint
  // When altitude unavailable (!n_is_zero, altitude<=0.5): rotation-only (3-DOF)
  // ================================================================
  Eigen::AngleAxisd aa(R_meas_cam);

  double angle = aa.angle();                // 旋转角度（弧度）
  Eigen::Vector3d axis = aa.axis();         // 旋转轴（单位向量）

  double angle_deg = angle * 180.0 / M_PI; // 转角度制
  PRINT_DEBUG("Decomposed homography: angle=%.2f°, axis=[%.2f, %.2f, %.2f]\n", angle_deg, axis[0], axis[1], axis[2]);

  // if(angle_deg < 0.5) {
  //   PRINT_DEBUG("[HOMO]: rotation yaw angle too small (%.2f°) — skipping update\n", angle_deg);
  //   return;
  // }
  // Always zero: pure rotation means no translation (physical ground truth).
  Eigen::Matrix3d R_meas_IMU = R_ItoC.transpose() * R_meas_cam * R_ItoC;
  Eigen::Vector3d res_rot = log_so3(R_meas_IMU * R_GtoI1 * R_GtoI2.transpose());
  std::stringstream ss;
  ss << "\nR_meas_cam =\n" << R_meas_cam << "\n";
  ss << "R_meas_IMU =\n" << R_meas_IMU << "\n";
  ss << "res_rot =\n" << res_rot.transpose() << "\n";
  PRINT_DEBUG("%s", ss.str().c_str());

  Eigen::Matrix3d R_GtoI1_fej = state->_options.do_fej ? state->_clones_IMU.at(t1)->Rot_fej() : R_GtoI1;
  Eigen::Matrix3d R_GtoI2_fej = state->_options.do_fej ? state->_clones_IMU.at(t2)->Rot_fej() : R_GtoI2;
  Eigen::Matrix3d A = R_ItoC * R_GtoI2_fej;

  std::vector<std::shared_ptr<Type>> Hx_order;
  Hx_order.push_back(state->_clones_IMU.at(t1));
  Hx_order.push_back(state->_clones_IMU.at(t2));

  // T_normalized=0 always, so t_actual=0 regardless of altitude — no altitude guard needed.
  // 6-DOF combined update (rotation + translation)
  Eigen::Vector3d p_I1inG = state->_clones_IMU.at(t1)->pos();
  Eigen::Vector3d p_I2inG = state->_clones_IMU.at(t2)->pos();
  Eigen::Vector3d t_actual = altitude * T_normalized; // zero when n_is_zero
  Eigen::Vector3d z_pred = R_ItoC * R_GtoI2 * (p_I1inG - p_I2inG);
  Eigen::Vector3d res_trans = t_actual - z_pred;
  std::stringstream ss_t;
  ss_t << "\nT_normalized =\n" << T_normalized.transpose() << "\n";
  ss_t << "t_actual =\n" << t_actual.transpose() << "\n";
  ss_t << "z_pred =\n" << z_pred.transpose() << "\n";
  ss_t << "res_trans =\n" << res_trans.transpose() << "\n";
  PRINT_DEBUG("%s", ss_t.str().c_str());


  Eigen::MatrixXd H_combined = Eigen::MatrixXd::Zero(6, 12);
  H_combined.block<3, 3>(0, 0) = R_meas_IMU;              // ∂rot/∂δθ_I1
  H_combined.block<3, 3>(0, 6) = -R_meas_IMU * R_GtoI1_fej; // ∂rot/∂δθ_I2
  H_combined.block<3, 3>(3, 3) = -A;                       // ∂trans/∂δp_I1
  H_combined.block<3, 3>(3, 9) = A;                        // ∂trans/∂δp_I2

  Eigen::VectorXd res_combined(6);
  res_combined.head<3>() = res_rot;
  res_combined.tail<3>() = res_trans;

  Eigen::MatrixXd R_combined = Eigen::MatrixXd::Zero(6, 6);
  R_combined.block<3, 3>(0, 0) = std::pow(_sigma_r, 2) * Eigen::Matrix3d::Identity();
  R_combined.block<3, 3>(3, 3) = std::pow(_sigma_trans, 2) * Eigen::Matrix3d::Identity();

  // Current IMU velocity before update (for divergence tracing)
  Eigen::Vector3d v_before = state->_imu->vel();
  PRINT_DEBUG("[HOMO]: pre-update v_IinG=[%.3f,%.3f,%.3f] |v|=%.3f m/s\n",
              v_before.x(), v_before.y(), v_before.z(), v_before.norm());
  PRINT_DEBUG("[HOMO]: p_I1inG=[%.3f,%.3f,%.3f]  p_I2inG=[%.3f,%.3f,%.3f]  dp=[%.3f,%.3f,%.3f]\n",
              p_I1inG.x(), p_I1inG.y(), p_I1inG.z(),
              p_I2inG.x(), p_I2inG.y(), p_I2inG.z(),
              (p_I1inG - p_I2inG).x(), (p_I1inG - p_I2inG).y(), (p_I1inG - p_I2inG).z());

  // Chi2 gate (6 DOF) — log rot and trans chi2 contributions separately
  double chi2_rot_only, chi2_trans_only, chi2_combined;
  {
    Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd S = H_combined * P * H_combined.transpose() + R_combined;
    Eigen::VectorXd s_sol = S.llt().solve(res_combined);
    chi2_combined = res_combined.dot(s_sol);

    // Rotation-only chi2 (top-left 3×3 of S)
    Eigen::MatrixXd H_rot = H_combined.topRows<3>();
    Eigen::MatrixXd R_rot = R_combined.block<3,3>(0,0);
    Eigen::MatrixXd S_rot = H_rot * P * H_rot.transpose() + R_rot;
    chi2_rot_only = res_rot.dot(S_rot.llt().solve(res_rot));

    // Translation-only chi2 (bottom-right 3×3 of S)
    Eigen::MatrixXd H_trans = H_combined.bottomRows<3>();
    Eigen::MatrixXd R_trans = R_combined.block<3,3>(3,3);
    Eigen::MatrixXd S_trans = H_trans * P * H_trans.transpose() + R_trans;
    chi2_trans_only = res_trans.dot(S_trans.llt().solve(res_trans));

    double chi2_check = chi_squared_table[6];
    PRINT_DEBUG("[HOMO]: chi2 combined=%.3f (thr=%.3f×%.3f=%.3f)  rot_only=%.3f (thr3=%.3f)  trans_only=%.3f (thr3=%.3f)\n",
                chi2_combined, _options.chi2_multipler, chi2_check, _options.chi2_multipler * chi2_check,
                chi2_rot_only, chi_squared_table[3], chi2_trans_only, chi_squared_table[3]);

    if (chi2_combined > _options.chi2_multipler * chi2_check) {
      PRINT_DEBUG(YELLOW "[HOMO]: chi2 rejected %.3f > %.3f (N_z=%.3f, alt=%.1f m)\n" RESET, chi2_combined,
                  _options.chi2_multipler * chi2_check, N_selected(2), altitude);
      return;
    }
  }

  StateHelper::EKFUpdate(state, Hx_order, H_combined, res_combined, R_combined);

  // Post-update velocity (trace velocity correction from HOMO)
  Eigen::Vector3d v_after = state->_imu->vel();
  Eigen::Vector3d dv = v_after - v_before;
  PRINT_INFO(CYAN "[HOMO]: update — rot|res|=%.4f rad, trans|res|=%.4f m (t_act=[%.3f,%.3f,%.3f]), N_z=%.3f, alt=%.1f m\n" RESET,
             res_rot.norm(), res_trans.norm(), t_actual.x(), t_actual.y(), t_actual.z(), N_selected(2), altitude);
  PRINT_DEBUG("[HOMO]: post-update v_IinG=[%.3f,%.3f,%.3f] |v|=%.3f  dv=[%.3f,%.3f,%.3f] |dv|=%.4f m/s\n",
              v_after.x(), v_after.y(), v_after.z(), v_after.norm(),
              dv.x(), dv.y(), dv.z(), dv.norm());
}
