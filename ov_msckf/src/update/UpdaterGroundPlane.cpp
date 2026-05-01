/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
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

#include "UpdaterGroundPlane.h"

#include "feat/Feature.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LandmarkRepresentation.h"
#include "update/UpdaterHelper.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/math/distributions/chi_squared.hpp>

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <unordered_set>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

// ============================================================
// File-scope helper: convert any SLAM landmark to global 3D
// ============================================================

// Returns false if the anchor clone is no longer in state (marginalized).
static bool lm_global_pos(const std::shared_ptr<State> &state, const std::shared_ptr<Landmark> &lm, Eigen::Vector3d &pos_out) {
  if (!LandmarkRepresentation::is_relative_representation(lm->_feat_representation)) {
    // GLOBAL_3D or GLOBAL_FULL_INVERSE_DEPTH: get_xyz returns global frame directly
    pos_out = lm->get_xyz(false);
    return true;
  }
  // Anchored representation — require anchor clone and calibration still in state
  if (lm->_anchor_cam_id < 0)
    return false;
  if (state->_clones_IMU.find(lm->_anchor_clone_timestamp) == state->_clones_IMU.end())
    return false;
  if (state->_calib_IMUtoCAM.find(lm->_anchor_cam_id) == state->_calib_IMUtoCAM.end())
    return false;

  Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(lm->_anchor_cam_id)->Rot();
  Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(lm->_anchor_cam_id)->pos();
  Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(lm->_anchor_clone_timestamp)->Rot();
  Eigen::Vector3d p_IinG = state->_clones_IMU.at(lm->_anchor_clone_timestamp)->pos();

  // get_xyz(false) returns position in anchor camera frame for all anchored types
  Eigen::Vector3d p_FinA = lm->get_xyz(false);
  pos_out = R_GtoI.transpose() * R_ItoC.transpose() * (p_FinA - p_IinC) + p_IinG;
  return true;
}

// ============================================================
// Construction
// ============================================================

UpdaterGroundPlane::UpdaterGroundPlane(const Options &opt) : _opt(opt) {}

// ============================================================
// Public interface
// ============================================================

void UpdaterGroundPlane::feed_height(double timestamp, double h_rel) {
  _height_buf.push_back({timestamp, h_rel});
  while (_height_buf.size() > kMaxHeightBufSize)
    _height_buf.pop_front();
}

Eigen::Vector3d UpdaterGroundPlane::get_cp() const {
  return _plane_initialized ? _plane->cp() : Eigen::Vector3d::Zero();
}

void UpdaterGroundPlane::try_height_update(std::shared_ptr<State> state, double timestamp) {
  if (!_opt.enable_plane)
    return;
  if (state->_clones_IMU.empty())
    return;

  calibrate_offset_if_needed(state, timestamp);

  if (_opt.enable_height && _offset_initialized) {
    do_height_update(state, timestamp);
  }
}

void UpdaterGroundPlane::try_update(std::shared_ptr<State> state, double timestamp,
                                     const std::vector<std::shared_ptr<ov_core::Feature>> &leftovers) {
  if (!_opt.enable_plane)
    return;
  if (state->_clones_IMU.empty())
    return;

  // Homography plane update runs as a one-shot pose update on height-recovered ground points.
  if (_opt.enable_homography_plane_update) {
    apply_homography_plane_update(state, timestamp, leftovers);
  }

  // 1. Waiting period: attempt plane initialisation
  if (!_plane_initialized) {
    try_initialize_plane(state);
    return;
  }

  // 2. Plane already initialised —————————————————————————

  // 2a. Add slow-walk process noise to the CP covariance
  propagate_plane_covariance(state, timestamp);

  // 2b. Associate newly promoted SLAM features with the plane (Event B)
  check_plane_association_on_birth(state);

  // 2c. Point-on-plane soft constraint (also handles Event C de-association)
  if (_opt.enable_point_on_plane) {
    apply_point_on_plane_updates(state);
  }

  // 2d. v3.1 mechanism C — leftover feature plane-depth recovery
  if (_opt.enable_plane_recovery) {
    apply_plane_recovery_update(state, leftovers);
  }

  // 2e. Health monitoring
  check_health(state);
}

// ============================================================
// Height observation helpers
// ============================================================

bool UpdaterGroundPlane::interp_height(double t, double &h_out) const {
  if (_height_buf.size() < 2)
    return false;

  // Find bracketing measurements
  auto it = _height_buf.begin();
  while (std::next(it) != _height_buf.end() && std::next(it)->first <= t)
    ++it;

  auto lo = it;
  if (lo == _height_buf.begin() && lo->first > t)
    return false; // before first measurement
  if (std::next(lo) == _height_buf.end() && lo->first < t)
    return false; // after last measurement (too stale)

  if (lo->first == t) {
    h_out = lo->second;
    return true;
  }

  auto hi = std::next(lo);
  if (hi == _height_buf.end())
    return false;

  double alpha = (t - lo->first) / (hi->first - lo->first);
  h_out = lo->second + alpha * (hi->second - lo->second);
  return true;
}

void UpdaterGroundPlane::calibrate_offset_if_needed(std::shared_ptr<State> state, double t) {
  if (_offset_initialized)
    return;

  double h_raw;
  if (!interp_height(t, h_raw))
    return;

  // Align barometric height to VIO p_z at startup
  _h_takeoff_offset = h_raw - state->_imu->pos()(2);
  _offset_initialized = true;
  PRINT_INFO("[GP] takeoff offset calibrated: h_raw=%.3f p_z=%.3f offset=%.3f\n", h_raw, state->_imu->pos()(2), _h_takeoff_offset);
}

void UpdaterGroundPlane::do_height_update(std::shared_ptr<State> state, double t) {
  double h_raw;
  if (!interp_height(t, h_raw))
    return;

  double h_rel = h_raw - _h_takeoff_offset;

  // Low-altitude gate: flight-controller baro is unreliable near the ground
  if (h_rel < _opt.min_height_for_update) {
    PRINT_DEBUG("[GP] skipping h_rel update (h=%.2f < min=%.2f)\n", h_rel, _opt.min_height_for_update);
    return;
  }

  double h_pred = state->_imu->pos()(2);
  double innov = h_rel - h_pred;

  // Jump detection
  if (_has_last_innov && std::abs(innov - _last_innov) > _opt.max_height_jump) {
    _last_innov = innov;
    _has_last_innov = true;
    PRINT_DEBUG("[GP] height jump detected (%.3f), skipping update\n", innov - _last_innov);
    return;
  }
  _last_innov = innov;
  _has_last_innov = true;

  // Segmented sigma: loose below transition altitude (baro less reliable), tight above
  double sigma = (h_rel < _opt.height_sigma_transition) ? _opt.sigma_height_low : _opt.sigma_height;

  // Jacobian: only p_z  (position Vec has 3 DOF, z is index 2)
  std::vector<std::shared_ptr<Type>> Hx_order;
  Hx_order.push_back(state->_imu->p());

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 3);
  H(0, 2) = 1.0;

  Eigen::VectorXd res(1);
  res(0) = innov;

  Eigen::MatrixXd R = Eigen::MatrixXd::Constant(1, 1, sigma * sigma);

  // Chi2 gate
  Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, Hx_order);
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  double chi2 = (res.transpose() * S.llt().solve(res))(0, 0);

  boost::math::chi_squared dist(1);
  double thresh = boost::math::quantile(dist, 0.95) * _opt.chi2_multiplier;
  if (chi2 > thresh) {
    PRINT_DEBUG("[GP] height chi2 fail: %.2f > %.2f\n", chi2, thresh);
    // return;
  }

  StateHelper::EKFUpdate(state, Hx_order, H, res, R);
  PRINT_DEBUG("[GP] h_rel update: p_z=%.3f h_rel=%.3f innov=%.3f chi2=%.2f sigma=%.2f\n", h_pred, h_rel, innov, chi2, sigma);
}

// ============================================================
// Plane initialisation
// ============================================================

bool UpdaterGroundPlane::should_attempt_init(std::shared_ptr<State> state) const {
  if (_plane_initialized)
    return false;
  if ((int)state->_features_SLAM.size() < _opt.init_min_slam_feats)
    return false;
  if (_last_init_attempt_time > 0 && state->_timestamp - _last_init_attempt_time < _opt.init_retry_interval)
    return false;
  return true;
}

bool UpdaterGroundPlane::ransac_fit_plane(const std::vector<Eigen::Vector3d> &pts, PlaneCandidate &out,
                                          double inlier_thresh) const {
  int N = (int)pts.size();
  if (N < 3)
    return false;

  int best_inliers = 0;
  PlaneCandidate best;

  for (int iter = 0; iter < _opt.ransac_iters; iter++) {
    // Random sample 3 distinct points
    int i0 = rand() % N;
    int i1, i2;
    do { i1 = rand() % N; } while (i1 == i0);
    do { i2 = rand() % N; } while (i2 == i0 || i2 == i1);

    Eigen::Vector3d v1 = pts[i1] - pts[i0];
    Eigen::Vector3d v2 = pts[i2] - pts[i0];
    Eigen::Vector3d normal = v1.cross(v2);
    if (normal.norm() < 1e-6)
      continue; // degenerate / collinear

    normal.normalize();
    double d = normal.dot(pts[i0]);
    Eigen::Vector3d cp = d * normal;

    std::vector<int> inliers;
    double sum_r = 0.0;
    for (int i = 0; i < N; i++) {
      double r = std::abs(normal.dot(pts[i]) - d);
      if (r < inlier_thresh) {
        inliers.push_back(i);
        sum_r += r;
      }
    }

    if ((int)inliers.size() > best_inliers) {
      best_inliers = (int)inliers.size();
      best.cp = cp;
      best.inlier_indices = inliers;
      best.avg_residual = sum_r / inliers.size();
    }
  }

  if (best_inliers < _opt.ransac_min_inliers)
    return false;
  if ((double)best_inliers / N < _opt.ransac_min_inlier_ratio)
    return false;

  out = best;
  return true;
}

void UpdaterGroundPlane::resolve_normal_direction(Eigen::Vector3d &cp, const std::vector<Eigen::Vector3d> &pts) const {
  // cp should point in the same direction as the centroid of the point cloud
  // (both origin→ground and origin→feature-centroid point toward the ground)
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto &p : pts)
    centroid += p;
  centroid /= (double)pts.size();

  if (cp.dot(centroid) < 0.0)
    cp = -cp;
}

void UpdaterGroundPlane::refine_plane_svd(const std::vector<Eigen::Vector3d> &pts, Eigen::Vector3d &cp) const {
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto &p : pts)
    centroid += p;
  centroid /= (double)pts.size();

  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  for (const auto &p : pts) {
    Eigen::Vector3d dp = p - centroid;
    C += dp * dp.transpose();
  }

  // Eigenvector of smallest eigenvalue = plane normal
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
  Eigen::Vector3d normal = es.eigenvectors().col(0);

  double d = normal.dot(centroid);
  cp = d * normal;

  resolve_normal_direction(cp, pts);
}

bool UpdaterGroundPlane::try_initialize_plane(std::shared_ptr<State> state) {
  if (!should_attempt_init(state))
    return false;

  _last_init_attempt_time = state->_timestamp;

  // Altitude-adaptive RANSAC inlier threshold: pct * altitude, clamped to [min, max]
  double altitude = std::abs(state->_imu->pos()(2));
  double h_baro;
  if (_offset_initialized && interp_height(state->_timestamp, h_baro))
    altitude = std::abs(h_baro - _h_takeoff_offset);
  double inlier_thresh = std::clamp(altitude * _opt.ransac_inlier_thresh_pct,
                                    _opt.ransac_inlier_thresh_min,
                                    _opt.ransac_inlier_thresh_max);

  // Collect SLAM feature positions in global frame (any representation)
  std::vector<Eigen::Vector3d> points;
  std::vector<std::shared_ptr<Landmark>> feats;
  for (auto &kv : state->_features_SLAM) {
    auto lm = kv.second;
    if (lm->_feat_representation == LandmarkRepresentation::Representation::UNKNOWN)
      continue;
    if (lm->id() < 0)
      continue; // not yet in the covariance
    Eigen::Vector3d p_FinG;
    if (!lm_global_pos(state, lm, p_FinG))
      continue; // anchor marginalized or missing
    points.push_back(p_FinG);
    feats.push_back(lm);
  }

  if ((int)points.size() < _opt.init_min_slam_feats) {
    PRINT_DEBUG("[GP-init] not enough feats (%zu) for plane init\n", points.size());
    return false;
  }

  // RANSAC with altitude-adaptive inlier threshold
  PlaneCandidate cand;
  if (!ransac_fit_plane(points, cand, inlier_thresh)) {
    PRINT_DEBUG("[GP-init] RANSAC failed (feats=%zu, thresh=%.3fm)\n", points.size(), inlier_thresh);
    return false;
  }

  // SVD refinement on inliers
  std::vector<Eigen::Vector3d> inlier_pts;
  inlier_pts.reserve(cand.inlier_indices.size());
  for (int i : cand.inlier_indices)
    inlier_pts.push_back(points[i]);

  Eigen::Vector3d cp_refined = cand.cp;
  refine_plane_svd(inlier_pts, cp_refined);

  // Sanity check: distance should be positive and reasonable
  double dist = cp_refined.norm();
  if (dist < 0.01 || dist > 50.0) {
    PRINT_DEBUG("[GP-init] unreasonable CP distance %.3f m, aborting\n", dist);
    return false;
  }

  // Normal/gravity consistency check: in OpenVINS global frame gravity is [0,0,+g],
  // so the CP direction n (from origin toward the ground) must satisfy n·[0,0,1] < threshold.
  // A value near -1 means correctly pointing anti-gravity (downward); near +1 means flipped.
  static const Eigen::Vector3d gravity_hat(0.0, 0.0, 1.0);
  double cos_ng = cp_refined.normalized().dot(gravity_hat);
  if (cos_ng > _opt.max_normal_gravity_cos) {
    PRINT_DEBUG("[GP-init] plane normal gravity check failed: n·g=%.2f > %.2f, aborting\n",
                cos_ng, _opt.max_normal_gravity_cos);
    return false;
  }

  // Create PlaneCP and add it to the state
  _plane = std::make_shared<PlaneCP>();
  Eigen::VectorXd v = cp_refined;
  _plane->set_value(v);
  _plane->set_fej(v);

  // Initial covariance — isotropic, derived from fit residual (scales with altitude)
  double sigma = std::max(inlier_thresh, cand.avg_residual);
  Eigen::MatrixXd P_init = Eigen::MatrixXd::Identity(3, 3) * sigma * sigma;

  // initialize_invertible with H_L = I, H_R = 0 (no cross-corr with existing state), res = 0
  std::vector<std::shared_ptr<Type>> H_order; // empty → independent of existing state
  Eigen::MatrixXd H_R = Eigen::MatrixXd::Zero(3, 0);
  Eigen::MatrixXd H_L = Eigen::MatrixXd::Identity(3, 3);
  Eigen::VectorXd res = Eigen::VectorXd::Zero(3);
  StateHelper::initialize_invertible(state, _plane, H_order, H_R, H_L, P_init, res);

  // Event A: mark RANSAC inliers as planar (initial plane-feature reserve)
  _plane_serial++;
  for (int i : cand.inlier_indices) {
    feats[i]->is_planar = true;
    feats[i]->plane_id = _plane_serial;
    feats[i]->planar_chi2_fail_count = 0;
  }

  _plane_initialized = true;
  _last_propagate_time = state->_timestamp;
  _consecutive_low_assoc = 0;
  _consecutive_chi2_fail = 0;
  _last_update_had_chi2_fail = false;

  PRINT_INFO("[GP-init] plane initialized: cp=[%.3f %.3f %.3f] |cp|=%.3f"
             " inliers=%zu/%zu sigma=%.3f thresh=%.3f alt=%.1f planar_feats=%zu\n",
             cp_refined(0), cp_refined(1), cp_refined(2), dist,
             cand.inlier_indices.size(), points.size(), sigma, inlier_thresh, altitude,
             cand.inlier_indices.size());
  return true;
}

// ============================================================
// Feature-plane association (Event B)
// ============================================================

void UpdaterGroundPlane::check_plane_association_on_birth(std::shared_ptr<State> state) {
  if (!_plane_initialized)
    return;

  Eigen::Vector3d cp = _plane->cp();
  double cp_norm = cp.norm();
  if (cp_norm < 1e-6)
    return;
  Eigen::Vector3d n = cp / cp_norm;

  int n_new = 0;
  for (auto &kv : state->_features_SLAM) {
    auto lm = kv.second;

    // Skip already-associated features (identity inheritance)
    if (lm->is_planar)
      continue;
    if (lm->_feat_representation == LandmarkRepresentation::Representation::UNKNOWN)
      continue;
    // Skip features not yet in the covariance
    if (lm->id() < 0)
      continue;

    Eigen::Vector3d p_f;
    if (!lm_global_pos(state, lm, p_f))
      continue; // anchor not available

    // Distance to the current best-estimate plane
    double dist = std::abs(n.dot(p_f) - cp_norm);
    if (dist < _opt.assoc_thresh_in) {
      lm->is_planar = true;
      lm->plane_id = _plane_serial;
      lm->planar_chi2_fail_count = 0;
      n_new++;
    }
  }

  if (n_new > 0) {
    PRINT_DEBUG("[GP] new planar feats on birth: +%d\n", n_new);
  }
}

// ============================================================
// Point-on-plane soft constraint
// ============================================================

void UpdaterGroundPlane::update_individual_chi2_counters(const std::vector<std::shared_ptr<Landmark>> &feats,
                                                         const Eigen::VectorXd &res, const Eigen::MatrixXd &S) {
  int N = (int)feats.size();
  for (int i = 0; i < N; i++) {
    double S_ii = S(i, i);
    if (S_ii < 1e-12)
      continue;
    double chi2_i = res(i) * res(i) / S_ii;

    if (chi2_i > _opt.individual_chi2_thresh) {
      feats[i]->planar_chi2_fail_count++;
      if (feats[i]->planar_chi2_fail_count > _opt.max_individual_chi2_fail) {
        // Event C: de-associate persistent outlier
        feats[i]->is_planar = false;
        feats[i]->plane_id = -1;
        feats[i]->planar_chi2_fail_count = 0;
        PRINT_DEBUG("[GP] feat %zu de-associated (persistent chi2 fail)\n", feats[i]->_featid);
      }
    } else {
      feats[i]->planar_chi2_fail_count = 0; // reset on success
    }
  }
}

void UpdaterGroundPlane::apply_point_on_plane_updates(std::shared_ptr<State> state) {
  if (!_plane_initialized)
    return;

  // Collect planar SLAM features (any representation) that are in the covariance
  std::vector<std::shared_ptr<Landmark>> planar_feats;
  for (auto &kv : state->_features_SLAM) {
    auto lm = kv.second;
    if (!lm->is_planar || lm->plane_id != _plane_serial)
      continue;
    if (lm->id() < 0)
      continue;
    if (lm->_feat_representation == LandmarkRepresentation::Representation::UNKNOWN)
      continue;
    if (LandmarkRepresentation::is_relative_representation(lm->_feat_representation)) {
      if (lm->_anchor_cam_id < 0)
        continue;
      if (state->_clones_IMU.find(lm->_anchor_clone_timestamp) == state->_clones_IMU.end())
        continue; // anchor clone marginalized
      if (state->_calib_IMUtoCAM.find(lm->_anchor_cam_id) == state->_calib_IMUtoCAM.end())
        continue;
    }
    planar_feats.push_back(lm);
  }

  int N = (int)planar_feats.size();

  // Health: track consecutive low-association frames
  if (N < _opt.min_planar_feats) {
    _consecutive_low_assoc++;
  } else {
    _consecutive_low_assoc = 0;
  }

  if (N == 0)
    return;

  Eigen::Vector3d cp = _plane->cp();
  double cp_norm = cp.norm();
  if (cp_norm < 1e-6)
    return;
  Eigen::Vector3d n = cp / cp_norm;

  // Per-feature: global position and representation Jacobians
  // H_f: ∂p_FinG/∂x_lm  (3 × lm_dim)
  // H_x, x_order: ∂p_FinG/∂x_other (anchor clone, calibration, etc.)
  std::vector<Eigen::Vector3d> p_FinG_vec(N);
  std::vector<Eigen::MatrixXd> H_f_vec(N);
  std::vector<std::vector<Eigen::MatrixXd>> H_x_vec(N);
  std::vector<std::vector<std::shared_ptr<Type>>> x_order_vec(N);

  for (int i = 0; i < N; i++) {
    auto lm = planar_feats[i];
    lm_global_pos(state, lm, p_FinG_vec[i]);

    UpdaterHelper::UpdaterHelperFeature uf;
    uf.feat_representation = lm->_feat_representation;
    uf.anchor_cam_id = lm->_anchor_cam_id;
    uf.anchor_clone_timestamp = lm->_anchor_clone_timestamp;
    uf.p_FinG = p_FinG_vec[i];
    uf.p_FinG_fej = p_FinG_vec[i];
    if (LandmarkRepresentation::is_relative_representation(lm->_feat_representation)) {
      uf.p_FinA = lm->get_xyz(false); // position in anchor camera frame
      uf.p_FinA_fej = lm->get_xyz(false);
    }
    UpdaterHelper::get_feature_jacobian_representation(state, uf, H_f_vec[i], H_x_vec[i], x_order_vec[i]);
  }

  // Build unified column ordering for H: features → their anchors/calibrations → plane
  // std::map uses pointer ordering, which is consistent across lookups.
  std::map<std::shared_ptr<Type>, int> col_map;
  std::vector<std::shared_ptr<Type>> Hx_order;
  int col_dim = 0;

  auto ensure_col = [&](const std::shared_ptr<Type> &var) -> int {
    auto it = col_map.find(var);
    if (it != col_map.end())
      return it->second;
    int c = col_dim;
    col_map.emplace(var, c);
    Hx_order.push_back(var);
    col_dim += var->size();
    return c;
  };

  for (int i = 0; i < N; i++) {
    ensure_col(planar_feats[i]);
    for (auto &xv : x_order_vec[i])
      ensure_col(xv);
  }
  ensure_col(_plane);

  // Build stacked H and residual
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, col_dim);
  Eigen::VectorXd res = Eigen::VectorXd::Zero(N);

  for (int i = 0; i < N; i++) {
    auto lm = planar_feats[i];
    const Eigen::Vector3d &p_f = p_FinG_vec[i];
    double q = n.dot(p_f);
    res(i) = -(q - cp_norm); // residual = 0 - h

    // Landmark state columns: ∂h/∂x_lm = n^T * H_f_i  (1 × lm_dim)
    int c_lm = col_map.at(lm);
    H.block(i, c_lm, 1, lm->size()).noalias() = n.transpose() * H_f_vec[i];

    // Anchor/calibration columns: ∂h/∂x_other = n^T * H_x_j
    for (int j = 0; j < (int)x_order_vec[i].size(); j++) {
      auto xv = x_order_vec[i][j];
      int c_xv = col_map.at(xv);
      H.block(i, c_xv, 1, xv->size()).noalias() += n.transpose() * H_x_vec[i][j];
    }

    // Plane CP columns: ∂h/∂cp = p_f_perp^T / cp_norm - n^T
    Eigen::Vector3d p_f_perp = p_f - q * n;
    int c_pl = col_map.at(_plane);
    H.block<1, 3>(i, c_pl) = p_f_perp.transpose() / cp_norm - n.transpose();
  }

  double sigma = _opt.sigma_point_on_plane;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(N, N) * sigma * sigma;

  // Joint chi2 gate
  Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, Hx_order);
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  double chi2 = (res.transpose() * S.llt().solve(res))(0, 0);

  boost::math::chi_squared dist_chi(N);
  double thresh = boost::math::quantile(dist_chi, 0.95) * _opt.chi2_multiplier;

  if (chi2 > thresh) {
    PRINT_DEBUG("[GP] point-on-plane chi2 fail: %.2f > %.2f (N=%d)\n", chi2, thresh, N);
    _last_update_had_chi2_fail = true;
    _consecutive_chi2_fail++;
    update_individual_chi2_counters(planar_feats, res, S);
    return;
  }

  _last_update_had_chi2_fail = false;
  _consecutive_chi2_fail = 0;

  StateHelper::EKFUpdate(state, Hx_order, H, res, R);
  update_individual_chi2_counters(planar_feats, res, S);

  Eigen::Vector3d cp_new = _plane->cp();
  PRINT_DEBUG("[GP] point-on-plane UPDATE: %d feats chi2=%.2f cp=[%.3f %.3f %.3f] |cp|=%.3f\n",
              N, chi2, cp_new(0), cp_new(1), cp_new(2), cp_new.norm());
}

// ============================================================
// Covariance propagation and health monitoring
// ============================================================

void UpdaterGroundPlane::propagate_plane_covariance(std::shared_ptr<State> state, double timestamp) {
  if (_last_propagate_time < 0) {
    _last_propagate_time = timestamp;
    return;
  }
  double dt = timestamp - _last_propagate_time;
  if (dt <= 0.0)
    return;
  _last_propagate_time = timestamp;

  // Phi = I_3 (plane doesn't move predictably), Q = sigma^2 * dt * I_3
  Eigen::MatrixXd Phi = Eigen::Matrix3d::Identity();
  Eigen::MatrixXd Q = Eigen::Matrix3d::Identity() * _opt.sigma_cp_walk * _opt.sigma_cp_walk * dt;
  StateHelper::EKFPropagation(state, {_plane}, {_plane}, Phi, Q);
}

void UpdaterGroundPlane::check_health(std::shared_ptr<State> state) {
  if (!_plane_initialized)
    return;

  // Normal/gravity consistency: reject a plane whose normal has drifted to the wrong hemisphere
  static const Eigen::Vector3d gravity_hat(0.0, 0.0, 1.0);
  double cos_ng = _plane->cp().normalized().dot(gravity_hat);
  if (cos_ng > _opt.max_normal_gravity_cos) {
    PRINT_WARNING(YELLOW "[GP] plane normal gravity check failed (n·g=%.2f > %.2f), resetting plane\n" RESET,
                  cos_ng, _opt.max_normal_gravity_cos);
    deinitialize_plane(state);
    return;
  }

  if (_consecutive_low_assoc > _opt.max_low_assoc_frames) {
    PRINT_WARNING(YELLOW "[GP] persistent low association (%d frames), resetting plane\n" RESET, _consecutive_low_assoc);
    // deinitialize_plane(state);
    return;
  }

  if (_consecutive_chi2_fail > _opt.max_chi2_fail_frames) {
    PRINT_WARNING(YELLOW "[GP] persistent chi2 fail (%d frames), resetting plane\n" RESET, _consecutive_chi2_fail);
    // deinitialize_plane(state);
    return;
  }
}

void UpdaterGroundPlane::deinitialize_plane(std::shared_ptr<State> state) {
  // Clear all feature plane identities
  for (auto &kv : state->_features_SLAM) {
    auto lm = kv.second;
    lm->is_planar = false;
    lm->plane_id = -1;
    lm->planar_chi2_fail_count = 0;
  }

  // Remove CP from the state covariance
  if (_plane && _plane->id() >= 0) {
    StateHelper::marginalize(state, _plane);
  }
  _plane.reset();
  _plane_initialized = false;
  _last_propagate_time = -1.0;
  _consecutive_low_assoc = 0;
  _consecutive_chi2_fail = 0;
  _last_update_had_chi2_fail = false;

  // The system returns to the waiting period and will re-attempt initialisation automatically
  PRINT_INFO("[GP] plane de-initialised, waiting for re-initialisation\n");
}

// ============================================================
// v3.1 Mechanism C: leftover feature plane-depth recovery
// ============================================================

bool UpdaterGroundPlane::can_use_plane_recovery(std::shared_ptr<State> state) const {
  if (!_plane_initialized)
    return false;
  if (_plane->id() < 0)
    return false;

  // Plane covariance must be sufficiently tight
  Eigen::MatrixXd P_plane = StateHelper::get_marginal_covariance(state, {_plane});
  double cov_trace = P_plane.trace();
  if (cov_trace > _opt.recovery_max_plane_cov)
    return false;

  // Need enough healthy planar SLAM features
  int n_planar = 0;
  for (auto &kv : state->_features_SLAM) {
    if (kv.second->is_planar && kv.second->plane_id == _plane_serial)
      n_planar++;
  }
  if (n_planar < _opt.min_slam_planar_for_recovery)
    return false;

  return true;
}

std::vector<std::shared_ptr<ov_core::Feature>>
UpdaterGroundPlane::filter_leftover_candidates(std::shared_ptr<State> state,
                                               const std::vector<std::shared_ptr<ov_core::Feature>> &leftovers) const {
  // Build SLAM feature ID set for exclusion
  std::unordered_set<size_t> slam_ids;
  for (auto &kv : state->_features_SLAM)
    slam_ids.insert(kv.second->_featid);

  std::vector<std::shared_ptr<ov_core::Feature>> out;
  out.reserve(leftovers.size());

  for (auto &feat : leftovers) {
    if (slam_ids.count(feat->featid))
      continue; // already a SLAM feature — would double-count

    if (feat->timestamps.empty())
      continue;

    // Use the first camera's timeline (monocular: cam 0)
    auto cam_it = feat->timestamps.begin();
    auto &times = cam_it->second;
    if ((int)times.size() < 2)
      continue;

    double t_anchor = times.front();
    double t_current = times.back();

    if (t_current - t_anchor < _opt.min_time_baseline)
      continue;

    // Anchor frame must still be in the state window
    if (state->_clones_IMU.find(t_anchor) == state->_clones_IMU.end())
      continue;
    if (state->_clones_IMU.find(t_current) == state->_clones_IMU.end())
      continue;

    out.push_back(feat);
    if ((int)out.size() >= _opt.max_recovery_features)
      break;
  }
  return out;
}

bool UpdaterGroundPlane::try_recover_3d_from_plane(std::shared_ptr<State> state,
                                                    const std::shared_ptr<ov_core::Feature> &feat,
                                                    RecoveredFeature &rf) const {
  auto cam_it = feat->timestamps.begin();
  size_t cam_id = cam_it->first;
  auto &times = cam_it->second;
  auto &uvs = feat->uvs_norm.at(cam_id);

  rf.feat = feat;
  rf.cam_id = cam_id;
  rf.t_anchor = times.front();
  rf.t_current = times.back();
  rf.uv_anchor_norm = uvs.front().head<2>().cast<double>();
  rf.uv_current_norm = uvs.back().head<2>().cast<double>();

  if (state->_calib_IMUtoCAM.find(cam_id) == state->_calib_IMUtoCAM.end())
    return false;

  Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(cam_id)->Rot();
  Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(cam_id)->pos();

  Eigen::Matrix3d R_GtoI_a = state->_clones_IMU.at(rf.t_anchor)->Rot();
  Eigen::Vector3d p_IainG  = state->_clones_IMU.at(rf.t_anchor)->pos();

  Eigen::Matrix3d R_GtoCa = R_ItoC * R_GtoI_a;
  Eigen::Vector3d p_CainG = p_IainG - R_GtoI_a.transpose() * R_ItoC.transpose() * p_IinC;

  // Plane params
  Eigen::Vector3d cp = _plane->cp();
  double cp_norm = cp.norm();
  if (cp_norm < 1e-6)
    return false;
  Eigen::Vector3d n = cp / cp_norm;

  // Plane in anchor camera frame
  Eigen::Vector3d n_Ca = R_GtoCa * n;
  double d_Ca = cp_norm - n.dot(p_CainG);

  // Ray from anchor camera
  Eigen::Vector3d x_a_bar(rf.uv_anchor_norm(0), rf.uv_anchor_norm(1), 1.0);
  double denom = n_Ca.dot(x_a_bar);
  if (std::abs(denom) < 1e-3)
    return false; // ray nearly parallel to plane

  rf.rho_at_anchor = d_Ca / denom;
  if (rf.rho_at_anchor < _opt.min_recovery_depth || rf.rho_at_anchor > _opt.max_recovery_depth)
    return false;

  // Recover 3D world position
  Eigen::Vector3d X_Ca = rf.rho_at_anchor * x_a_bar;
  rf.X_G = R_GtoCa.transpose() * X_Ca + p_CainG;

  return true;
}

bool UpdaterGroundPlane::compute_recovery_jacobian(std::shared_ptr<State> state,
                                                    const RecoveredFeature &rf,
                                                    RecoveryJacobian &out) const {
  if (state->_calib_IMUtoCAM.find(rf.cam_id) == state->_calib_IMUtoCAM.end())
    return false;

  Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(rf.cam_id)->Rot();
  Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(rf.cam_id)->pos();

  Eigen::Matrix3d R_GtoI_a = state->_clones_IMU.at(rf.t_anchor)->Rot();
  Eigen::Vector3d p_IainG  = state->_clones_IMU.at(rf.t_anchor)->pos();

  Eigen::Matrix3d R_GtoI_c = state->_clones_IMU.at(rf.t_current)->Rot();
  Eigen::Vector3d p_IcinG  = state->_clones_IMU.at(rf.t_current)->pos();

  Eigen::Matrix3d R_GtoCa = R_ItoC * R_GtoI_a;
  Eigen::Vector3d p_CainG = p_IainG - R_GtoI_a.transpose() * R_ItoC.transpose() * p_IinC;

  Eigen::Matrix3d R_GtoCc = R_ItoC * R_GtoI_c;
  Eigen::Vector3d p_CcinG = p_IcinG - R_GtoI_c.transpose() * R_ItoC.transpose() * p_IinC;

  // Plane
  Eigen::Vector3d cp = _plane->cp();
  double cp_norm = cp.norm();
  if (cp_norm < 1e-6)
    return false;
  Eigen::Vector3d n = cp / cp_norm;

  Eigen::Vector3d n_Ca = R_GtoCa * n;
  double d_Ca = cp_norm - n.dot(p_CainG);
  Eigen::Vector3d x_a_bar(rf.uv_anchor_norm(0), rf.uv_anchor_norm(1), 1.0);
  double denom = n_Ca.dot(x_a_bar);
  if (std::abs(denom) < 1e-6)
    return false;

  double rho = d_Ca / denom;
  Eigen::Vector3d X_Ca = rho * x_a_bar;

  // Global and current-camera 3D positions
  Eigen::Vector3d X_G = R_GtoCa.transpose() * X_Ca + p_CainG;
  Eigen::Vector3d X_Cc = R_GtoCc * (X_G - p_CcinG);
  if (X_Cc(2) < 1e-4)
    return false;

  double Z = X_Cc(2);
  // Projection Jacobian: ∂[u,v]/∂X_Cc
  Eigen::Matrix<double, 2, 3> J_proj;
  J_proj << 1.0 / Z, 0, -X_Cc(0) / (Z * Z),
            0, 1.0 / Z, -X_Cc(1) / (Z * Z);

  // Residual: observed − predicted
  out.residual = rf.uv_current_norm - Eigen::Vector2d(X_Cc(0) / Z, X_Cc(1) / Z);

  // ∂r/∂X_Cc = −J_proj  (2×3)
  Eigen::Matrix<double, 2, 3> drdXcc = -J_proj;

  // --- Current clone Jacobians ---
  // ∂X_Cc/∂δθ_c = R_ItoC * [R_GtoI_c*(X_G - p_IcinG)]×  (JPL left perturbation)
  Eigen::Vector3d v_c = R_GtoI_c * (X_G - p_IcinG);
  out.J_theta_c = drdXcc * (R_ItoC * skew_x(v_c));
  // ∂X_Cc/∂δp_c = −R_GtoCc
  out.J_p_c = drdXcc * (-R_GtoCc);

  // --- Shared: ∂r/∂X_G = drdXcc * R_GtoCc  (2×3)
  Eigen::Matrix<double, 2, 3> dr_dXG = drdXcc * R_GtoCc;

  // --- Anchor clone Jacobians ---
  // A = R_ItoC^T * X_Ca (X_Ca in IMU frame), B = R_ItoC^T * p_IinC
  Eigen::Vector3d A = R_ItoC.transpose() * X_Ca;
  Eigen::Vector3d B = R_ItoC.transpose() * p_IinC;
  // m_a = R_GtoCa^T * x_a_bar / denom  (direction vector)
  Eigen::Vector3d m_a = R_GtoCa.transpose() * x_a_bar / denom;

  // ∂X_G/∂δθ_a (JPL):
  //   = R_GtoI_a^T([B]× − [A]×) − m_a * (R_GtoI_a*n)^T * [B]×
  Eigen::Matrix3d dXG_dtha = R_GtoI_a.transpose() * (skew_x(B) - skew_x(A)) -
                              m_a * (R_GtoI_a * n).transpose() * skew_x(B);
  out.J_theta_a = dr_dXG * dXG_dtha;

  // ∂X_G/∂δp_a = I − m_a * n^T
  Eigen::Matrix3d dXG_dpa = Eigen::Matrix3d::Identity() - m_a * n.transpose();
  out.J_p_a = dr_dXG * dXG_dpa;

  // --- Plane CP Jacobian ---
  // ∂d_Ca/∂cp = n^T − p_CainG^T*(I − n*n^T)/cp_norm
  Eigen::RowVector3d e_d = n.transpose() -
                           p_CainG.transpose() * (Eigen::Matrix3d::Identity() - n * n.transpose()) / cp_norm;
  // ∂denom/∂cp = x_a_bar^T * R_GtoCa * (I − n*n^T) / cp_norm
  Eigen::RowVector3d e_denom = x_a_bar.transpose() * R_GtoCa *
                               (Eigen::Matrix3d::Identity() - n * n.transpose()) / cp_norm;
  // ∂rho/∂cp = (e_d − rho*e_denom) / denom
  Eigen::RowVector3d drho_dcp = (e_d - rho * e_denom) / denom;
  // ∂X_G/∂cp = R_GtoCa^T*x_a_bar * drho_dcp  = (denom*m_a) * drho_dcp → m_a * (e_d − rho*e_denom)
  Eigen::Matrix3d dXG_dcp = m_a * (e_d - rho * e_denom); // outer product (3×1)(1×3)
  out.J_cp = dr_dXG * dXG_dcp;

  return true;
}

void UpdaterGroundPlane::build_and_apply_recovery_update(std::shared_ptr<State> state,
                                                          const std::vector<RecoveredFeature> &recovered) {
  int N = (int)recovered.size();
  if (N < _opt.min_recovery_features)
    return;

  // Collect involved clone timestamps (ordered)
  std::map<double, std::shared_ptr<ov_type::PoseJPL>> involved_clones;
  for (auto &rf : recovered) {
    involved_clones[rf.t_anchor]  = state->_clones_IMU.at(rf.t_anchor);
    involved_clones[rf.t_current] = state->_clones_IMU.at(rf.t_current);
  }

  // Build column mapping: clone → column offset (6 DOF each), plane → last 3
  std::map<std::shared_ptr<ov_type::Type>, int> col_map;
  std::vector<std::shared_ptr<ov_type::Type>> Hx_order;
  int col_dim = 0;

  auto ensure_col = [&](std::shared_ptr<ov_type::Type> var) {
    if (col_map.find(var) == col_map.end()) {
      col_map[var] = col_dim;
      Hx_order.push_back(var);
      col_dim += var->size();
    }
  };
  for (auto &kv : involved_clones)
    ensure_col(kv.second);
  ensure_col(_plane);

  // Build H (2N × col_dim) and res (2N)
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2 * N, col_dim);
  Eigen::VectorXd res = Eigen::VectorXd::Zero(2 * N);
  int good_rows = 0;

  for (int i = 0; i < N; i++) {
    const auto &rf = recovered[i];

    RecoveryJacobian jac;
    if (!compute_recovery_jacobian(state, rf, jac))
      continue;

    auto clone_a = state->_clones_IMU.at(rf.t_anchor);
    auto clone_c = state->_clones_IMU.at(rf.t_current);
    int ca = col_map.at(clone_a);
    int cc = col_map.at(clone_c);
    int cp_col = col_map.at(_plane);

    int row = good_rows * 2;
    // Rotation (first 3 cols of PoseJPL), position (next 3 cols)
    H.block<2, 3>(row, ca + 0) += jac.J_theta_a;
    H.block<2, 3>(row, ca + 3) += jac.J_p_a;
    H.block<2, 3>(row, cc + 0) += jac.J_theta_c;
    H.block<2, 3>(row, cc + 3) += jac.J_p_c;
    H.block<2, 3>(row, cp_col) += jac.J_cp;
    res.segment<2>(row) = jac.residual;
    good_rows++;
  }

  if (good_rows < _opt.min_recovery_features)
    return;

  int M = good_rows * 2;
  H.conservativeResize(M, col_dim);
  res.conservativeResize(M);

  double sigma = _opt.sigma_pixel_recovery;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(M, M) * sigma * sigma;

  // Chi2 gate
  Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, Hx_order);
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  double chi2 = (res.transpose() * S.llt().solve(res))(0, 0);

  boost::math::chi_squared dist(M);
  double thresh = boost::math::quantile(dist, 0.95) * _opt.recovery_chi2_multiplier;
  if (chi2 > thresh) {
    PRINT_DEBUG("[GP-recovery] chi2 fail: %.2f > %.2f (N=%d)\n", chi2, thresh, good_rows);
    return;
  }

  StateHelper::EKFUpdate(state, Hx_order, H, res, R);
  PRINT_DEBUG("[GP-recovery] UPDATE: %d feats chi2=%.2f (thresh=%.2f)\n", good_rows, chi2, thresh);
}

void UpdaterGroundPlane::apply_plane_recovery_update(std::shared_ptr<State> state,
                                                      const std::vector<std::shared_ptr<ov_core::Feature>> &leftovers) {
  if (!_plane_initialized || leftovers.empty())
    return;
  if (!can_use_plane_recovery(state))
    return;

  // 1. Filter candidates
  auto candidates = filter_leftover_candidates(state, leftovers);
  PRINT_DEBUG("[GP-recovery] candidates: leftover=%zu → after_filter=%zu\n", leftovers.size(), candidates.size());
  if ((int)candidates.size() < _opt.min_recovery_features)
    return;

  // 2. Recover 3D from plane for each candidate
  std::vector<RecoveredFeature> recovered;
  recovered.reserve(candidates.size());
  for (auto &feat : candidates) {
    RecoveredFeature rf;
    if (try_recover_3d_from_plane(state, feat, rf))
      recovered.push_back(rf);
  }
  PRINT_DEBUG("[GP-recovery] recovered: %zu features\n", recovered.size());
  if ((int)recovered.size() < _opt.min_recovery_features)
    return;

  // 3. Optional pre-filter: reject obvious reprojection outliers
  if (_opt.enable_recovery_prefilter) {
    std::vector<RecoveredFeature> filtered;
    filtered.reserve(recovered.size());
    for (auto &rf : recovered) {
      if (state->_clones_IMU.find(rf.t_current) == state->_clones_IMU.end())
        continue;
      if (state->_calib_IMUtoCAM.find(rf.cam_id) == state->_calib_IMUtoCAM.end())
        continue;
      Eigen::Matrix3d R_ItoC  = state->_calib_IMUtoCAM.at(rf.cam_id)->Rot();
      Eigen::Vector3d p_IinC  = state->_calib_IMUtoCAM.at(rf.cam_id)->pos();
      Eigen::Matrix3d R_GtoI_c = state->_clones_IMU.at(rf.t_current)->Rot();
      Eigen::Vector3d p_IcinG  = state->_clones_IMU.at(rf.t_current)->pos();
      Eigen::Matrix3d R_GtoCc = R_ItoC * R_GtoI_c;
      Eigen::Vector3d p_CcinG = p_IcinG - R_GtoI_c.transpose() * R_ItoC.transpose() * p_IinC;
      Eigen::Vector3d X_Cc = R_GtoCc * (rf.X_G - p_CcinG);
      if (X_Cc(2) < 0.1)
        continue;
      Eigen::Vector2d uv_pred(X_Cc(0) / X_Cc(2), X_Cc(1) / X_Cc(2));
      if ((rf.uv_current_norm - uv_pred).norm() < _opt.recovery_prefilter_thresh)
        filtered.push_back(rf);
    }
    recovered = std::move(filtered);
    PRINT_DEBUG("[GP-recovery] after prefilter: %zu features\n", recovered.size());
    if ((int)recovered.size() < _opt.min_recovery_features)
      return;
  }

  // 4. EKF update
  build_and_apply_recovery_update(state, recovered);
}

// ============================================================
// Homography plane update
// ============================================================

void UpdaterGroundPlane::promote_homography_points_to_slam(std::shared_ptr<State> state, const std::vector<size_t> &feat_ids,
                                                            const std::vector<Eigen::Vector3d> &pts3d, double h_rel) {
  if (!_opt.enable_homography_persistent_landmarks)
    return;
  if (feat_ids.empty() || pts3d.empty() || feat_ids.size() != pts3d.size())
    return;

  const int aruco_reserved_max = 4 * state->_options.max_aruco_features;
  int slam_non_aruco_count = 0;
  for (const auto &kv : state->_features_SLAM) {
    if ((int)kv.first > aruco_reserved_max)
      slam_non_aruco_count++;
  }

  int promoted = 0;
  int promoted_limit = std::max(0, _opt.homography_persistent_max_add_per_update);
  double sigma_init = std::max(_opt.homography_persistent_sigma_min,
                               _opt.homography_persistent_sigma_h_rel_scale * std::abs(h_rel));

  for (size_t i = 0; i < feat_ids.size(); i++) {
    if (promoted >= promoted_limit)
      break;
    if ((int)feat_ids[i] <= aruco_reserved_max)
      continue;
    if (state->_features_SLAM.find(feat_ids[i]) != state->_features_SLAM.end())
      continue;
    if (slam_non_aruco_count >= state->_options.max_slam_features)
      break;

    auto landmark = std::make_shared<Landmark>(3);
    landmark->_featid = feat_ids[i];
    landmark->_feat_representation = LandmarkRepresentation::Representation::GLOBAL_3D;
    landmark->_unique_camera_id = 0;
    landmark->set_from_xyz(pts3d[i], false);
    landmark->set_from_xyz(pts3d[i], true);

    std::vector<std::shared_ptr<Type>> H_order;
    Eigen::MatrixXd H_R = Eigen::MatrixXd::Zero(3, 0);
    Eigen::MatrixXd H_L = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd res = Eigen::VectorXd::Zero(3);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3) * sigma_init * sigma_init;
    StateHelper::initialize_invertible(state, landmark, H_order, H_R, H_L, R, res);

    state->_features_SLAM.insert({feat_ids[i], landmark});
    slam_non_aruco_count++;
    promoted++;
  }

  if (promoted > 0) {
    PRINT_DEBUG("[GP-homo] promoted %d persistent SLAM landmarks (sigma_init=%.2f h_rel=%.1f)\n",
                promoted, sigma_init, h_rel);
  }
}

void UpdaterGroundPlane::apply_homography_plane_update(
    std::shared_ptr<State> state, double timestamp,
    const std::vector<std::shared_ptr<ov_core::Feature>> &leftovers) {

  _homography_plane_points.clear();

  if (leftovers.empty())
    return;

  // Need baro height with offset calibrated
  if (!_offset_initialized)
    return;
  double h_raw;
  if (!interp_height(timestamp, h_raw))
    return;
  double h_rel = h_raw - _h_takeoff_offset;
  if (h_rel < _opt.min_height_for_update)
    return;

  // Need current and previous clone
  if ((int)state->_clones_IMU.size() < 2)
    return;
  auto it_end = state->_clones_IMU.end();
  --it_end;
  double t_cur = it_end->first;
  --it_end;
  double t_prev = it_end->first;

  // Camera extrinsics (cam 0)
  const size_t cam_id = 0;
  if (state->_calib_IMUtoCAM.find(cam_id) == state->_calib_IMUtoCAM.end())
    return;
  Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(cam_id)->Rot();
  Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(cam_id)->pos();

  // Current clone pose
  Eigen::Matrix3d R_GtoI_c = state->_clones_IMU.at(t_cur)->Rot();
  Eigen::Vector3d p_IcinG  = state->_clones_IMU.at(t_cur)->pos();
  Eigen::Matrix3d R_GtoC   = R_ItoC * R_GtoI_c;
  Eigen::Vector3d p_CcinG  = p_IcinG - R_GtoI_c.transpose() * R_ItoC.transpose() * p_IinC;

  // Collect normalised coordinate pairs from leftover features at t_prev and t_cur
  std::vector<cv::Point2f> pts_prev_norm, pts_cur_norm;
  std::vector<size_t> feat_ids_norm;

  for (const auto &feat : leftovers) {
    auto cam_it = feat->timestamps.find(cam_id);
    if (cam_it == feat->timestamps.end())
      continue;
    if (feat->uvs_norm.find(cam_id) == feat->uvs_norm.end())
      continue;
    const auto &ts   = cam_it->second;
    const auto &uvs  = feat->uvs_norm.at(cam_id);

    int idx_cur = -1, idx_prev = -1;
    for (int i = 0; i < (int)ts.size(); i++) {
      if (std::abs(ts[i] - t_cur)  < 1e-6) idx_cur  = i;
      if (std::abs(ts[i] - t_prev) < 1e-6) idx_prev = i;
    }
    if (idx_cur < 0 || idx_prev < 0)
      continue;

    pts_cur_norm.emplace_back((float)uvs[idx_cur](0),  (float)uvs[idx_cur](1));
    pts_prev_norm.emplace_back((float)uvs[idx_prev](0), (float)uvs[idx_prev](1));
    feat_ids_norm.push_back(feat->featid);
  }

  if ((int)pts_cur_norm.size() < _opt.homography_min_inliers)
    return;

  // findHomography with RANSAC on normalised coords
  std::vector<uchar> mask;
  cv::Mat H_cv = cv::findHomography(pts_prev_norm, pts_cur_norm, cv::RANSAC,
                                     _opt.homography_ransac_thresh, mask, 2000, 0.999);
  if (H_cv.empty())
    return;

  int n_inliers = 0;
  for (auto m : mask) n_inliers += (m != 0);

  double inlier_ratio = (double)n_inliers / (double)pts_cur_norm.size();
  PRINT_DEBUG("[GP-homo] feats=%zu inliers=%d ratio=%.2f h_rel=%.2f\n",
              pts_cur_norm.size(), n_inliers, inlier_ratio, h_rel);

  if (n_inliers < _opt.homography_min_inliers || inlier_ratio < _opt.homography_min_inlier_ratio)
    return;

  // Recover 3D positions for each inlier using FC/baro height
  std::vector<Eigen::Vector3d> pts3d;
  std::vector<Eigen::Vector2d> pts_cur_2d;
  std::vector<Eigen::Vector2d> pts_prev_2d;
  std::vector<size_t> pts_feat_ids;
  pts3d.reserve(n_inliers);
  pts_cur_2d.reserve(n_inliers);
  pts_prev_2d.reserve(n_inliers);
  pts_feat_ids.reserve(n_inliers);

  for (size_t i = 0; i < pts_cur_norm.size(); i++) {
    if (!mask[i])
      continue;

    Eigen::Vector3d m_cam((double)pts_cur_norm[i].x, (double)pts_cur_norm[i].y, 1.0);
    Eigen::Vector3d ray_G = R_GtoC.transpose() * m_cam;
    double beta = ray_G(2);

    // For a downward-facing camera, the ray's global z-component is negative (points toward ground)
    if (beta >= -1e-4)
      continue;

    double t_depth = -h_rel / beta; // positive: h_rel > 0, beta < 0
    if (t_depth < 0.1 || t_depth > _opt.max_recovery_depth)
      continue;

    pts3d.push_back(p_CcinG + t_depth * ray_G);
    pts_cur_2d.emplace_back((double)pts_cur_norm[i].x, (double)pts_cur_norm[i].y);
    pts_prev_2d.emplace_back((double)pts_prev_norm[i].x, (double)pts_prev_norm[i].y);
    pts_feat_ids.push_back(feat_ids_norm[i]);
  }

  // Validation: reproject the height-recovered 3D points back into the current frame.
  double reproj_err_sum = 0.0;
  double reproj_err_max = 0.0;
  int reproj_err_count = 0;
  for (int i = 0; i < (int)pts3d.size(); i++) {
    Eigen::Vector3d X_Cc = R_GtoC * (pts3d[i] - p_CcinG);
    if (X_Cc(2) < 1e-6)
      continue;

    Eigen::Vector2d uv_pred(X_Cc(0) / X_Cc(2), X_Cc(1) / X_Cc(2));
    double err = (pts_cur_2d[i] - uv_pred).norm();
    reproj_err_sum += err;
    reproj_err_max = std::max(reproj_err_max, err);
    reproj_err_count++;
  }

  // Store for visualisation regardless of EKF update
  _homography_plane_points = pts3d;
  double reproj_err_mean = reproj_err_count > 0 ? reproj_err_sum / reproj_err_count : 0.0;
  PRINT_DEBUG("[GP-homo] recovered %zu 3D points self-reproj(mean=%.6f, max=%.6f, n=%d)\n",
              pts3d.size(), reproj_err_mean, reproj_err_max, reproj_err_count);

  // One-shot pose-only EKF update: reproject height-recovered 3D points into previous frame.
  if ((int)pts3d.size() < _opt.homography_min_inliers)
    return;

  auto prev_clone = state->_clones_IMU.at(t_prev);
  auto cur_clone = state->_clones_IMU.at(t_cur);

  Eigen::Matrix3d R_GtoI_p = prev_clone->Rot();
  Eigen::Vector3d p_IpinG = prev_clone->pos();
  Eigen::Matrix3d R_GtoC_p = R_ItoC * R_GtoI_p;
  Eigen::Vector3d p_CpinG = p_IpinG - R_GtoI_p.transpose() * R_ItoC.transpose() * p_IinC;

  std::vector<std::shared_ptr<Type>> Hx_order;
  Hx_order.push_back(prev_clone->q()); // cols 0-2: previous IMU rotation
  Hx_order.push_back(prev_clone->p()); // cols 3-5: previous IMU position
  Hx_order.push_back(cur_clone->q());  // cols 6-8: current IMU rotation
  Hx_order.push_back(cur_clone->p());  // cols 9-11: current IMU position

  int N = (int)pts3d.size();
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2 * N, 12);
  Eigen::VectorXd res = Eigen::VectorXd::Zero(2 * N);
  int good_rows = 0;
  std::vector<size_t> good_feat_ids;
  std::vector<Eigen::Vector3d> good_pts3d;
  good_feat_ids.reserve(N);
  good_pts3d.reserve(N);

  for (int i = 0; i < N; i++) {
    const Eigen::Vector3d &X_G = pts3d[i];
    const Eigen::Vector2d &uv_cur = pts_cur_2d[i];
    const Eigen::Vector2d &uv_prev = pts_prev_2d[i];

    Eigen::Vector3d X_Cp = R_GtoC_p * (X_G - p_CpinG);
    if (X_Cp(2) < 1e-4)
      continue;

    double Zp = X_Cp(2);
    Eigen::Matrix<double, 2, 3> J_proj;
    J_proj << 1.0 / Zp, 0, -X_Cp(0) / (Zp * Zp),
              0, 1.0 / Zp, -X_Cp(1) / (Zp * Zp);

    Eigen::Matrix<double, 2, 3> drdXcp = -J_proj;
    Eigen::Matrix<double, 2, 3> drdXG = drdXcp * R_GtoC_p;

    // Current-view ray derivatives used by the height-based 3D reconstruction.
    Eigen::Vector3d m_cam(uv_cur(0), uv_cur(1), 1.0);
    Eigen::Vector3d u = R_ItoC.transpose() * m_cam;
    Eigen::Vector3d v = R_ItoC.transpose() * p_IinC;
    Eigen::Vector3d rG = R_GtoI_c.transpose() * u;
    double beta = rG(2);
    if (std::abs(beta) < 1e-6)
      continue;

    Eigen::Matrix3d dP = R_GtoI_c.transpose() * skew_x(v);
    Eigen::Matrix3d dR = -R_GtoI_c.transpose() * skew_x(u);
    double t_d = -h_rel / beta;
    Eigen::Matrix3d dX_dtheta_c = dP + (h_rel / (beta * beta)) * rG * (Eigen::RowVector3d(0, 0, 1) * dR) + t_d * dR;

    Eigen::Vector3d v_prev = R_GtoI_p * (X_G - p_IpinG);
    Eigen::Matrix<double, 2, 3> J_theta_p = drdXcp * (R_ItoC * skew_x(v_prev));
    Eigen::Matrix<double, 2, 3> J_p_p = drdXcp * (-R_GtoC_p);
    Eigen::Matrix<double, 2, 3> J_theta_c = drdXG * dX_dtheta_c;
    Eigen::Matrix<double, 2, 3> J_p_c = drdXG;

    int row = 2 * good_rows;
    res.segment<2>(row) = uv_prev - Eigen::Vector2d(X_Cp(0) / Zp, X_Cp(1) / Zp);
    H.block<2, 3>(row, 0) = J_theta_p;
    H.block<2, 3>(row, 3) = J_p_p;
    H.block<2, 3>(row, 6) = J_theta_c;
    H.block<2, 3>(row, 9) = J_p_c;
    good_feat_ids.push_back(pts_feat_ids[i]);
    good_pts3d.push_back(X_G);
    good_rows++;
  }

  if (good_rows < _opt.homography_min_inliers)
    return;

  int M = 2 * good_rows;
  H.conservativeResize(M, 12);
  res.conservativeResize(M);

  double sigma = _opt.sigma_pixel_recovery;
  Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);

  // Per-feature chi2 gating: rejects individual outliers (e.g. features on elevated terrain at
  // high altitude) while preserving flat-ground features. This replaces the joint gate, which
  // fails whenever even a few terrain outliers are present.
  boost::math::chi_squared dist_chi2_2(2);
  double per_feat_thresh = boost::math::quantile(dist_chi2_2, 0.95) * _opt.recovery_chi2_multiplier;

  std::vector<int> kept_rows;
  for (int i = 0; i < good_rows; i++) {
    Eigen::Matrix<double, 2, 12> H_i = H.block<2, 12>(2 * i, 0);
    Eigen::Matrix2d S_i = H_i * P_marg * H_i.transpose() + sigma * sigma * Eigen::Matrix2d::Identity();
    Eigen::Vector2d r_i = res.segment<2>(2 * i);
    double chi2_i = (r_i.transpose() * S_i.llt().solve(r_i))(0, 0);
    if (chi2_i <= per_feat_thresh)
      kept_rows.push_back(i);
  }

  int n_kept = (int)kept_rows.size();
  PRINT_DEBUG("[GP-homo] per-feat chi2 gate: kept %d/%d (h_rel=%.1fm thresh=%.2f)\n",
              n_kept, good_rows, h_rel, per_feat_thresh);
  if (n_kept < _opt.homography_min_inliers)
    return;

  // Rebuild H and res from kept features
  int M_kept = 2 * n_kept;
  Eigen::MatrixXd H_kept = Eigen::MatrixXd::Zero(M_kept, 12);
  Eigen::VectorXd res_kept = Eigen::VectorXd::Zero(M_kept);
  std::vector<size_t> kept_feat_ids;
  std::vector<Eigen::Vector3d> kept_pts3d;
  kept_feat_ids.reserve(n_kept);
  kept_pts3d.reserve(n_kept);
  for (int k = 0; k < n_kept; k++) {
    H_kept.block<2, 12>(2 * k, 0) = H.block<2, 12>(2 * kept_rows[k], 0);
    res_kept.segment<2>(2 * k)     = res.segment<2>(2 * kept_rows[k]);
    kept_feat_ids.push_back(good_feat_ids[kept_rows[k]]);
    kept_pts3d.push_back(good_pts3d[kept_rows[k]]);
  }

  Eigen::MatrixXd R_mat = Eigen::MatrixXd::Identity(M_kept, M_kept) * sigma * sigma;
  Eigen::MatrixXd S_kept = H_kept * P_marg * H_kept.transpose() + R_mat;
  double chi2_kept = (res_kept.transpose() * S_kept.llt().solve(res_kept))(0, 0);

  // StateHelper::EKFUpdate(state, Hx_order, H_kept, res_kept, R_mat);
  promote_homography_points_to_slam(state, kept_feat_ids, kept_pts3d, h_rel);
  PRINT_DEBUG("[GP-homo] UPDATE: kept=%d/%d pts chi2=%.2f h_rel=%.1fm\n",
              n_kept, good_rows, chi2_kept, h_rel);
}
