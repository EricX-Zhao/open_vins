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

#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LandmarkRepresentation.h"
#include "utils/print.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <random>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

namespace {

// Fit plane n·X + d = 0 through pts via centroid + JacobiSVD.
// n_out is forced to point upward (z > 0). Returns false if pts is empty.
bool svd_fit_plane(const std::vector<Eigen::Vector3d> &pts, Eigen::Vector3d &n_out, double &d_out) {
  if (pts.empty())
    return false;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto &p : pts)
    centroid += p;
  centroid /= (double)pts.size();

  Eigen::MatrixXd A((int)pts.size(), 3);
  for (size_t i = 0; i < pts.size(); i++)
    A.row((int)i) = (pts[i] - centroid).transpose();

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  n_out = svd.matrixV().col(2);
  if (n_out(2) < 0)
    n_out = -n_out;
  d_out = -n_out.dot(centroid);
  return true;
}

// Retrieve the camera pose in global frame from the IMU clone at t_cur.
// Returns false if the calibration or clone is unavailable.
bool get_camera_pose(const std::shared_ptr<State> &state, size_t cam_id, double t_cur,
                     Eigen::Matrix3d &R_GtoC, Eigen::Vector3d &p_CcinG) {
  if (state->_calib_IMUtoCAM.find(cam_id) == state->_calib_IMUtoCAM.end())
    return false;
  const Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(cam_id)->Rot();
  const Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(cam_id)->pos();
  const Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(t_cur)->Rot();
  const Eigen::Vector3d p_IinG = state->_clones_IMU.at(t_cur)->pos();
  R_GtoC = R_ItoC * R_GtoI;
  p_CcinG = p_IinG - R_GtoI.transpose() * R_ItoC.transpose() * p_IinC;
  return true;
}

// Build normalised-coord correspondences (prev→cur) from leftover features at timestamps t_prev / t_cur.
void collect_feature_correspondences(const std::vector<std::shared_ptr<ov_core::Feature>> &leftovers,
                                     size_t cam_id, double t_prev, double t_cur,
                                     std::vector<cv::Point2f> &pts_prev, std::vector<cv::Point2f> &pts_cur,
                                     std::vector<size_t> &ids) {
  for (const auto &feat : leftovers) {
    auto cam_it = feat->timestamps.find(cam_id);
    if (cam_it == feat->timestamps.end())
      continue;
    if (feat->uvs_norm.find(cam_id) == feat->uvs_norm.end())
      continue;
    const auto &ts = cam_it->second;
    const auto &uvs = feat->uvs_norm.at(cam_id);

    int idx_cur = -1, idx_prev = -1;
    for (int i = 0; i < (int)ts.size(); i++) {
      if (std::abs(ts[i] - t_cur) < 1e-6)
        idx_cur = i;
      if (std::abs(ts[i] - t_prev) < 1e-6)
        idx_prev = i;
    }
    if (idx_cur < 0 || idx_prev < 0)
      continue;

    pts_cur.emplace_back((float)uvs[idx_cur](0), (float)uvs[idx_cur](1));
    pts_prev.emplace_back((float)uvs[idx_prev](0), (float)uvs[idx_prev](1));
    ids.push_back(feat->featid);
  }
}

} // namespace

UpdaterGroundPlane::UpdaterGroundPlane(const Options &opt) : _opt(opt) {}

void UpdaterGroundPlane::feed_height(double timestamp, double h_rel) {
  _height_buf.push_back({timestamp, h_rel});
  while (_height_buf.size() > kMaxHeightBufSize)
    _height_buf.pop_front();
}

Eigen::Vector3d UpdaterGroundPlane::get_cp() const {
  if (!_plane_valid)
    return Eigen::Vector3d::Zero();
  return -_plane_d * _plane_normal;
}

void UpdaterGroundPlane::fit_plane_from_slam_points(std::shared_ptr<State> state, double h_rel) {
  _plane_valid = false;
  _plane_inlier_pts.clear();

  // Collect 3D positions from all non-aruco SLAM landmarks.
  // get_xyz() handles all representations (GLOBAL_3D, ANCHORED_MSCKF_INVERSE_DEPTH, etc.)
  // and returns global-frame coordinates in every case.
  const int aruco_max = 4 * state->_options.max_aruco_features;
  std::vector<Eigen::Vector3d> pts;
  pts.reserve(state->_features_SLAM.size());
  for (const auto &kv : state->_features_SLAM) {
    if ((int)kv.first <= aruco_max)
      continue;
    pts.push_back(kv.second->get_xyz(false));
  }

  if ((int)pts.size() < _opt.slam_plane_min_points) {
    PRINT_DEBUG("[GP-plane] not enough SLAM pts (%zu < %d)\n", pts.size(), _opt.slam_plane_min_points);
    return;
  }

  const double thresh = _opt.slam_plane_ransac_thresh_base + _opt.slam_plane_ransac_thresh_h_scale * std::abs(h_rel);
  const int N = (int)pts.size();

  // RANSAC: sample 3 points, fit plane, count inliers
  int best_inliers = 0;
  Eigen::Vector3d best_n = Eigen::Vector3d::UnitZ();
  double best_d = 0.0;

  std::mt19937 rng(static_cast<uint32_t>(pts.size()));
  std::uniform_int_distribution<int> dist(0, N - 1);

  for (int iter = 0; iter < _opt.slam_plane_ransac_iterations; iter++) {
    int i0 = dist(rng);
    int i1, i2;
    do { i1 = dist(rng); } while (i1 == i0);
    do { i2 = dist(rng); } while (i2 == i0 || i2 == i1);

    Eigen::Vector3d n_cand = (pts[i1] - pts[i0]).cross(pts[i2] - pts[i0]);
    if (n_cand.norm() < 1e-9)
      continue;
    n_cand.normalize();
    if (n_cand(2) < 0)
      n_cand = -n_cand;
    double d_cand = -n_cand.dot(pts[i0]);

    int cnt = 0;
    for (const auto &p : pts)
      cnt += (std::abs(n_cand.dot(p) + d_cand) < thresh) ? 1 : 0;

    if (cnt > best_inliers) {
      best_inliers = cnt;
      best_n = n_cand;
      best_d = d_cand;
    }
  }

  double inlier_ratio = (double)best_inliers / (double)N;
  PRINT_DEBUG("[GP-plane] RANSAC: pts=%d inliers=%d ratio=%.2f thresh=%.3f\n", N, best_inliers, inlier_ratio, thresh);

  if (inlier_ratio < _opt.slam_plane_min_inlier_ratio)
    return;

  // Reject non-horizontal planes (|n_z| < 0.7 ≈ inclination > 45°)
  if (std::abs(best_n(2)) < 0.7)
    return;

  // Collect RANSAC inliers, then SVD-refine the plane through them.
  _plane_inlier_pts.reserve(best_inliers);
  for (const auto &p : pts) {
    if (std::abs(best_n.dot(p) + best_d) < thresh)
      _plane_inlier_pts.push_back(p);
  }
  svd_fit_plane(_plane_inlier_pts, _plane_normal, _plane_d);

  // Sanity check: drone's height above the fitted plane must match barometric h_rel.
  // If features are clustered at drone altitude (not on the ground), the fitted plane
  // passes through them and n·p_drone + d ≈ 0, which will diverge from h_rel by ~h_rel.
  Eigen::Vector3d p_drone = state->_imu->pos();
  double h_from_plane = _plane_normal.dot(p_drone) + _plane_d;
  double h_err = std::abs(h_from_plane - h_rel);
  double h_tol = std::max(5.0, _opt.slam_plane_max_h_err_ratio * h_rel);
  if (h_err > h_tol) {
    PRINT_DEBUG("[GP-plane] Stage A rejected (h_check=%.1f h_rel=%.1f err=%.1f tol=%.1f)\n",
                h_from_plane, h_rel, h_err, h_tol);
    _plane_inlier_pts.clear();
    return;
  }

  _plane_valid = true;

  PRINT_DEBUG("[GP-plane] Stage A: n=[%.3f,%.3f,%.3f] d=%.3f inliers=%d/%d (h_rel=%.1f)\n",
              _plane_normal(0), _plane_normal(1), _plane_normal(2), _plane_d,
              (int)_plane_inlier_pts.size(), N, h_rel);
}

void UpdaterGroundPlane::refine_plane_stage_b(const Eigen::Matrix3d &R_GtoC, const Eigen::Vector3d &p_CcinG,
                                               const std::vector<cv::Point2f> &pts_cur_norm,
                                               const std::vector<uchar> &mask) {
  if (!_plane_valid || _plane_inlier_pts.empty())
    return;

  // Start from Stage A inlier points (ground-verified), then add current-frame inlier projections.
  std::vector<Eigen::Vector3d> refine_pts = _plane_inlier_pts;
  for (size_t i = 0; i < pts_cur_norm.size(); i++) {
    if (!mask[i])
      continue;
    Eigen::Vector3d ray_G = R_GtoC.transpose() * Eigen::Vector3d((double)pts_cur_norm[i].x, (double)pts_cur_norm[i].y, 1.0);
    double denom = _plane_normal.dot(ray_G);
    if (denom >= -1e-4)
      continue;
    double t = -(_plane_normal.dot(p_CcinG) + _plane_d) / denom;
    if (t < 0.1 || t > _opt.max_recovery_depth)
      continue;
    refine_pts.push_back(p_CcinG + t * ray_G);
  }

  if ((int)refine_pts.size() < _opt.slam_plane_min_points)
    return;

  Eigen::Vector3d n_new;
  double d_new;
  if (!svd_fit_plane(refine_pts, n_new, d_new) || std::abs(n_new(2)) < 0.7)
    return;

  _plane_normal = n_new;
  _plane_d = d_new;
  PRINT_DEBUG("[GP-plane] Stage B: n=[%.3f,%.3f,%.3f] d=%.3f pts=%zu\n",
              _plane_normal(0), _plane_normal(1), _plane_normal(2), _plane_d, refine_pts.size());
}

void UpdaterGroundPlane::recover_3d_points(const Eigen::Matrix3d &R_GtoC, const Eigen::Vector3d &p_CcinG,
                                            const std::vector<cv::Point2f> &pts_cur_norm,
                                            const std::vector<size_t> &feat_ids,
                                            const std::vector<uchar> &mask, double h_rel,
                                            std::vector<Eigen::Vector3d> &pts3d,
                                            std::vector<size_t> &out_ids) const {
  pts3d.clear();
  out_ids.clear();
  pts3d.reserve(pts_cur_norm.size());
  out_ids.reserve(pts_cur_norm.size());

  for (size_t i = 0; i < pts_cur_norm.size(); i++) {
    if (!mask[i])
      continue;
    Eigen::Vector3d ray_G = R_GtoC.transpose() * Eigen::Vector3d((double)pts_cur_norm[i].x, (double)pts_cur_norm[i].y, 1.0);

    double t_depth;
    if (_plane_valid) {
      double denom = _plane_normal.dot(ray_G);
      if (denom >= -1e-4)
        continue;
      t_depth = -(_plane_normal.dot(p_CcinG) + _plane_d) / denom;
    } else {
      double beta = ray_G(2);
      if (beta >= -1e-4)
        continue;
      t_depth = -h_rel / beta;
    }

    if (t_depth < 0.1 || t_depth > _opt.max_recovery_depth)
      continue;

    pts3d.push_back(p_CcinG + t_depth * ray_G);
    out_ids.push_back(feat_ids[i]);
  }
}

void UpdaterGroundPlane::try_height_update(std::shared_ptr<State> state, double timestamp) {
  if (state->_clones_IMU.empty())
    return;

  calibrate_offset_if_needed(state, timestamp);

  if (_offset_initialized) {
    do_height_update(state, timestamp);
  }
}

bool UpdaterGroundPlane::interp_height(double t, double &h_out) const {
  if (_height_buf.size() < 2)
    return false;

  auto it = _height_buf.begin();
  while (std::next(it) != _height_buf.end() && std::next(it)->first <= t)
    ++it;

  auto lo = it;
  if (lo == _height_buf.begin() && lo->first > t)
    return false;
  if (std::next(lo) == _height_buf.end() && lo->first < t)
    return false;

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

  _h_takeoff_offset = h_raw - state->_imu->pos()(2);
  _offset_initialized = true;
  PRINT_INFO("[GP] takeoff offset calibrated: h_raw=%.3f p_z=%.3f offset=%.3f\n", h_raw, state->_imu->pos()(2), _h_takeoff_offset);
}

void UpdaterGroundPlane::do_height_update(std::shared_ptr<State> state, double t) {
  double h_raw;
  if (!interp_height(t, h_raw))
    return;

  double h_rel = h_raw - _h_takeoff_offset;

  if (h_rel < _opt.min_height_for_update) {
    PRINT_DEBUG("[GP] skipping h_rel update (h=%.2f < min=%.2f)\n", h_rel, _opt.min_height_for_update);
    return;
  }

  double h_pred = state->_imu->pos()(2);
  double innov = h_rel - h_pred;

  if (_has_last_innov && std::abs(innov - _last_innov) > _opt.max_height_jump) {
    _last_innov = innov;
    _has_last_innov = true;
    PRINT_DEBUG("[GP] height jump detected (%.3f), skipping update\n", innov - _last_innov);
    return;
  }
  _last_innov = innov;
  _has_last_innov = true;

  double sigma = (h_rel < _opt.height_sigma_transition) ? _opt.sigma_height_low : _opt.sigma_height;

  std::vector<std::shared_ptr<Type>> Hx_order;
  Hx_order.push_back(state->_imu->p());

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 3);
  H(0, 2) = 1.0;

  Eigen::VectorXd res(1);
  res(0) = innov;

  Eigen::MatrixXd R = Eigen::MatrixXd::Constant(1, 1, sigma * sigma);

  Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, Hx_order);
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  double chi2 = (res.transpose() * S.llt().solve(res))(0, 0);

  boost::math::chi_squared dist(1);
  double thresh = boost::math::quantile(dist, 0.95) * _opt.height_chi2_multiplier;
  if (chi2 > thresh) {
    PRINT_DEBUG("[GP] height chi2 fail: %.2f > %.2f\n", chi2, thresh);
    // Keep behavior: only log, still apply update.
  }

  StateHelper::EKFUpdate(state, Hx_order, H, res, R);
  PRINT_DEBUG("[GP] h_rel update: p_z=%.3f h_rel=%.3f innov=%.3f chi2=%.2f sigma=%.2f\n", h_pred, h_rel, innov, chi2, sigma);
}

void UpdaterGroundPlane::promote_homography_points_to_slam(std::shared_ptr<State> state, double timestamp,
                                                            const std::vector<std::shared_ptr<ov_core::Feature>> &leftovers) {
  _homography_plane_points.clear();
  if (leftovers.empty())
    return;

  // Height gate — always enforced
  if (!_offset_initialized)
    return;
  double h_raw;
  if (!interp_height(timestamp, h_raw))
    return;
  double h_rel = h_raw - _h_takeoff_offset;
  if (h_rel < _opt.min_height_for_update)
    return;

  // Stage A: fit ground plane from existing SLAM landmarks
  fit_plane_from_slam_points(state, h_rel);

  // Camera pose at the two most-recent clones
  if ((int)state->_clones_IMU.size() < 2)
    return;
  auto it = state->_clones_IMU.end();
  const double t_cur = (--it)->first;
  const double t_prev = (--it)->first;

  constexpr size_t cam_id = 0;
  Eigen::Matrix3d R_GtoC;
  Eigen::Vector3d p_CcinG;
  if (!get_camera_pose(state, cam_id, t_cur, R_GtoC, p_CcinG))
    return;

  // 2D correspondences from leftover features
  std::vector<cv::Point2f> pts_prev_norm, pts_cur_norm;
  std::vector<size_t> feat_ids_norm;
  collect_feature_correspondences(leftovers, cam_id, t_prev, t_cur, pts_prev_norm, pts_cur_norm, feat_ids_norm);
  if ((int)pts_cur_norm.size() < _opt.homography_min_inliers)
    return;

  // 2D-2D homography RANSAC inlier check
  auto rT0 = boost::posix_time::microsec_clock::local_time();
  std::vector<uchar> mask;
  cv::Mat H_cv = cv::findHomography(pts_prev_norm, pts_cur_norm, cv::RANSAC,
                                    _opt.homography_ransac_thresh, mask, 2000, 0.999);
  auto rT1 = boost::posix_time::microsec_clock::local_time();
  if (H_cv.empty())
    return;

  int n_inliers = std::count(mask.begin(), mask.end(), (uchar)1);
  double inlier_ratio = (double)n_inliers / (double)pts_cur_norm.size();
  PRINT_DEBUG("[GP-homo-promote] feats=%zu inliers=%d ratio=%.2f h_rel=%.2f plane_valid=%d\n",
              pts_cur_norm.size(), n_inliers, inlier_ratio, h_rel, (int)_plane_valid);
  if (n_inliers < _opt.homography_min_inliers || inlier_ratio < _opt.homography_min_inlier_ratio)
    return;

  // Stage B: refine plane with current-frame inlier reprojections
  refine_plane_stage_b(R_GtoC, p_CcinG, pts_cur_norm, mask);

  // 3D recovery via plane-ray intersection (or baro fallback)
  std::vector<Eigen::Vector3d> pts3d;
  std::vector<size_t> pts_feat_ids;
  recover_3d_points(R_GtoC, p_CcinG, pts_cur_norm, feat_ids_norm, mask, h_rel, pts3d, pts_feat_ids);

  _homography_plane_points = pts3d;
  if ((int)pts3d.size() < _opt.homography_min_inliers)
    return;

  promote_homography_points_to_slam_from_points(state, pts_feat_ids, pts3d, h_rel);
  auto rT2 = boost::posix_time::microsec_clock::local_time();
  PRINT_DEBUG("[GP-homo-promote] promoted from %zu recovered points (h_rel=%.1f plane=%d)\n",
              pts3d.size(), h_rel, (int)_plane_valid);
  PRINT_DEBUG("[GP-homo-timing] findHomography=%.4fs promote_to_slam=%.4fs\n",
              (rT1 - rT0).total_microseconds() * 1e-6, (rT2 - rT1).total_microseconds() * 1e-6);
}

void UpdaterGroundPlane::promote_homography_points_to_slam_from_points(std::shared_ptr<State> state, const std::vector<size_t> &feat_ids,
                                                                        const std::vector<Eigen::Vector3d> &pts3d, double h_rel) {
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
