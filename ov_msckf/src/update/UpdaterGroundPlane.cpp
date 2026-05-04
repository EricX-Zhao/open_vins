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

#include <boost/math/distributions/chi_squared.hpp>
#include <opencv2/calib3d.hpp>

#include <cmath>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterGroundPlane::UpdaterGroundPlane(const Options &opt) : _opt(opt) {}

void UpdaterGroundPlane::feed_height(double timestamp, double h_rel) {
  _height_buf.push_back({timestamp, h_rel});
  while (_height_buf.size() > kMaxHeightBufSize)
    _height_buf.pop_front();
}

Eigen::Vector3d UpdaterGroundPlane::get_cp() const {
  return Eigen::Vector3d::Zero();
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

  if (!_offset_initialized)
    return;
  double h_raw;
  if (!interp_height(timestamp, h_raw))
    return;
  double h_rel = h_raw - _h_takeoff_offset;
  if (h_rel < _opt.min_height_for_update)
    return;

  if ((int)state->_clones_IMU.size() < 2)
    return;
  auto it_end = state->_clones_IMU.end();
  --it_end;
  double t_cur = it_end->first;
  --it_end;
  double t_prev = it_end->first;

  const size_t cam_id = 0;
  if (state->_calib_IMUtoCAM.find(cam_id) == state->_calib_IMUtoCAM.end())
    return;
  Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(cam_id)->Rot();
  Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(cam_id)->pos();

  Eigen::Matrix3d R_GtoI_c = state->_clones_IMU.at(t_cur)->Rot();
  Eigen::Vector3d p_IcinG = state->_clones_IMU.at(t_cur)->pos();
  Eigen::Matrix3d R_GtoC = R_ItoC * R_GtoI_c;
  Eigen::Vector3d p_CcinG = p_IcinG - R_GtoI_c.transpose() * R_ItoC.transpose() * p_IinC;

  std::vector<cv::Point2f> pts_prev_norm, pts_cur_norm;
  std::vector<size_t> feat_ids_norm;
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

    pts_cur_norm.emplace_back((float)uvs[idx_cur](0), (float)uvs[idx_cur](1));
    pts_prev_norm.emplace_back((float)uvs[idx_prev](0), (float)uvs[idx_prev](1));
    feat_ids_norm.push_back(feat->featid);
  }

  if ((int)pts_cur_norm.size() < _opt.homography_min_inliers)
    return;

  std::vector<uchar> mask;
  cv::Mat H_cv = cv::findHomography(pts_prev_norm, pts_cur_norm, cv::RANSAC,
                                    _opt.homography_ransac_thresh, mask, 2000, 0.999);
  if (H_cv.empty())
    return;

  int n_inliers = 0;
  for (auto m : mask)
    n_inliers += (m != 0);

  double inlier_ratio = (double)n_inliers / (double)pts_cur_norm.size();
  PRINT_DEBUG("[GP-homo-promote] feats=%zu inliers=%d ratio=%.2f h_rel=%.2f\n",
              pts_cur_norm.size(), n_inliers, inlier_ratio, h_rel);

  if (n_inliers < _opt.homography_min_inliers || inlier_ratio < _opt.homography_min_inlier_ratio)
    return;

  std::vector<Eigen::Vector3d> pts3d;
  std::vector<size_t> pts_feat_ids;
  pts3d.reserve(n_inliers);
  pts_feat_ids.reserve(n_inliers);

  for (size_t i = 0; i < pts_cur_norm.size(); i++) {
    if (!mask[i])
      continue;

    Eigen::Vector3d m_cam((double)pts_cur_norm[i].x, (double)pts_cur_norm[i].y, 1.0);
    Eigen::Vector3d ray_G = R_GtoC.transpose() * m_cam;
    double beta = ray_G(2);
    if (beta >= -1e-4)
      continue;

    double t_depth = -h_rel / beta;
    if (t_depth < 0.1 || t_depth > _opt.max_recovery_depth)
      continue;

    pts3d.push_back(p_CcinG + t_depth * ray_G);
    pts_feat_ids.push_back(feat_ids_norm[i]);
  }

  _homography_plane_points = pts3d;
  if ((int)pts3d.size() < _opt.homography_min_inliers)
    return;

  promote_homography_points_to_slam_from_points(state, pts_feat_ids, pts3d, h_rel);
  PRINT_DEBUG("[GP-homo-promote] promoted from %zu recovered points (h_rel=%.1f)\n",
              pts3d.size(), h_rel);
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
