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

#ifndef OV_MSCKF_UPDATER_GROUND_PLANE_H
#define OV_MSCKF_UPDATER_GROUND_PLANE_H

#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Eigen>
#include <opencv2/core/types.hpp>

#include "feat/Feature.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

namespace ov_msckf {

class State;

/**
 * @brief Height and homography-promotion helper for downward-facing monocular VIO.
 */
class UpdaterGroundPlane {

public:
  /// Configuration parameters
  struct Options {
    // --- height observation ---
    double sigma_height = 0.5;           ///< stddev of baro noise above height_sigma_transition [m]
    double sigma_height_low = 2.0;       ///< stddev of baro noise below height_sigma_transition [m] (less reliable near ground)
    double height_sigma_transition = 15.0; ///< altitude [m] at which sigma switches from sigma_height_low to sigma_height
    double max_height_jump = 2.0;        ///< innovation jump threshold for outlier rejection [m]
    double min_height_for_update = 10.0;  ///< skip height update when h_rel below this [m] (baro totally unreliable near ground)
    double height_chi2_multiplier = 2.0; ///< multiplier on chi2 gate threshold

    // --- homography promotion ---
    double homography_ransac_thresh = 0.005;   ///< RANSAC inlier threshold in normalised image coords
    double homography_min_inlier_ratio = 0.7;  ///< minimum fraction of features that must be inliers
    int homography_min_inliers = 8;            ///< minimum absolute inlier count
    double max_recovery_depth = 400.0;         ///< maximum valid recovered depth [m]

    int homography_persistent_max_add_per_update = 15; ///< hard cap per frame
    double homography_persistent_sigma_min = 3.0;     ///< minimum init stddev in global xyz [m]
    double homography_persistent_sigma_h_rel_scale = 0.03; ///< extra init stddev scaling with altitude [m/m]

    // --- SLAM-point plane fitting ---
    int slam_plane_min_points = 5;                    ///< minimum SLAM landmarks to attempt plane fit
    double slam_plane_ransac_thresh_base = 0.1;       ///< base inlier distance to plane [m]
    double slam_plane_ransac_thresh_h_scale = 0.02;   ///< additional threshold per metre of altitude [m/m]
    int slam_plane_ransac_iterations = 100;           ///< RANSAC iterations for plane fitting
    double slam_plane_min_inlier_ratio = 0.7;         ///< minimum inlier ratio for a valid plane fit
    double slam_plane_max_h_err_ratio = 0.3;          ///< reject plane if |n·p_drone + d - h_rel| > ratio * h_rel

    /**
     * @brief Load parameters from a YAML parser and print current values.
     * @param parser If not null, values are read from the config file; otherwise defaults are printed.
     */
    void print_and_load(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
      if (parser != nullptr) {
        // height
        parser->parse_config("gp_sigma_height", sigma_height, false);
        parser->parse_config("gp_sigma_height_low", sigma_height_low, false);
        parser->parse_config("gp_height_sigma_transition", height_sigma_transition, false);
        parser->parse_config("gp_max_height_jump", max_height_jump, false);
        parser->parse_config("gp_min_height_for_update", min_height_for_update, false);
        parser->parse_config("gp_height_chi2_multiplier", height_chi2_multiplier, false);
        // homography
        parser->parse_config("gp_homography_ransac_thresh", homography_ransac_thresh, false);
        parser->parse_config("gp_homography_min_inlier_ratio", homography_min_inlier_ratio, false);
        parser->parse_config("gp_homography_min_inliers", homography_min_inliers, false);
        parser->parse_config("gp_max_recovery_depth", max_recovery_depth, false);
        parser->parse_config("gp_homography_persistent_max_add_per_update", homography_persistent_max_add_per_update, false);
        parser->parse_config("gp_homography_persistent_sigma_min", homography_persistent_sigma_min, false);
        parser->parse_config("gp_homography_persistent_sigma_h_rel_scale", homography_persistent_sigma_h_rel_scale, false);
        // SLAM plane fitting
        parser->parse_config("gp_slam_plane_min_points", slam_plane_min_points, false);
        parser->parse_config("gp_slam_plane_ransac_thresh_base", slam_plane_ransac_thresh_base, false);
        parser->parse_config("gp_slam_plane_ransac_thresh_h_scale", slam_plane_ransac_thresh_h_scale, false);
        parser->parse_config("gp_slam_plane_ransac_iterations", slam_plane_ransac_iterations, false);
        parser->parse_config("gp_slam_plane_min_inlier_ratio", slam_plane_min_inlier_ratio, false);
        parser->parse_config("gp_slam_plane_max_h_err_ratio", slam_plane_max_h_err_ratio, false);
      }
      PRINT_DEBUG("GROUND PLANE PARAMETERS:\n");
      PRINT_DEBUG("  - gp_sigma_height: %.3f\n", sigma_height);
      PRINT_DEBUG("  - gp_sigma_height_low: %.3f\n", sigma_height_low);
      PRINT_DEBUG("  - gp_height_sigma_transition: %.1f\n", height_sigma_transition);
      PRINT_DEBUG("  - gp_max_height_jump: %.2f\n", max_height_jump);
      PRINT_DEBUG("  - gp_min_height_for_update: %.2f\n", min_height_for_update);
      PRINT_DEBUG("  - gp_height_chi2_multiplier: %.2f\n", height_chi2_multiplier);
      PRINT_DEBUG("  - gp_homography_ransac_thresh: %.4f\n", homography_ransac_thresh);
      PRINT_DEBUG("  - gp_homography_min_inlier_ratio: %.2f\n", homography_min_inlier_ratio);
      PRINT_DEBUG("  - gp_homography_min_inliers: %d\n", homography_min_inliers);
      PRINT_DEBUG("  - gp_max_recovery_depth: %.1f\n", max_recovery_depth);
      PRINT_DEBUG("  - gp_persistent_max_add_per_update: %d\n", homography_persistent_max_add_per_update);
      PRINT_DEBUG("  - gp_persistent_sigma_min: %.2f\n", homography_persistent_sigma_min);
      PRINT_DEBUG("  - gp_persistent_sigma_h_rel_scale: %.4f\n", homography_persistent_sigma_h_rel_scale);
      PRINT_DEBUG("  - gp_slam_plane_min_points: %d\n", slam_plane_min_points);
      PRINT_DEBUG("  - gp_slam_plane_ransac_thresh_base: %.3f\n", slam_plane_ransac_thresh_base);
      PRINT_DEBUG("  - gp_slam_plane_ransac_thresh_h_scale: %.4f\n", slam_plane_ransac_thresh_h_scale);
      PRINT_DEBUG("  - gp_slam_plane_ransac_iterations: %d\n", slam_plane_ransac_iterations);
      PRINT_DEBUG("  - gp_slam_plane_min_inlier_ratio: %.2f\n", slam_plane_min_inlier_ratio);
      PRINT_DEBUG("  - gp_slam_plane_max_h_err_ratio: %.2f\n", slam_plane_max_h_err_ratio);
    }
  };

  explicit UpdaterGroundPlane(const Options &opt);

  /**
   * @brief Buffer a barometric / flight-controller height measurement.
   * @param timestamp Sensor timestamp [s]
   * @param h_rel     Height above takeoff point [m]
   */
  void feed_height(double timestamp, double h_rel);

  /**
   * @brief Height-only update — call right after propagation, before MSCKF/SLAM updates.
   *        Calibrates the takeoff offset on the first call and applies the barometric p_z constraint.
   */
  void try_height_update(std::shared_ptr<State> state, double timestamp);

  /// Returns true when a valid ground plane has been fitted from SLAM landmarks.
  bool is_plane_initialized() const { return _plane_valid; }

  /// Closest point on the fitted plane to the origin; zero if plane is not yet valid.
  Eigen::Vector3d get_cp() const;

  /// Unit normal of the fitted plane (pointing upward); UnitZ when not yet valid.
  Eigen::Vector3d get_plane_normal() const { return _plane_normal; }

  /// Plane offset parameter satisfying n^T * X + d = 0; 0 when not yet valid.
  double get_plane_d() const { return _plane_d; }

  /// 3D points recovered by the most recent homography update (empty if not run yet)
  const std::vector<Eigen::Vector3d> &get_homography_plane_points() const { return _homography_plane_points; }

  /**
   * @brief Public promotion-only entrypoint.
   *        Can be called directly from VioManager to run homography-based 3D recovery and
   *        promote kept ground points into persistent SLAM landmarks.
   */
  void promote_homography_points_to_slam(std::shared_ptr<State> state, double timestamp,
                                         const std::vector<std::shared_ptr<ov_core::Feature>> &leftovers);

private:
  // ---- height helpers ----

  bool interp_height(double t, double &h_out) const;
  void calibrate_offset_if_needed(std::shared_ptr<State> state, double t);
  void do_height_update(std::shared_ptr<State> state, double t);

  /// Fit a ground plane from current SLAM landmarks via RANSAC + SVD.
  /// Results are stored in _plane_normal, _plane_d, _plane_valid.
  void fit_plane_from_slam_points(std::shared_ptr<State> state, double h_rel);

  /// Stage B: refine _plane_normal/_plane_d using Stage A inliers + reprojected 2D inliers.
  void refine_plane_stage_b(const Eigen::Matrix3d &R_GtoC, const Eigen::Vector3d &p_CcinG,
                             const std::vector<cv::Point2f> &pts_cur_norm, const std::vector<uchar> &mask);

  /// Recover 3D positions for homography inliers via plane-ray intersection (or baro fallback).
  void recover_3d_points(const Eigen::Matrix3d &R_GtoC, const Eigen::Vector3d &p_CcinG,
                          const std::vector<cv::Point2f> &pts_cur_norm, const std::vector<size_t> &feat_ids,
                          const std::vector<uchar> &mask, double h_rel,
                          std::vector<Eigen::Vector3d> &pts3d, std::vector<size_t> &out_ids) const;

  /// Internal helper: inject already recovered points into the SLAM state.
  void promote_homography_points_to_slam_from_points(std::shared_ptr<State> state, const std::vector<size_t> &feat_ids,
                                                      const std::vector<Eigen::Vector3d> &pts3d, double h_rel);

  // ---- member state ----

  Options _opt;

  // Height measurement buffer: (timestamp, h_rel)
  std::deque<std::pair<double, double>> _height_buf;
  static constexpr size_t kMaxHeightBufSize = 500;

  // Takeoff-point offset calibration
  bool _offset_initialized = false;
  double _h_takeoff_offset = 0.0;

  // Jump detection for height update
  bool _has_last_innov = false;
  double _last_innov = 0.0;

  // 3D points from the most recent homography update (for visualization)
  std::vector<Eigen::Vector3d> _homography_plane_points;

  // Ground plane fitted from SLAM landmarks: n^T * X + d = 0, ||n|| = 1
  Eigen::Vector3d _plane_normal = Eigen::Vector3d::UnitZ();
  double _plane_d = 0.0;
  bool _plane_valid = false;
  // RANSAC inlier points from the most recent Stage A fit (used by Stage B)
  std::vector<Eigen::Vector3d> _plane_inlier_pts;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_GROUND_PLANE_H
