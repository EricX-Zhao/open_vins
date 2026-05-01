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

#include "feat/Feature.h"
#include "types/Landmark.h"
#include "types/PlaneCP.h"

namespace ov_msckf {

class State;

/**
 * @brief Ground-plane enhancement for high-altitude downward-facing monocular VIO.
 *
 * Implements Method C v3.0:
 *  - CP (closest-point) plane parameterisation
 *  - Data-driven initialisation (RANSAC + SVD)
 *  - Feature-plane identity inheritance via KLT feature IDs
 *  - Point-on-plane soft constraint
 *  - Barometric / flight-controller height observation (only constrains p_z)
 */
class UpdaterGroundPlane {

public:
  /// Configuration parameters
  struct Options {
    // --- master switches ---
    bool enable_plane = true;
    bool enable_height = true;
    bool enable_point_on_plane = true;

    // --- height observation ---
    double sigma_height = 0.5;           ///< stddev of baro noise above height_sigma_transition [m]
    double sigma_height_low = 2.0;       ///< stddev of baro noise below height_sigma_transition [m] (less reliable near ground)
    double height_sigma_transition = 15.0; ///< altitude [m] at which sigma switches from sigma_height_low to sigma_height
    double max_height_jump = 2.0;        ///< innovation jump threshold for outlier rejection [m]
    double min_height_for_update = 5.0;  ///< skip height update when h_rel below this [m] (baro totally unreliable near ground)

    // --- plane initialisation ---
    int init_min_slam_feats = 10;       ///< minimum SLAM features before trying init
    double init_retry_interval = 0.5;   ///< minimum seconds between init attempts
    int ransac_iters = 100;
    double ransac_inlier_thresh_pct = 0.01; ///< inlier threshold as fraction of altitude (e.g. 0.01 = 1%)
    double ransac_inlier_thresh_min = 0.05; ///< floor: never go below this [m]
    double ransac_inlier_thresh_max = 2.00; ///< ceiling: never exceed this [m]
    int ransac_min_inliers = 8;
    double ransac_min_inlier_ratio = 0.6;

    // --- feature-plane association ---
    double assoc_thresh_in = 0.05;        ///< on-birth association threshold [m]
    double individual_chi2_thresh = 5.99; ///< single-feature chi2 de-association threshold
    int max_individual_chi2_fail = 5;     ///< consecutive failures before de-association

    // --- point-on-plane soft constraint ---
    double sigma_point_on_plane = 0.05; ///< plane thickness tolerance [m]
    double chi2_multiplier = 2.0;       ///< multiplier on chi2 gate threshold

    // --- normal / gravity consistency ---
    double max_normal_gravity_cos = -0.5; ///< plane CP normal must satisfy n·g_hat < this; rejects flipped/tilted planes

    // --- plane slow-walk process noise ---
    double sigma_cp_walk = 0.001;       ///< CP random walk stddev per sqrt-second

    // --- health monitoring ---
    int max_low_assoc_frames = 30;  ///< consecutive low-association frames before reset
    int max_chi2_fail_frames = 20;  ///< consecutive chi2-fail frames before reset
    int min_planar_feats = 3;       ///< minimum planar features to keep plane healthy

    // --- v3.1: leftover plane-depth recovery (mechanism C) ---
    bool enable_plane_recovery = true; ///< default off; enable once v3.0 is stable

    // candidate filtering
    double min_time_baseline = 0.1;    ///< minimum time span between first and last observation [s]
    int min_recovery_features = 5;     ///< minimum candidates required to attempt update
    int max_recovery_features = 30;    ///< cap to prevent H matrix from blowing up

    // depth sanity
    double min_recovery_depth = 0.5;   ///< minimum valid recovered depth [m]
    double max_recovery_depth = 300.0; ///< maximum valid recovered depth [m]

    // plane maturity gate
    double recovery_max_plane_cov = 0.01; ///< max trace of CP covariance (≈ σ 5 cm per axis)
    int min_slam_planar_for_recovery = 5; ///< minimum active planar SLAM features

    // optional pre-filter
    bool enable_recovery_prefilter = false;
    double recovery_prefilter_thresh = 0.02; ///< reprojection threshold in normalised coords (~10 px)

    // EKF update noise
    double sigma_pixel_recovery = 0.01;   ///< pixel noise for recovery features (slightly loose)
    double recovery_chi2_multiplier = 2.0;

    // --- homography plane update ---
    bool enable_homography_plane_update = true;
    double homography_ransac_thresh = 0.005;   ///< RANSAC inlier threshold in normalised image coords
    double homography_min_inlier_ratio = 0.4;  ///< minimum fraction of features that must be inliers
    int homography_min_inliers = 8;            ///< minimum absolute inlier count

    // --- optional: promote homography-recovered points to persistent SLAM landmarks ---
    bool enable_homography_persistent_landmarks = true;
    int homography_persistent_max_add_per_update = 6; ///< hard cap per frame
    double homography_persistent_sigma_min = 3.0;     ///< minimum init stddev in global xyz [m]
    double homography_persistent_sigma_h_rel_scale = 0.03; ///< extra init stddev scaling with altitude [m/m]
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

  /**
   * @brief Main entry point — call every camera frame after UpdaterSLAM::delayed_init().
   * @param leftovers Triangulation-failed features from MSCKF/SLAM (v3.1 mechanism C input)
   */
  void try_update(std::shared_ptr<State> state, double timestamp,
                  const std::vector<std::shared_ptr<ov_core::Feature>> &leftovers = {});

  /// True once the plane has been successfully initialised
  bool is_plane_initialized() const { return _plane_initialized; }

  /// Current CP vector (zero if not initialised)
  Eigen::Vector3d get_cp() const;

  /// 3D points recovered by the most recent homography update (empty if not run yet)
  const std::vector<Eigen::Vector3d> &get_homography_plane_points() const { return _homography_plane_points; }

  /**
   * @brief Homography-based one-shot planar feature update.
   *        Runs findHomography on normalised coords, recovers 3D points from FC height, stores them for
   *        visualisation, and directly applies a pose-only EKF update using previous-frame reprojection residuals.
   */
  void apply_homography_plane_update(std::shared_ptr<State> state, double timestamp,
                                      const std::vector<std::shared_ptr<ov_core::Feature>> &leftovers);

  /// Promote a subset of homography-recovered ground points into persistent SLAM landmarks.
  void promote_homography_points_to_slam(std::shared_ptr<State> state, const std::vector<size_t> &feat_ids,
                                         const std::vector<Eigen::Vector3d> &pts3d, double h_rel);

private:
  // ---- height helpers ----

  bool interp_height(double t, double &h_out) const;
  void calibrate_offset_if_needed(std::shared_ptr<State> state, double t);
  void do_height_update(std::shared_ptr<State> state, double t);

  // ---- plane initialisation ----

  struct PlaneCandidate {
    Eigen::Vector3d cp = Eigen::Vector3d::Zero();
    std::vector<int> inlier_indices;
    double avg_residual = 0.0;
  };

  bool should_attempt_init(std::shared_ptr<State> state) const;
  bool ransac_fit_plane(const std::vector<Eigen::Vector3d> &pts, PlaneCandidate &out, double inlier_thresh) const;
  void resolve_normal_direction(Eigen::Vector3d &cp, const std::vector<Eigen::Vector3d> &pts) const;
  void refine_plane_svd(const std::vector<Eigen::Vector3d> &pts, Eigen::Vector3d &cp) const;
  bool try_initialize_plane(std::shared_ptr<State> state);

  // ---- feature-plane association ----

  void check_plane_association_on_birth(std::shared_ptr<State> state);

  // ---- point-on-plane constraint ----

  void update_individual_chi2_counters(const std::vector<std::shared_ptr<ov_type::Landmark>> &feats,
                                       const Eigen::VectorXd &res, const Eigen::MatrixXd &S);
  void apply_point_on_plane_updates(std::shared_ptr<State> state);

  // ---- covariance propagation + health ----

  void propagate_plane_covariance(std::shared_ptr<State> state, double timestamp);
  void check_health(std::shared_ptr<State> state);
  void deinitialize_plane(std::shared_ptr<State> state);

  // ---- v3.1: mechanism C — leftover plane-depth recovery ----

  struct RecoveredFeature {
    std::shared_ptr<ov_core::Feature> feat;
    double t_anchor;
    double t_current;
    size_t cam_id;
    Eigen::Vector2d uv_anchor_norm;
    Eigen::Vector2d uv_current_norm;
    double rho_at_anchor; ///< recovered depth at anchor frame
    Eigen::Vector3d X_G;  ///< recovered world-frame position
  };

  struct RecoveryJacobian {
    Eigen::Matrix<double, 2, 3> J_theta_a; ///< wrt anchor IMU rotation (JPL)
    Eigen::Matrix<double, 2, 3> J_p_a;     ///< wrt anchor IMU position
    Eigen::Matrix<double, 2, 3> J_theta_c; ///< wrt current IMU rotation (JPL)
    Eigen::Matrix<double, 2, 3> J_p_c;     ///< wrt current IMU position
    Eigen::Matrix<double, 2, 3> J_cp;      ///< wrt plane CP
    Eigen::Vector2d residual;
  };

  bool can_use_plane_recovery(std::shared_ptr<State> state) const;

  std::vector<std::shared_ptr<ov_core::Feature>>
  filter_leftover_candidates(std::shared_ptr<State> state,
                             const std::vector<std::shared_ptr<ov_core::Feature>> &leftovers) const;

  bool try_recover_3d_from_plane(std::shared_ptr<State> state,
                                  const std::shared_ptr<ov_core::Feature> &feat,
                                  RecoveredFeature &rf) const;

  bool compute_recovery_jacobian(std::shared_ptr<State> state,
                                  const RecoveredFeature &rf,
                                  RecoveryJacobian &out) const;

  void build_and_apply_recovery_update(std::shared_ptr<State> state,
                                        const std::vector<RecoveredFeature> &recovered);

  void apply_plane_recovery_update(std::shared_ptr<State> state,
                                    const std::vector<std::shared_ptr<ov_core::Feature>> &leftovers);

  // ---- member state ----

  Options _opt;
  std::shared_ptr<PlaneCP> _plane;
  bool _plane_initialized = false;
  int _plane_serial = 0; ///< incremented each (re-)initialisation; used as plane_id

  // Height measurement buffer: (timestamp, h_rel)
  std::deque<std::pair<double, double>> _height_buf;
  static constexpr size_t kMaxHeightBufSize = 500;

  // Takeoff-point offset calibration
  bool _offset_initialized = false;
  double _h_takeoff_offset = 0.0;

  // Jump detection for height update
  bool _has_last_innov = false;
  double _last_innov = 0.0;

  // Init retry throttle
  double _last_init_attempt_time = -1.0;

  // Covariance propagation
  double _last_propagate_time = -1.0;

  // Health counters
  int _consecutive_low_assoc = 0;
  int _consecutive_chi2_fail = 0;
  bool _last_update_had_chi2_fail = false;

  // 3D points from the most recent homography update (for visualization)
  std::vector<Eigen::Vector3d> _homography_plane_points;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_GROUND_PLANE_H
