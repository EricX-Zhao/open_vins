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

#ifndef OV_MSCKF_UPDATER_PUREYAW_H
#define OV_MSCKF_UPDATER_PUREYAW_H

#include <memory>
#include <vector>

#include "utils/sensor_data.h"
#include "utils/NoiseManager.h"

namespace ov_msckf {

class State;

/**
 * @brief Detects pure-yaw rotation at altitude and updates rotation + SLAM feature depths.
 *
 * When the vehicle is performing a pure yaw rotation at high altitude, SLAM features tend to
 * fail the chi2 check repeatedly because the feature positions in the image move significantly
 * with each frame. Two consecutive chi2 failures cause update_fail_count > 1, which triggers
 * marginalization of all SLAM features.
 *
 * This updater detects pure-yaw conditions (large rotation mostly around the vertical axis,
 * at altitude above _min_altitude) and, instead of adding a new clone:
 *   1. Updates rotation bias using the gyroscope constraint.
 *   2. Pins SLAM feature depths via zero-residual pseudo-measurements to prevent covariance growth.
 *   3. Resets update_fail_count for all SLAM features.
 *
 * Returning true from try_update causes VioManager to skip do_feature_propagate_update,
 * so no new clone is added for this frame.
 */
class UpdaterPureYaw {

public:
  /**
   * @brief Constructor.
   * @param noises         IMU noise characteristics (continuous time)
   * @param min_altitude   Minimum altitude (m) above which pure-yaw detection is active
   * @param gyro_min_rate  Minimum angular rate (rad/s) required to consider as "rotating"
   * @param yaw_dom_ratio  Minimum fraction of |w_G(2)| / |w_G| to classify as "pure yaw"
   * @param sigma_depth    Depth uncertainty (m) used for the depth-pin pseudo-measurement
   */
  UpdaterPureYaw(const NoiseManager &noises, double min_altitude = 20.0, double gyro_min_rate = 0.3,
                 double yaw_dom_ratio = 0.70, double sigma_depth = 5.0);

  /**
   * @brief Feed function for inertial data.
   * @param message     Contains our timestamp and inertial information
   * @param oldest_time Time that we can discard measurements before
   */
  void feed_imu(const ov_core::ImuData &message, double oldest_time = -1) {
    imu_data.emplace_back(message);
    clean_old_imu_measurements(oldest_time - 0.10);
  }

  /**
   * @brief Remove any IMU measurements older than the given time.
   * @param oldest_time Time before which measurements are discarded (IMU clock)
   */
  void clean_old_imu_measurements(double oldest_time) {
    if (oldest_time < 0)
      return;
    auto it0 = imu_data.begin();
    while (it0 != imu_data.end()) {
      if (it0->timestamp < oldest_time) {
        it0 = imu_data.erase(it0);
      } else {
        it0++;
      }
    }
  }

  /**
   * @brief Detect pure-yaw and, if confirmed, apply EKF updates.
   * @param state     Current filter state
   * @param timestamp Next camera timestamp
   * @param altitude  Vehicle altitude above ground (m); negative means unknown / disabled
   * @return True if pure-yaw was detected and updates were applied (caller should skip propagate_and_clone)
   */
  bool try_update(std::shared_ptr<State> state, double timestamp, double altitude);

protected:
  /// IMU noise characteristics
  NoiseManager _noises;

  /// Minimum altitude for the updater to activate (m)
  double _min_altitude;

  /// Minimum total angular rate (rad/s) to be considered "rotating"
  double _gyro_min_rate;

  /// Minimum |w_G_z| / |w_G| fraction to be classified as pure yaw
  double _yaw_dom_ratio;

  /// Depth uncertainty (m) for the depth-pin pseudo-measurement
  double _sigma_depth;

  /// Buffered IMU measurements
  std::vector<ov_core::ImuData> imu_data;

  /// Last IMU-camera time offset used for propagation bookkeeping
  double last_prop_time_offset = 0.0;

  /// Whether last_prop_time_offset has been initialised
  bool have_last_prop_time_offset = false;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_PUREYAW_H
