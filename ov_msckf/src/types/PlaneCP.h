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

#ifndef OV_MSCKF_PLANE_CP_H
#define OV_MSCKF_PLANE_CP_H

#include "types/Vec.h"

namespace ov_msckf {

/**
 * @brief Closest-point (CP) plane parameterization.
 *
 * The plane is represented by the vector from the world-frame origin to the
 * closest point on the plane.  Given cp = p_pi^G the plane equation is:
 *
 *   cp · x = ||cp||^2   (equivalently  n · x = d,  n = cp/||cp||,  d = ||cp||)
 *
 * This is a simple 3-vector in R^3.  The update operation is ordinary vector
 * addition, so no rebase / re-parameterisation is needed.
 */
class PlaneCP : public ov_type::Vec {

public:
  PlaneCP() : Vec(3) {}

  /// Closest-point vector (the 3-DOF plane state)
  Eigen::Vector3d cp() const { return _value; }

  /// Distance from the world-frame origin to the plane  (= ||cp||)
  double distance() const { return _value.norm(); }

  /// Unit normal pointing from the origin toward the plane  (= cp / ||cp||)
  Eigen::Vector3d normal() const {
    double d = _value.norm();
    return d > 1e-9 ? _value / d : Eigen::Vector3d(0, 0, 1);
  }

  std::shared_ptr<Type> clone() override {
    auto c = std::make_shared<PlaneCP>();
    c->set_value(value());
    c->set_fej(fej());
    return c;
  }
};

} // namespace ov_msckf

#endif // OV_MSCKF_PLANE_CP_H
