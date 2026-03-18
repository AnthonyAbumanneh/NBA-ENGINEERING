"""
Court geometry module for the NBA Court Re-Engineering Optimizer.

Coordinate system:
  - Basket at origin (0, 0)
  - Positive y toward half court
  - x along baseline (positive = right side from offensive team's perspective)
  - Angles measured from positive y-axis: negative = left, positive = right
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Zone Taxonomy
# ---------------------------------------------------------------------------

class Zone(Enum):
    Z01 = "Z01"   # restricted_area:           0–4 ft, all angles, 2PT
    Z02 = "Z02"   # paint_non_restricted:       4–8 ft, within paint width, 2PT
    Z03 = "Z03"   # left_short_corner_floater:  4–14 ft, angle < -60°, 2PT
    Z04 = "Z04"   # right_short_corner_floater: 4–14 ft, angle > +60°, 2PT
    Z05 = "Z05"   # left_elbow_mid_range:       14–arc_r ft, -60° to -30°, 2PT
    Z06 = "Z06"   # right_elbow_mid_range:      14–arc_r ft, +30° to +60°, 2PT
    Z07 = "Z07"   # center_mid_range:           14–arc_r ft, -30° to +30°, 2PT
    Z08 = "Z08"   # left_mid_range_baseline:    14–arc_r ft, angle < -60°, 2PT
    Z09 = "Z09"   # right_mid_range_baseline:   14–arc_r ft, angle > +60°, 2PT
    Z10 = "Z10"   # left_corner_3:              >= arc_r, angle < -65°, 3PT (or 2PT if eliminated)
    Z11 = "Z11"   # right_corner_3:             >= arc_r, angle > +65°, 3PT (or 2PT if eliminated)
    Z12 = "Z12"   # left_wing_3:                >= arc_r, -65° to -20°, 3PT
    Z13 = "Z13"   # right_wing_3:               >= arc_r, +20° to +65°, 3PT
    Z14 = "Z14"   # top_of_arc_3:               >= arc_r, -20° to +20°, 3PT


# Default point values (before corner3 elimination adjustment)
_ZONE_DEFAULT_POINTS: Dict[Zone, int] = {
    Zone.Z01: 2,
    Zone.Z02: 2,
    Zone.Z03: 2,
    Zone.Z04: 2,
    Zone.Z05: 2,
    Zone.Z06: 2,
    Zone.Z07: 2,
    Zone.Z08: 2,
    Zone.Z09: 2,
    Zone.Z10: 3,
    Zone.Z11: 3,
    Zone.Z12: 3,
    Zone.Z13: 3,
    Zone.Z14: 3,
}

_CORNER_ZONES = {Zone.Z10, Zone.Z11}


# ---------------------------------------------------------------------------
# corner3_eliminated — Task 3.2
# ---------------------------------------------------------------------------

def corner3_eliminated(
    arc_radius: float,
    baseline_width: float,
    basket_to_baseline: float = 5.25,
) -> bool:
    """
    Determine whether the corner 3-point zone is eliminated for a given court config.

    The corner 3 exists only when the arc does not reach the baseline within the
    half-width of the court. Derived analytically — no lookup tables.

    Args:
        arc_radius: 3-point arc radius in feet.
        baseline_width: Full court width in feet.
        basket_to_baseline: Distance from basket center to baseline (default 5.25 ft).

    Returns:
        True if the corner 3 zone is eliminated (arc hits baseline before sideline).
    """
    half_w = baseline_width / 2
    if arc_radius < half_w:
        # Arc never reaches the sideline; corner 3 always exists
        return False
    y_at_sideline = math.sqrt(arc_radius ** 2 - half_w ** 2)
    # If the arc at the sideline is at or below the baseline level,
    # the arc hits the baseline before the sideline → no straight corner segment
    return y_at_sideline <= basket_to_baseline


# ---------------------------------------------------------------------------
# CourtConfig dataclass — Task 3.1
# ---------------------------------------------------------------------------

@dataclass
class CourtConfig:
    """
    Configuration for a court geometry variant.

    Attributes:
        arc_radius: 3-point arc radius in feet.
        baseline_width: Full court width in feet.
        basket_to_baseline: Distance from basket center to baseline (constant 5.25 ft).
        corner3_eliminated: Computed property — True when the corner 3 zone is eliminated.
    """
    arc_radius: float
    baseline_width: float
    basket_to_baseline: float = 5.25

    @property
    def corner3_eliminated(self) -> bool:
        """Derived analytically from arc_radius, baseline_width, basket_to_baseline."""
        return corner3_eliminated(
            self.arc_radius, self.baseline_width, self.basket_to_baseline
        )


# ---------------------------------------------------------------------------
# zone_point_value — Task 3.1
# ---------------------------------------------------------------------------

def zone_point_value(zone_id: str, court_config: CourtConfig) -> int:
    """
    Return the point value (2 or 3) for a zone under the given court config.

    Z10 and Z11 return 2 when corner3_eliminated is True.

    Args:
        zone_id: Zone identifier string, e.g. "Z01", "Z14".
        court_config: The court configuration to evaluate against.

    Returns:
        2 or 3.
    """
    zone = Zone(zone_id)
    if zone in _CORNER_ZONES and court_config.corner3_eliminated:
        return 2
    return _ZONE_DEFAULT_POINTS[zone]


# ---------------------------------------------------------------------------
# classify_zone — Task 3.3
# ---------------------------------------------------------------------------

def classify_zone(x: float, y: float, court_config: CourtConfig) -> str:
    """
    Classify a shot location (x, y) into one of the 14 zones under the given court config.

    Coordinate system:
        - Basket at (0, 0)
        - Positive y toward half court
        - Angle measured from positive y-axis: atan2(x, y) in degrees
          Negative = left side, positive = right side

    Args:
        x: Horizontal coordinate (feet), positive = right.
        y: Vertical coordinate (feet), positive = toward half court.
        court_config: The court configuration defining arc radius and baseline width.

    Returns:
        Zone ID string, e.g. "Z01", "Z14".
    """
    arc_r = court_config.arc_radius
    dist = math.sqrt(x ** 2 + y ** 2)
    # Angle from positive y-axis; negative = left, positive = right
    angle = math.degrees(math.atan2(x, y))

    # 1. Restricted area
    if dist <= 4:
        return Zone.Z01.value

    # 2. Paint non-restricted (4–8 ft, within paint width ~8 ft each side)
    if dist <= 8 and abs(x) <= 8:
        return Zone.Z02.value

    # 3. Short range (4–14 ft)
    if dist <= 14:
        if angle < -60:
            return Zone.Z03.value
        if angle > 60:
            return Zone.Z04.value
        return Zone.Z07.value

    # 4. Mid-range (14 ft to arc_r)
    if dist < arc_r:
        if angle < -60:
            return Zone.Z08.value
        if angle > 60:
            return Zone.Z09.value
        if angle < -30:
            return Zone.Z05.value
        if angle > 30:
            return Zone.Z06.value
        return Zone.Z07.value

    # 5. Three-point territory (dist >= arc_r)
    if court_config.corner3_eliminated:
        # Corner zones are reclassified as wing zones
        if angle < -65:
            return Zone.Z12.value
        if angle > 65:
            return Zone.Z13.value
    else:
        if angle < -65:
            return Zone.Z10.value
        if angle > 65:
            return Zone.Z11.value

    if angle < -20:
        return Zone.Z12.value
    if angle > 20:
        return Zone.Z13.value
    return Zone.Z14.value


# ---------------------------------------------------------------------------
# derive_zone_boundaries — Task 3.3
# ---------------------------------------------------------------------------

def derive_zone_boundaries(court_config: CourtConfig) -> Dict[str, Any]:
    """
    Return the boundary parameters for each zone under the given court config.

    Useful for documentation, debugging, and visualization.

    Args:
        court_config: The court configuration to derive boundaries for.

    Returns:
        Dict mapping zone_id -> dict of boundary parameters.
    """
    arc_r = court_config.arc_radius
    half_w = court_config.baseline_width / 2
    c3_elim = court_config.corner3_eliminated

    boundaries: Dict[str, Any] = {
        "Z01": {
            "name": "restricted_area",
            "dist_min": 0, "dist_max": 4,
            "angle_min": -180, "angle_max": 180,
            "point_value": 2,
        },
        "Z02": {
            "name": "paint_non_restricted",
            "dist_min": 4, "dist_max": 8,
            "x_max": 8,  # within paint width
            "point_value": 2,
        },
        "Z03": {
            "name": "left_short_corner_floater",
            "dist_min": 4, "dist_max": 14,
            "angle_max": -60,
            "point_value": 2,
        },
        "Z04": {
            "name": "right_short_corner_floater",
            "dist_min": 4, "dist_max": 14,
            "angle_min": 60,
            "point_value": 2,
        },
        "Z05": {
            "name": "left_elbow_mid_range",
            "dist_min": 14, "dist_max": arc_r,
            "angle_min": -60, "angle_max": -30,
            "point_value": 2,
        },
        "Z06": {
            "name": "right_elbow_mid_range",
            "dist_min": 14, "dist_max": arc_r,
            "angle_min": 30, "angle_max": 60,
            "point_value": 2,
        },
        "Z07": {
            "name": "center_mid_range",
            "dist_min": 14, "dist_max": arc_r,
            "angle_min": -30, "angle_max": 30,
            "point_value": 2,
        },
        "Z08": {
            "name": "left_mid_range_baseline",
            "dist_min": 14, "dist_max": arc_r,
            "angle_max": -60,
            "point_value": 2,
        },
        "Z09": {
            "name": "right_mid_range_baseline",
            "dist_min": 14, "dist_max": arc_r,
            "angle_min": 60,
            "point_value": 2,
        },
        "Z10": {
            "name": "left_corner_3",
            "dist_min": arc_r,
            "angle_max": -65,
            "corner3_eliminated": c3_elim,
            "point_value": 2 if c3_elim else 3,
        },
        "Z11": {
            "name": "right_corner_3",
            "dist_min": arc_r,
            "angle_min": 65,
            "corner3_eliminated": c3_elim,
            "point_value": 2 if c3_elim else 3,
        },
        "Z12": {
            "name": "left_wing_3",
            "dist_min": arc_r,
            "angle_min": -65, "angle_max": -20,
            "point_value": 3,
        },
        "Z13": {
            "name": "right_wing_3",
            "dist_min": arc_r,
            "angle_min": 20, "angle_max": 65,
            "point_value": 3,
        },
        "Z14": {
            "name": "top_of_arc_3",
            "dist_min": arc_r,
            "angle_min": -20, "angle_max": 20,
            "point_value": 3,
        },
    }

    # Annotate with court-level metadata
    boundaries["_meta"] = {
        "arc_radius": arc_r,
        "baseline_width": court_config.baseline_width,
        "basket_to_baseline": court_config.basket_to_baseline,
        "half_width": half_w,
        "corner3_eliminated": c3_elim,
    }

    return boundaries
