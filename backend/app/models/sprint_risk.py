"""
SprintGuard - Sprint Risk Engine

Mamdani Fuzzy Inference System implemented with scikit-fuzzy (skfuzzy).
Computes a sprint risk score in [0, 1] from three inputs:
  - bug_hours_added   : hours added to sprint by the new bug  [0, 25]
  - velocity_trend    : sprint velocity delta                 [-10, 10]
  - days_remaining    : days left in the sprint               [0, 15]

Output:
  sprint_risk : continuous risk score in [0, 1], mapped to LOW / MEDIUM / HIGH
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SprintRiskEngine:
    """
    Mamdani FIS via scikit-fuzzy.

    Usage::

        engine = SprintRiskEngine()
        engine.build()
        result = engine.compute(bug_hours_added=10.0, velocity_trend=-3.0, days_remaining=2.0)
        # {"risk_score": 0.82, "risk_level": "HIGH", "factors": [...]}
    """

    def __init__(self) -> None:
        self._control_system = None
        self._simulation = None

        # Store antecedents/consequent as instance attributes (optional, for introspection)
        self._bug_hours_added = None
        self._velocity_trend = None
        self._days_remaining = None
        self._sprint_risk = None

    # ------------------------------------------------------------------
    # build()
    # ------------------------------------------------------------------

    def build(self) -> None:
        """
        Define universe of discourse, membership functions, rules, and
        build the ControlSystem and ControlSystemSimulation.
        """
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        # ----------------------------------------------------------------
        # Antecedents (inputs)
        # ----------------------------------------------------------------

        bug_hours_added = ctrl.Antecedent(np.arange(0, 25.1, 0.1), "bug_hours_added")
        bug_hours_added["Low"]    = fuzz.trimf(bug_hours_added.universe, [0, 2, 4])
        bug_hours_added["Medium"] = fuzz.trimf(bug_hours_added.universe, [3, 6, 9])
        bug_hours_added["High"]   = fuzz.trimf(bug_hours_added.universe, [8, 15, 20])

        velocity_trend = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), "velocity_trend")
        velocity_trend["Declining"]  = fuzz.trimf(velocity_trend.universe, [-10, -5, 0])
        velocity_trend["Stable"]     = fuzz.trimf(velocity_trend.universe, [-2, 0, 2])
        velocity_trend["Increasing"] = fuzz.trimf(velocity_trend.universe, [0, 5, 10])

        days_remaining = ctrl.Antecedent(np.arange(0, 15.1, 0.1), "days_remaining")
        days_remaining["Critical"]    = fuzz.trimf(days_remaining.universe, [0, 1, 3])
        days_remaining["Tight"]       = fuzz.trimf(days_remaining.universe, [2, 4, 6])
        days_remaining["Comfortable"] = fuzz.trimf(days_remaining.universe, [5, 8, 14])

        # ----------------------------------------------------------------
        # Consequent (output)
        # ----------------------------------------------------------------

        sprint_risk = ctrl.Consequent(np.arange(0, 1.01, 0.01), "sprint_risk")
        sprint_risk["Low"]    = fuzz.trimf(sprint_risk.universe, [0, 0.2, 0.4])
        sprint_risk["Medium"] = fuzz.trimf(sprint_risk.universe, [0.3, 0.5, 0.7])
        sprint_risk["High"]   = fuzz.trimf(sprint_risk.universe, [0.6, 0.8, 1.0])

        # ----------------------------------------------------------------
        # Rules (15 total)
        # ----------------------------------------------------------------

        # --- Three required rules ---
        # Rule 1: worst case → High
        rule1 = ctrl.Rule(
            bug_hours_added["High"] & velocity_trend["Declining"] & days_remaining["Critical"],
            sprint_risk["High"],
        )
        # Rule 2: best case → Low
        rule2 = ctrl.Rule(
            bug_hours_added["Low"] & velocity_trend["Increasing"] & days_remaining["Comfortable"],
            sprint_risk["Low"],
        )
        # Rule 3: moderate case → Medium
        rule3 = ctrl.Rule(
            bug_hours_added["Medium"] & velocity_trend["Stable"] & days_remaining["Tight"],
            sprint_risk["Medium"],
        )

        # --- 12 additional rules to cover remaining combinations ---

        # High bug hours with any declining/tight conditions → High
        rule4 = ctrl.Rule(
            bug_hours_added["High"] & velocity_trend["Declining"] & days_remaining["Tight"],
            sprint_risk["High"],
        )
        rule5 = ctrl.Rule(
            bug_hours_added["High"] & velocity_trend["Stable"] & days_remaining["Critical"],
            sprint_risk["High"],
        )
        rule6 = ctrl.Rule(
            bug_hours_added["High"] & velocity_trend["Declining"] & days_remaining["Comfortable"],
            sprint_risk["High"],
        )

        # High bug hours + positive conditions → Medium (not low — too many hours)
        rule7 = ctrl.Rule(
            bug_hours_added["High"] & velocity_trend["Increasing"] & days_remaining["Comfortable"],
            sprint_risk["Medium"],
        )
        rule8 = ctrl.Rule(
            bug_hours_added["High"] & velocity_trend["Stable"] & days_remaining["Tight"],
            sprint_risk["High"],
        )
        rule9 = ctrl.Rule(
            bug_hours_added["High"] & velocity_trend["Increasing"] & days_remaining["Tight"],
            sprint_risk["Medium"],
        )

        # Low bug hours with bad conditions → Medium
        rule10 = ctrl.Rule(
            bug_hours_added["Low"] & velocity_trend["Declining"] & days_remaining["Critical"],
            sprint_risk["Medium"],
        )
        rule11 = ctrl.Rule(
            bug_hours_added["Low"] & velocity_trend["Declining"] & days_remaining["Tight"],
            sprint_risk["Medium"],
        )
        rule12 = ctrl.Rule(
            bug_hours_added["Low"] & velocity_trend["Stable"] & days_remaining["Critical"],
            sprint_risk["Medium"],
        )

        # Low bug hours + comfortable/stable → Low
        rule13 = ctrl.Rule(
            bug_hours_added["Low"] & velocity_trend["Stable"] & days_remaining["Comfortable"],
            sprint_risk["Low"],
        )
        rule14 = ctrl.Rule(
            bug_hours_added["Low"] & velocity_trend["Increasing"] & days_remaining["Tight"],
            sprint_risk["Low"],
        )

        # Medium bug hours + bad conditions → High
        rule15 = ctrl.Rule(
            bug_hours_added["Medium"] & velocity_trend["Declining"] & days_remaining["Critical"],
            sprint_risk["High"],
        )

        # ----------------------------------------------------------------
        # Build control system
        # ----------------------------------------------------------------

        all_rules = [
            rule1, rule2, rule3, rule4, rule5,
            rule6, rule7, rule8, rule9, rule10,
            rule11, rule12, rule13, rule14, rule15,
        ]

        control_system = ctrl.ControlSystem(all_rules)
        simulation = ctrl.ControlSystemSimulation(control_system)

        # Store on instance
        self._control_system = control_system
        self._simulation = simulation
        self._bug_hours_added = bug_hours_added
        self._velocity_trend = velocity_trend
        self._days_remaining = days_remaining
        self._sprint_risk = sprint_risk

        logger.info("SprintRiskEngine built successfully with %d rules.", len(all_rules))

    # ------------------------------------------------------------------
    # compute()
    # ------------------------------------------------------------------

    def compute(
        self,
        bug_hours_added: float,
        velocity_trend: float,
        days_remaining: float,
    ) -> dict:
        """
        Run Mamdani defuzzification for the given sprint inputs.

        Inputs are clamped to their respective universe ranges before being
        passed to the simulation to avoid skfuzzy ValueError on out-of-range
        values.

        Returns:
            {
                "risk_score":  float in [0, 1],
                "risk_level":  "LOW" | "MEDIUM" | "HIGH",
                "factors":     list[str]  (human-readable explanations)
            }
        """
        if self._simulation is None:
            logger.warning("SprintRiskEngine.compute() called before build().")
            return {
                "risk_score": 0.5,
                "risk_level": "MEDIUM",
                "factors": ["Risk computation unavailable"],
            }

        # Clamp inputs to universe ranges
        bha = float(np.clip(bug_hours_added, 0.0, 25.0))
        vt  = float(np.clip(velocity_trend,  -10.0, 10.0))
        dr  = float(np.clip(days_remaining,   0.0, 15.0))

        try:
            self._simulation.input["bug_hours_added"] = bha
            self._simulation.input["velocity_trend"]  = vt
            self._simulation.input["days_remaining"]  = dr
            self._simulation.compute()
            risk_score = float(self._simulation.output["sprint_risk"])
        except Exception as exc:  # noqa: BLE001
            logger.error("SprintRiskEngine defuzzification failed: %s", exc)
            return {
                "risk_score": 0.5,
                "risk_level": "MEDIUM",
                "factors": ["Risk computation unavailable"],
            }

        # Determine categorical risk level
        risk_level = self._score_to_level(risk_score)

        # Build human-readable factor list
        factors: list[str] = []
        if bha > 8:
            factors.append(f"High bug hours added ({bha:.1f}h)")
        if vt < -2:
            factors.append(f"Declining velocity ({vt:.1f})")
        if dr < 3:
            factors.append(f"Critical time remaining ({dr:.0f} days)")
        if not factors:
            factors.append("No significant risk factors detected")

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "factors": factors,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_level(score: float) -> str:
        """Map a continuous risk score to a categorical label."""
        if score < 0.4:
            return "LOW"
        elif score < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True if the FIS has been built and is ready for inference."""
        return self._simulation is not None
