"""
SprintGuard - GA Replanner

DEAP-based Genetic Algorithm that selects an optimal subset of sprint stories
to keep when sprint risk is HIGH.

Chromosome encoding: binary vector of len(stories), gene=1 means "keep in sprint",
gene=0 means "move to backlog".

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.sprint_risk import SprintRiskEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DEAP global creator registration (guarded against re-registration)
# ---------------------------------------------------------------------------

def _register_deap_types() -> None:
    """
    Register DEAP creator types exactly once.

    DEAP's creator module uses module-level globals, so importing this module
    multiple times (e.g., in tests) would raise an error if we try to re-create
    the same class names.  The `hasattr` guard prevents that.
    """
    from deap import base, creator  # noqa: PLC0415

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SprintStory:
    """
    Lightweight data class representing a single sprint story for GA optimization.

    Attributes:
        id:            Unique story identifier (e.g. "JIRA-440").
        story_points:  Story size in points (>= 0).
        priority:      Business priority 1–5 (5 = highest).
        effort_hours:  Estimated effort in hours.
        must_have:     When True, the gene for this story is always pinned to 1.
    """

    id: str
    story_points: int
    priority: int          # 1–5 (5 = highest)
    effort_hours: float
    must_have: bool = False


# ---------------------------------------------------------------------------
# GA Replanner
# ---------------------------------------------------------------------------

class GAReplanner:
    """
    DEAP-based Genetic Algorithm for sprint re-planning.

    The GA evolves a population of binary chromosomes, where each gene
    represents one sprint story.  The fitness function maximises
    Σ priority × story_points for all kept stories while enforcing a hard
    capacity constraint on total effort hours.

    Must-have stories are repaired after every genetic operator to ensure
    their gene values are never cleared to 0.

    Args:
        risk_engine: A built :class:`SprintRiskEngine` instance used to score
                     each candidate plan.
    """

    def __init__(self, risk_engine: "SprintRiskEngine") -> None:
        _register_deap_types()
        self._risk_engine = risk_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replan(
        self,
        stories: list[SprintStory],
        available_capacity_hours: float,
        sprint_state: dict,
        population_size: int = 50,
        generations: int = 100,
    ) -> list[dict]:
        """
        Run the GA and return sorted re-planning suggestions.

        Args:
            stories:                   List of sprint stories to optimise over.
            available_capacity_hours:  Total team capacity in hours.
            sprint_state:              Dict with at least ``velocity_trend`` and
                                       ``days_remaining`` keys (from SprintService).
            population_size:           Number of chromosomes in the population.
            generations:               Number of GA generations to evolve.

        Returns:
            List of suggestion dicts sorted ascending by ``projected_risk_score``.
            Contains at least 2 entries (or all unique solutions if fewer exist).
            Each dict has keys:
                id, action, story_points_removed,
                projected_risk, projected_risk_score, [note]
        """
        from deap import algorithms, base, creator, tools  # noqa: PLC0415

        if not stories:
            return []

        n = len(stories)
        must_have_indices = {i for i, s in enumerate(stories) if s.must_have}

        # ---- Fitness function ----------------------------------------

        def evaluate(chromosome: list[int]) -> tuple[float]:
            """
            Return (fitness,) for a chromosome.

            Returns (-inf,) if the total effort hours of kept stories exceeds
            available_capacity_hours, otherwise returns (Σ priority * story_points,).
            """
            total_effort = sum(
                stories[i].effort_hours
                for i in range(n)
                if chromosome[i] == 1
            )
            if total_effort > available_capacity_hours:
                return (-math.inf,)

            fitness = sum(
                stories[i].priority * stories[i].story_points
                for i in range(n)
                if chromosome[i] == 1
            )
            return (float(fitness),)

        # ---- Repair function -----------------------------------------

        def repair(chromosome: list[int]) -> list[int]:
            """Pin must-have genes to 1 after crossover/mutation."""
            for idx in must_have_indices:
                chromosome[idx] = 1
            return chromosome

        # ---- Build toolbox -------------------------------------------

        toolbox = base.Toolbox()

        # Individual generator: random binary vector, then repair
        def make_individual() -> creator.Individual:
            chrom = creator.Individual(
                random.randint(0, 1) for _ in range(n)
            )
            repair(chrom)
            return chrom

        toolbox.register("individual", make_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate",   evaluate)
        toolbox.register("select",     tools.selTournament, tournsize=3)
        toolbox.register("mate",       tools.cxTwoPoint)
        toolbox.register("mutate",     tools.mutFlipBit, indpb=0.05)

        # ---- Evolve --------------------------------------------------

        population = toolbox.population(n=population_size)

        # Evaluate initial population
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        for _gen in range(generations):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8:
                    toolbox.mate(child1, child2)
                    repair(child1)
                    repair(child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < 0.05 * n:
                    toolbox.mutate(mutant)
                    repair(mutant)
                    del mutant.fitness.values

            # Evaluate any individual whose fitness was invalidated
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            population[:] = offspring

        # ---- Extract suggestions -------------------------------------

        return self._build_suggestions(
            population=population,
            stories=stories,
            available_capacity_hours=available_capacity_hours,
            sprint_state=sprint_state,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_suggestions(
        self,
        population: list,
        stories: list[SprintStory],
        available_capacity_hours: float,
        sprint_state: dict,
    ) -> list[dict]:
        """
        Convert the final population into de-duplicated, ranked suggestion dicts.

        Steps:
        1. Collect unique, feasible chromosomes (sorted by fitness descending).
        2. For each unique chromosome compute projected risk via SprintRiskEngine.
        3. Sort suggestions by projected_risk_score ascending.
        4. Return at least 2 suggestions (or all available if fewer exist).
           If no suggestion reduces risk < 0.70, return best available + note.
        """
        # Sort by fitness (descending), filter infeasible
        feasible = sorted(
            [ind for ind in population if math.isfinite(ind.fitness.values[0])],
            key=lambda ind: ind.fitness.values[0],
            reverse=True,
        )

        # De-duplicate by chromosome content
        seen: set[tuple[int, ...]] = set()
        unique: list = []
        for ind in feasible:
            key = tuple(ind)
            if key not in seen:
                seen.add(key)
                unique.append(ind)

        if not unique:
            logger.warning("GAReplanner: no feasible chromosomes found after evolution.")
            unique = [list(1 for _ in stories)]  # fallback: keep all stories

        # Build suggestion objects
        velocity_trend  = float(sprint_state.get("velocity_trend", 0.0))
        days_remaining  = float(sprint_state.get("days_remaining", 7.0))

        suggestions: list[dict] = []
        for idx, chrom in enumerate(unique[:max(10, len(unique))]):
            kept_stories    = [stories[i] for i in range(len(stories)) if chrom[i] == 1]
            removed_stories = [stories[i] for i in range(len(stories)) if chrom[i] == 0]

            kept_hours   = sum(s.effort_hours for s in kept_stories)
            kept_points  = sum(s.story_points for s in removed_stories)  # removed points

            # action string
            if removed_stories:
                ids_str = ", ".join(s.id for s in removed_stories)
                pts_str = sum(s.story_points for s in removed_stories)
                action  = f"Remove stories: {ids_str} ({pts_str} story points)"
            else:
                action = "Keep all stories (no changes)"

            # Compute projected risk: kept_hours represent the new bug burden
            try:
                risk_result = self._risk_engine.compute(
                    bug_hours_added=kept_hours,
                    velocity_trend=velocity_trend,
                    days_remaining=days_remaining,
                )
                projected_risk_score = risk_result["risk_score"]
                projected_risk       = risk_result["risk_level"]
            except Exception as exc:
                logger.error("GAReplanner: risk computation failed for suggestion %d: %s", idx, exc)
                projected_risk_score = 1.0
                projected_risk       = "HIGH"

            suggestion: dict = {
                "id":                    str(uuid.uuid4()),
                "action":                action,
                "story_points_removed":  kept_points,
                "projected_risk":        projected_risk,
                "projected_risk_score":  projected_risk_score,
            }
            suggestions.append(suggestion)

        # Sort ascending by risk score
        suggestions.sort(key=lambda s: s["projected_risk_score"])

        # Ensure at least 2 suggestions
        if len(suggestions) < 2 and len(unique) < 2:
            # Duplicate the best suggestion as a second entry with a note
            if suggestions:
                duplicate = dict(suggestions[0])
                duplicate["id"]   = str(uuid.uuid4())
                duplicate["note"] = "Only one distinct solution found; showing duplicate."
                suggestions.append(duplicate)
            else:
                suggestions.append({
                    "id":                   str(uuid.uuid4()),
                    "action":               "Keep all stories (no feasible alternative found)",
                    "story_points_removed": 0,
                    "projected_risk":       "HIGH",
                    "projected_risk_score": 1.0,
                    "note":                 "No feasible alternative found.",
                })

        # If no suggestion reduces risk below 0.70, annotate with note
        if all(s["projected_risk_score"] >= 0.70 for s in suggestions):
            for s in suggestions:
                s.setdefault(
                    "note",
                    "No re-planning option reduces sprint risk below HIGH threshold",
                )

        return suggestions
