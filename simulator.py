"""
Game Simulator for the NBA Court Re-Engineering Optimizer.

Simulates 100 games between the Warriors and Cavs for a given CourtConfig.
Uses NN-weighted KDE shot location sampling and GB make/miss prediction.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.court_geometry.court_geometry import CourtConfig, classify_zone, zone_point_value

logger = logging.getLogger(__name__)

# Maximum resampling attempts when a zone has 0 historical attempts
_MAX_RESAMPLE = 50


class Simulator:
    """
    Simulates NBA games under a given CourtConfig.

    Parameters
    ----------
    stat_calculator : StatCalculator
        Provides player_stats, config_zone_attempts, resolve_zone_pct().
    nn_model : NeuralNetworkModel
        Trained NN model for shot location weighting.
    gb_model : GradientBoostModel
        Trained GB model for make/miss prediction.
    shot_sampler : ShotLocationSampler
        Per-player KDE sampler.
    defensive_ratings : dict
        {team_id: defensive_rating}.
    warriors_team_id :
        Team identifier for the Warriors.
    cavs_team_id :
        Team identifier for the Cavs.
    n_games : int
        Number of games to simulate per config (default 100).
    kde_candidates : int
        Number of KDE candidates per shot attempt (default 100).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        stat_calculator,
        nn_model,
        gb_model,
        shot_sampler,
        defensive_ratings: dict,
        warriors_team_id,
        cavs_team_id,
        n_games: int = 100,
        kde_candidates: int = 100,
        seed: Optional[int] = None,
    ):
        self.stat_calculator = stat_calculator
        self.nn_model = nn_model
        self.gb_model = gb_model
        self.shot_sampler = shot_sampler
        self.defensive_ratings = defensive_ratings
        self.warriors_team_id = warriors_team_id
        self.cavs_team_id = cavs_team_id
        self.n_games = n_games
        self.kde_candidates = kde_candidates
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Task 10.1 — Shot attempt allocation
    # ------------------------------------------------------------------

    def _sample_attempts(self, player_id, court_config: CourtConfig) -> int:
        """
        Sample the number of shot attempts for a player in one game.

        Formula:
            player_attempts_per_game = usage_rate * estimated_minutes
                                       * team_possessions_per_minute

        Returns a Poisson-sampled integer around this expected value.
        """
        stats = self.stat_calculator.player_stats.get(player_id, {})
        usage_rate = stats.get("usage_rate", 0.0)
        est_minutes = stats.get("estimated_minutes", 0.0)

        # Determine team for possessions_per_minute
        player_df = self.stat_calculator.player_dfs.get(player_id)
        if player_df is not None and not player_df.empty:
            team_id = player_df["team_id"].mode().iloc[0]
        else:
            team_id = self.warriors_team_id

        ppm = self.stat_calculator.team_possessions_per_minute.get(team_id, 1.0)

        expected_attempts = usage_rate * est_minutes * ppm
        # Poisson sampling for integer count; floor at 0
        n = int(self.rng.poisson(max(expected_attempts, 0.0)))
        return n

    # ------------------------------------------------------------------
    # Task 10.2 — NN-weighted KDE shot location sampling
    # ------------------------------------------------------------------

    def _sample_shot_location(
        self, player_id, court_config: CourtConfig
    ) -> Tuple[float, float, str]:
        """
        Sample one (x, y, zone) for a shot attempt using NN-weighted KDE.

        Steps:
          1. Draw n_candidates from player's KDE.
          2. Weight by NN P(make | x, y, player).
          3. Weighted-sample one (x, y).
          4. Classify to zone under current config.
          5. Resample if zone has 0 historical attempts (up to _MAX_RESAMPLE times).

        Returns
        -------
        (x, y, zone_id)
        """
        config_key = (court_config.arc_radius, court_config.baseline_width)
        zone_attempts = (
            self.stat_calculator.config_zone_attempts
            .get(player_id, {})
            .get(config_key, {})
        )

        for _ in range(_MAX_RESAMPLE):
            x, y = self.shot_sampler.sample_weighted(
                player_id=player_id,
                nn_model=self.nn_model,
                n_candidates=self.kde_candidates,
                rng=self.rng,
            )
            zone = classify_zone(x, y, court_config)
            if zone_attempts.get(zone, 0) > 0:
                return x, y, zone

        # If we exhaust resamples, return the last sampled location regardless
        logger.debug(
            "[Simulator] Player %s: exhausted resample attempts for zone exclusion.", player_id
        )
        return x, y, zone

    # ------------------------------------------------------------------
    # Task 10.4 — Shot outcome simulation
    # ------------------------------------------------------------------

    def _simulate_shot(
        self,
        player_id,
        zone: str,
        court_config: CourtConfig,
        opp_team_id,
    ) -> Tuple[bool, int]:
        """
        Simulate a single shot outcome using the GB model.

        Returns
        -------
        (made: bool, points: int)
        """
        stats = self.stat_calculator.player_stats.get(player_id, {})
        config_key = (court_config.arc_radius, court_config.baseline_width)

        attempts_in_zone = (
            self.stat_calculator.config_zone_attempts
            .get(player_id, {})
            .get(config_key, {})
            .get(zone, 0)
        )

        resolved_pct = self.stat_calculator.resolve_zone_pct(player_id, zone, court_config)
        if resolved_pct is None:
            resolved_pct = 0.0

        opp_def = self.defensive_ratings.get(opp_team_id, 110.0)

        p_make = self.gb_model.predict_proba(
            player_id=player_id,
            zone=zone,
            court_config=court_config,
            opp_defensive_rating=opp_def,
            attempts_in_zone=attempts_in_zone,
            zone_pct=resolved_pct,
            overall_2pt_pct=stats.get("two_pt_pct", 0.0),
            overall_3pt_pct=stats.get("three_pt_pct", 0.0),
        )

        made = bool(self.rng.random() < p_make)
        points = zone_point_value(zone, court_config) if made else 0
        return made, points

    # ------------------------------------------------------------------
    # Task 10.5 — 100-game simulation loop
    # ------------------------------------------------------------------

    def run(self, court_config: CourtConfig) -> dict:
        """
        Simulate n_games games between Warriors and Cavs under the given config.

        Returns
        -------
        dict with keys:
            cavs_3pt_pct, warriors_3pt_pct, combined_3pt_pct,
            cavs_ppg, warriors_ppg,
            per_player: {player_id: {ppg, three_pt_pct, three_pa_per_game}},
            corner3_eliminated: bool
        """
        # Ensure zone stats are computed for this config
        self.stat_calculator.compute_zone_stats(court_config)

        warriors_players = self._get_team_players(self.warriors_team_id)
        cavs_players = self._get_team_players(self.cavs_team_id)

        # Accumulators: per-player totals across all games
        player_totals: Dict = defaultdict(lambda: {
            "points": 0, "attempts": 0,
            "three_pt_attempts": 0, "three_pt_makes": 0,
        })
        team_totals: Dict = {
            self.warriors_team_id: {"points": 0, "three_pt_attempts": 0, "three_pt_makes": 0},
            self.cavs_team_id:     {"points": 0, "three_pt_attempts": 0, "three_pt_makes": 0},
        }

        for _ in range(self.n_games):
            self._simulate_one_game(
                warriors_players, cavs_players,
                court_config, player_totals, team_totals,
            )

        return self._aggregate_results(
            player_totals, team_totals, court_config,
            warriors_players, cavs_players,
        )

    def _get_team_players(self, team_id) -> List:
        """Return list of player_ids for the given team."""
        players = []
        for pid in self.stat_calculator.eligible_players:
            pdf = self.stat_calculator.player_dfs.get(pid)
            if pdf is not None and not pdf.empty:
                if pdf["team_id"].mode().iloc[0] == team_id:
                    players.append(pid)
        return players

    def _simulate_one_game(
        self,
        warriors_players: List,
        cavs_players: List,
        court_config: CourtConfig,
        player_totals: Dict,
        team_totals: Dict,
    ) -> None:
        """Simulate one game and accumulate results into totals dicts."""
        # Warriors shoot against Cavs defense; Cavs shoot against Warriors defense
        matchups = [
            (warriors_players, self.warriors_team_id, self.cavs_team_id),
            (cavs_players, self.cavs_team_id, self.warriors_team_id),
        ]

        for players, team_id, opp_team_id in matchups:
            for pid in players:
                if pid not in self.shot_sampler.player_kdes:
                    continue  # no KDE fitted (< 2 shots)

                n_attempts = self._sample_attempts(pid, court_config)
                for _ in range(n_attempts):
                    try:
                        x, y, zone = self._sample_shot_location(pid, court_config)
                    except (KeyError, Exception) as exc:
                        logger.debug("[Simulator] Shot sampling failed for %s: %s", pid, exc)
                        continue

                    made, points = self._simulate_shot(pid, zone, court_config, opp_team_id)
                    pv = zone_point_value(zone, court_config)
                    is_3pt = pv == 3

                    player_totals[pid]["points"] += points
                    player_totals[pid]["attempts"] += 1
                    if is_3pt:
                        player_totals[pid]["three_pt_attempts"] += 1
                        if made:
                            player_totals[pid]["three_pt_makes"] += 1

                    team_totals[team_id]["points"] += points
                    if is_3pt:
                        team_totals[team_id]["three_pt_attempts"] += 1
                        if made:
                            team_totals[team_id]["three_pt_makes"] += 1

    def _aggregate_results(
        self,
        player_totals: Dict,
        team_totals: Dict,
        court_config: CourtConfig,
        warriors_players: List,
        cavs_players: List,
    ) -> dict:
        """Convert raw totals into per-game averages and 3PT%."""
        n = self.n_games

        def team_3pt_pct(team_id: str) -> float:
            t = team_totals[team_id]
            return (
                t["three_pt_makes"] / t["three_pt_attempts"]
                if t["three_pt_attempts"] > 0 else 0.0
            )

        def team_ppg(team_id: str) -> float:
            return team_totals[team_id]["points"] / n

        w_3pt = team_3pt_pct(self.warriors_team_id)
        c_3pt = team_3pt_pct(self.cavs_team_id)

        per_player = {}
        for pid in warriors_players + cavs_players:
            t = player_totals[pid]
            three_att = t["three_pt_attempts"]
            three_makes = t["three_pt_makes"]
            per_player[pid] = {
                "ppg": t["points"] / n,
                "three_pt_pct": three_makes / three_att if three_att > 0 else 0.0,
                "three_pa_per_game": three_att / n,
            }

        return {
            "cavs_3pt_pct": c_3pt,
            "warriors_3pt_pct": w_3pt,
            "combined_3pt_pct": (c_3pt + w_3pt) / 2,
            "cavs_ppg": team_ppg(self.cavs_team_id),
            "warriors_ppg": team_ppg(self.warriors_team_id),
            "per_player": per_player,
            "corner3_eliminated": court_config.corner3_eliminated,
        }
