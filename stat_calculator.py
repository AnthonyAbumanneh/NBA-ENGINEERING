"""
Stat_Calculator: computes per-player and per-team statistics from the
warriors_cavs_2014 dataset and team defensive ratings from the NN_Dataset.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import ZONE_ATTEMPT_THRESHOLD, KNOWN_PLAYOFF_MINUTES
from src.court_geometry.court_geometry import CourtConfig, classify_zone, zone_point_value

logger = logging.getLogger(__name__)


class StatCalculator:
    """
    Computes eligible-player filtering, per-player base stats, per-team stats,
    and team defensive ratings.

    Parameters
    ----------
    primary_df : pd.DataFrame
        The warriors_cavs_2014 dataset loaded by DataLoader.
    """

    def __init__(self, primary_df: pd.DataFrame):
        self.primary_df = primary_df

        # Populated by _filter_eligible_players (called in __init__)
        self.warriors_team_id: Optional[str | int] = None
        self.cavs_team_id: Optional[str | int] = None
        self.eligible_players: list = []          # list of player_ids
        self.player_dfs: dict = {}                # {player_id: filtered DataFrame}

        # Populated by compute_player_stats()
        self.player_stats: dict = {}              # {player_id: stats dict}

        # Populated by compute_team_stats()
        self.team_stats: dict = {}                # {team_id: stats dict}
        self.team_possessions_per_minute: dict = {}  # {team_id: float}

        # Populated by compute_defensive_ratings()
        self.defensive_ratings: dict = {}         # {team_id: float}

        # Populated by compute_baseline_zone_stats()
        self.baseline_zone_attempts: dict = {}    # {player_id: {zone_label: int}}
        self.baseline_zone_pct: dict = {}         # {player_id: {zone_label: float}}

        # Populated by compute_zone_stats()
        self.config_zone_attempts: dict = {}      # {player_id: {config_key: {zone: int}}}
        self.config_zone_pct: dict = {}           # {player_id: {config_key: {zone: float}}}

        # Run filtering immediately on construction
        self._filter_eligible_players()

    # ------------------------------------------------------------------
    # Task 2.1 — Eligible player filtering
    # ------------------------------------------------------------------

    def _filter_eligible_players(self) -> None:
        """
        Identify Warriors and Cavs rosters, then build per-player filtered
        DataFrames that include only Warriors/Cavs rows for each player.

        Uses 'team_abbrev' column (from TEAM in raw CSV) when available,
        which only contains 'Warriors' / 'Cavaliers'. Falls back to
        auto-discovering the two most common team_ids.
        """
        df = self.primary_df

        # Prefer the 'team_abbrev' column (TEAM in raw CSV) — it only contains
        # the two target teams and is the most reliable filter.
        if "team_abbrev" in df.columns:
            abbrevs = df["team_abbrev"].dropna().unique().tolist()
            warriors_abbrev = next(
                (a for a in abbrevs if "warrior" in str(a).lower() or "thunder" in str(a).lower() or "oklahoma" in str(a).lower()),
                None,
            )
            cavs_abbrev = next(
                (a for a in abbrevs if "cavalier" in str(a).lower() or "cav" in str(a).lower() or "pacer" in str(a).lower() or "indiana" in str(a).lower()),
                None,
            )

            if warriors_abbrev and cavs_abbrev:
                warriors_rows = df[df["team_abbrev"] == warriors_abbrev]
                cavs_rows = df[df["team_abbrev"] == cavs_abbrev]

                # Map abbrev → team_id (use most common team_id for that abbrev)
                self.warriors_team_id = warriors_rows["team_id"].mode().iloc[0]
                self.cavs_team_id = cavs_rows["team_id"].mode().iloc[0]

                filtered = pd.concat([warriors_rows, cavs_rows]).copy()
                self.eligible_players = sorted(
                    filtered["player_id"].unique().tolist(), key=str
                )
                for pid in self.eligible_players:
                    pdf = filtered[filtered["player_id"] == pid].copy()
                    # Keep only rows where the player's team_id matches Warriors or Cavs
                    # (some players like LeBron have rows from other teams in the dataset)
                    eligible_tids = {self.warriors_team_id, self.cavs_team_id}
                    pdf_clean = pdf[pdf["team_id"].isin(eligible_tids)]
                    # Use cleaned version if it has rows, otherwise keep all
                    self.player_dfs[pid] = (
                        pdf_clean.reset_index(drop=True)
                        if not pdf_clean.empty
                        else pdf.reset_index(drop=True)
                    )

                logger.info(
                    "[StatCalculator] Identified %d eligible players "
                    "(Warriors=%s, Cavs=%s) via team_abbrev column.",
                    len(self.eligible_players), warriors_abbrev, cavs_abbrev,
                )
                return

        # Fallback: auto-discover the two team_ids present in the dataset
        team_ids = df["team_id"].unique().tolist()
        sorted_teams = sorted(team_ids, key=lambda t: str(t))
        if len(sorted_teams) >= 2:
            self.warriors_team_id = sorted_teams[0]
            self.cavs_team_id = sorted_teams[1]
        elif len(sorted_teams) == 1:
            self.warriors_team_id = sorted_teams[0]
            self.cavs_team_id = sorted_teams[0]
        else:
            raise ValueError("[StatCalculator] No team_id values found in primary dataset.")

        eligible_team_ids = set(sorted_teams[:2])
        filtered = df[df["team_id"].isin(eligible_team_ids)].copy()
        self.eligible_players = sorted(filtered["player_id"].unique().tolist(), key=str)
        for pid in self.eligible_players:
            self.player_dfs[pid] = filtered[
                filtered["player_id"] == pid
            ].reset_index(drop=True)

        logger.info(
            "[StatCalculator] Identified %d eligible players across teams %s.",
            len(self.eligible_players), eligible_team_ids,
        )

    # ------------------------------------------------------------------
    # Task 2.2 — Per-player base stat computation
    # ------------------------------------------------------------------

    def compute_player_stats(self) -> dict:
        """
        Compute per-player base statistics for all eligible players.

        Returns
        -------
        dict
            {player_id: {ppg, fg_pct, three_pt_pct, two_pt_pct,
                         total_attempts, three_pt_makes, three_pt_attempts,
                         two_pt_makes, two_pt_attempts,
                         usage_rate, estimated_minutes}}
        """
        self.player_stats = {}

        for pid in self.eligible_players:
            pdf = self.player_dfs[pid]
            stats = self._compute_single_player_stats(pid, pdf)
            self.player_stats[pid] = stats

        logger.info(
            "[StatCalculator] Computed base stats for %d players.", len(self.player_stats)
        )
        return self.player_stats

    def _detect_shot_type_column(self, df: pd.DataFrame) -> Optional[str]:
        """Return the column name that distinguishes 2PT vs 3PT shots, or None."""
        candidates = ["shot_type", "shot_zone_basic", "shot_zone_range", "action_type"]
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _classify_3pt(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a boolean Series: True where the shot is a 3-point attempt.

        Strategy (in priority order):
        1. 'shot_type' column with '3PT Field Goal' / '2PT Field Goal' values
        2. 'shot_zone_basic' column containing '3' or 'Above the Break 3' etc.
        3. Euclidean distance from basket (x, y) — shots beyond 23.75 ft are 3PT
        """
        if "shot_type" in df.columns:
            return df["shot_type"].str.contains("3PT", case=False, na=False)

        if "shot_zone_basic" in df.columns:
            three_keywords = ["3", "above the break", "corner 3", "backcourt"]
            pattern = "|".join(three_keywords)
            return df["shot_zone_basic"].str.contains(pattern, case=False, na=False)

        # Fallback: use distance from basket
        if "x" in df.columns and "y" in df.columns:
            dist = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
            return dist > 23.75

        # Last resort: all shots treated as 2PT
        logger.warning(
            "[StatCalculator] Cannot determine shot type; treating all shots as 2PT."
        )
        return pd.Series(False, index=df.index)

    def _compute_single_player_stats(self, pid, pdf: pd.DataFrame) -> dict:
        """Compute stats for a single player from their filtered DataFrame."""
        if pdf.empty:
            return self._empty_player_stats()

        is_3pt = self._classify_3pt(pdf)
        made = pdf["shot_made_flag"].astype(int)

        total_attempts = len(pdf)
        total_makes = made.sum()

        three_pt_mask = is_3pt
        two_pt_mask = ~is_3pt

        three_pt_attempts = int(three_pt_mask.sum())
        three_pt_makes = int((made[three_pt_mask]).sum())

        two_pt_attempts = int(two_pt_mask.sum())
        two_pt_makes = int((made[two_pt_mask]).sum())

        fg_pct = total_makes / total_attempts if total_attempts > 0 else 0.0
        three_pt_pct = three_pt_makes / three_pt_attempts if three_pt_attempts > 0 else 0.0
        two_pt_pct = two_pt_makes / two_pt_attempts if two_pt_attempts > 0 else 0.0

        # Points: 3PT made = 3 pts, 2PT made = 2 pts
        total_points = three_pt_makes * 3 + two_pt_makes * 2

        # PPG — group by game identifier if available
        ppg = self._compute_ppg(pdf, three_pt_makes, two_pt_makes)

        # Estimated minutes
        estimated_minutes = self._compute_estimated_minutes(pdf)

        # Usage rate
        usage_rate = self._compute_usage_rate(pid, pdf)

        return {
            "ppg": ppg,
            "fg_pct": fg_pct,
            "three_pt_pct": three_pt_pct,
            "two_pt_pct": two_pt_pct,
            "total_attempts": total_attempts,
            "total_makes": int(total_makes),
            "total_points": total_points,
            "three_pt_makes": three_pt_makes,
            "three_pt_attempts": three_pt_attempts,
            "two_pt_makes": two_pt_makes,
            "two_pt_attempts": two_pt_attempts,
            "usage_rate": usage_rate,
            "estimated_minutes": estimated_minutes,
        }

    def _compute_ppg(self, pdf: pd.DataFrame, three_pt_makes: int, two_pt_makes: int) -> float:
        """Compute points per game. Uses game_id column if available."""
        total_points = three_pt_makes * 3 + two_pt_makes * 2

        game_col = self._find_game_col(pdf)
        if game_col:
            n_games = pdf[game_col].nunique()
            return total_points / n_games if n_games > 0 else 0.0

        # No game column — return total points (single-game assumption)
        return float(total_points)

    def _find_game_col(self, df: pd.DataFrame) -> Optional[str]:
        """Return the game identifier column name if present."""
        for col in ["game_id", "game_date", "match_id"]:
            if col in df.columns:
                return col
        return None

    def _compute_estimated_minutes(self, pdf: pd.DataFrame) -> float:
        """
        Return average playoff minutes per game for this player.

        Priority:
        1. KNOWN_PLAYOFF_MINUTES lookup by player name (hardcoded from StatMuse/BBRef)
        2. 'minutes_played' / 'min' / 'minutes' column in the DataFrame
        3. Fallback: 24.0 minutes
        """
        # Priority 1: known lookup by player name
        name_col = next((c for c in ["player_name", "player", "name"] if c in pdf.columns), None)
        if name_col and not pdf.empty:
            player_name = str(pdf[name_col].iloc[0]).strip()
            if player_name in KNOWN_PLAYOFF_MINUTES:
                return KNOWN_PLAYOFF_MINUTES[player_name]

        # Priority 2: explicit minutes column in data
        for col in ["minutes_played", "min", "minutes"]:
            if col in pdf.columns:
                game_col = self._find_game_col(pdf)
                if game_col:
                    per_game = pdf.groupby(game_col)[col].max()
                    return float(per_game.mean())
                return float(pdf[col].mean())

        # Priority 3: fallback
        return 24.0

    def _compute_usage_rate(self, pid, pdf: pd.DataFrame) -> float:
        """
        Compute Usage_Rate = player's share of team possessions while on floor.

        Formula (full):
          (FGA + 0.44 * FTA + TOV) / (team_FGA + 0.44 * team_FTA + team_TOV)
          while player is on floor.

        Approximation when TOV/FTA not available:
          FGA / team_FGA while player is on floor.
        """
        if pdf.empty:
            return 0.0

        # Determine which team this player belongs to (use most frequent team_id)
        player_team = pdf["team_id"].mode().iloc[0]
        team_df = self.primary_df[self.primary_df["team_id"] == player_team]

        fga_player = len(pdf)

        has_fta = "fta" in pdf.columns and "fta" in team_df.columns
        has_tov = "tov" in pdf.columns and "tov" in team_df.columns

        if has_fta and has_tov:
            fta_player = pdf["fta"].sum()
            tov_player = pdf["tov"].sum()
            numerator = fga_player + 0.44 * fta_player + tov_player

            fga_team = len(team_df)
            fta_team = team_df["fta"].sum()
            tov_team = team_df["tov"].sum()
            denominator = fga_team + 0.44 * fta_team + tov_team
        else:
            # Approximation: FGA / team_FGA
            numerator = fga_player
            denominator = len(team_df)

        return float(numerator / denominator) if denominator > 0 else 0.0

    @staticmethod
    def _empty_player_stats() -> dict:
        return {
            "ppg": 0.0,
            "fg_pct": 0.0,
            "three_pt_pct": 0.0,
            "two_pt_pct": 0.0,
            "total_attempts": 0,
            "total_makes": 0,
            "total_points": 0,
            "three_pt_makes": 0,
            "three_pt_attempts": 0,
            "two_pt_makes": 0,
            "two_pt_attempts": 0,
            "usage_rate": 0.0,
            "estimated_minutes": 0.0,
        }

    # ------------------------------------------------------------------
    # Task 2.3 — Per-team stat computation and defensive ratings
    # ------------------------------------------------------------------

    def compute_team_stats(self) -> dict:
        """
        Compute per-team statistics for Warriors and Cavs.

        Returns
        -------
        dict
            {team_id: {ppg, fg_pct, three_pt_pct, total_attempts,
                       total_3pt_attempts, team_possessions_per_minute}}
        """
        self.team_stats = {}
        self.team_possessions_per_minute = {}

        eligible_team_ids = [self.warriors_team_id, self.cavs_team_id]
        # Deduplicate in case both are the same (edge case)
        eligible_team_ids = list(dict.fromkeys(eligible_team_ids))

        df = self.primary_df

        for team_id in eligible_team_ids:
            team_df = df[df["team_id"] == team_id].copy()
            if team_df.empty:
                logger.warning("[StatCalculator] No rows found for team_id=%s", team_id)
                self.team_stats[team_id] = self._empty_team_stats()
                self.team_possessions_per_minute[team_id] = 0.0
                continue

            is_3pt = self._classify_3pt(team_df)
            made = team_df["shot_made_flag"].astype(int)

            total_attempts = len(team_df)
            total_makes = int(made.sum())

            three_pt_attempts = int(is_3pt.sum())
            three_pt_makes = int(made[is_3pt].sum())
            two_pt_makes = int(made[~is_3pt].sum())

            fg_pct = total_makes / total_attempts if total_attempts > 0 else 0.0
            three_pt_pct = three_pt_makes / three_pt_attempts if three_pt_attempts > 0 else 0.0

            total_points = three_pt_makes * 3 + two_pt_makes * 2

            # PPG
            game_col = self._find_game_col(team_df)
            if game_col:
                n_games = team_df[game_col].nunique()
                ppg = total_points / n_games if n_games > 0 else float(total_points)
            else:
                ppg = float(total_points)

            # team_possessions_per_minute from historical pace
            ppm = self._compute_possessions_per_minute(team_df, game_col)
            self.team_possessions_per_minute[team_id] = ppm

            self.team_stats[team_id] = {
                "ppg": ppg,
                "fg_pct": fg_pct,
                "three_pt_pct": three_pt_pct,
                "total_attempts": total_attempts,
                "total_3pt_attempts": three_pt_attempts,
                "team_possessions_per_minute": ppm,
            }

        logger.info(
            "[StatCalculator] Computed team stats for teams: %s", list(self.team_stats.keys())
        )
        return self.team_stats

    def _compute_possessions_per_minute(
        self, team_df: pd.DataFrame, game_col: Optional[str]
    ) -> float:
        """
        Derive team_possessions_per_minute from historical pace.

        The shot chart dataset only contains field goal attempts — it excludes
        free throw trips, and-ones, and other possessions. Real NBA playoff teams
        average ~88 FGA per game; this dataset shows ~60-63. We apply a scale
        factor so simulated shot totals match real NBA pace.
        """
        # Target: ~104 FGA per game (real NBA playoff pace)
        TARGET_FGA_PER_GAME = 104.0

        fga = len(team_df)

        if game_col:
            n_games = team_df[game_col].nunique()
        else:
            n_games = 1

        # Check for explicit minutes column
        for col in ["minutes_played", "min", "minutes"]:
            if col in team_df.columns:
                if game_col:
                    total_minutes = team_df.groupby(game_col)[col].max().sum()
                else:
                    total_minutes = team_df[col].sum()
                if total_minutes > 0:
                    raw_ppm = fga / total_minutes
                    # Scale to match real pace
                    actual_fga_per_game = raw_ppm * 48.0
                    scale = TARGET_FGA_PER_GAME / actual_fga_per_game if actual_fga_per_game > 0 else 1.0
                    return raw_ppm * scale

        # Fallback: use target pace directly
        total_minutes = n_games * 48.0
        raw_ppm = fga / total_minutes if total_minutes > 0 else 0.0
        actual_fga_per_game = raw_ppm * 48.0
        scale = TARGET_FGA_PER_GAME / actual_fga_per_game if actual_fga_per_game > 0 else 1.0
        return raw_ppm * scale

    @staticmethod
    def _empty_team_stats() -> dict:
        return {
            "ppg": 0.0,
            "fg_pct": 0.0,
            "three_pt_pct": 0.0,
            "total_attempts": 0,
            "total_3pt_attempts": 0,
            "team_possessions_per_minute": 0.0,
        }

    def compute_defensive_ratings(self, secondary_df: pd.DataFrame) -> dict:
        """
        Read team defensive ratings from the NN_Dataset (secondary DataFrame).

        Parameters
        ----------
        secondary_df : pd.DataFrame
            The nn_dataset loaded by DataLoader. Must have 'team_name' and
            'defensive_rating' columns (team_name already stripped of year suffix).

        Returns
        -------
        dict
            {team_id: defensive_rating}
        """
        self.defensive_ratings = {}

        team_col = "team_name" if "team_name" in secondary_df.columns else "team_id"
        if team_col not in secondary_df.columns or "defensive_rating" not in secondary_df.columns:
            logger.warning(
                "[StatCalculator] secondary_df missing team or 'defensive_rating' columns. "
                "Defensive ratings will be empty."
            )
            return self.defensive_ratings

        # Build a name→team_id lookup using the known team IDs and name keywords
        name_to_id: dict = {}
        if self.warriors_team_id is not None:
            name_to_id["warriors"] = self.warriors_team_id
            name_to_id["warrior"] = self.warriors_team_id
            name_to_id["gsw"] = self.warriors_team_id
            name_to_id["golden state"] = self.warriors_team_id
            name_to_id["thunder"] = self.warriors_team_id
            name_to_id["oklahoma"] = self.warriors_team_id
            name_to_id["okc"] = self.warriors_team_id
        if self.cavs_team_id is not None:
            name_to_id["cavaliers"] = self.cavs_team_id
            name_to_id["cavalier"] = self.cavs_team_id
            name_to_id["cavs"] = self.cavs_team_id
            name_to_id["cle"] = self.cavs_team_id
            name_to_id["cleveland"] = self.cavs_team_id
            name_to_id["pacers"] = self.cavs_team_id
            name_to_id["pacer"] = self.cavs_team_id
            name_to_id["indiana"] = self.cavs_team_id
            name_to_id["ind"] = self.cavs_team_id

        # Aggregate: average defensive_rating per team_name (multiple rows per team)
        agg = (
            secondary_df.groupby(team_col)["defensive_rating"]
            .mean()
            .reset_index()
        )

        for _, row in agg.iterrows():
            raw_name = str(row[team_col]).strip().lower()
            rating = float(row["defensive_rating"])
            raw_name_orig = str(row[team_col]).strip()

            matched_id = None

            # Priority 1: raw value directly matches a known team_id (handles test data)
            all_team_ids = self.primary_df["team_id"].unique().tolist()
            if raw_name_orig in all_team_ids:
                matched_id = raw_name_orig

            # Priority 2: keyword match against known Warriors/Cavs names
            if matched_id is None:
                matched_id = name_to_id.get(raw_name)

            # Priority 3: substring match
            if matched_id is None:
                for name, tid in name_to_id.items():
                    if raw_name in name or name in raw_name:
                        matched_id = tid
                        break

            if matched_id is not None:
                self.defensive_ratings[matched_id] = rating

        if not self.defensive_ratings:
            logger.warning(
                "[StatCalculator] No defensive ratings matched for Warriors/Cavs in NN_Dataset."
            )

        logger.info(
            "[StatCalculator] Loaded defensive ratings: %s", self.defensive_ratings
        )
        return self.defensive_ratings

    # ------------------------------------------------------------------
    # Task 5.1 — Baseline zone stats (standard court, pre-labeled zones)
    # ------------------------------------------------------------------

    def _detect_zone_label_column(self, df: pd.DataFrame) -> Optional[str]:
        """Return the zone label column present in the DataFrame, or None."""
        for col in ["shot_zone_basic", "shot_zone_area", "zone_name"]:
            if col in df.columns:
                return col
        return None

    def compute_baseline_zone_stats(self) -> dict:
        """
        Compute per-player zone attempt counts and shooting percentages using
        the pre-labeled zone column from the warriors_cavs_2014 dataset.

        This is the standard court baseline used for GB model training and
        sparse zone fallback.

        Returns
        -------
        dict
            {player_id: {zone_label: {"attempts": int, "pct": float}}}
        """
        zone_col = self._detect_zone_label_column(self.primary_df)
        if zone_col is None:
            logger.warning(
                "[StatCalculator] No zone label column found "
                "(expected 'shot_zone_basic', 'shot_zone_area', or 'zone_name'). "
                "baseline_zone_stats will be empty."
            )
            self.baseline_zone_attempts = {}
            self.baseline_zone_pct = {}
            return {}

        self.baseline_zone_attempts = {}
        self.baseline_zone_pct = {}

        for pid in self.eligible_players:
            pdf = self.player_dfs[pid]
            if pdf.empty or zone_col not in pdf.columns:
                self.baseline_zone_attempts[pid] = {}
                self.baseline_zone_pct[pid] = {}
                continue

            zone_attempts: dict = {}
            zone_pct: dict = {}

            for zone_label, group in pdf.groupby(zone_col):
                attempts = len(group)
                makes = int(group["shot_made_flag"].astype(int).sum())
                zone_attempts[zone_label] = attempts
                zone_pct[zone_label] = makes / attempts if attempts > 0 else 0.0

            self.baseline_zone_attempts[pid] = zone_attempts
            self.baseline_zone_pct[pid] = zone_pct

        logger.info(
            "[StatCalculator] Computed baseline zone stats for %d players using column '%s'.",
            len(self.baseline_zone_attempts),
            zone_col,
        )
        return {
            pid: {
                zone: {
                    "attempts": self.baseline_zone_attempts[pid].get(zone, 0),
                    "pct": self.baseline_zone_pct[pid].get(zone, 0.0),
                }
                for zone in self.baseline_zone_attempts[pid]
            }
            for pid in self.eligible_players
        }

    # ------------------------------------------------------------------
    # Task 5.2 — Zone stats for new CourtConfig (geometric reclassification)
    # ------------------------------------------------------------------

    def compute_zone_stats(self, court_config: CourtConfig) -> dict:
        """
        Reclassify all shots geometrically for the given CourtConfig and
        compute per-player zone attempt counts and shooting percentages.

        Parameters
        ----------
        court_config : CourtConfig
            The court configuration to evaluate.

        Returns
        -------
        dict
            {player_id: {zone_id: {"attempts": int, "pct": float}}}
        """
        if "x" not in self.primary_df.columns or "y" not in self.primary_df.columns:
            logger.warning(
                "[StatCalculator] Shot coordinate columns 'x' and 'y' not found. "
                "compute_zone_stats will return empty results."
            )
            return {}

        config_key = (court_config.arc_radius, court_config.baseline_width)

        result: dict = {}

        for pid in self.eligible_players:
            pdf = self.player_dfs[pid]
            if pdf.empty:
                if pid not in self.config_zone_attempts:
                    self.config_zone_attempts[pid] = {}
                    self.config_zone_pct[pid] = {}
                self.config_zone_attempts[pid][config_key] = {}
                self.config_zone_pct[pid][config_key] = {}
                result[pid] = {}
                continue

            # Reclassify each shot geometrically
            zones = pdf.apply(
                lambda row: classify_zone(row["x"], row["y"], court_config), axis=1
            )
            made = pdf["shot_made_flag"].astype(int)

            zone_attempts: dict = {}
            zone_pct_map: dict = {}

            for zone_label in zones.unique():
                mask = zones == zone_label
                attempts = int(mask.sum())
                makes = int(made[mask].sum())
                zone_attempts[zone_label] = attempts
                zone_pct_map[zone_label] = makes / attempts if attempts > 0 else 0.0

            if pid not in self.config_zone_attempts:
                self.config_zone_attempts[pid] = {}
                self.config_zone_pct[pid] = {}

            self.config_zone_attempts[pid][config_key] = zone_attempts
            self.config_zone_pct[pid][config_key] = zone_pct_map

            result[pid] = {
                zone: {"attempts": zone_attempts[zone], "pct": zone_pct_map[zone]}
                for zone in zone_attempts
            }

        logger.info(
            "[StatCalculator] Computed zone stats for config %s (%d players).",
            config_key,
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Task 5.3 — Sparse zone fallback
    # ------------------------------------------------------------------

    def resolve_zone_pct(
        self, player_id, zone: str, court_config: CourtConfig
    ) -> Optional[float]:
        """
        Resolve the shooting percentage for a player in a zone under the given
        CourtConfig, applying sparse zone fallback logic.

        Parameters
        ----------
        player_id :
            The player identifier.
        zone : str
            Zone ID string, e.g. "Z10".
        court_config : CourtConfig
            The court configuration being evaluated.

        Returns
        -------
        float or None
            - Player's actual zone_pct if attempts >= ZONE_ATTEMPT_THRESHOLD
            - overall_3pt_pct if 1 <= attempts < ZONE_ATTEMPT_THRESHOLD and zone is 3PT
            - overall_2pt_pct if 1 <= attempts < ZONE_ATTEMPT_THRESHOLD and zone is 2PT
            - None if attempts == 0 (zone excluded from shot distribution)
        """
        config_key = (court_config.arc_radius, court_config.baseline_width)

        # Retrieve attempt count for this player/config/zone
        attempts = (
            self.config_zone_attempts
            .get(player_id, {})
            .get(config_key, {})
            .get(zone, 0)
        )

        if attempts >= ZONE_ATTEMPT_THRESHOLD:
            return (
                self.config_zone_pct
                .get(player_id, {})
                .get(config_key, {})
                .get(zone, 0.0)
            )

        if attempts == 0:
            return None

        # 1 <= attempts < ZONE_ATTEMPT_THRESHOLD — use overall pct based on zone type
        point_value = zone_point_value(zone, court_config)
        player_stats = self.player_stats.get(player_id, {})

        if point_value == 3:
            return player_stats.get("three_pt_pct", 0.0)
        else:
            return player_stats.get("two_pt_pct", 0.0)
