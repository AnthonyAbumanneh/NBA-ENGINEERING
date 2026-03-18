"""
Data_Loader: reads and validates the warriors_cavs_2014 and nn_dataset CSV files.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Required columns that must be present and non-null in the primary dataset
# (after column normalization)
PRIMARY_REQUIRED_COLS = ["x", "y", "player_id", "team_id", "shot_made_flag"]

# Required columns for the secondary (NN) dataset (after normalization)
SECONDARY_REQUIRED_COLS = ["team_name", "defensive_rating"]

# Maps actual CSV column names → internal names used throughout the codebase
_PRIMARY_COL_MAP = {
    "LOC_X": "x",
    "LOC_Y": "y",
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ID": "team_id",
    "TEAM_NAME": "team_name",
    "TEAM": "team_abbrev",          # 'Warriors' / 'Cavaliers' — used for filtering
    "SHOT_MADE_FLAG": "shot_made_flag",
    "SHOT_TYPE": "shot_type",
    "SHOT_ZONE_BASIC": "shot_zone_basic",
    "SHOT_ZONE_AREA": "shot_zone_area",
    "SHOT_ZONE_RANGE": "shot_zone_range",
    "SHOT_DISTANCE": "shot_distance",
    "GAME_ID": "game_id",
    "GAME_DATE": "game_date",
    "PERIOD": "period",
    "ACTION_TYPE": "action_type",
    "EVENT_TYPE": "event_type",
    "MINUTES_REMAINING": "minutes_remaining",
    "SECONDS_REMAINING": "seconds_remaining",
}

# Secondary dataset: 'team' → 'team_name', derive defensive_rating from available cols
_SECONDARY_COL_MAP = {
    "team": "team_name",
    # opp_3par_allowed is the closest proxy for defensive rating in this dataset
    "opp_3par_allowed": "defensive_rating",
}

# Strip trailing year digits from team names like 'Cavaliers2016' → 'Cavaliers'
_SECONDARY_TEAM_YEAR_RE = __import__("re").compile(r"\d+$")


class DataLoader:
    """Loads and validates the primary and secondary datasets."""

    def __init__(self, primary_path: str, secondary_path: str):
        self.primary_path = primary_path
        self.secondary_path = secondary_path
        self._primary: pd.DataFrame | None = None
        self._secondary: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_primary(self) -> pd.DataFrame:
        """Load and validate warriors_cavs shot dataset."""
        df = self._read_csv(self.primary_path, label="primary")
        df = self._normalize_columns(df, _PRIMARY_COL_MAP)
        df = self._validate_and_clean(df, PRIMARY_REQUIRED_COLS, label="primary")
        # NBA shot chart coordinates: LOC_X/LOC_Y are in tenths of a foot
        # Convert to feet if values look like tenths (typical range ±250 for x, 0-470 for y)
        if df["x"].abs().max() > 50:
            df["x"] = df["x"] / 10.0
            df["y"] = df["y"] / 10.0
            logger.info("[DataLoader] Converted x/y from tenths-of-foot to feet.")
        self._primary = df
        return self._primary

    def load_secondary(self) -> pd.DataFrame:
        """Load and validate nn_dataset."""
        df = self._read_csv(self.secondary_path, label="secondary")
        df = self._normalize_columns(df, _SECONDARY_COL_MAP)
        # Normalise team names: strip trailing year digits ('Cavaliers2016' → 'Cavaliers')
        if "team_name" in df.columns:
            df["team_name"] = df["team_name"].str.strip().apply(
                lambda v: _SECONDARY_TEAM_YEAR_RE.sub("", str(v)).strip()
            )
        df = self._validate_and_clean(df, SECONDARY_REQUIRED_COLS, label="secondary")
        self._secondary = df
        return self._secondary

    @property
    def primary(self) -> pd.DataFrame:
        if self._primary is None:
            raise RuntimeError("Primary dataset not loaded. Call load_primary() first.")
        return self._primary

    @property
    def secondary(self) -> pd.DataFrame:
        if self._secondary is None:
            raise RuntimeError("Secondary dataset not loaded. Call load_secondary() first.")
        return self._secondary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv(path: str, label: str) -> pd.DataFrame:
        """Read CSV, raise descriptive error on missing file."""
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"[DataLoader] {label} dataset not found at path: '{path}'. "
                "Please update the path in config.py."
            )
        except Exception as exc:
            raise IOError(
                f"[DataLoader] Failed to read {label} dataset at '{path}': {exc}"
            ) from exc

    @staticmethod
    def _normalize_columns(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
        """Rename columns according to col_map (only renames columns that exist)."""
        rename = {k: v for k, v in col_map.items() if k in df.columns}
        return df.rename(columns=rename)

    @staticmethod
    def _validate_and_clean(
        df: pd.DataFrame, required_cols: list[str], label: str
    ) -> pd.DataFrame:
        """Check required columns exist, drop rows with nulls in them."""
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"[DataLoader] {label} dataset is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        before = len(df)
        df = df.dropna(subset=required_cols)
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(
                "[DataLoader] Dropped %d rows from %s dataset due to missing values "
                "in required columns %s.",
                dropped, label, required_cols,
            )

        logger.info(
            "[DataLoader] Loaded %s dataset: %d rows.", label, len(df)
        )
        return df.reset_index(drop=True)
