"""
Gradient Boost Model for the NBA Court Re-Engineering Optimizer.

Predicts P(make | player, zone, team_defense) for use during game simulation.
Uses XGBoost (default) or LightGBM (configurable).

Features per shot:
  player_id (encoded int), zone_id (Z01–Z14 encoded), opp_defensive_rating,
  attempts_in_zone, zone_pct (resolved via sparse logic), overall_2pt_pct,
  overall_3pt_pct, zone_point_value
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.court_geometry.court_geometry import CourtConfig, zone_point_value

logger = logging.getLogger(__name__)

# Zone label → integer encoding (Z01=0 … Z14=13)
_ZONE_ENCODING = {f"Z{i:02d}": i - 1 for i in range(1, 15)}

# Feature column order (must match between training and inference)
FEATURE_COLS = [
    "player_idx",
    "zone_idx",
    "opp_defensive_rating",
    "attempts_in_zone",
    "zone_pct",
    "overall_2pt_pct",
    "overall_3pt_pct",
    "zone_point_value",
]


class GradientBoostModel:
    """
    XGBoost / LightGBM binary classifier for shot make/miss prediction.

    Parameters
    ----------
    backend : str
        'xgboost' (default) or 'lightgbm'.
    n_estimators : int
    max_depth : int
    learning_rate : float
    subsample : float
    """

    def __init__(
        self,
        backend: str = "xgboost",
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
    ):
        self.backend = backend.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample

        self.model = None
        # Maps raw player_id → integer index
        self.player_index: dict = {}

    # ------------------------------------------------------------------
    # Task 7.1 — Feature engineering
    # ------------------------------------------------------------------

    def build_feature_vector(
        self,
        player_id,
        zone: str,
        court_config: CourtConfig,
        opp_defensive_rating: float,
        attempts_in_zone: int,
        zone_pct: float,
        overall_2pt_pct: float,
        overall_3pt_pct: float,
    ) -> np.ndarray:
        """
        Build the 8-feature vector for a single shot.

        Parameters
        ----------
        player_id :
            Raw player identifier.
        zone : str
            Zone ID, e.g. 'Z10'.
        court_config : CourtConfig
            Current court configuration.
        opp_defensive_rating : float
            Opponent team defensive rating.
        attempts_in_zone : int
            Player's historical attempts in this zone under this config.
        zone_pct : float
            Resolved zone shooting % (from resolve_zone_pct).
        overall_2pt_pct : float
            Player's overall 2PT%.
        overall_3pt_pct : float
            Player's overall 3PT%.

        Returns
        -------
        np.ndarray of shape (8,)
        """
        player_idx = self.player_index.get(player_id, 0)
        zone_idx = _ZONE_ENCODING.get(zone, 0)
        pv = zone_point_value(zone, court_config)

        return np.array([
            player_idx,
            zone_idx,
            opp_defensive_rating,
            attempts_in_zone,
            zone_pct,
            overall_2pt_pct,
            overall_3pt_pct,
            pv,
        ], dtype=np.float32)

    def build_training_dataset(
        self,
        shots_df: pd.DataFrame,
        stat_calculator,
        standard_config: CourtConfig,
        defensive_ratings: dict,
    ) -> tuple:
        """
        Build (X, y) training arrays from eligible-player shot data.

        Parameters
        ----------
        shots_df : pd.DataFrame
            All eligible-player shots with columns: x, y, player_id, shot_made_flag.
        stat_calculator : StatCalculator
            Provides player_stats, config_zone_attempts, config_zone_pct,
            and resolve_zone_pct().
        standard_config : CourtConfig
            The standard NBA court config (used for zone classification).
        defensive_ratings : dict
            {team_id: defensive_rating} from NN_Dataset.

        Returns
        -------
        (X, y) : (np.ndarray of shape (n, 8), np.ndarray of shape (n,))
        """
        rows = []
        labels = []

        # Build player index
        unique_players = sorted(shots_df["player_id"].unique().tolist(), key=str)
        self.player_index = {pid: idx for idx, pid in enumerate(unique_players)}

        # Ensure zone stats are computed for the standard config
        config_key = (standard_config.arc_radius, standard_config.baseline_width)
        if not stat_calculator.config_zone_attempts.get(
            unique_players[0] if unique_players else None, {}
        ).get(config_key):
            stat_calculator.compute_zone_stats(standard_config)

        for _, shot in shots_df.iterrows():
            pid = shot["player_id"]
            player_stats = stat_calculator.player_stats.get(pid, {})
            if not player_stats:
                continue

            # Determine opponent team for defensive rating
            # Use the opposing team's defensive rating; fall back to mean
            opp_def = self._get_opp_defensive_rating(shot, defensive_ratings)

            # Zone under standard config
            from src.court_geometry.court_geometry import classify_zone
            zone = classify_zone(float(shot["x"]), float(shot["y"]), standard_config)

            attempts = (
                stat_calculator.config_zone_attempts
                .get(pid, {})
                .get(config_key, {})
                .get(zone, 0)
            )

            resolved_pct = stat_calculator.resolve_zone_pct(pid, zone, standard_config)
            if resolved_pct is None:
                # Zone excluded (0 attempts) — skip this shot for training
                continue

            feat = self.build_feature_vector(
                player_id=pid,
                zone=zone,
                court_config=standard_config,
                opp_defensive_rating=opp_def,
                attempts_in_zone=attempts,
                zone_pct=resolved_pct,
                overall_2pt_pct=player_stats.get("two_pt_pct", 0.0),
                overall_3pt_pct=player_stats.get("three_pt_pct", 0.0),
            )
            rows.append(feat)
            labels.append(int(shot["shot_made_flag"]))

        if not rows:
            raise ValueError("[GradientBoostModel] No valid training rows built.")

        X = np.vstack(rows).astype(np.float32)
        y = np.array(labels, dtype=np.int32)
        return X, y

    def _get_opp_defensive_rating(self, shot_row, defensive_ratings: dict) -> float:
        """Return opponent defensive rating for a shot row, or mean if unavailable."""
        if not defensive_ratings:
            return 110.0  # league-average fallback

        # Try to find opponent team from shot row
        for col in ["opp_team_id", "opponent_team_id", "opp_id"]:
            if col in shot_row.index:
                opp_id = shot_row[col]
                if opp_id in defensive_ratings:
                    return float(defensive_ratings[opp_id])

        # Fall back to mean of available ratings
        return float(np.mean(list(defensive_ratings.values())))

    # ------------------------------------------------------------------
    # Task 7.2 — Model training
    # ------------------------------------------------------------------

    def train(
        self,
        shots_df: pd.DataFrame,
        stat_calculator,
        standard_config: CourtConfig,
        defensive_ratings: dict,
        val_split: float = 0.2,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Train the GB model on eligible-player shot data.

        Parameters
        ----------
        shots_df : pd.DataFrame
        stat_calculator : StatCalculator
        standard_config : CourtConfig
        defensive_ratings : dict
        val_split : float
        save_path : str, optional
            If provided, save the trained model to this path.
        """
        X, y = self.build_training_dataset(
            shots_df, stat_calculator, standard_config, defensive_ratings
        )

        # Stratified split by player (approximate: random split on full dataset)
        n_val = max(1, int(len(X) * val_split))
        rng = np.random.default_rng(42)
        val_idx = rng.choice(len(X), size=n_val, replace=False)
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[val_idx] = False

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[~train_mask], y[~train_mask]

        self.model = self._build_model()
        self._fit_model(X_train, y_train, X_val, y_val)

        if save_path:
            self._save_model(save_path)

        logger.info("[GradientBoostModel] Training complete.")

    def _build_model(self):
        """Instantiate the XGBoost or LightGBM classifier."""
        if self.backend == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    early_stopping_rounds=20,
                    random_state=42,
                )
            except ImportError:
                logger.warning("[GradientBoostModel] xgboost not found; trying lightgbm.")
                self.backend = "lightgbm"

        if self.backend == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=42,
            )

        raise ValueError(f"[GradientBoostModel] Unknown backend: {self.backend}")

    def _fit_model(self, X_train, y_train, X_val, y_val) -> None:
        """Fit the model with early stopping."""
        if self.backend == "xgboost":
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:  # lightgbm
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[],
            )

    def _save_model(self, path: str) -> None:
        import pickle
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("[GradientBoostModel] Model saved to %s", path)

    def load_model(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("[GradientBoostModel] Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Task 7.3 — Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        player_id,
        zone: str,
        court_config: CourtConfig,
        opp_defensive_rating: float,
        attempts_in_zone: int,
        zone_pct: float,
        overall_2pt_pct: float,
        overall_3pt_pct: float,
    ) -> float:
        """
        Return P(make) ∈ [0, 1] for a single shot.

        Parameters
        ----------
        player_id :
            Raw player identifier.
        zone : str
            Zone ID under the current court config.
        court_config : CourtConfig
        opp_defensive_rating : float
        attempts_in_zone : int
        zone_pct : float
            Resolved zone % (from resolve_zone_pct).
        overall_2pt_pct : float
        overall_3pt_pct : float

        Returns
        -------
        float in [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        feat = self.build_feature_vector(
            player_id=player_id,
            zone=zone,
            court_config=court_config,
            opp_defensive_rating=opp_defensive_rating,
            attempts_in_zone=attempts_in_zone,
            zone_pct=zone_pct,
            overall_2pt_pct=overall_2pt_pct,
            overall_3pt_pct=overall_3pt_pct,
        )
        X = feat.reshape(1, -1)
        prob = float(self.model.predict_proba(X)[0][1])
        return float(np.clip(prob, 0.0, 1.0))
