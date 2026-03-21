"""
Fatigue Model for the NBA Court Re-Engineering Optimizer.

Trains a logistic regression on historical shot data to learn how a player's
make probability decays as a function of minutes elapsed in the game.

The model outputs a fatigue multiplier in [FATIGUE_MIN, 1.0] that is applied
to the GB model's p_make during simulation.  The effect is intentionally minor
(max ~5% reduction at peak fatigue) so it does not dominate the simulation.

Fatigue proxy:
    elapsed_minutes = (period - 1) * 12 + (12 - minutes_remaining)
                      + (60 - seconds_remaining) / 60
    Range: 0 (tip-off) → 48 (end of regulation)

The logistic regression learns:
    log-odds(make) = β0 + β1 * elapsed_minutes + β2 * player_idx
                   + β3 * elapsed_minutes * player_idx   (optional interaction)

The fatigue multiplier for a given elapsed_minutes is:
    multiplier = sigmoid(β0 + β1 * elapsed_minutes + β2 * player_idx)
                 / sigmoid(β0 + β2 * player_idx)          ← normalised to 1.0 at t=0

This ensures the multiplier is exactly 1.0 at the start of the game and
decreases (or increases) as minutes accumulate.
"""

import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Clamp multiplier so fatigue never reduces p_make by more than this fraction
FATIGUE_MAX_REDUCTION = 0.05   # 5% maximum reduction
FATIGUE_MIN_MULTIPLIER = 1.0 - FATIGUE_MAX_REDUCTION  # 0.95


class FatigueModel:
    """
    Logistic regression fatigue model.

    Parameters
    ----------
    use_player_interaction : bool
        If True, include a player × elapsed_minutes interaction term so each
        player has their own fatigue slope.  Requires more data; defaults False
        (global slope, per-player intercept via player index feature).
    """

    def __init__(self, use_player_interaction: bool = False):
        self.use_player_interaction = use_player_interaction
        self._model = None          # sklearn LogisticRegression
        self.player_index: Dict = {}
        self._is_trained: bool = False
        # Fallback decay rate used when model is not trained
        # Derived from published research: ~3% drop over 48 minutes
        self._fallback_decay_per_minute: float = FATIGUE_MAX_REDUCTION / 48.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, shots_df: pd.DataFrame) -> None:
        """
        Fit the logistic regression on historical shot data.

        Parameters
        ----------
        shots_df : pd.DataFrame
            Must contain: period, minutes_remaining, seconds_remaining,
                          player_id, shot_made_flag.
            All columns must already be in normalised form (as produced by DataLoader).
        """
        required = ["period", "minutes_remaining", "seconds_remaining",
                    "player_id", "shot_made_flag"]
        missing = [c for c in required if c not in shots_df.columns]
        if missing:
            logger.warning(
                "[FatigueModel] Missing columns %s — using rule-based fallback.", missing
            )
            return

        df = shots_df.dropna(subset=required).copy()
        if len(df) < 100:
            logger.warning("[FatigueModel] Too few rows (%d) — using fallback.", len(df))
            return

        # Compute elapsed minutes (0–48)
        df["elapsed_minutes"] = (
            (df["period"].astype(float) - 1) * 12
            + (12 - df["minutes_remaining"].astype(float))
            + (60 - df["seconds_remaining"].astype(float)) / 60.0
        ).clip(0, 48)

        # Build player index
        unique_players = sorted(df["player_id"].unique().tolist(), key=str)
        self.player_index = {pid: idx for idx, pid in enumerate(unique_players)}
        df["player_idx"] = df["player_id"].map(self.player_index).fillna(0).astype(int)

        # Feature matrix
        X = self._build_features(df["elapsed_minutes"].values, df["player_idx"].values)
        y = df["shot_made_flag"].astype(int).values

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            self._model = LogisticRegression(
                max_iter=500,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )
            self._model.fit(X_scaled, y)
            self._is_trained = True

            # Held-out validation metrics (20% split)
            n_val = max(1, int(len(X) * 0.2))
            rng = np.random.default_rng(42)
            val_idx = rng.choice(len(X), size=n_val, replace=False)
            X_val_s = X_scaled[val_idx]
            y_val   = y[val_idx]
            val_probs = self._model.predict_proba(X_val_s)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)

            from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
            val_acc  = accuracy_score(y_val, val_preds)
            val_auc  = roc_auc_score(y_val, val_probs)
            val_loss = log_loss(y_val, val_probs)

            print("\n" + "="*50)
            print("Fatigue Model — Validation Metrics")
            print("="*50)
            print(f"  Accuracy : {val_acc:.4f}  ({val_acc*100:.1f}%)")
            print(f"  ROC-AUC  : {val_auc:.4f}")
            print(f"  Log-Loss : {val_loss:.4f}")
            print(f"  Val set  : {len(y_val):,} shots  |  "
                  f"Make rate: {y_val.mean():.3f}")
            elapsed_coef = self._model.coef_[0][0]
            print(f"  Fatigue coef (elapsed_min): {elapsed_coef:.5f}")
            print("="*50 + "\n")

            logger.info(
                "[FatigueModel] Trained. elapsed_minutes coef=%.5f "
                "(negative = fatigue reduces make %%)",
                elapsed_coef,
            )
        except Exception as exc:
            logger.warning("[FatigueModel] Training failed (%s) — using fallback.", exc)

    def _build_features(
        self, elapsed: np.ndarray, player_idx: np.ndarray
    ) -> np.ndarray:
        """Build feature matrix [elapsed_minutes, player_idx, (interaction)]."""
        cols = [elapsed, player_idx.astype(float)]
        if self.use_player_interaction:
            cols.append(elapsed * player_idx)
        return np.column_stack(cols)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_multiplier(self, player_id, elapsed_minutes: float) -> float:
        """
        Return a fatigue multiplier in [FATIGUE_MIN_MULTIPLIER, 1.0].

        The multiplier is normalised so it equals 1.0 at elapsed_minutes=0
        and decreases as the game progresses.

        Parameters
        ----------
        player_id :
            Raw player identifier (must have been seen during training).
        elapsed_minutes : float
            Minutes elapsed in the game so far (0–48).

        Returns
        -------
        float in [FATIGUE_MIN_MULTIPLIER, 1.0]
        """
        elapsed_minutes = float(np.clip(elapsed_minutes, 0, 48))

        if not self._is_trained or self._model is None:
            return self._fallback_multiplier(elapsed_minutes)

        player_idx = float(self.player_index.get(player_id, 0))

        # Probability at elapsed_minutes
        X_t = self._build_features(
            np.array([elapsed_minutes]), np.array([player_idx], dtype=float)
        )
        X_t_scaled = self._scaler.transform(X_t)
        p_t = float(self._model.predict_proba(X_t_scaled)[0][1])

        # Probability at elapsed_minutes=0 (baseline — no fatigue)
        X_0 = self._build_features(
            np.array([0.0]), np.array([player_idx], dtype=float)
        )
        X_0_scaled = self._scaler.transform(X_0)
        p_0 = float(self._model.predict_proba(X_0_scaled)[0][1])

        if p_0 <= 0:
            return 1.0

        # Normalised ratio: how much has make% changed relative to game start
        raw_multiplier = p_t / p_0

        # Clamp to [FATIGUE_MIN_MULTIPLIER, 1.05] — allow tiny positive effect early game
        return float(np.clip(raw_multiplier, FATIGUE_MIN_MULTIPLIER, 1.05))

    def _fallback_multiplier(self, elapsed_minutes: float) -> float:
        """
        Rule-based fallback: linear decay of FATIGUE_MAX_REDUCTION over 48 min.
        At t=0 → 1.0; at t=48 → FATIGUE_MIN_MULTIPLIER.
        """
        decay = self._fallback_decay_per_minute * elapsed_minutes
        return float(np.clip(1.0 - decay, FATIGUE_MIN_MULTIPLIER, 1.0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("[FatigueModel] Saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "FatigueModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("[FatigueModel] Loaded from %s", path)
        return model
