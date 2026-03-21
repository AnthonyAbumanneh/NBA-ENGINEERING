"""
Neural Network Model for the NBA Court Re-Engineering Optimizer.

Roles:
  1. Heatmap generation — evaluates P(make | x, y, player) on a dense grid.
  2. Simulation weighting — provides importance weights for KDE-sampled shot
     location candidates during the grid search simulation.

Architecture:
  Input: [x_norm, y_norm, player_embedding(8)]
  Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.2) → Dense(32, relu)
  Output: Dense(1, sigmoid) → P(make)
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional TF import — graceful fallback so the module can be imported even
# when TensorFlow is not installed (e.g., during unit tests with mocks).
try:
    import tensorflow as tf
    from tensorflow import keras
    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False
    logger.warning("[NeuralNetworkModel] TensorFlow not available. Model cannot be trained.")


class NeuralNetworkModel:
    """
    Spatial shot-make probability model.

    Parameters
    ----------
    n_players : int
        Number of unique players (determines embedding table size).
    embedding_dim : int
        Dimensionality of the player embedding (default 8).
    """

    def __init__(self, n_players: int, embedding_dim: int = 8):
        if not _TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for NeuralNetworkModel. "
                "Install it with: pip install tensorflow"
            )
        self.n_players = n_players
        self.embedding_dim = embedding_dim
        self.model: Optional["keras.Model"] = None
        # Maps raw player_id values → integer indices [0, n_players)
        self.player_index: dict = {}
        # Court coordinate normalisation bounds (set during training)
        self._x_scale: float = 25.0   # half-court width
        self._y_scale: float = 47.0   # court length from basket to half-court

    # ------------------------------------------------------------------
    # Task 6.1 — Model architecture
    # ------------------------------------------------------------------

    def _build_model(self) -> "keras.Model":
        """Construct and compile the Keras model."""
        # Coordinate inputs
        coord_input = keras.Input(shape=(2,), name="coords")          # [x_norm, y_norm]
        player_input = keras.Input(shape=(1,), name="player_id")      # integer index

        # Player embedding
        embedding = keras.layers.Embedding(
            input_dim=self.n_players,
            output_dim=self.embedding_dim,
            name="player_embedding",
        )(player_input)
        embedding_flat = keras.layers.Flatten(name="embedding_flat")(embedding)

        # Concatenate coordinates + embedding
        x = keras.layers.Concatenate(name="concat")([coord_input, embedding_flat])

        # Hidden layers
        x = keras.layers.Dense(128, activation="relu", name="dense_128")(x)
        x = keras.layers.Dropout(0.3, name="dropout_1")(x)
        x = keras.layers.Dense(64, activation="relu", name="dense_64")(x)
        x = keras.layers.Dropout(0.2, name="dropout_2")(x)
        x = keras.layers.Dense(32, activation="relu", name="dense_32")(x)

        # Output
        output = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

        model = keras.Model(
            inputs=[coord_input, player_input],
            outputs=output,
            name="shot_make_nn",
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------
    # Task 6.2 — Training pipeline
    # ------------------------------------------------------------------

    def train(
        self,
        shots_df: pd.DataFrame,
        val_split: float = 0.2,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        save_path: Optional[str] = None,
    ) -> "keras.callbacks.History":
        """
        Train the model on eligible-player shot data.

        Parameters
        ----------
        shots_df : pd.DataFrame
            Must contain columns: x, y, player_id, shot_made_flag.
        val_split : float
            Fraction of data held out for validation (stratified by player).
        batch_size : int
        max_epochs : int
        patience : int
            Early stopping patience on validation loss.
        save_path : str, optional
            If provided, save model weights to this path after training.

        Returns
        -------
        keras.callbacks.History
        """
        shots_df = shots_df.dropna(subset=["x", "y", "player_id", "shot_made_flag"]).copy()

        # Build player index mapping
        unique_players = sorted(shots_df["player_id"].unique().tolist(), key=str)
        self.player_index = {pid: idx for idx, pid in enumerate(unique_players)}
        # Update n_players in case it differs from constructor arg
        self.n_players = max(self.n_players, len(unique_players))

        # Encode player ids
        shots_df["player_idx"] = shots_df["player_id"].map(self.player_index)

        # Stratified 80/20 split by player
        train_mask = self._stratified_split_mask(shots_df, val_split)
        train_df = shots_df[train_mask]
        val_df = shots_df[~train_mask]

        x_train, pid_train, y_train = self._prepare_inputs(train_df)
        x_val, pid_val, y_val = self._prepare_inputs(val_df)

        self.model = self._build_model()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        ]

        history = self.model.fit(
            [x_train, pid_train],
            y_train,
            validation_data=([x_val, pid_val], y_val),
            batch_size=batch_size,
            epochs=max_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            self.model.save_weights(save_path)
            logger.info("[NeuralNetworkModel] Weights saved to %s", save_path)

        best_val_loss = min(history.history["val_loss"])
        best_val_acc  = max(history.history["val_accuracy"])

        # Compute AUC and log-loss on val set
        import tensorflow as tf
        val_probs = self.model(
            [tf.constant(x_val), tf.constant(pid_val)], training=False
        ).numpy().ravel()
        try:
            from sklearn.metrics import roc_auc_score, log_loss as sk_log_loss
            val_auc  = roc_auc_score(y_val, val_probs)
            val_logloss = sk_log_loss(y_val, val_probs)
        except Exception:
            val_auc = float("nan")
            val_logloss = best_val_loss

        print("\n" + "="*50)
        print("Neural Network Model — Validation Metrics")
        print("="*50)
        print(f"  Accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.1f}%)")
        print(f"  ROC-AUC  : {val_auc:.4f}")
        print(f"  Log-Loss : {val_logloss:.4f}")
        print(f"  Val set  : {len(y_val):,} shots  |  "
              f"Make rate: {y_val.mean():.3f}")
        print("="*50 + "\n")

        logger.info(
            "[NeuralNetworkModel] Training complete. "
            "Best val_loss: %.4f  val_accuracy: %.4f",
            best_val_loss, best_val_acc,
        )
        return history

    def _stratified_split_mask(self, df: pd.DataFrame, val_split: float) -> pd.Series:
        """Return a boolean mask (True = train) with stratification by player."""
        mask = pd.Series(True, index=df.index)
        for pid, group in df.groupby("player_id"):
            n_val = max(1, int(len(group) * val_split))
            val_indices = group.sample(n=n_val, random_state=42).index
            mask.loc[val_indices] = False
        return mask

    def _prepare_inputs(self, df: pd.DataFrame):
        """Return (coord_array, player_idx_array, label_array) from a DataFrame."""
        x_norm = df["x"].values / self._x_scale
        y_norm = df["y"].values / self._y_scale
        coords = np.column_stack([x_norm, y_norm]).astype(np.float32)
        player_ids = df["player_idx"].values.astype(np.int32)
        labels = df["shot_made_flag"].values.astype(np.float32)
        return coords, player_ids, labels

    # ------------------------------------------------------------------
    # Task 6.3 — Inference
    # ------------------------------------------------------------------

    def predict_batch(self, xs: np.ndarray, ys: np.ndarray, player_id) -> np.ndarray:
        """
        Return P(make | x, y, player_id) for a batch of shot locations.

        Parameters
        ----------
        xs, ys : np.ndarray of shape (N,)
            Shot coordinates in feet.
        player_id :
            Raw player identifier.

        Returns
        -------
        np.ndarray of shape (N,) with probabilities in [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        player_idx = self.player_index.get(player_id, 0)
        coords = np.column_stack([
            xs / self._x_scale,
            ys / self._y_scale,
        ]).astype(np.float32)
        pid_arr = np.full((len(xs), 1), player_idx, dtype=np.int32)
        # Use direct model call instead of model.predict() to avoid Keras shape inference bug
        import tensorflow as tf
        coords_t = tf.constant(coords)
        pid_t = tf.constant(pid_arr)
        probs = self.model([coords_t, pid_t], training=False).numpy().ravel()
        return probs.astype(np.float64)

    def predict(self, x: float, y: float, player_id) -> float:
        """
        Return P(make | x, y, player_id) for a single shot location.

        Parameters
        ----------
        x, y : float
            Shot coordinates in feet (basket at origin).
        player_id :
            Raw player identifier (must have been seen during training).

        Returns
        -------
        float in [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        player_idx = self.player_index.get(player_id, 0)
        coord = np.array([[x / self._x_scale, y / self._y_scale]], dtype=np.float32)
        pid_arr = np.array([[player_idx]], dtype=np.int32)
        import tensorflow as tf
        prob = float(self.model([tf.constant(coord), tf.constant(pid_arr)], training=False).numpy()[0][0])
        return prob

    def predict_grid(self, player_id, x_step: float = 0.5, y_step: float = 0.5) -> np.ndarray:
        """
        Evaluate P(make | x, y, player_id) on a dense grid covering the half-court.

        Parameters
        ----------
        player_id :
            Raw player identifier.
        x_step : float
            Grid spacing in the x direction (feet).
        y_step : float
            Grid spacing in the y direction (feet).

        Returns
        -------
        np.ndarray of shape (n_y, n_x) with P(make) values.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        xs = np.arange(-25, 25 + x_step, x_step)
        ys = np.arange(0, 47 + y_step, y_step)

        xx, yy = np.meshgrid(xs, ys)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        player_idx = self.player_index.get(player_id, 0)
        coords = np.column_stack([
            x_flat / self._x_scale,
            y_flat / self._y_scale,
        ]).astype(np.float32)
        pid_arr = np.full((len(x_flat), 1), player_idx, dtype=np.int32)

        import tensorflow as tf
        coords_t = tf.constant(coords)
        pid_t = tf.constant(pid_arr)
        probs = self.model([coords_t, pid_t], training=False).numpy().ravel()
        return probs.reshape(xx.shape)

    def load_weights(self, path: str) -> None:
        """Load previously saved model weights from disk."""
        if self.model is None:
            self.model = self._build_model()
        self.model.load_weights(path)
        logger.info("[NeuralNetworkModel] Weights loaded from %s", path)


# ---------------------------------------------------------------------------
# Task 6.4 — Per-player KDE for shot location sampling
# ---------------------------------------------------------------------------

class ShotLocationSampler:
    """
    Fits a per-player KDE on historical (x, y) shot locations and provides
    NN-weighted sampling for use during simulation.

    Parameters
    ----------
    bandwidth : float or str
        KDE bandwidth. Use 'scott' for Scott's rule (default), or a float.
    """

    def __init__(self, bandwidth="scott", nn_temperature: float = 0.3):
        """
        Parameters
        ----------
        bandwidth : float or str
            KDE bandwidth. Use 'scott' for Scott's rule (default), or a float.
        nn_temperature : float
            Controls how strongly NN weights influence location sampling.
            weights = raw_nn_weights ** nn_temperature
            - 1.0 = full NN influence (biases toward high-make-% spots)
            - 0.3 = flattened (preserves NN signal but respects KDE distribution)
            - 0.0 = pure KDE (no NN influence)
        """
        from sklearn.neighbors import KernelDensity  # noqa: F401 — validate import
        self.bandwidth = bandwidth
        self.nn_temperature = nn_temperature
        self.player_kdes: dict = {}   # {player_id: KernelDensity}
        self._player_shots: dict = {} # {player_id: np.ndarray of shape (n, 2)}
        # Precomputed weighted candidate pool: {player_id: (candidates, weights)}
        self._cached_candidates: dict = {}
        self._cached_weights: dict = {}

    def fit(self, shots_df: pd.DataFrame) -> None:
        """
        Fit a KDE for every player in shots_df.

        Parameters
        ----------
        shots_df : pd.DataFrame
            Must contain columns: x, y, player_id.
        """
        from sklearn.neighbors import KernelDensity

        shots_df = shots_df.dropna(subset=["x", "y", "player_id"])

        for pid, group in shots_df.groupby("player_id"):
            coords = group[["x", "y"]].values.astype(np.float64)
            if len(coords) < 2:
                logger.warning(
                    "[ShotLocationSampler] Player %s has < 2 shots; skipping KDE.", pid
                )
                continue

            bw = self.bandwidth
            if bw == "scott":
                # Scott's rule: n^(-1/(d+4)), d=2
                bw = len(coords) ** (-1.0 / 6.0)

            kde = KernelDensity(bandwidth=bw, kernel="gaussian")
            kde.fit(coords)
            self.player_kdes[pid] = kde
            self._player_shots[pid] = coords

        logger.info(
            "[ShotLocationSampler] Fitted KDEs for %d players.", len(self.player_kdes)
        )

    def precompute_nn_weights(
        self, nn_model: "NeuralNetworkModel", pool_size: int = 5000
    ) -> None:
        """
        Precompute a large pool of KDE candidates + NN weights per player.
        Called once after training; simulation then samples from this cache
        instead of calling the NN on every shot.

        Parameters
        ----------
        nn_model : NeuralNetworkModel
        pool_size : int
            Number of candidates to precompute per player (default 5000).
        """
        logger.info(
            "[ShotLocationSampler] Precomputing NN weights for %d players (pool=%d)...",
            len(self.player_kdes), pool_size,
        )
        rng = np.random.default_rng(42)
        for pid, kde in self.player_kdes.items():
            candidates = kde.sample(pool_size, random_state=int(rng.integers(1 << 31)))
            weights = nn_model.predict_batch(
                candidates[:, 0].astype(float),
                candidates[:, 1].astype(float),
                pid,
            )
            # Apply temperature to flatten NN influence on location distribution.
            # temperature=1.0 → full NN bias; temperature=0.3 → flattened; 0.0 → uniform
            if self.nn_temperature != 1.0:
                weights = np.power(np.clip(weights, 1e-10, None), self.nn_temperature)
            w_sum = weights.sum()
            if w_sum <= 0:
                weights = np.ones(pool_size) / pool_size
            else:
                weights = weights / w_sum
            self._cached_candidates[pid] = candidates
            self._cached_weights[pid] = weights
        logger.info("[ShotLocationSampler] NN weight precomputation complete.")

    def sample_weighted(
        self,
        player_id,
        nn_model: "NeuralNetworkModel",
        n_candidates: int = 100,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple:
        """
        Draw one (x, y) location using NN-weighted KDE sampling.

        Steps:
          1. Sample n_candidates from the player's KDE.
          2. Query NN for P(make | x, y, player) at each candidate.
          3. Normalise NN probabilities as importance weights.
          4. Sample one candidate via weighted selection.

        Parameters
        ----------
        player_id :
            Raw player identifier.
        nn_model : NeuralNetworkModel
            Trained NN model.
        n_candidates : int
            Number of KDE candidates to draw.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        (x, y) : tuple of float
        """
        if rng is None:
            rng = np.random.default_rng()

        kde = self.player_kdes.get(player_id)
        if kde is None:
            raise KeyError(f"[ShotLocationSampler] No KDE fitted for player {player_id}.")

        # Use precomputed cache if available (fast path — no NN call needed)
        if player_id in self._cached_candidates:
            candidates = self._cached_candidates[player_id]
            weights = self._cached_weights[player_id]
            idx = rng.choice(len(candidates), p=weights)
            return float(candidates[idx, 0]), float(candidates[idx, 1])

        # Fallback: sample fresh candidates and call NN (slow path)
        candidates = kde.sample(n_candidates, random_state=int(rng.integers(1 << 31)))

        # Step 2: NN importance weights — batch all candidates in one forward pass
        weights = nn_model.predict_batch(
            candidates[:, 0].astype(float),
            candidates[:, 1].astype(float),
            player_id,
        )

        # Step 3: apply temperature + normalise (guard against all-zero weights)
        if self.nn_temperature != 1.0:
            weights = np.power(np.clip(weights, 1e-10, None), self.nn_temperature)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            weights = np.ones(len(candidates)) / len(candidates)
        else:
            weights = weights / weight_sum

        # Step 4: weighted sample
        idx = rng.choice(len(candidates), p=weights)
        x, y = float(candidates[idx, 0]), float(candidates[idx, 1])
        return x, y
