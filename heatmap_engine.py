"""
Heatmap Engine for the NBA Court Re-Engineering Optimizer.

Responsibilities:
  - Build historical 2D shot-density heatmaps per player (actual shot locations).
  - Build new-court NN probability surface heatmaps per player (combined-optimal config).
  - Render side-by-side matplotlib figures for every eligible player.
  - Reclassify shots per CourtConfig for zone-level analysis.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; switch to "TkAgg" for display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle
from matplotlib.colors import LogNorm

from src.court_geometry.court_geometry import CourtConfig, classify_zone, zone_point_value

logger = logging.getLogger(__name__)

# Grid resolution for NN probability surface
GRID_X_STEP = 0.5   # feet
GRID_Y_STEP = 0.5   # feet

# Heatmap bin sizes for historical density
HIST_BINS_X = 50
HIST_BINS_Y = 50


class HeatmapEngine:
    """
    Generates and renders shot heatmaps for eligible players.

    Parameters
    ----------
    player_names : dict, optional
        {player_id: display_name} for plot titles.
    """

    def __init__(self, player_names: Optional[Dict] = None):
        self.player_names = player_names or {}
        # {player_id: np.ndarray (H, W)} — raw 2D histogram counts
        self.historical_heatmaps: Dict = {}
        # {player_id: np.ndarray (H, W)} — NN P(make) surface on combined-optimal config
        self.new_court_heatmaps: Dict = {}
        # Grid extents used for NN surface (set in generate_new_court_heatmaps)
        self._grid_xs: Optional[np.ndarray] = None
        self._grid_ys: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Task 9.1 — Historical heatmap generation
    # ------------------------------------------------------------------

    def build_historical_heatmaps(self, shots_df: pd.DataFrame) -> Dict:
        """
        Bin each player's shot attempts into a 2D histogram on the standard court.

        Parameters
        ----------
        shots_df : pd.DataFrame
            Must contain columns: x, y, player_id.

        Returns
        -------
        dict {player_id: np.ndarray of shape (HIST_BINS_Y, HIST_BINS_X)}
        """
        shots_df = shots_df.dropna(subset=["x", "y", "player_id"])

        for pid, group in shots_df.groupby("player_id"):
            xs = group["x"].values.astype(float)
            ys = group["y"].values.astype(float)

            heatmap, _, _ = np.histogram2d(
                xs, ys,
                bins=[HIST_BINS_X, HIST_BINS_Y],
                range=[[-25, 25], [0, 47]],
            )
            # Transpose so shape is (y_bins, x_bins) — row = y, col = x
            self.historical_heatmaps[pid] = heatmap.T

        logger.info(
            "[HeatmapEngine] Built historical heatmaps for %d players.",
            len(self.historical_heatmaps),
        )
        return self.historical_heatmaps

    # ------------------------------------------------------------------
    # Task 9.3 — New court heatmap generation (combined-optimal config)
    # ------------------------------------------------------------------

    def generate_new_court_heatmaps(
        self,
        eligible_players,
        nn_model,
        combined_optimal_config: CourtConfig,
    ) -> Dict:
        """
        Evaluate the NN on a dense grid for every eligible player under the
        combined-optimal court config.

        Parameters
        ----------
        eligible_players : list
            List of player_ids.
        nn_model : NeuralNetworkModel
            Trained NN model.
        combined_optimal_config : CourtConfig

        Returns
        -------
        dict {player_id: np.ndarray of shape (n_y, n_x)}
        """
        xs = np.arange(-25, 25 + GRID_X_STEP, GRID_X_STEP)
        ys = np.arange(0, 47 + GRID_Y_STEP, GRID_Y_STEP)
        self._grid_xs = xs
        self._grid_ys = ys

        for pid in eligible_players:
            try:
                surface = nn_model.predict_grid(
                    player_id=pid,
                    x_step=GRID_X_STEP,
                    y_step=GRID_Y_STEP,
                )
                self.new_court_heatmaps[pid] = surface
            except Exception as exc:
                logger.warning(
                    "[HeatmapEngine] Could not generate new-court heatmap for player %s: %s",
                    pid, exc,
                )

        logger.info(
            "[HeatmapEngine] Generated new-court heatmaps for %d players.",
            len(self.new_court_heatmaps),
        )
        return self.new_court_heatmaps

    # ------------------------------------------------------------------
    # Task 9.4 — Zone reclassification per CourtConfig
    # ------------------------------------------------------------------

    def reclassify_shots(
        self,
        shots_df: pd.DataFrame,
        court_config: CourtConfig,
    ) -> pd.DataFrame:
        """
        Reclassify every shot's zone and point value under the given court config.

        Parameters
        ----------
        shots_df : pd.DataFrame
            Must contain columns: x, y.
        court_config : CourtConfig

        Returns
        -------
        pd.DataFrame with added columns: zone_<key>, point_value_<key>
        """
        key = f"{court_config.arc_radius}_{court_config.baseline_width}"
        zone_col = f"zone_{key}"
        pv_col = f"point_value_{key}"

        shots_df = shots_df.copy()
        shots_df[zone_col] = shots_df.apply(
            lambda r: classify_zone(float(r["x"]), float(r["y"]), court_config), axis=1
        )
        shots_df[pv_col] = shots_df[zone_col].apply(
            lambda z: zone_point_value(z, court_config)
        )
        return shots_df

    # ------------------------------------------------------------------
    # Task 9.2 — Rendering
    # ------------------------------------------------------------------

    def render_heatmap(
        self,
        heatmap_array: np.ndarray,
        title: str,
        court_config: Optional[CourtConfig] = None,
        ax: Optional[plt.Axes] = None,
        cmap: str = "hot",
        is_probability: bool = False,
    ) -> plt.Axes:
        """
        Render a single heatmap on a matplotlib Axes, with court lines overlaid.

        Parameters
        ----------
        heatmap_array : np.ndarray
            2D array of shape (n_y, n_x).
        title : str
        court_config : CourtConfig, optional
            If provided, draws the 3PT arc for this config.
        ax : plt.Axes, optional
            Axes to draw on; creates a new figure if None.
        cmap : str
        is_probability : bool
            If True, auto-scale colormap to actual prediction range.

        Returns
        -------
        plt.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 7))

        if is_probability:
            # Auto-scale to actual prediction range so small differences are visible
            vmin = float(np.nanmin(heatmap_array))
            vmax = float(np.nanmax(heatmap_array))
            # Add a small margin so the colorbar doesn't clip
            margin = (vmax - vmin) * 0.05
            vmin = max(0.0, vmin - margin)
            vmax = min(1.0, vmax + margin)
            norm = None
            im = ax.imshow(
                heatmap_array,
                origin="lower",
                extent=[-25, 25, 0, 47],
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
            )
            plt.colorbar(im, ax=ax, label="P(make)", fraction=0.03, pad=0.04)
        else:
            # Log-norm for shot density so arc shots aren't washed out by basket spike
            data = heatmap_array.copy().astype(float)
            data_min = data[data > 0].min() if (data > 0).any() else 1.0
            norm = LogNorm(vmin=data_min, vmax=data.max())
            im = ax.imshow(
                data,
                origin="lower",
                extent=[-25, 25, 0, 47],
                aspect="auto",
                cmap=cmap,
                norm=norm,
                interpolation="bilinear",
            )
            plt.colorbar(im, ax=ax, label="Shot count (log)", fraction=0.03, pad=0.04)

        self._draw_court_lines(ax, court_config)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (ft)")
        ax.set_ylabel("y (ft)")
        ax.set_xlim(-25, 25)
        ax.set_ylim(-6, 47)  # show baseline (y ≈ -5.25) and full court
        return ax

    def render_side_by_side(
        self,
        left_heatmap: np.ndarray,
        right_heatmap: np.ndarray,
        player_name: str,
        left_title: str,
        right_title: str,
        left_config: Optional[CourtConfig] = None,
        right_config: Optional[CourtConfig] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Render two heatmaps side by side in a single figure.

        Parameters
        ----------
        left_heatmap : np.ndarray
            Historical shot density (original court).
        right_heatmap : np.ndarray
            NN probability surface (new court).
        player_name : str
        left_title : str
        right_title : str
        left_config : CourtConfig, optional
        right_config : CourtConfig, optional
        save_path : str, optional
            If provided, save figure to this path.

        Returns
        -------
        plt.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(player_name, fontsize=13, fontweight="bold")

        self.render_heatmap(
            left_heatmap, left_title,
            court_config=left_config,
            ax=axes[0],
            cmap="YlOrRd",
            is_probability=False,
        )
        self.render_heatmap(
            right_heatmap, right_title,
            court_config=right_config,
            ax=axes[1],
            cmap="RdYlGn",
            is_probability=True,
        )

        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            logger.info("[HeatmapEngine] Saved heatmap to %s", save_path)

        return fig

    def render_all_players(
        self,
        eligible_players,
        combined_optimal_config: CourtConfig,
        output_dir: str = "output/heatmaps",
    ) -> None:
        """
        Render and save side-by-side heatmaps for every eligible player.

        Parameters
        ----------
        eligible_players : list
            List of player_ids.
        combined_optimal_config : CourtConfig
        output_dir : str
            Directory to save PNG files.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for pid in eligible_players:
            hist = self.historical_heatmaps.get(pid)
            new_court = self.new_court_heatmaps.get(pid)

            if hist is None or new_court is None:
                logger.warning(
                    "[HeatmapEngine] Missing heatmap data for player %s; skipping.", pid
                )
                continue

            name = self.player_names.get(pid, str(pid))
            safe_name = str(pid).replace(" ", "_").replace("/", "_")
            save_path = os.path.join(output_dir, f"{safe_name}_heatmap.png")

            fig = self.render_side_by_side(
                left_heatmap=hist,
                right_heatmap=new_court,
                player_name=name,
                left_title="Original Court\n(Historical Shot Locations)",
                right_title=(
                    f"Combined-Optimal Court\n"
                    f"(NN P(make) | arc={combined_optimal_config.arc_radius}ft, "
                    f"baseline={combined_optimal_config.baseline_width}ft)"
                ),
                left_config=None,
                right_config=combined_optimal_config,
                save_path=save_path,
            )
            plt.close(fig)

        logger.info(
            "[HeatmapEngine] Rendered heatmaps for %d players to '%s'.",
            len(eligible_players), output_dir,
        )

    # ------------------------------------------------------------------
    # Court line drawing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_court_lines(
        ax: plt.Axes,
        court_config: Optional[CourtConfig] = None,
    ) -> None:
        """
        Draw NBA court lines on the given Axes.

        The shot data uses basket-at-origin coordinates (y=0 at basket).
        Court lines are drawn in the same coordinate system.
        basket_to_baseline is the distance from basket to the baseline (y < 0).
        """
        arc_r = court_config.arc_radius if court_config else 23.75
        baseline_w = court_config.baseline_width if court_config else 50.0
        half_w = baseline_w / 2
        # basket_to_baseline: how far below y=0 the baseline sits
        btb = court_config.basket_to_baseline if court_config else 5.25

        color = "white"
        lw = 1.5

        # Basket at origin
        basket = Circle((0, 0), radius=0.75, color=color, fill=False, linewidth=lw)
        ax.add_patch(basket)

        # Paint (key) — 16 ft wide, 19 ft tall, bottom at -btb
        paint = mpatches.Rectangle((-8, -btb), 16, 19,
                                   fill=False, edgecolor=color, linewidth=lw)
        ax.add_patch(paint)

        # Free throw circle — center at top of paint (y = 19 - btb)
        ft_circle = Arc((0, 19 - btb), 16, 16,
                        angle=0, theta1=0, theta2=180,
                        color=color, linewidth=lw)
        ax.add_patch(ft_circle)

        # 3PT arc — centered at basket (0, 0)
        # Compute angle where arc meets the sideline at x = ±half_w
        if arc_r >= half_w:
            theta_side = np.degrees(np.arcsin(half_w / arc_r))
        else:
            theta_side = 90.0

        arc_patch = Arc(
            (0, 0), 2 * arc_r, 2 * arc_r,
            angle=90,
            theta1=theta_side,
            theta2=180 - theta_side,
            color=color, linewidth=lw,
        )
        ax.add_patch(arc_patch)

        # Corner 3 straight segments (if not eliminated)
        if court_config is None or not court_config.corner3_eliminated:
            y_arc_start = np.sqrt(max(arc_r ** 2 - half_w ** 2, 0)) if arc_r >= half_w else 0
            ax.plot([-half_w, -half_w], [-btb, y_arc_start], color=color, linewidth=lw)
            ax.plot([half_w, half_w], [-btb, y_arc_start], color=color, linewidth=lw)

        # Baseline
        ax.plot([-half_w, half_w], [-btb, -btb], color=color, linewidth=lw)
