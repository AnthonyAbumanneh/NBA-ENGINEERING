"""
Optimizer for the NBA Court Re-Engineering Optimizer.

Runs the 231-config grid search, stores per-config simulation results,
and ranks configurations by 3PT% for Cavs, Warriors, and combined.
"""

import logging
import time
from typing import Dict, List, Tuple

from config import ARC_RADII, BASELINE_WIDTHS
from src.court_geometry.court_geometry import CourtConfig

logger = logging.getLogger(__name__)


class Optimizer:
    """
    Grid search over all (arc_radius, baseline_width) combinations.

    Parameters
    ----------
    simulator : Simulator
        Configured simulator instance.
    stat_calculator : StatCalculator
        For computing zone stats per config.
    """

    def __init__(self, simulator, stat_calculator):
        self.simulator = simulator
        self.stat_calculator = stat_calculator
        # {(arc_radius, baseline_width): result_dict}
        self.results: Dict[Tuple[float, float], dict] = {}

    # ------------------------------------------------------------------
    # Task 12.1 — Grid enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def enumerate_configs() -> List[CourtConfig]:
        """
        Return all 231 CourtConfig combinations (11 arc × 21 baseline).

        arc_radii:       23.75, 24.00, ..., 26.00  (11 values)
        baseline_widths: 50.00, 50.25, ..., 55.00  (21 values)
        """
        configs = []
        for arc_r in ARC_RADII:
            for bw in BASELINE_WIDTHS:
                configs.append(CourtConfig(arc_radius=arc_r, baseline_width=bw))
        return configs

    # ------------------------------------------------------------------
    # Task 12.2 — Per-config execution loop
    # ------------------------------------------------------------------

    def run(self, configs: List[CourtConfig] = None) -> Dict:
        """
        Execute the grid search: simulate all configs and store results.

        Parameters
        ----------
        configs : list of CourtConfig, optional
            Defaults to all 231 configs from enumerate_configs().

        Returns
        -------
        dict keyed by (arc_radius, baseline_width)
        """
        if configs is None:
            configs = self.enumerate_configs()

        total = len(configs)
        logger.info("[Optimizer] Starting grid search over %d configs.", total)

        for i, config in enumerate(configs, 1):
            key = (config.arc_radius, config.baseline_width)
            t0 = time.time()

            # Compute zone stats for this config
            self.stat_calculator.compute_zone_stats(config)

            # Run simulation
            result = self.simulator.run(config)

            # Store
            self.store(config, result)

            elapsed = time.time() - t0
            logger.info(
                "[Optimizer] Config %d/%d done in %.1fs: arc=%.2f, baseline=%.2f, "
                "combined_3pt=%.3f, corner3_elim=%s",
                i, total, elapsed, config.arc_radius, config.baseline_width,
                result.get("combined_3pt_pct", 0.0), config.corner3_eliminated,
            )

        logger.info("[Optimizer] Grid search complete. %d results stored.", len(self.results))
        return self.results

    # ------------------------------------------------------------------
    # Task 12.3 — Result storage
    # ------------------------------------------------------------------

    def store(self, config: CourtConfig, result: dict) -> None:
        """
        Store a simulation result for the given config.

        Ensures combined_3pt_pct = (cavs_3pt_pct + warriors_3pt_pct) / 2.
        """
        key = (config.arc_radius, config.baseline_width)
        cavs = result.get("cavs_3pt_pct", 0.0)
        warriors = result.get("warriors_3pt_pct", 0.0)
        result["combined_3pt_pct"] = (cavs + warriors) / 2
        self.results[key] = result

    # ------------------------------------------------------------------
    # Task 12.4 — top5_with_ties ranking
    # ------------------------------------------------------------------

    @staticmethod
    def top5_with_ties(
        results: Dict[Tuple[float, float], dict],
        key: str,
    ) -> List[Tuple[int, Tuple[float, float], dict]]:
        """
        Rank configs descending by `key`, including all ties at each rank
        position up to rank 5.

        Parameters
        ----------
        results : dict
            {(arc_radius, baseline_width): result_dict}
        key : str
            Result field to rank by, e.g. 'cavs_3pt_pct'.

        Returns
        -------
        list of (rank, (arc_radius, baseline_width), result_dict)
        """
        sorted_cfgs = sorted(
            results.items(),
            key=lambda kv: kv[1].get(key, 0.0),
            reverse=True,
        )

        top5 = []
        rank = 0
        prev_val = None

        for cfg, r in sorted_cfgs:
            val = r.get(key, 0.0)
            if val != prev_val:
                rank += 1
                if rank > 5:
                    break
                prev_val = val
            top5.append((rank, cfg, r))

        return top5

    def rank_all(self) -> dict:
        """
        Produce top-5 rankings for Cavs, Warriors, and combined.

        Returns
        -------
        dict with keys: cavs_top5, warriors_top5, combined_top5,
                        combined_optimal_cfg, combined_optimal_result
        """
        cavs_top5 = self.top5_with_ties(self.results, "cavs_3pt_pct")
        warriors_top5 = self.top5_with_ties(self.results, "warriors_3pt_pct")
        combined_top5 = self.top5_with_ties(self.results, "combined_3pt_pct")

        combined_optimal_cfg = combined_top5[0][1] if combined_top5 else None
        combined_optimal_result = combined_top5[0][2] if combined_top5 else {}

        return {
            "cavs_top5": cavs_top5,
            "warriors_top5": warriors_top5,
            "combined_top5": combined_top5,
            "combined_optimal_cfg": combined_optimal_cfg,
            "combined_optimal_result": combined_optimal_result,
        }
