"""
NBA Court Re-Engineering Optimizer — main entry point.

Pipeline:
  1. Load data
  2. Filter eligible players, compute stats
  3. Train NN + GB models; fit per-player KDEs
  4. Run 231-config grid search (simulate 100 games per config)
  5. Identify combined-optimal config; generate heatmaps
  6. Print results
"""

import logging
import os
import time
start_time = time.time()


from config import (
    PRIMARY_DATA_PATH,
    SECONDARY_DATA_PATH,
    STANDARD_ARC_RADIUS,
    STANDARD_BASELINE_WIDTH,
    N_GAMES,
    KDE_CANDIDATES,
    NN_TEMPERATURE,
    NN_BATCH_SIZE,
    NN_MAX_EPOCHS,
    NN_PATIENCE,
    TRAIN_VAL_SPLIT,
)
from src.data_loader.data_loader import DataLoader
from src.stat_calculator.stat_calculator import StatCalculator
from src.court_geometry.court_geometry import CourtConfig
from src.models.neural_network import NeuralNetworkModel, ShotLocationSampler
from src.models.gradient_boost import GradientBoostModel
from src.heatmap_engine.heatmap_engine import HeatmapEngine
from src.simulator.simulator import Simulator
from src.optimizer.optimizer import Optimizer
from src.output_formatter import print_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading datasets...")
    loader = DataLoader(PRIMARY_DATA_PATH, SECONDARY_DATA_PATH)
    primary_df = loader.load_primary()
    secondary_df = loader.load_secondary()

    # ------------------------------------------------------------------
    # 2. Eligible players + stats
    # ------------------------------------------------------------------
    logger.info("Computing player and team stats...")
    stat_calc = StatCalculator(primary_df)
    stat_calc.compute_player_stats()
    stat_calc.compute_team_stats()
    stat_calc.compute_defensive_ratings(secondary_df)
    stat_calc.compute_baseline_zone_stats()

    # Historical baseline 3PT% (from raw data, before any simulation)
    warriors_id = stat_calc.warriors_team_id
    cavs_id = stat_calc.cavs_team_id
    hist_warriors_3pt = stat_calc.team_stats.get(warriors_id, {}).get("three_pt_pct", 0.0)
    hist_cavs_3pt = stat_calc.team_stats.get(cavs_id, {}).get("three_pt_pct", 0.0)

    # ------------------------------------------------------------------
    # 2b. Print per-player usage rates (before model training)
    # ------------------------------------------------------------------
    print("\n" + "="*75)
    print("PLAYER USAGE RATES & SHOT ALLOCATION")
    print("="*75)
    print(f"{'Player':<28} {'Team':<10} {'Usage%':>7} {'Min/G':>6} {'PPM':>6} {'Exp FGA/G':>10} {'Hist 3PA/G':>11}")
    print("-"*75)
    for pid in stat_calc.eligible_players:
        pdf = stat_calc.player_dfs.get(pid)
        if pdf is None or pdf.empty:
            continue
        stats = stat_calc.player_stats.get(pid, {})
        team_id = pdf["team_id"].mode().iloc[0]
        ppm = stat_calc.team_possessions_per_minute.get(team_id, 0.0)
        usage = stats.get("usage_rate", 0.0)
        mins = stats.get("estimated_minutes", 0.0)
        exp_fga = usage * mins * ppm
        three_pa = stats.get("three_pt_attempts", 0)
        game_col = next((c for c in ["game_id", "game_date"] if c in pdf.columns), None)
        n_games_hist = pdf[game_col].nunique() if game_col else 1
        hist_3pa_per_game = three_pa / n_games_hist if n_games_hist > 0 else 0.0
        name_col = next((c for c in ["player_name", "player", "name"] if c in pdf.columns), None)
        pname = str(pdf[name_col].iloc[0]) if name_col else str(pid)
        team_label = "Warriors" if team_id == warriors_id else "Cavs"
        print(f"{pname:<28} {team_label:<10} {usage:>7.1%} {mins:>6.1f} {ppm:>6.3f} {exp_fga:>10.1f} {hist_3pa_per_game:>11.1f}")
    print("="*75 + "\n")

    # ------------------------------------------------------------------
    # 3. Train models
    # ------------------------------------------------------------------
    shots_df = primary_df  # all eligible shots already filtered in stat_calc

    logger.info("Training Neural Network model...")
    n_players = len(stat_calc.eligible_players)
    nn_model = NeuralNetworkModel(n_players=n_players)
    nn_model.train(
        shots_df=shots_df,
        val_split=TRAIN_VAL_SPLIT,
        batch_size=NN_BATCH_SIZE,
        max_epochs=NN_MAX_EPOCHS,
        patience=NN_PATIENCE,
        save_path=os.path.join("output", "models", "nn_weights.weights.h5"),
    )

    logger.info("Fitting per-player KDEs...")
    shot_sampler = ShotLocationSampler(nn_temperature=NN_TEMPERATURE)
    shot_sampler.fit(shots_df)
    logger.info("Precomputing NN weights for simulation (one-time cost)...")
    shot_sampler.precompute_nn_weights(nn_model, pool_size=5000)

    logger.info("Training Gradient Boost model...")
    standard_config = CourtConfig(
        arc_radius=STANDARD_ARC_RADIUS,
        baseline_width=STANDARD_BASELINE_WIDTH,
    )
    gb_model = GradientBoostModel()
    gb_model.train(
        shots_df=shots_df,
        stat_calculator=stat_calc,
        standard_config=standard_config,
        defensive_ratings=stat_calc.defensive_ratings,
        val_split=TRAIN_VAL_SPLIT,
        save_path=os.path.join("output", "models", "gb_model.pkl"),
    )

    # ------------------------------------------------------------------
    # 4. Grid search
    # ------------------------------------------------------------------
    logger.info("Starting grid search (231 configs × %d games)...", N_GAMES)
    simulator = Simulator(
        stat_calculator=stat_calc,
        nn_model=nn_model,
        gb_model=gb_model,
        shot_sampler=shot_sampler,
        defensive_ratings=stat_calc.defensive_ratings,
        warriors_team_id=warriors_id,
        cavs_team_id=cavs_id,
        n_games=N_GAMES,
        kde_candidates=KDE_CANDIDATES,
    )

    optimizer = Optimizer(simulator=simulator, stat_calculator=stat_calc)
    optimizer.run()
    rankings = optimizer.rank_all()

    combined_optimal_cfg_key = rankings["combined_optimal_cfg"]
    combined_optimal_result = rankings["combined_optimal_result"]

    # ------------------------------------------------------------------
    # 5. Heatmaps
    # ------------------------------------------------------------------
    logger.info("Generating heatmaps...")
    # Build player name lookup if available
    player_names = {}
    for col in ["player_name", "player", "name"]:
        if col in primary_df.columns:
            for pid in stat_calc.eligible_players:
                pdf = stat_calc.player_dfs.get(pid)
                if pdf is not None and not pdf.empty and col in pdf.columns:
                    player_names[pid] = str(pdf[col].iloc[0])
            break

    heatmap_engine = HeatmapEngine(player_names=player_names)
    heatmap_engine.build_historical_heatmaps(shots_df)

    if combined_optimal_cfg_key:
        arc_r, bw = combined_optimal_cfg_key
        combined_optimal_config = CourtConfig(arc_radius=arc_r, baseline_width=bw)
        heatmap_engine.generate_new_court_heatmaps(
            eligible_players=stat_calc.eligible_players,
            nn_model=nn_model,
            combined_optimal_config=combined_optimal_config,
        )
        heatmap_engine.render_all_players(
            eligible_players=stat_calc.eligible_players,
            combined_optimal_config=combined_optimal_config,
            output_dir=os.path.join("output", "heatmaps"),
        )
    else:
        combined_optimal_config = None

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------
    print_results(
        historical_warriors_3pt=hist_warriors_3pt,
        historical_cavs_3pt=hist_cavs_3pt,
        cavs_top5=rankings["cavs_top5"],
        warriors_top5=rankings["warriors_top5"],
        combined_top5=rankings["combined_top5"],
        combined_optimal_cfg=combined_optimal_cfg_key,
        combined_optimal_result=combined_optimal_result,
        player_names=player_names,
    )

    logger.info("Done. Heatmaps saved to output/heatmaps/")


if __name__ == "__main__":
    main()
    
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")