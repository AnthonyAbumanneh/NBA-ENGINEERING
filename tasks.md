# Implementation Plan: NBA Court Re-Engineering Optimizer

## Overview

Python-based pipeline: ingest shot data → filter eligible players → compute stats → train NN + GB models → run 231-config grid search with 100-game simulation per config → rank and output optimal court dimensions.

## Tasks

- [-] 1. Project setup and data ingestion
  - [x] 1.1 Create project structure and configuration
    - Create `src/` package with `__init__.py` files for each module: `data_loader`, `stat_calculator`, `court_geometry`, `heatmap_engine`, `models`, `simulator`, `optimizer`
    - Create `config.py` with configurable file paths for `warriors_cavs_2014.csv` and `nn_dataset.csv`, and constants: `BASKET_TO_BASELINE = 5.25`, `ARC_RADII`, `BASELINE_WIDTHS`
    - Create `requirements.txt` with: pandas, numpy, matplotlib, xgboost or lightgbm, scikit-learn, tensorflow or torch
    - _Requirements: 1.1, 1.2_

  - [x] 1.2 Implement `Data_Loader.load_primary()` and `Data_Loader.load_secondary()`
    - Read `warriors_cavs_2014.csv` and `nn_dataset.csv` from configurable paths
    - Raise a descriptive `FileNotFoundError` when a file is missing or unreadable
    - Log a warning and drop rows with missing values in required columns: shot coordinates (x, y), player_id, team_id, shot_result
    - Expose loaded data as pandas DataFrames accessible to all downstream components
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [~] 1.3 Write unit tests for Data_Loader
    - Test missing file raises descriptive error
    - Test rows with missing required columns are dropped with a warning logged
    - Test clean data loads without modification
    - _Requirements: 1.3, 1.4_

- [-] 2. Eligible player filtering and base stat calculation
  - [x] 2.1 Implement eligible player filtering in `Stat_Calculator`
    - Identify Warriors and Cavs rosters from `warriors_cavs_2014` dataset by `team_id`
    - For players appearing on multiple teams, retain only rows where `team_id` is Warriors or Cavs
    - Expose `eligible_players` list and per-player filtered shot DataFrames
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 2.2 Implement per-player base stat computation
    - Compute per `Eligible_Player`: PPG, FG%, 3PT%, 2PT%, total attempts, 3PT makes/attempts, 2PT makes/attempts
    - Compute `Usage_Rate` as player's share of team possessions while on floor
    - Compute `Estimated_Minutes` from historical average minutes in `warriors_cavs_2014`
    - _Requirements: 3.1, 3.3, 3.4_

  - [x] 2.3 Implement per-team stat computation and defensive ratings
    - Compute per team (Warriors, Cavs): team PPG, FG%, 3PT%, total attempts, total 3PT attempts
    - Compute team defensive ratings for Warriors and Cavs from `NN_Dataset`
    - Derive `team_possessions_per_minute` from historical pace in `warriors_cavs_2014`
    - _Requirements: 3.2, 3.5_

  - [~] 2.4 Write unit tests for Stat_Calculator base stats
    - Test multi-team player filtering keeps only Warriors/Cavs rows
    - Test PPG, FG%, 3PT% calculations against known values
    - Test defensive rating derivation from NN_Dataset
    - _Requirements: 2.2, 3.1, 3.5_

- [ ] 3. Court geometry module
  - [~] 3.1 Implement `CourtConfig` dataclass and zone taxonomy
    - Create `CourtConfig(arc_radius, baseline_width)` dataclass with `basket_to_baseline = 5.25` constant
    - Define the 14-zone taxonomy (Z01–Z14) as named constants with distance ranges, angle ranges, and default point values
    - Implement `zone_point_value(zone_id, court_config)` returning 2 or 3, accounting for Corner_3 elimination
    - _Requirements: 5.1, 5.2, 6.1_

  - [~] 3.2 Implement `corner3_eliminated()` geometric derivation
    - Implement `corner3_eliminated(arc_radius, baseline_width, basket_to_baseline=5.25) -> bool`
    - Compute `half_w = baseline_width / 2`; if `arc_radius < half_w` return `False`
    - Compute `y_at_sideline = sqrt(arc_radius² - half_w²)`; return `y_at_sideline <= basket_to_baseline`
    - No hardcoded lookup tables; condition must be derived analytically for every config
    - _Requirements: 6.1, 6.2, 6.3, 6.5_

  - [~] 3.3 Implement zone boundary derivation per `CourtConfig`
    - Implement `classify_zone(x, y, court_config) -> zone_id` using angular and distance thresholds from the zone taxonomy
    - Angles measured from positive y-axis (straight ahead); basket at origin (0, 0)
    - Apply Corner_3 reclassification: when `corner3_eliminated` is True, shots in Z10/Z11 region return point value 2
    - Implement `derive_zone_boundaries(court_config)` returning per-zone boundary parameters for the given arc radius and baseline width
    - _Requirements: 5.1, 5.2, 5.3, 6.4_

  - [~] 3.4 Write property test for Corner_3_Elimination_Condition
    - **Property 1: Geometric monotonicity — for fixed baseline_width, corner3_eliminated transitions from False to True as arc_radius increases and never reverts**
    - **Validates: Requirements 6.1, 6.2, 6.3**

  - [~] 3.5 Write unit tests for court geometry
    - Test `corner3_eliminated` against known standard court values (arc=23.75, baseline=50 → False; large arc → True)
    - Test `classify_zone` places shots in correct zones for boundary coordinates
    - Test `zone_point_value` returns 2 for Z10/Z11 when corner is eliminated
    - _Requirements: 6.1, 6.2, 5.3_

- [~] 4. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Per-player per-zone stats and sparse zone logic
  - [~] 5.1 Implement `Stat_Calculator.compute_baseline_zone_stats()`
    - Read zone labels directly from the `warriors_cavs_2014` dataset (pre-labeled per shot row)
    - For each `Eligible_Player`, count `zone_attempts[player][zone]` and compute `zone_pct[player][zone] = makes / attempts` using the dataset's zone label column
    - Store as the standard court baseline — used for GB model training and sparse zone fallback
    - _Requirements: 3.6, 8.5_

  - [~] 5.2 Implement `Stat_Calculator.compute_zone_stats(court_config)` for new configs
    - For each `Eligible_Player`, reclassify all shots geometrically using `classify_zone(x, y, court_config)` — pre-labeled zones are invalid for non-standard court geometry
    - Count `zone_attempts[player][config][zone]` and compute `zone_pct[player][config][zone]`
    - Store attempt counts and percentages for use by GB model and Simulator during grid search
    - _Requirements: 3.6, 8.5_

  - [~] 5.2 Implement `resolve_zone_pct(player, zone, court_config)` sparse zone fallback
    - If `attempts >= 5`: return `player.zone_pct[config][zone]`
    - If `1 <= attempts <= 4`: return `overall_3pt_pct` if zone is 3PT under current config, else `overall_2pt_pct`
    - If `attempts == 0`: return `None` (zone excluded from shot distribution)
    - Use current `Court_Config` zone classification for fallback, not original classification
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [~] 5.3 Write property test for sparse zone logic
    - **Property 2: Fallback consistency — when Corner_3 is eliminated and attempts are 1–4, resolve_zone_pct returns overall_2pt_pct (not overall_3pt_pct)**
    - **Validates: Requirements 8.4**

  - [~] 5.4 Write unit tests for zone stats and sparse logic
    - Test zone attempt counts sum to total player attempts
    - Test sparse fallback returns correct overall stat for each threshold band
    - Test zones with 0 attempts return None
    - _Requirements: 8.1, 8.2, 8.3_

- [ ] 6. Neural Network model
  - [~] 6.1 Implement `Neural_Network_Model` architecture
    - Input: `[x_norm, y_norm, player_embedding]` where `player_id` → `Embedding(n_players, 8)`
    - Hidden layers: `Dense(128, relu)` → `Dropout(0.3)` → `Dense(64, relu)` → `Dropout(0.2)` → `Dense(32, relu)`
    - Output: `Dense(1, sigmoid)` → P(make)
    - Compile with `binary_crossentropy` loss and `Adam(lr=1e-3)`
    - _Requirements: 7.1, 7.2_

  - [ ] 6.2 Implement NN training pipeline
    - Build training dataset from all `Eligible_Player` shots: features `(x_norm, y_norm, player_id)`, label `made`
    - 80/20 stratified train/validation split by player
    - Train with `EarlyStopping(patience=10)` on validation loss, batch size 256, max 100 epochs
    - Save trained model weights to disk
    - _Requirements: 7.6, 7.8_

  - [ ] 6.3 Implement NN inference for heatmap generation
    - Implement `predict(x, y, player_id) -> float` returning P(make) for a single coordinate
    - Evaluate model on dense grid `x ∈ [-25, 25] step 0.5`, `y ∈ [0, 47] step 0.5` per player
    - Return 2D probability surface array per player
    - _Requirements: 7.2_

  - [ ] 6.4 Fit per-player KDE for shot location sampling
    - For each `Eligible_Player`, fit a `sklearn.neighbors.KernelDensity` on their historical `(x, y)` shot locations from `warriors_cavs_2014`
    - Store as `player.shot_location_kde`; use bandwidth selection via cross-validation or Scott's rule
    - Expose `kde.sample(n)` and `kde.score_samples(coords)` for use by the Simulator's Shot_Location_Sampler
    - _Requirements: 7.4, 10.3_

  - [ ] 6.5 Write unit tests for Neural Network model
    - Test model output is in [0, 1] for arbitrary inputs
    - Test training loop runs without error on a small synthetic dataset
    - Test inference grid produces correct shape output
    - Test KDE `sample()` returns coordinates within the historical shot bounding box
    - _Requirements: 7.1, 7.2, 7.9_

- [ ] 7. Gradient Boost model
  - [ ] 7.1 Implement GB feature engineering
    - Build feature vector per shot: `player_id` (encoded int), `zone_id` (Z01–Z14), `opp_defensive_rating`, `attempts_in_zone`, `zone_pct` (resolved via sparse logic), `overall_2pt_pct`, `overall_3pt_pct`, `zone_point_value`
    - Apply `resolve_zone_pct` to populate `zone_pct` feature; exclude shots from zones with `None` (0 attempts)
    - Use reclassified zone assignments from current `Court_Config` at inference time
    - _Requirements: 7.3, 7.10, 8.1, 8.2_

  - [ ] 7.2 Implement GB model training
    - Train XGBoost or LightGBM (configurable) on `Eligible_Player` shots with engineered features
    - 80/20 stratified train/validation split by player
    - Hyperparameters: `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`
    - Early stopping on validation log-loss; save trained model to disk
    - _Requirements: 7.4, 7.5, 7.7, 7.9_

  - [ ] 7.3 Implement GB inference
    - Implement `predict_proba(player, zone, court_config, opp_defense) -> float`
    - Build feature vector using current `Court_Config` zone classification and sparse zone fallback
    - Return P(make) ∈ [0, 1]
    - _Requirements: 7.5, 7.10_

  - [ ] 7.4 Write property test for Gradient Boost model
    - **Property 3: Output bounds — predict_proba always returns a value in [0, 1] for any valid player/zone/config/defense combination**
    - **Validates: Requirements 7.5**

  - [ ] 7.5 Write unit tests for Gradient Boost model
    - Test feature vector construction includes all 8 required features
    - Test sparse zone fallback is applied correctly in feature construction
    - Test training runs without error on synthetic data
    - _Requirements: 7.3, 7.4, 8.1, 8.2_

- [ ] 8. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Heatmap Engine
  - [ ] 9.1 Implement historical heatmap generation
    - For each `Eligible_Player`, bin all shot attempts from `warriors_cavs_2014` into a 2D histogram of `(x, y)` cells on the standard NBA court
    - Store as `historical_heatmap[player_id]` — raw density, no model inference
    - _Requirements: 4.1, 4.3_

  - [ ] 9.2 Implement heatmap rendering and side-by-side display
    - Implement `render_heatmap(heatmap_array, title, court_config)` using matplotlib, drawing court lines for the given config
    - Implement `render_side_by_side(left_heatmap, right_heatmap, player_name, left_title, right_title)` placing both plots in a single figure
    - Render and print historical heatmap for every `Eligible_Player`
    - _Requirements: 4.2, 4.7_

  - [ ] 9.3 Implement new court heatmap generation for combined-optimal config
    - After grid search completes, accept `combined_optimal_config` as input
    - For each `Eligible_Player`, evaluate `NN_Model.predict(x, y, player_id)` on dense grid `x ∈ [-25, 25] step 0.5`, `y ∈ [0, 47] step 0.5`
    - Store as `new_court_heatmap[player_id]`
    - Call `render_side_by_side(historical_heatmap[p], new_court_heatmap[p], ...)` for every player
    - _Requirements: 4.6, 4.7_

  - [ ] 9.4 Implement zone reclassification and shot value update per Court_Config
    - For each shot in each player's history, compute `shot.zone[config] = classify_zone(x, y, config)` and `shot.point_value[config] = zone_point_value(zone, config)`
    - Apply Corner_3 reclassification when `corner3_eliminated` is True
    - _Requirements: 4.4, 4.5, 4.8, 5.1, 5.2, 5.3_

  - [ ] 9.5 Write unit tests for Heatmap Engine
    - Test historical heatmap bins sum to total player shot count
    - Test zone reclassification flips Z10/Z11 to 2PT when corner is eliminated
    - Test side-by-side render produces a figure with two subplots
    - _Requirements: 4.1, 4.8, 4.7_

- [ ] 10. Game Simulator
  - [ ] 10.1 Implement shot attempt allocation
    - Compute `player_attempts_per_game = usage_rate * estimated_minutes * team_possessions_per_minute` per player per config
    - Sample integer attempt count per game using this rate
    - _Requirements: 10.2_

  - [ ] 10.2 Implement NN-weighted KDE shot location sampling
    - Implement `Shot_Location_Sampler.sample(player, court_config, nn_model) -> (x, y, zone)`
    - Draw `n=100` candidate coordinates from `player.shot_location_kde.sample(n=100)`
    - Query `nn_model.predict(x, y, player.id)` for each candidate to get importance weights
    - Select one `(x, y)` via `weighted_sample(candidates, nn_weights)`
    - Classify to zone: `zone = classify_zone(x, y, court_config)`
    - Apply sparse zone check: while `player.zone_attempts[config][zone] == 0`, resample
    - _Requirements: 7.3, 7.4, 7.5, 10.3, 10.4_

  - [ ] 10.3 Write property test for NN-weighted shot location sampling
    - **Property 4a: High-NN-probability bias — NN-weighted sampling produces higher shot attempt rates from high-NN-probability zones compared to uniform KDE sampling (without NN weights)**
    - **Validates: Requirements 7.3, 7.5, 10.3**

  - [ ] 10.4 Implement shot outcome simulation
    - Call `gb_model.predict_proba(player, zone, config, opp_defense)` → `p_make`
    - Draw `made = random.random() < p_make`
    - Compute `points = zone_point_value(zone, config) if made else 0`
    - Apply ePPG classification: 2-point or 3-point based on zone under current config
    - _Requirements: 10.5, 10.6, 5.4_

  - [ ] 10.5 Implement per-game tracking and 100-game simulation loop
    - Track per player per game: total points, total attempts, 3PT attempts, 3PT makes
    - Track per team per game: same aggregates
    - Run outer loop `for game in range(100)` over all players in both rosters
    - Aggregate PPG and 3PT% as averages over 100 games
    - _Requirements: 10.1, 10.8, 11.1, 11.2_

  - [ ] 10.6 Write property test for Simulator zone exclusion
    - **Property 4b: Zone exclusion — a player never attempts a shot from a zone with 0 historical attempts across all 100 simulated games**
    - **Validates: Requirements 8.3, 10.4**

  - [ ] 10.7 Write unit tests for Simulator
    - Test attempt allocation is proportional to usage_rate and estimated_minutes
    - Test NN-weighted sampling never selects a zero-weight zone
    - Test 3PT% aggregation is correct given known made/attempt counts
    - _Requirements: 10.2, 10.8_

- [ ] 11. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Grid search and Optimizer
  - [ ] 12.1 Implement grid search enumeration
    - Generate `arc_radii = [23.75 + 0.25*i for i in range(11)]` (11 values: 23.75–26.00)
    - Generate `baseline_widths = [50.00 + 0.25*i for i in range(21)]` (21 values: 50.00–55.00)
    - Enumerate all 231 `CourtConfig` combinations
    - _Requirements: 9.1, 9.2, 9.3_

  - [ ] 12.2 Implement per-config execution loop
    - For each config: compute `corner3_eliminated`, call `stat_calculator.compute_zone_stats(config)`, call `simulator.run(config, zone_stats, n_games=100)`, store result
    - Pass `corner3_eliminated` flag to both `Heatmap_Engine` and `Simulator` for consistent reclassification
    - _Requirements: 9.3, 9.4, 6.4_

  - [ ] 12.3 Implement result storage
    - Store per-config results in a dict keyed by `(arc_radius, baseline_width)` with fields: `cavs_3pt_pct`, `warriors_3pt_pct`, `combined_3pt_pct`, `cavs_ppg`, `warriors_ppg`, `per_player`, `corner3_eliminated`
    - Compute `combined_3pt_pct = (cavs_3pt_pct + warriors_3pt_pct) / 2` for every config
    - _Requirements: 11.3, 12.4_

  - [ ] 12.4 Implement `top5_with_ties` ranking
    - Implement `top5_with_ties(results, key)` that sorts configs descending by `key`, assigns ranks, and includes all tied configs at each rank position up to rank 5
    - Produce `cavs_top5`, `warriors_top5`, `combined_top5` lists
    - Identify `combined_optimal_cfg` as the rank-1 entry from `combined_top5`
    - _Requirements: 12.1, 12.2, 12.3, 12.5, 12.7_

  - [ ] 12.5 Write property test for Optimizer ranking
    - **Property 5: Tie completeness — if N configs share the same 3PT% at rank K, all N are included in the output and none are omitted**
    - **Validates: Requirements 12.7**

  - [ ] 12.6 Write unit tests for Optimizer
    - Test grid produces exactly 231 configs (11 × 21)
    - Test `top5_with_ties` includes all tied entries at each rank
    - Test `combined_3pt_pct` is correctly computed as simple average
    - _Requirements: 9.1, 9.2, 12.4, 12.7_

- [ ] 13. Output formatting and final integration
  - [ ] 13.1 Implement final output formatter
    - Print `=== Historical Baseline (Standard NBA Court) ===` block showing Warriors and Cavs historical 3PT% computed from the warriors_cavs_2014 data
    - Print `=== Optimal Court Configurations ===` header
    - For each of `cavs_top5`, `warriors_top5`, `combined_top5`: print ranked entries with arc radius, baseline width, relevant 3PT%, and corner-3 elimination status
    - Print `=== Best Court (Combined-Optimal #1) ===` block with full detail: arc radius, baseline width, Cavs 3PT%, Warriors 3PT%, combined 3PT%, corner-3 status
    - _Requirements: 12.6, 12.8_

  - [ ] 13.2 Implement per-player stats output per config
    - For the combined-optimal config, print per-player: PPG, 3PT%, 3PA per game
    - _Requirements: 11.2_

  - [ ] 13.3 Wire all components into `main.py` end-to-end pipeline
    - Instantiate `Data_Loader` → load both datasets
    - Instantiate `Stat_Calculator` → filter eligible players, compute base stats and defensive ratings
    - Train `Neural_Network_Model` and `Gradient_Boost_Model` (once, on standard court)
    - Fit per-player KDE (`Shot_Location_Sampler`) on historical `(x, y)` shot locations
    - Run grid search loop: for each of 231 configs, compute zone stats, run simulator (using NN-weighted KDE sampling + GB make/miss), store results
    - After grid search: identify combined-optimal config, generate and render all heatmaps (historical + new court side-by-side for every eligible player)
    - Call output formatter to print top-5 rankings and combined-optimal detail
    - _Requirements: all_

  - [ ] 13.4 Write integration test for end-to-end pipeline
    - Run pipeline on a small synthetic dataset (5 players, 2 configs)
    - Assert output contains top-5 rankings for Cavs, Warriors, combined
    - Assert combined-optimal config is printed with all required fields
    - Assert heatmap render is called for every eligible player
    - _Requirements: 12.6, 12.8, 4.2, 4.6_

- [ ] 14. Final checkpoint — Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The Neural_Network_Model is trained once and serves two roles: heatmap generation (P(make) surface) and shot location sampling weights during simulation
- Per-player KDEs are fitted once after model training and reused across all 231 grid search configs
- Models are trained once on the standard court; zone reclassification is applied at inference/feature-construction time, not by retraining
- Heatmaps for non-combined-optimal configs are generated internally but not rendered or printed
- Property tests validate universal correctness invariants; unit tests validate specific examples and edge cases
