# Design Document

## Overview

The NBA Court Re-Engineering Optimizer is a Python-based data pipeline and simulation system. It ingests historical Warriors/Cavs playoff shot data (2014–2024), trains three complementary ML models, and runs a grid search over 231 court dimension combinations to identify the arc radius and baseline width that maximizes 3-point shooting percentage for the Cavs, Warriors, and both teams combined.

---

## System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ Data_Loader │────▶│  Stat_Calculator │────▶│   Heatmap_Engine     │
└─────────────┘     └──────────────────┘     └──────────────────────┘
       │                     │                          │
       │                     ▼                          ▼
       │          ┌──────────────────────┐   ┌──────────────────────┐
       │          │ Neural_Network_Model │──▶│ Gradient_Boost_Model │
       │          │  + ShotLocationSampler   └──────────────────────┘
       │          └──────────────────────┘            │ make/miss
       │                     │ weighted (x,y)          │
       │                     ▼                         │
       │          ┌──────────────────────┐             │
       │          │   Fatigue_Model      │─────────────┤ p_make multiplier
       │          └──────────────────────┘             │
       │                                               ▼
       │                       ┌──────────────────────────┐
       │                       │        Simulator         │
       │                       └──────────────────────────┘
       │                                    │
       │                                    ▼
       └───────────────────────▶  ┌──────────────────────┐
                                  │       Optimizer       │
                                  └──────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| Data_Loader | CSV ingestion, column normalization, coordinate unit conversion, row filtering |
| Stat_Calculator | Per-player/team stats, zone percentages, attempt counts, defensive ratings, possessions/min |
| Heatmap_Engine | Historical shot density maps (log-normalized), NN probability surfaces (auto-scaled), side-by-side rendering |
| Neural_Network_Model | Spatial P(make \| x, y, player) — heatmap generation + KDE importance weighting |
| ShotLocationSampler | Per-player Gaussian KDE; precomputes 5,000 NN-weighted candidates per player before grid search |
| Gradient_Boost_Model | P(make \| player, zone, defense) — per-shot make/miss during simulation |
| Fatigue_Model | Logistic regression fatigue multiplier — minor p_make decay as elapsed game minutes increase |
| Simulator | N-game Monte Carlo simulation per CourtConfig using KDE/NN location sampling, GB make/miss, fatigue adjustment |
| Optimizer | Exhaustive grid search over 231 configs; ranks by Cavs-optimal, Warriors-optimal, combined-optimal |

---

## Data Pipeline Flow

```
1. Load CSVs
   warriors_cavs_playoff_shots_MASTER_2014_2024.csv  ──▶  DataLoader.load_primary()
   final_nn_input_full_grid.csv                      ──▶  DataLoader.load_secondary()

2. Column Normalization
   LOC_X → x, LOC_Y → y (converted from tenths-of-foot to feet if |x| > 50)
   PLAYER_ID → player_id, TEAM_ID → team_id, SHOT_MADE_FLAG → shot_made_flag
   PERIOD → period, MINUTES_REMAINING → minutes_remaining,
   SECONDS_REMAINING → seconds_remaining  (used by FatigueModel)

3. Filter Eligible Players
   Keep only rows where team_id ∈ {Warriors ID, Cavs ID}
   Post-filter: drop rows where team_id does not match Warriors or Cavs
   (prevents LeBron's Lakers rows from inflating Cavs usage)

4. Compute Base Stats (StatCalculator)
   Per player: PPG, FG%, 3PT%, 2PT%, total attempts/makes, usage_rate, estimated_minutes
   Per team: team PPG, FG%, 3PT%, possessions_per_minute (scaled to ~88 FGA/game)
   Defensive ratings: from secondary dataset (opp_3par_allowed proxy)

5. Compute Zone Stats (StatCalculator)
   Standard court (baseline):
     - Read zone labels from dataset column (pre-labeled per shot)
     - Count attempts per player per zone; compute zone_pct with sparse fallback
   Per new CourtConfig (grid search):
     - Reclassify geometrically using classify_zone(x, y, court_config)
     - Recount attempts and recompute zone_pct per player per zone

6. Train Models (once, before grid search)
   NeuralNetworkModel.train(shots: x, y, player_id, shot_made_flag)
   ShotLocationSampler.fit(shots: x, y, player_id)
   ShotLocationSampler.precompute_nn_weights(nn_model, pool_size=5000)  ← one-time cost
   GradientBoostModel.train(shots: zone, player_id, opp_def_rating, ...)
   FatigueModel.train(shots: period, minutes_remaining, seconds_remaining, player_id, shot_made_flag)

7. Grid Search (Optimizer)
   For arc_radius in [23.75, 24.00, ..., 26.25]:       # 11 values
     For baseline_width in [50.00, 50.25, ..., 55.00]: # 21 values
       config = CourtConfig(arc_radius, baseline_width)
       zone_stats = StatCalculator.compute_zone_stats(config)
       result = Simulator.run(config, n_games=25)
       Optimizer.store(config, result)

8. Output
   Rank by cavs_3pt_pct, warriors_3pt_pct, combined_3pt_pct → top 5 each
   Generate heatmaps for combined-optimal config
   Print per-player ePPG, 3PT%, 3PA/G
```

---

## Court Geometry Module

### Coordinate System

The basket is at the origin (0, 0). The court extends in the positive y direction toward half court. The x-axis runs along the baseline. `basket_to_baseline = 5.25 ft` is held constant across all configs.

```
  baseline  x = -baseline_width/2 ... +baseline_width/2   (y = -5.25)
  basket    (0, 0)
  arc       circle of radius arc_radius centered at (0, 0)
```

### Corner_3_Elimination_Condition

```python
def corner3_eliminated(arc_radius, baseline_width, basket_to_baseline=5.25):
    half_w = baseline_width / 2
    if arc_radius < half_w:
        return False  # arc never reaches sideline; corner 3 always exists
    y_at_sideline = sqrt(arc_radius**2 - half_w**2)
    return y_at_sideline <= basket_to_baseline
```

When True, all shots in the corner region are reclassified as 2-point attempts.

### Zone Taxonomy (14 Zones)

| Zone ID | Name | Distance | Angle Range | Point Value |
|---|---|---|---|---|
| Z01 | restricted_area | 0–4 ft | all | 2 |
| Z02 | paint_non_restricted | 4–8 ft | within paint | 2 |
| Z03 | left_short_corner_floater | 4–14 ft | < -60° | 2 |
| Z04 | right_short_corner_floater | 4–14 ft | > +60° | 2 |
| Z05 | left_elbow_mid_range | 14–arc_r | -60° to -30° | 2 |
| Z06 | right_elbow_mid_range | 14–arc_r | +30° to +60° | 2 |
| Z07 | center_mid_range | 14–arc_r | -30° to +30° | 2 |
| Z08 | left_mid_range_baseline | 14–arc_r | < -60° (above paint) | 2 |
| Z09 | right_mid_range_baseline | 14–arc_r | > +60° (above paint) | 2 |
| Z10 | left_corner_3 | ≥ arc_r | < -65° (corner) | 3 (or 2*) |
| Z11 | right_corner_3 | ≥ arc_r | > +65° (corner) | 3 (or 2*) |
| Z12 | left_wing_3 | ≥ arc_r | -65° to -20° | 3 |
| Z13 | right_wing_3 | ≥ arc_r | +20° to +65° | 3 |
| Z14 | top_of_arc_3 | ≥ arc_r | -20° to +20° | 3 |

*Z10/Z11 become 2-point zones when corner3_eliminated is True.

---

## Neural Network Model

### Architecture

```
Input:
  coords:    [x_norm, y_norm]     (normalized: x/25, y/47)
  player_id: integer index

Player Embedding: Embedding(n_players, 8) → Flatten

Concatenate → Dense(128, relu) → Dropout(0.3)
            → Dense(64,  relu) → Dropout(0.2)
            → Dense(32,  relu)
            → Dense(1, sigmoid) → P(make) ∈ [0, 1]

Loss: binary_crossentropy  |  Optimizer: Adam(lr=1e-3)
Early stopping: patience=10 on val_loss
```

### Training
- Data: all eligible player shots (x, y, player_id, shot_made_flag)
- Split: 80/20 stratified by player
- Batch size: 256, max epochs: 100
- Weights saved to: `output/models/nn_weights.weights.h5`

### Dual Role
1. **Heatmap generation**: evaluated on a dense 0.5ft grid per player to produce P(make) surface
2. **Simulation weighting**: scores 5,000 KDE candidates per player; weights cached before grid search

---

## Shot Location Sampler (KDE)

### KDE Fitting
- One Gaussian KDE per player fitted on historical (x, y) shot coordinates
- Bandwidth: Scott's rule — `h = n^(-1/6)` where n = number of shots

### Precomputation (one-time before grid search)
```python
for each player:
    candidates = kde.sample(5000)
    weights = nn_model.predict_batch(candidates)  # P(make) for each candidate
    weights = weights ** nn_temperature            # temperature=0.3 flattens NN bias
    weights = weights / weights.sum()              # normalize
    cache[player] = (candidates, weights)
```

### Simulation Sampling
Each shot draws from the cached pool — zero NN calls during the grid search. This is the primary performance optimization enabling the full 231-config search to complete in reasonable time.

---

## Gradient Boost Model

### Features

| Feature | Type | Description |
|---|---|---|
| player_id | categorical | encoded integer |
| zone_id | categorical | Z01–Z14 (reclassified per CourtConfig) |
| opp_defensive_rating | float | opponent team defensive rating |
| attempts_in_zone | int | player's historical attempts in this zone |
| zone_pct | float | resolved zone shooting % (with sparse fallback) |
| overall_2pt_pct | float | player's overall 2PT% |
| overall_3pt_pct | float | player's overall 3PT% |
| zone_point_value | int | 2 or 3 under current CourtConfig |

### Sparse Zone Logic
```python
def resolve_zone_pct(player, zone, court_config):
    attempts = player.zone_attempts[court_config][zone]
    if attempts >= 5:   return player.zone_pct[court_config][zone]
    elif attempts >= 1: return player.overall_3pt_pct if zone is 3PT else player.overall_2pt_pct
    else:               return None  # zone excluded from shot distribution
```

### Training
- XGBoost: n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8
- Split: 80/20 stratified by player
- Saved to: `output/models/gb_model.pkl`

---

## Fatigue Model

### Purpose
Applies a minor p_make multiplier that decays as elapsed game minutes increase. Maximum effect: 5% reduction at end of game (minute 48). This captures the real NBA phenomenon where shooting efficiency drops slightly in the 4th quarter.

### Method
Logistic regression trained on historical shot data using game clock features:

```python
elapsed_minutes = (period - 1) * 12 + (12 - minutes_remaining) + (60 - seconds_remaining) / 60
# Range: 0 (tip-off) → 48 (end of regulation)

Features: [elapsed_minutes, player_idx]
Target:   shot_made_flag
```

### Multiplier Calculation
```python
# Normalised ratio: make probability at time t vs. game start
multiplier = P(make | elapsed=t, player) / P(make | elapsed=0, player)
multiplier = clip(multiplier, 0.95, 1.05)  # max 5% reduction, allow tiny early-game boost
```

### Fallback
If training data is insufficient or columns are missing, a rule-based linear decay is used:
```python
multiplier = 1.0 - (FATIGUE_MAX_REDUCTION / 48.0) * elapsed_minutes
# = 1.0 at t=0, = 0.95 at t=48
```

### Integration in Simulator
```python
p_make = gb_model.predict_proba(...)
fatigue_mult = fatigue_model.get_multiplier(player_id, elapsed_minutes)
p_make = clip(p_make * fatigue_mult, 0.0, 1.0)
```

Saved to: `output/models/fatigue_model.pkl`

---

## Heatmap Engine

### Historical Heatmaps (left panel)
- 2D histogram of (x, y) shot attempts per player on the standard court
- Rendered with **log-normalized** color scale (`LogNorm`) so arc shots are visible alongside the basket hotspot
- Colormap: `YlOrRd`

### New Court Heatmaps (right panel)
- NN P(make) surface evaluated on a 0.5ft grid for the combined-optimal config
- Rendered with **auto-scaled** color range (actual prediction min/max, not hardcoded 0–1) so spatial gradients are visible
- Colormap: `RdYlGn` (green = high P(make), red = low)
- Colorbar shows actual P(make) range

### Court Line Drawing
- Basket at (0, 0); y-axis extends from -6 to 47 ft to show baseline below basket
- Paint: 16ft wide × 19ft tall, bottom at y = -basket_to_baseline
- Free throw circle: centered at y = 19 - basket_to_baseline
- 3PT arc: centered at basket, angle computed from arc_radius and half_width
- Corner segments drawn when corner3_eliminated = False

### Rendering
```python
render_side_by_side(
    left=historical_heatmap[player],   # log-normalized shot density
    right=new_court_heatmap[player],   # auto-scaled NN P(make) surface
    title=f"{player.name}"
)
```
Saved to: `output/heatmaps/{player_id}_heatmap.png`

---

## Simulator

### Shot Attempt Allocation
```python
expected_attempts = usage_rate * estimated_minutes * team_possessions_per_minute
n_attempts ~ Poisson(expected_attempts)
```
`team_possessions_per_minute` is scaled so total team FGA/game ≈ 88 (real NBA playoff pace).

### Known Playoff Minutes
Minutes are hardwired from StatMuse/Basketball Reference for all 20 players rather than estimated from the dataset. Examples: LeBron 41.3, Curry 37.2, Kyrie 37.4, Klay 36.2, Draymond 37.3.

### Game Clock Tracking
```python
MINUTES_PER_SHOT = 48.0 / 88.0  # ≈ 0.545 min/shot
elapsed_minutes += MINUTES_PER_SHOT  # advances after each shot
```
`elapsed_minutes` is passed to the FatigueModel to compute the per-shot multiplier.

### Shot Simulation Loop
```python
for game in range(N_GAMES):  # N_GAMES = 25
    elapsed_minutes = 0.0
    for player in team_roster:
        n_attempts = Poisson(usage_rate * minutes * ppm)
        for _ in range(n_attempts):
            x, y = shot_sampler.sample_weighted(player)   # from precomputed KDE/NN cache
            zone  = classify_zone(x, y, config)           # geometric reclassification
            fatigue_mult = fatigue_model.get_multiplier(player, elapsed_minutes)
            p_make = gb_model.predict_proba(...) * fatigue_mult
            made  = random() < p_make
            points = zone_point_value(zone, config) if made else 0
            elapsed_minutes += MINUTES_PER_SHOT
```

### ePPG Calculation
```python
ePPG[player] = total_simulated_points_across_all_games / N_GAMES
```
Point value per shot: 3 if zone is 3PT under current config, 2 if 2PT, 0 if missed.

---

## Grid Search (Optimizer)

### Search Space
```python
arc_radii       = [23.75 + 0.25*i for i in range(11)]   # 23.75 to 26.25 ft, 11 values
baseline_widths = [50.00 + 0.25*i for i in range(21)]   # 50.00 to 55.00 ft, 21 values
total_configs   = 11 * 21 = 231
```

### Three Optimization Objectives
```python
combined_3pt_pct = (cavs_3pt_pct + warriors_3pt_pct) / 2

cavs_top5     = top 5 configs by cavs_3pt_pct
warriors_top5 = top 5 configs by warriors_3pt_pct
combined_top5 = top 5 configs by combined_3pt_pct
```

### Result Storage
```python
results[(arc_r, bw)] = {
    "cavs_3pt_pct":       float,
    "warriors_3pt_pct":   float,
    "combined_3pt_pct":   float,
    "cavs_ppg":           float,
    "warriors_ppg":       float,
    "per_player":         {player_id: {ppg, three_pt_pct, three_pa_per_game}},
    "corner3_eliminated": bool,
}
```

---

## Key Design Decisions

- **Three ML models**: NN for spatial location weighting + heatmaps, XGBoost for make/miss prediction, logistic regression for fatigue adjustment. Each model handles what it does best.
- **KDE precomputation**: 5,000 candidates per player are scored by the NN once before the grid search. During simulation, each shot draws from this cache — zero NN calls during the 231-config search. This is the primary speed optimization.
- **Fatigue as a multiplier**: the fatigue model outputs a value in [0.95, 1.05] that scales `p_make`. It is intentionally minor — it adds realism without dominating the simulation outcome.
- **Log-normalized heatmaps**: the basket hotspot would otherwise crush the color scale, making arc shots invisible. LogNorm makes the full court visible.
- **Auto-scaled NN heatmap**: the NN predicts in a narrow range (e.g. 0.42–0.58). Hardcoding vmin=0, vmax=1 made the surface look flat. Auto-scaling to actual min/max reveals the spatial gradient.
- **Y-axis from -6 to 47**: the basket is at y=0 and the baseline is at y=-5.25. Extending the y-axis below 0 ensures the baseline and paint bottom are visible in heatmaps.
- **Usage rate normalization**: the dataset has ~60–63 FGA/game vs real ~88. A scale factor is applied to `possessions_per_minute` so simulated shot totals match real NBA pace.
- **Team ID post-filter**: LeBron's Lakers rows have `TEAM_NAME='Cavaliers'` in the CSV but `TEAM_ID` pointing to the Lakers. A post-filter on `team_id` (not team name) prevents these rows from inflating Cavs usage rates.
- **N_GAMES = 25**: balances statistical stability with compute time across 231 configs (5,775 total game simulations).
- **combined_3pt_pct = (cavs + warriors) / 2**: simple average; the combined-optimal config is the primary printed result.
- **Top 5 per objective**: all three rankings (Cavs, Warriors, combined) report top 5 configs with arc radius, baseline width, 3PT%, and corner-3 elimination status.
