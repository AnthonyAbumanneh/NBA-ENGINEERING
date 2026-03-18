# Design Document

## Overview

The NBA Court Re-Engineering Optimizer is a Python-based data pipeline and simulation system. It ingests historical shot data, trains two complementary ML models, and runs a grid search over 231 court dimension combinations to identify the arc radius and baseline width that maximizes 3-point shooting percentage for the Cavs, Warriors, and both teams combined.

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
       │          └──────────────────────┘   └──────────────────────┘
       │            shot location (x,y)  │            │
       │            + NN weights         │            │make/miss
       │                                 ▼            │
       │                       ┌──────────────────────┐
       │                       │      Simulator       │◀─────────────┘
       │                       └──────────────────────┘
       │                                 │
       │                                 ▼
       │                       ┌──────────────────────┐
       └──────────────────────▶│      Optimizer       │
                               └──────────────────────┘
```

### Component Responsibilities

- Data_Loader: CSV ingestion, validation, row filtering
- Stat_Calculator: per-player/team stats, zone percentages, attempt counts, defensive ratings
- Heatmap_Engine: zone boundary computation, shot reclassification, heatmap rendering
- Neural_Network_Model: (1) spatial P(make | x, y, player) for heatmap generation; (2) shot location sampling weights for simulation — both roles use the same trained model
- Gradient_Boost_Model: P(make | player, zone, team_defense) for simulation make/miss outcomes
- Simulator: 100-game simulation per Court_Config using NN-weighted KDE shot location sampling and GB make/miss prediction
- Optimizer: grid search enumeration, result aggregation, optimal config identification

---

## Data Pipeline Flow

```
1. Load CSVs
   warriors_cavs_2014.csv  ──▶  Data_Loader.load_primary()
   nn_dataset.csv          ──▶  Data_Loader.load_secondary()

2. Filter Eligible Players
   All rows where team_id in {Warriors, Cavs}
   If player appeared on multiple teams, keep only Warriors/Cavs rows

3. Compute Base Stats (Stat_Calculator)
   Per player: PPG, FG%, 3PT%, 2PT%, total attempts, makes by type
   Per player: Usage_Rate, Estimated_Minutes
   Per team: team PPG, FG%, 3PT%, defensive rating (from NN_Dataset)

4. Compute Zone Stats (Stat_Calculator)
   Standard court (baseline):
     - Read zone labels directly from warriors_cavs_2014 dataset (pre-labeled per shot)
     - Count attempts per player per zone from zone label column
     - Compute zone_pct[player][zone] = makes / attempts using zone labels
     - Apply sparse zone logic (see Sparse Zone section)

   Per new Court_Config (grid search):
     - Pre-labeled zone labels are invalid for new geometry — reclassify geometrically
     - For each shot, compute classify_zone(x, y, court_config) using (x, y) coordinates
     - Recount attempts and recompute zone_pct per player per zone for that config
     - Apply sparse zone logic with new zone assignments

5. Train Models (once, on standard court zone assignments)
   Neural_Network_Model.train(shots with x, y, player_id)
   Gradient_Boost_Model.train(shots with zone, player_id, opp_def_rating, ...)
   Fit per-player KDE on historical (x, y) shot locations (Shot_Location_Sampler)

6. Grid Search (Optimizer)
   For arc_radius in [23.75, 24.00, ..., 26.00]:       # 11 values
     For baseline_width in [50.00, 50.25, ..., 55.00]: # 21 values
       court = CourtConfig(arc_radius, baseline_width)
       court.derive_corner3_elimination()
       zone_stats = Stat_Calculator.compute_zone_stats(court)
       heatmaps  = Heatmap_Engine.generate(court)
       results   = Simulator.run(court, zone_stats, 100 games)
       Optimizer.store(court, results)

7. Output
   Optimizer.rank_by_3pt_pct(team="Cavs")
   Optimizer.rank_by_3pt_pct(team="Warriors")
   Optimizer.rank_by_3pt_pct(team="combined")
   Print optimal configs + tied configs
```

---

## Court Geometry Module

### Coordinate System

The basket is placed at the origin (0, 0). The court extends in the positive y direction toward half court. The x-axis runs along the baseline. Positive x is the right side of the court from the offensive team's perspective.

```
  baseline  x = -baseline_width/2 ... +baseline_width/2
  basket    (0, 0)
  arc       circle of radius arc_radius centered at (0, 0)
```

### Zone Boundary Derivation

Zone boundaries are recomputed for every Court_Config. The primary inputs are:

- `arc_radius` (r): the 3-point arc radius in feet
- `baseline_width` (w): full court width in feet; half-width = w/2
- `basket_to_baseline` (d): perpendicular distance from basket to baseline

On the standard NBA court, `basket_to_baseline` = 5.25 ft (basket is 5.25 ft from the baseline). This value is held constant across all Court_Config combinations; only the arc radius and baseline width vary.

### Corner_3_Elimination_Condition

The standard court has a straight sideline segment at x = ±22 ft running from the baseline up to where the arc begins. This segment exists because the arc at 23.75 ft radius would extend beyond the sideline (22 ft) before reaching the baseline.

For a given Court_Config, the Corner_3 zone exists only if the arc does not reach the baseline within the half-width of the court. The condition is derived as follows:

```
# y-coordinate where the arc crosses x = half_width
# arc equation: x² + y² = r²  →  y = sqrt(r² - x²)
# The arc intersects the baseline (y = -basket_to_baseline) when:
#   r² - half_width² < basket_to_baseline²
#   i.e., sqrt(r² - half_width²) <= basket_to_baseline

half_width = baseline_width / 2

def corner3_eliminated(arc_radius, baseline_width, basket_to_baseline=5.25):
    half_w = baseline_width / 2
    if arc_radius < half_w:
        # arc never reaches the sideline; corner 3 always exists
        return False
    y_at_sideline = math.sqrt(arc_radius**2 - half_w**2)
    # If the arc at the sideline is at or below the baseline level,
    # the arc hits the baseline before the sideline → no straight corner segment
    return y_at_sideline <= basket_to_baseline
```

When `corner3_eliminated` is True, all shots in the corner region are reclassified as 2-point attempts.

### Zone Taxonomy (14 Zones)

Zones are defined by angular and distance thresholds relative to the basket. Angles are measured from the positive y-axis (straight ahead), with negative angles to the left and positive to the right.

| Zone ID | Name                        | Distance      | Angle Range         | Point Value |
|---------|-----------------------------|---------------|---------------------|-------------|
| Z01     | restricted_area             | 0 – 4 ft      | all                 | 2           |
| Z02     | paint_non_restricted        | 4 – 8 ft      | within paint width  | 2           |
| Z03     | left_short_corner_floater   | 4 – 14 ft     | < -60°              | 2           |
| Z04     | right_short_corner_floater  | 4 – 14 ft     | > +60°              | 2           |
| Z05     | left_elbow_mid_range        | 14 – arc_r    | -60° to -30°        | 2           |
| Z06     | right_elbow_mid_range       | 14 – arc_r    | +30° to +60°        | 2           |
| Z07     | center_mid_range            | 14 – arc_r    | -30° to +30°        | 2           |
| Z08     | left_mid_range_baseline     | 14 – arc_r    | < -60° (above paint)| 2           |
| Z09     | right_mid_range_baseline    | 14 – arc_r    | > +60° (above paint)| 2           |
| Z10     | left_corner_3               | ≥ arc_r       | < -65° (corner)     | 3 (or 2*)   |
| Z11     | right_corner_3              | ≥ arc_r       | > +65° (corner)     | 3 (or 2*)   |
| Z12     | left_wing_3                 | ≥ arc_r       | -65° to -20°        | 3           |
| Z13     | right_wing_3                | ≥ arc_r       | +20° to +65°        | 3           |
| Z14     | top_of_arc_3                | ≥ arc_r       | -20° to +20°        | 3           |

*Z10 and Z11 become 2-point zones when Corner_3_Elimination_Condition is met.

Corner zone boundary: a shot is in the corner region if its x-coordinate satisfies `|x| > half_width - corner_depth` where `corner_depth` is derived from the geometry of the straight sideline segment. When the corner is eliminated, this region no longer exists as a 3-point zone.

Zone boundaries are recomputed per Court_Config. The `arc_r` threshold in the distance column refers to the Court_Config's arc_radius.

---

## Heatmap Engine

### Historical Heatmaps

For each Eligible_Player, a historical heatmap is built by binning all shot attempts from the warriors_cavs_2014 dataset into a 2D grid of (x, y) cells plotted on the standard NBA court. This is a raw density plot — no model inference involved.

```
historical_heatmap[player] = 2D histogram of (x, y) shot attempts on standard court
```

These are rendered and printed for every Eligible_Player as the "original court" view.

### New Court Heatmaps (combined-optimal Court_Config)

After the grid search completes and the combined-optimal Court_Config is identified (the single config with the highest combined_3pt_pct = (cavs_3pt_pct + warriors_3pt_pct) / 2), the Heatmap_Engine evaluates the Neural_Network_Model on a dense grid of (x, y) points for every Eligible_Player on that config.

```
grid = [(x, y) for x in range(-25, 26, 0.5) for y in range(0, 48, 0.5)]
for player in eligible_players:
    for (x, y) in grid:
        p_make[player][x][y] = NN_Model.predict(x, y, player_id=player.id)
    new_court_heatmap[player] = p_make[player]
```

Both heatmaps are then rendered and printed side by side (or sequentially) for every Eligible_Player:

```
for player in eligible_players:
    render_side_by_side(
        left=historical_heatmap[player],   # original court, actual shot locations
        right=new_court_heatmap[player],   # combined-optimal court, NN probability surface
        title=f"{player.name}: Original vs. Combined-Optimal Court"
    )
```

Heatmaps for other Court_Config instances (non-combined-optimal) are generated internally during the grid search but are not rendered or printed.

### Zone Reclassification

When a new Court_Config is evaluated, every historical shot is reclassified into the new zone taxonomy:

```
for shot in player.shots:
    shot.zone[court_config] = classify_zone(shot.x, shot.y, court_config)
    shot.point_value[court_config] = zone_point_value(shot.zone[court_config], court_config)
```

---

## Neural Network Model

### Purpose

Generates continuous spatial shot probability heatmaps P(make | x, y, player). Used by the Heatmap_Engine to visualize how each player's shooting probability changes across the re-engineered court surface.

### Architecture

```
Input layer:  [x, y, player_id_embedding]
              x, y: normalized court coordinates (float)
              player_id: integer index → Embedding(n_players, 8)

Hidden layers:
  Dense(128, activation='relu')
  Dropout(0.3)
  Dense(64, activation='relu')
  Dropout(0.2)
  Dense(32, activation='relu')

Output layer: Dense(1, activation='sigmoid')  → P(make)

Loss:    binary_crossentropy
Optimizer: Adam(lr=1e-3)
```

### Training

- Input: all Eligible_Player shots from warriors_cavs_2014 with (x, y, player_id, made)
- Train/validation split: 80/20 stratified by player
- Early stopping on validation loss (patience=10)
- Batch size: 256, max epochs: 100

### Inference

At inference time, the model is evaluated on the dense (x, y) grid for each player to produce the heatmap surface.

### Role in Simulation

The NN also participates directly in the simulation pipeline as a shot location weighting function. For each simulated shot attempt, the Shot_Location_Sampler:

1. Draws `n=100` candidate (x, y) coordinates from the player's KDE (their historical spatial shooting tendency)
2. Queries the NN for `P(make | x, y, player)` at each candidate location
3. Computes a combined weight for each candidate: `weight(x, y) = KDE_density(x, y, player) × NN_P(make | x, y, player)`
4. Normalizes the weights and samples one (x, y) location via weighted sampling

This means players are more likely to attempt shots from locations where they both tend to shoot (high KDE density) AND have higher make probability (high NN output). The same trained NN model serves both roles — heatmap generation and simulation weighting — without any retraining.

---

## Gradient Boost Model

### Purpose

Predicts P(make | player, zone, team_defense) for use during game simulation. Provides interpretable, defense-conditioned shot probabilities at the zone level.

### Features

| Feature                | Type        | Description                                              |
|------------------------|-------------|----------------------------------------------------------|
| player_id              | categorical | encoded integer                                          |
| zone_id                | categorical | one of Z01–Z14 (remapped per Court_Config)               |
| opp_defensive_rating   | float       | opponent team defensive rating from NN_Dataset           |
| attempts_in_zone       | int         | player's historical attempts in this zone                |
| zone_pct               | float       | player's shooting % in zone (or fallback, see below)     |
| overall_2pt_pct        | float       | player's overall 2PT%                                    |
| overall_3pt_pct        | float       | player's overall 3PT%                                    |
| zone_point_value       | int         | 2 or 3, based on Court_Config classification             |

### Sparse Zone Logic (applied before feature construction)

```python
def resolve_zone_pct(player, zone, court_config):
    attempts = player.zone_attempts[court_config][zone]
    if attempts >= 5:
        return player.zone_pct[court_config][zone]
    elif 1 <= attempts <= 4:
        # Use classification under the CURRENT Court_Config, not original
        if court_config.zone_point_value(zone) == 3:
            return player.overall_3pt_pct
        else:
            return player.overall_2pt_pct
    else:  # attempts == 0
        return None  # zone excluded from shot distribution
```

The `zone_pct` feature passed to the model is the resolved value from `resolve_zone_pct`. Zones returning `None` are excluded from the player's shot distribution weight vector in the Simulator.

### Output

- Binary classification: P(make) ∈ [0, 1]
- Implementation: XGBoost or LightGBM (configurable)

### Training

- Input: all Eligible_Player shots from warriors_cavs_2014 with engineered features
- Train/validation split: 80/20 stratified by player
- Hyperparameters: n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8
- Early stopping on validation log-loss

---

## Grid Search

### Enumeration

```python
arc_radii       = [23.75 + 0.25*i for i in range(11)]   # 23.75 to 26.00, 11 values
baseline_widths = [50.00 + 0.25*i for i in range(21)]   # 50.00 to 55.00, 21 values
total_configs   = 11 * 21  # = 231
```

### Per-Config Execution

```python
for arc_r in arc_radii:
    for bw in baseline_widths:
        config = CourtConfig(arc_radius=arc_r, baseline_width=bw)
        config.corner3_eliminated = corner3_eliminated(arc_r, bw)
        zone_stats = stat_calculator.compute_zone_stats(config)
        heatmaps   = heatmap_engine.generate(config)
        result     = simulator.run(config, zone_stats, n_games=100)
        optimizer.store(config, result)
```

---

## Simulator

### Shot Allocation

For each simulated game, each player's shot attempts are allocated proportionally to their Usage_Rate and Estimated_Minutes:

```python
player_attempts_per_game = (
    player.usage_rate
    * player.estimated_minutes
    * team_possessions_per_minute
)
```

`team_possessions_per_minute` is derived from the team's historical pace in the warriors_cavs_2014 dataset.

### Shot Location Sampling

For each shot attempt, the (x, y) coordinate is sampled using the player's KDE weighted by the NN's spatial probability surface. This replaces direct zone sampling and provides full spatial precision:

```python
# Step 1: Sample candidate (x, y) locations from the player's KDE
kde = player.shot_location_kde  # fitted on historical (x, y) shots
candidates = kde.sample(n=100)  # sample candidate locations

# Step 2: Weight candidates by NN's P(make | x, y, player)
nn_weights = [nn_model.predict(x, y, player.id) for x, y in candidates]
# Combined weight = KDE_density × NN_P(make) — already implicit since
# candidates are drawn from KDE; nn_weights serve as importance weights
(shot_x, shot_y) = weighted_sample(candidates, nn_weights)

# Step 3: Classify to zone under current Court_Config
zone = classify_zone(shot_x, shot_y, court_config)

# Step 4: Apply sparse zone check — if zone has 0 attempts, resample
while player.zone_attempts[config][zone] == 0:
    (shot_x, shot_y) = weighted_sample(candidates, nn_weights)
    zone = classify_zone(shot_x, shot_y, court_config)

# Step 5: GB predicts make probability
p_make = gb_model.predict_proba(player, zone, config, opp_defense)
made = random.random() < p_make
```

The per-player KDE is fitted once (after model training) on the player's historical (x, y) shot locations from warriors_cavs_2014 using `sklearn.neighbors.KernelDensity`.

### Shot Outcome

```python
# zone is already classified from the NN-weighted KDE sampling step above
features = build_features(player, zone, config, opp_team)
p_make   = gb_model.predict_proba(features)
made     = random.random() < p_make
points   = config.zone_point_value(zone) if made else 0
```

### Per-Game Tracking

For each simulated game, the Simulator records per player and per team:
- total points
- total shot attempts
- total 3-point attempts and makes
- PPG (averaged over 100 games)
- 3PT% (averaged over 100 games)

### Simulation Loop

```python
for game in range(100):
    for player in warriors_roster + cavs_roster:
        n_attempts = sample_attempts(player, config)
        for _ in range(n_attempts):
            (x, y) = sample_shot_location(player, config, nn_model)  # NN-weighted KDE
            zone   = classify_zone(x, y, config)                     # zone under current config
            made   = simulate_shot(player, zone, config, opp_defense) # GB make/miss
            record(player, zone, made, config)
aggregate_results(config)
```

---

## Output and Ranking

### Storage

All per-config results are stored in a structured dict (or DataFrame) keyed by `(arc_radius, baseline_width)`:

```python
results[(arc_r, bw)] = {
    "cavs_3pt_pct":       float,
    "warriors_3pt_pct":   float,
    "combined_3pt_pct":   float,  # = (cavs_3pt_pct + warriors_3pt_pct) / 2
    "cavs_ppg":           float,
    "warriors_ppg":       float,
    "per_player":         {player_id: {ppg, 3pt_pct, 3pa_per_game}},
    "corner3_eliminated": bool,
}
```

### Optimal Config Identification

```python
# combined_3pt_pct is defined as the simple average of both teams
for cfg, r in results.items():
    r["combined_3pt_pct"] = (r["cavs_3pt_pct"] + r["warriors_3pt_pct"]) / 2

# Sort all configs by each metric descending
def top5_with_ties(results, key):
    sorted_cfgs = sorted(results.items(), key=lambda kv: kv[1][key], reverse=True)
    top5 = []
    rank = 0
    prev_val = None
    for cfg, r in sorted_cfgs:
        val = r[key]
        if val != prev_val:
            rank += 1
            if rank > 5:
                break
            prev_val = val
        top5.append((rank, cfg, r))
    return top5

cavs_top5     = top5_with_ties(results, "cavs_3pt_pct")
warriors_top5 = top5_with_ties(results, "warriors_3pt_pct")
combined_top5 = top5_with_ties(results, "combined_3pt_pct")

# The combined-optimal is the rank-1 entry from combined_top5
combined_optimal_cfg = combined_top5[0][1]
```

### Final Output Format

```
=== Historical Baseline (Standard NBA Court) ===
  Warriors 3PT%: 35.2%
  Cavs 3PT%:     33.8%

=== Optimal Court Configurations ===

--- Cavs-Optimal: Top 5 ---
Rank 1: Arc radius: 24.50 ft | Baseline: 52.00 ft | Cavs 3PT%: 38.4% | Corner 3 eliminated: No
Rank 2: Arc radius: 24.75 ft | Baseline: 52.00 ft | Cavs 3PT%: 38.1% | Corner 3 eliminated: No
Rank 3: Arc radius: 24.50 ft | Baseline: 52.25 ft | Cavs 3PT%: 37.9% | Corner 3 eliminated: No
Rank 4: Arc radius: 25.00 ft | Baseline: 51.75 ft | Cavs 3PT%: 37.7% | Corner 3 eliminated: No
Rank 5: Arc radius: 25.25 ft | Baseline: 51.50 ft | Cavs 3PT%: 37.5% | Corner 3 eliminated: Yes
  [ties at any rank are all listed]

--- Warriors-Optimal: Top 5 ---
Rank 1: Arc radius: 25.00 ft | Baseline: 51.50 ft | Warriors 3PT%: 41.2% | Corner 3 eliminated: Yes
  ...

--- Combined-Optimal: Top 5 ---
  combined_3pt_pct = (cavs_3pt_pct + warriors_3pt_pct) / 2
Rank 1: Arc radius: 24.75 ft | Baseline: 51.75 ft | Combined 3PT%: 39.7% | Corner 3 eliminated: No
  ...

=== Best Court (Combined-Optimal #1) ===
  Arc radius:          24.75 ft
  Baseline width:      51.75 ft
  Cavs 3PT%:           39.1%
  Warriors 3PT%:       40.3%
  Combined 3PT%:       39.7%
  Corner 3 eliminated: No
```

---

## Key Design Decisions

- The Neural_Network_Model serves two roles: (1) heatmap generation — evaluating P(make | x, y, player) on a dense grid for visualization; and (2) simulation weighting — providing importance weights for KDE-sampled shot location candidates. The same trained model is used for both without retraining.
- The Shot_Location_Sampler uses a per-player KDE fitted on historical (x, y) shot locations, weighted by the NN's P(make | x, y, player) surface. This means shot attempts are preferentially sampled from locations where the player both tends to shoot (high KDE density) and has higher make probability (high NN output).
- The Gradient_Boost_Model and Simulator are trained once on the standard court zone assignments. Zone reclassification per Court_Config is applied at inference/feature-construction time, not by retraining.
- The sparse zone fallback uses the Court_Config's current zone classification (2PT or 3PT), not the original classification. This ensures that when a corner zone is reclassified as 2PT, the fallback correctly uses overall_2pt_pct.
- The Corner_3_Elimination_Condition is derived analytically per Court_Config — no lookup tables.
- Simulation uses 100 games per config to balance statistical stability with compute time across 231 configs (23,100 total game simulations).
- combined_3pt_pct is defined as the simple average: (cavs_3pt_pct + warriors_3pt_pct) / 2. The combined-optimal Court_Config is the single config with the highest combined_3pt_pct and is the primary printed result.
- Historical heatmaps (original court, actual shot locations) are rendered and printed for every Eligible_Player. New court heatmaps (NN probability surface on the combined-optimal config) are also rendered and printed for every Eligible_Player, side by side with the historical heatmap.
- The Optimizer outputs the top 5 configs (with ties) for each of the three optimization targets (Cavs-optimal, Warriors-optimal, combined-optimal), each entry including arc radius, baseline width, the relevant 3PT%, and corner-3 elimination status.
