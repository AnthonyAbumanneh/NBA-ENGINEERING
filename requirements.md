# Requirements Document

## Introduction

The NBA Court Re-Engineering Optimizer is a data-driven simulation system that re-engineers NBA court dimensions to find the optimal 3-point line arc and baseline width that maximizes team 3-point shooting percentage. The system ingests historical shot data from the 2014 Warriors vs. Cavs dataset, builds player shot heatmaps, trains two complementary prediction models, and runs a grid-search simulation across all valid court dimension combinations. The output identifies the court configuration that maximizes 3pt% for the Cavs, the Warriors, and both teams combined.

## Glossary

- **System**: The NBA Court Re-Engineering Optimizer application
- **Data_Loader**: The component responsible for reading and validating raw CSV/dataset files
- **Stat_Calculator**: The component that computes per-player and per-team statistics from raw data
- **Heatmap_Engine**: The component that generates and transforms shot heatmaps per player per court configuration
- **Neural_Network_Model**: A neural network trained on raw shot coordinates (x, y) and player identity features; serves two roles: (1) generates continuous spatial shot probability heatmaps P(make | x, y, player) for each player on each new Court_Config, and (2) provides spatial probability weights used to bias shot location sampling toward high-efficiency zones during simulation
- **Gradient_Boost_Model**: An XGBoost or LightGBM model trained on engineered zone features, player identity, and team defensive ratings; used during game simulation to predict P(make | player, zone, team_defense); provides interpretable, defense-conditioned probabilities suited for zone-based simulation
- **Shot_Model**: The collective term for the Neural_Network_Model and Gradient_Boost_Model working together in the pipeline
- **KDE (Kernel Density Estimator)**: A non-parametric statistical model fitted per player on their historical (x, y) shot locations; used by the Simulator to model each player's spatial shooting tendencies and sample candidate shot coordinates during simulation
- **Shot_Location_Sampler**: The sub-component of the Simulator that samples (x, y) shot coordinates for each simulated attempt using a per-player KDE weighted by the Neural_Network_Model's P(make | x, y, player) surface
- **Court_Config**: A specific combination of 3-point arc radius (in feet) and baseline width (in feet)
- **Zone**: A named spatial region of the court used to aggregate shot attempts and compute per-player shooting percentages; zone boundaries are recalculated for each Court_Config based on the new arc radius and baseline width
- **Corner_3_Elimination_Condition**: The geometric condition under which the Corner_3 zone ceases to exist for a given Court_Config; this occurs when the 3-point arc radius is large enough that the arc reaches the baseline before the straight sideline segment (which runs parallel to the baseline at 22 ft from the basket on the standard court) can begin, meaning the arc physically intersects or touches the baseline and no straight corner section exists
- **Simulator**: The component that runs game simulations between the Warriors and Cavs for a given Court_Config
- **Optimizer**: The component that evaluates all Court_Config results and identifies the optimal dimensions
- **Warriors_Cavs_Dataset**: The primary dataset containing shot-level data from Warriors and Cavs games (warriors_cavs_2014)
- **NN_Dataset**: The secondary dataset used to compute supplementary stats and team defensive ratings
- **Eligible_Player**: A player who appears in the Warriors or Cavs roster; if a player appeared on multiple teams, only their Warriors/Cavs game rows are used
- **ePPG**: Expected points per game, calculated using r(2pt) or r(3pt) based on whether a shot location falls inside or outside the 3-point boundary for a given Court_Config
- **Corner_3**: A 3-point attempt taken from the corner region of the court, which becomes a 2-point attempt when the Corner_3_Elimination_Condition is met for a given Court_Config
- **Usage_Rate**: A per-player metric estimating the proportion of team possessions used by that player while on the floor
- **Estimated_Minutes**: The projected per-game minutes for a player, derived from Usage_Rate and historical playing time

---

## Requirements

### Requirement 1: Data Ingestion

**User Story:** As a data analyst, I want the system to load the Warriors/Cavs dataset and the NN dataset, so that all downstream calculations have access to clean, validated input data.

#### Acceptance Criteria

1. THE Data_Loader SHALL read the warriors_cavs_2014 dataset from a configurable file path.
2. THE Data_Loader SHALL read the NN_Dataset from a configurable file path.
3. WHEN a dataset file is missing or unreadable, THE Data_Loader SHALL raise a descriptive error identifying the missing file path.
4. WHEN a dataset row contains missing values in required columns (shot coordinates, player ID, team ID, shot result), THE Data_Loader SHALL log a warning and exclude that row from further processing.
5. THE Data_Loader SHALL expose the loaded data as structured, queryable records for use by all other components.

---

### Requirement 2: Eligible Player Filtering

**User Story:** As a data analyst, I want the system to restrict stat calculations to Warriors and Cavs players only, so that the simulation reflects only the rosters of those two teams.

#### Acceptance Criteria

1. THE Stat_Calculator SHALL identify the Warriors roster and the Cavs roster from the warriors_cavs_2014 dataset.
2. WHEN a player appears in rows associated with both a Warriors/Cavs team and another team, THE Stat_Calculator SHALL include only the rows where that player's team is the Warriors or the Cavs.
3. THE Stat_Calculator SHALL use all rows in the dataset (not just Warriors vs. Cavs matchups) when computing stats for Eligible_Players.

---

### Requirement 3: Per-Player and Per-Team Statistics

**User Story:** As a data analyst, I want the system to compute comprehensive shooting and usage statistics for every Eligible_Player and their respective teams, so that the simulation has accurate baselines.

#### Acceptance Criteria

1. THE Stat_Calculator SHALL compute the following per Eligible_Player: points per game (PPG), field goal percentage (FG%), 3-point percentage (3PT%), total shot attempts, total 3-point attempts, total 3-point makes, total 2-point attempts, total 2-point makes.
2. THE Stat_Calculator SHALL compute the following per team (Warriors and Cavs): team PPG, team FG%, team 3PT%, team total shot attempts, team total 3-point attempts.
3. THE Stat_Calculator SHALL compute a Usage_Rate for each Eligible_Player based on their share of team possessions while on the floor.
4. THE Stat_Calculator SHALL compute Estimated_Minutes per game for each Eligible_Player using their historical average minutes from the warriors_cavs_2014 dataset.
5. THE Stat_Calculator SHALL use the NN_Dataset to compute team defensive ratings for both the Warriors and the Cavs.
6. THE Stat_Calculator SHALL compute per-player, per-Zone shooting percentages using the warriors_cavs_2014 data, where Zone boundaries are defined by the current Court_Config.

---

### Requirement 4: Shot Heatmap Generation

**User Story:** As a simulation engineer, I want the system to generate shot heatmaps for every Eligible_Player, so that spatial shot distributions are available for court re-engineering analysis.

#### Acceptance Criteria

1. THE Heatmap_Engine SHALL generate a historical shot heatmap for each Eligible_Player by plotting all shot attempts recorded for that player in the warriors_cavs_2014 dataset using (x, y) court coordinates on the standard NBA court.
2. THE Heatmap_Engine SHALL render and print the historical shot heatmap for every Eligible_Player.
3. THE Heatmap_Engine SHALL represent each heatmap as a 2D spatial distribution of shot attempts using (x, y) court coordinates.
4. WHEN a new Court_Config is evaluated, THE Heatmap_Engine SHALL recalculate Zone boundaries based on the new arc radius and baseline width, reclassify each historical shot into the new Zones, and recompute per-player per-Zone shooting percentages for that Court_Config.
5. WHEN a new Court_Config is evaluated, THE Heatmap_Engine SHALL generate new heatmap visualizations reflecting the re-engineered court geometry for that Court_Config.
6. THE Heatmap_Engine SHALL render and print, for every Eligible_Player, a new court heatmap showing the Neural_Network_Model-predicted shot probability surface for that player on the combined-optimal Court_Config (the single Court_Config that maximizes combined_3pt_pct as defined in Requirement 12).
7. THE Heatmap_Engine SHALL present the historical shot heatmap and the new court heatmap for each Eligible_Player side by side or sequentially so that both are visible together.
8. WHEN the Corner_3_Elimination_Condition is met for a Court_Config, THE Heatmap_Engine SHALL reclassify all Corner_3 shot locations as 2-point attempts for that Court_Config.

---

### Requirement 5: Shot Value Classification (ePPG)

**User Story:** As a simulation engineer, I want the system to correctly classify each historical shot as a 2-point or 3-point attempt under a given Court_Config, so that expected scoring is accurately computed.

#### Acceptance Criteria

1. WHEN evaluating a shot under a given Court_Config, THE Heatmap_Engine SHALL classify the shot as a 3-point attempt if the shot's distance from the basket exceeds the 3-point arc radius for that Court_Config at the shot's angular position.
2. WHEN evaluating a shot under a given Court_Config, THE Heatmap_Engine SHALL classify the shot as a 2-point attempt if the shot's distance from the basket is less than or equal to the 3-point arc radius for that Court_Config at the shot's angular position.
3. WHEN the Corner_3_Elimination_Condition is met for a Court_Config, THE Heatmap_Engine SHALL classify all shots in the corner region as 2-point attempts regardless of their distance from the basket.
4. THE Heatmap_Engine SHALL compute ePPG for each shot as: 2 × P(make) for 2-point attempts and 3 × P(make) for 3-point attempts, where P(make) is derived from the Gradient_Boost_Model's predicted probability conditioned on player, Zone, and team defensive rating.

---

### Requirement 6: Corner 3 Elimination Geometry

**User Story:** As a simulation engineer, I want the system to derive the Corner_3_Elimination_Condition geometrically from the arc radius and baseline width, so that corner zone reclassification is physically accurate for each Court_Config.

#### Acceptance Criteria

1. THE System SHALL derive the Corner_3_Elimination_Condition for each Court_Config by computing whether the 3-point arc, centered at the basket, intersects or touches the baseline given the Court_Config's baseline width.
2. THE System SHALL determine that the Corner_3_Elimination_Condition is met when the arc radius is large enough that the arc reaches the baseline before the straight sideline segment (parallel to the baseline at 22 ft from the basket) can begin, meaning no straight corner section exists.
3. THE System SHALL compute the Corner_3_Elimination_Condition analytically using the arc radius and the perpendicular distance from the basket to the baseline derived from the Court_Config's baseline width.
4. WHEN the Corner_3_Elimination_Condition is met, THE Optimizer SHALL pass this flag to the Heatmap_Engine and Simulator so Corner_3 reclassification is applied consistently across all components.
5. THE System SHALL NOT rely on hardcoded lookup tables for the Corner_3_Elimination_Condition; the condition MUST be derived geometrically for every Court_Config evaluated.

---

### Requirement 7: Shot Prediction Models

**User Story:** As a simulation engineer, I want the system to train two complementary models with distinct roles, so that spatial heatmap generation and zone-based game simulation each use the most appropriate modeling approach.

#### Acceptance Criteria

1. THE Neural_Network_Model SHALL be trained on raw shot coordinates (x, y) and player identity features from the warriors_cavs_2014 Eligible_Player data.
2. THE Neural_Network_Model SHALL output a continuous spatial shot probability P(make | x, y, player) for each (x, y) location per player, used to generate shot probability heatmaps for each new Court_Config.
3. THE Neural_Network_Model SHALL provide spatial shot probability weights P(make | x, y, player) used to weight shot location sampling during simulation, so that shot attempts are preferentially sampled from high-efficiency spatial locations.
4. THE Simulator SHALL use a per-player Kernel Density Estimator (KDE) fitted on historical (x, y) shot locations to model each player's spatial shooting tendencies during simulation.
5. THE Simulator SHALL weight KDE samples by the Neural_Network_Model's P(make | x, y, player) so that the sampling probability for each candidate location equals KDE_density(x, y, player) × NN_P(make | x, y, player), normalized across candidates.
6. THE Gradient_Boost_Model SHALL be trained on engineered Zone features, player identity features, and team defensive ratings from the warriors_cavs_2014 Eligible_Player data.
7. THE Gradient_Boost_Model SHALL be implemented using XGBoost or LightGBM.
8. THE Gradient_Boost_Model SHALL predict P(make | player, zone, team_defense) for use during game simulation.
9. WHEN training data is split, THE Neural_Network_Model SHALL use a held-out validation set to evaluate model performance and prevent overfitting.
10. WHEN training data is split, THE Gradient_Boost_Model SHALL use a held-out validation set to evaluate model performance and prevent overfitting.
11. THE Neural_Network_Model SHALL be trained only on Eligible_Player shot data from the warriors_cavs_2014 dataset.
12. THE Gradient_Boost_Model SHALL be trained only on Eligible_Player shot data from the warriors_cavs_2014 dataset.
13. WHEN predicting shot outcomes for a new Court_Config, THE Gradient_Boost_Model SHALL use the reclassified Zone assignments derived from the new court geometry as input features.

---

### Requirement 8: Sparse Zone Probability Logic

**User Story:** As a simulation engineer, I want the system to apply statistically appropriate fallback logic when a player has few or no attempts in a zone, so that shot probability estimates remain valid across all Court_Config combinations.

#### Acceptance Criteria

1. WHEN a player has 5 or more recorded attempts in a Zone under a given Court_Config, THE Gradient_Boost_Model SHALL use that player's actual per-Zone shooting percentage as the primary probability input for that Zone.
2. WHEN a player has 1 to 4 recorded attempts in a Zone under a given Court_Config, THE Gradient_Boost_Model SHALL fall back to the player's overall 2PT% if the Zone is classified as a 2-point Zone under that Court_Config, or the player's overall 3PT% if the Zone is classified as a 3-point Zone under that Court_Config.
3. WHEN a player has 0 recorded attempts in a Zone under a given Court_Config, THE Simulator SHALL assign zero weight to that Zone in the player's shot distribution, meaning the player will never attempt a shot from that Zone in simulation.
4. WHEN a Court_Config reclassifies a Zone (e.g., a Corner_3 Zone becomes a 2-point Zone due to the Corner_3_Elimination_Condition), THE Gradient_Boost_Model SHALL use the new Zone classification (2PT% or 3PT%) when applying the sparse fallback for players with 1 to 4 attempts in that Zone, not the original classification.
5. THE Stat_Calculator SHALL compute and store per-player per-Zone attempt counts alongside per-Zone shooting percentages so that the sparse zone threshold logic can be applied consistently by the Gradient_Boost_Model and Simulator.

---

### Requirement 9: Court Configuration Grid Search

**User Story:** As an optimizer, I want the system to enumerate all valid court dimension combinations, so that every candidate Court_Config is evaluated during simulation.

#### Acceptance Criteria

1. THE Optimizer SHALL enumerate Court_Config combinations by varying the 3-point arc radius from 23.75 ft to 26 ft in increments of 0.25 ft.
2. THE Optimizer SHALL enumerate Court_Config combinations by varying the baseline width from 50 ft to 55 ft in increments of 0.25 ft.
3. THE Optimizer SHALL evaluate every combination produced by the grid search.
4. WHEN the Corner_3_Elimination_Condition is met for a Court_Config, THE Optimizer SHALL pass this flag to the Heatmap_Engine and Simulator so Corner_3 reclassification is applied consistently.

---

### Requirement 10: Game Simulation

**User Story:** As a simulation engineer, I want the system to simulate 100 games between the Warriors and Cavs for each Court_Config, so that results are statistically stable.

#### Acceptance Criteria

1. THE Simulator SHALL simulate exactly 100 games between the Warriors and the Cavs for each Court_Config.
2. THE Simulator SHALL use each Eligible_Player's Estimated_Minutes and Usage_Rate to determine shot attempt allocation per simulated game.
3. FOR each simulated shot attempt, THE Simulator SHALL sample a candidate (x, y) coordinate from the player's KDE, weight each candidate by the Neural_Network_Model's P(make | x, y, player), and select the shot location via weighted sampling.
4. THE Simulator SHALL classify the sampled (x, y) coordinate into a Zone under the current Court_Config and apply the sparse zone check: if the sampled Zone has 0 historical attempts for that player, the Simulator SHALL resample until a Zone with at least 1 historical attempt is selected.
5. THE Simulator SHALL query the Gradient_Boost_Model with the classified Zone, player identity, and opponent defensive rating to obtain P(make) for the shot attempt.
6. THE Simulator SHALL apply the ePPG shot classification rules from Requirement 5 to determine whether each simulated shot is worth 2 or 3 points.
7. THE Simulator SHALL incorporate team defensive ratings (from the NN_Dataset) as a conditioning input to the Gradient_Boost_Model during simulation.
8. WHEN simulating a game, THE Simulator SHALL track per-player and per-team: total points, total shot attempts, total 3-point attempts, total 3-point makes, PPG, and 3PT%.

---

### Requirement 11: Per-Configuration Statistics Output

**User Story:** As a data analyst, I want the system to compute and store aggregate statistics for each Court_Config, so that configurations can be compared and ranked.

#### Acceptance Criteria

1. THE Simulator SHALL compute the following per Court_Config per team: expected team PPG, team 3PT%, team 3-point attempts per game.
2. THE Simulator SHALL compute the following per Court_Config per Eligible_Player: expected PPG, 3PT%, and 3-point attempts per game.
3. THE Optimizer SHALL store all per-configuration results in a structured format suitable for ranking and export.

---

### Requirement 12: Optimal Court Identification

**User Story:** As a basketball analyst, I want the system to identify the court dimensions that maximize 3-point shooting percentage, so that I can recommend evidence-based rule changes.

#### Acceptance Criteria

1. THE Optimizer SHALL identify the top 5 Court_Config combinations that produce the highest simulated 3PT% for the Cavs, ranked from highest to lowest.
2. THE Optimizer SHALL identify the top 5 Court_Config combinations that produce the highest simulated 3PT% for the Warriors, ranked from highest to lowest.
3. THE Optimizer SHALL identify the top 5 Court_Config combinations that produce the highest combined simulated 3PT% across both the Cavs and the Warriors, ranked from highest to lowest.
4. THE Optimizer SHALL compute combined_3pt_pct for each Court_Config as: combined_3pt_pct = (cavs_3pt_pct + warriors_3pt_pct) / 2.
5. THE Optimizer SHALL designate the single Court_Config with the highest combined_3pt_pct as the combined-optimal Court_Config; this is the primary "best court" result.
6. THE Optimizer SHALL output, for each entry in each top-5 list: the 3-point arc radius in feet, the baseline width in feet, the relevant 3PT% (Cavs 3PT%, Warriors 3PT%, or combined_3pt_pct as appropriate), and whether the Corner_3_Elimination_Condition is met.
7. WHEN multiple Court_Config combinations produce the same 3PT% at any rank position, THE Optimizer SHALL report all tied configurations at that rank.
8. THE Optimizer SHALL print the combined-optimal Court_Config with full detail as the primary output result.
