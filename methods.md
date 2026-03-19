# Methods & Techniques — NBA Court Re-Engineering Optimizer

This document explains every machine learning model, statistical method, and algorithmic technique used in this project. Each section covers what the method is, why it was chosen, and exactly how it is applied here.

---

## 1. Kernel Density Estimation (KDE)

### What it is
Kernel Density Estimation is a non-parametric statistical method for estimating the probability distribution of a dataset without assuming any fixed shape (like a normal distribution). Given a set of observed data points, KDE places a smooth "kernel" (in our case, a Gaussian bell curve) over each point and sums them all together to produce a continuous probability surface.

### How it works
For a set of shot locations (x, y), KDE produces a smooth 2D density surface where regions with many historical shots have high density and regions with few shots have low density. The **bandwidth** parameter controls how wide each Gaussian kernel is — a narrow bandwidth produces a spiky, detailed surface; a wide bandwidth produces a smoother, more generalized surface. We use **Scott's rule** to automatically select bandwidth: `h = n^(-1/6)` where n is the number of shots for that player.

### Why we use it
We need to simulate realistic shot locations for each player. Rather than just replaying historical shots, KDE lets us sample new (x, y) coordinates that follow the same spatial distribution as a player's real shot chart. This means Klay Thompson will still tend to shoot from the corners and the top of the arc, but the exact coordinates vary each simulation.

### How it is applied here
- A separate KDE is fitted per player using their historical (x, y) shot coordinates
- During the precomputation phase, 5,000 candidate locations are sampled from each player's KDE
- These candidates are then scored by the Neural Network (see Section 2) to create a weighted pool
- During simulation, each shot location is drawn from this precomputed weighted pool — making simulation fast while preserving both historical shot patterns and NN-informed make probability

---

## 2. Feedforward Neural Network (FFNN) with Player Embeddings

### What it is
A Feedforward Neural Network (also called a Multilayer Perceptron or MLP) is a type of deep learning model where information flows in one direction — from input through hidden layers to output — with no loops or recurrence. Each layer applies a linear transformation followed by a non-linear activation function, allowing the network to learn complex patterns.

### Architecture
```
Input:
  - coords:    [x_norm, y_norm]         (2 values, normalized to [-1, 1] and [0, 1])
  - player_id: integer index            (1 value)

Player Embedding Layer:
  - Embedding(n_players, 8) → Flatten  (learns an 8-dim vector per player)

Concatenate coords + embedding → 10-dim vector

Hidden Layers:
  - Dense(128, activation=ReLU)
  - Dropout(0.3)
  - Dense(64,  activation=ReLU)
  - Dropout(0.2)
  - Dense(32,  activation=ReLU)

Output:
  - Dense(1, activation=Sigmoid) → P(make) ∈ [0, 1]
```

### Player Embeddings
Instead of one-hot encoding each player (which would be sparse and high-dimensional), we use an **embedding layer** that learns a dense 8-dimensional vector for each player. This vector captures that player's shooting tendencies — players with similar shooting profiles will have similar embedding vectors. The embedding is learned jointly with the rest of the network during training.

### Training
- Loss function: Binary Cross-Entropy (standard for binary classification)
- Optimizer: Adam with learning rate 1e-3
- Regularization: Dropout layers (0.3 and 0.2) to prevent overfitting
- Early stopping: training halts when validation loss stops improving (patience = 10 epochs)
- Data split: stratified 80/20 train/validation split per player

### What it predicts
Given a court location (x, y) and a player identity, the NN outputs **P(make)** — the probability that this specific player makes a shot from that location. This is a spatial make-probability surface unique to each player.

### How it is applied here
- Trained once on all eligible players' historical shot data
- Used to score the 5,000 KDE candidates per player during precomputation (importance weighting)
- Used to generate the NN probability surface heatmaps (right panel in output images)
- Does NOT directly decide make/miss during simulation — that role belongs to XGBoost

---

## 3. Gradient Boosted Trees — XGBoost

### What it is
Gradient Boosting is an ensemble machine learning technique that builds a sequence of decision trees, where each new tree corrects the errors of the previous ones. **XGBoost** (Extreme Gradient Boosting) is a highly optimized implementation of this algorithm known for speed and strong performance on tabular data.

### How gradient boosting works
1. Start with a simple prediction (e.g., the mean of the target)
2. Fit a decision tree to the residual errors (the difference between predictions and actual values)
3. Add this tree to the model with a small learning rate (shrinkage)
4. Repeat — each tree focuses on the mistakes the previous ensemble made
5. The final prediction is the sum of all trees' outputs

This process minimizes a loss function (binary cross-entropy for classification) using gradient descent in function space — hence "gradient boosting."

### Features used
The XGBoost model is trained on features including:
- Shot distance from the basket
- Shot zone (paint, mid-range, corner 3, above-the-break 3, etc.)
- Player usage rate
- Player historical zone shooting percentage
- Court configuration parameters (arc radius, baseline width)

### What it predicts
Given a specific shot attempt with its features, XGBoost outputs a **probability of making the shot**. A random draw against this probability determines the binary make/miss outcome in simulation.

### Why XGBoost instead of the NN for simulation
XGBoost is faster for tabular inference and handles structured features (distance, zone, stats) better than the NN, which is optimized for spatial (x, y) patterns. The two models are complementary: NN handles spatial location sampling, XGBoost handles the final make/miss decision.

### How it is applied here
- Trained on historical shot data with the features listed above
- Called once per simulated shot attempt to determine make or miss
- Runs across all 25 simulated games × all court configurations in the grid search

---

## 4. Monte Carlo Simulation

### What it is
Monte Carlo simulation is a computational technique that uses repeated random sampling to estimate outcomes that are difficult to calculate analytically. By running thousands of random trials and averaging the results, you get a stable statistical estimate of the true expected outcome.

### The core idea
Instead of trying to mathematically derive "what 3PT% would the Warriors shoot on a court with arc=24.5ft?", we simulate 25 full games under that court configuration and measure the actual 3PT% across those games. The randomness in shot selection, location sampling, and make/miss outcomes averages out over many games to give a reliable estimate.

### How a single simulated game works
1. For each possession, a player is selected proportional to their **usage rate** (how often they take shots relative to their team)
2. The player's **shot location** (x, y) is drawn from their precomputed KDE/NN weighted candidate pool
3. The shot is **geometrically classified** under the current court config — is it a 2PT or 3PT attempt given this arc radius and baseline width?
4. The **XGBoost model** predicts P(make) for this shot given its features
5. A random draw determines make or miss
6. Points are recorded; the game continues for the appropriate number of possessions (based on minutes × possessions per minute)

### Averaging across games
Each court configuration is evaluated over 50 simulated games. The 3PT% for each team is averaged across all games to produce a stable estimate. More games = more stable estimate, but slower runtime (50 is a balance between accuracy and speed).

### Why Monte Carlo
The shot-making process is inherently stochastic — even Steph Curry doesn't make every open three. Monte Carlo naturally captures this variability and produces realistic distributions of outcomes rather than a single deterministic prediction.

### How it is applied here
- 25 games simulated per court configuration
- Both teams (Warriors and Cavs) simulated independently per game
- Output: average 3PT% for Warriors, average 3PT% for Cavs, combined average
- These outputs feed directly into the optimizer

---

## 5. Exhaustive Grid Search Optimizer

### What it is
Grid search is an optimization strategy that evaluates every possible combination of parameters within a defined search space. Unlike gradient-based optimization (which follows gradients to find a local minimum) or evolutionary algorithms (which evolve solutions over generations), grid search simply tries everything and picks the best.

### The search space
The optimizer searches over two court dimensions:
- **Arc radius**: from 22.0 ft to 27.0 ft in 0.25 ft increments
- **Baseline width**: from 47.0 ft to 55.0 ft in 0.25 ft increments

This produces approximately **1,700+ configurations** to evaluate. For each configuration, the simulator runs 25 games for both teams.

### Geometric constraint
A configuration is only valid if the 3PT arc physically intersects the baseline — i.e., the arc radius must be less than or equal to half the baseline width. If `arc_radius > baseline_width / 2`, corner 3-pointers are geometrically impossible and the configuration is flagged as "corner 3 eliminated."

### Three optimization objectives
The optimizer tracks three separate rankings:
1. **Cavs-Optimal**: court configs that maximize Cleveland's 3PT%
2. **Warriors-Optimal**: court configs that maximize Golden State's 3PT%
3. **Combined-Optimal**: court configs that maximize `(Cavs_3PT% + Warriors_3PT%) / 2`

Top 5 results are reported for each objective.

### Why exhaustive search instead of gradient descent
The objective function (3PT% from Monte Carlo simulation) is noisy and non-differentiable — you can't take a gradient of it. Exhaustive search guarantees finding the global optimum within the search space and gives a complete picture of how 3PT% varies across all configurations, which is valuable for analysis beyond just the top result.

### How it is applied here
- All valid (arc, baseline) pairs are enumerated upfront
- For each pair, the simulator is called and results are stored
- After all configs are evaluated, results are sorted by each objective
- Top 5 per objective are printed and the combined-optimal config is used for heatmap generation

---

## 7. Expected Points Per Game (ePPG)

### What it is
ePPG (Expected Points Per Game) is the per-player scoring output produced by the Monte Carlo simulation. It represents how many points a player is expected to score per game under a given court configuration, based on their simulated shot attempts, locations, and make/miss outcomes.

### How it is calculated
ePPG is computed in two stages — a historical baseline from real data, and a simulated value from the Monte Carlo runs.

**Historical PPG (from real data):**
```
total_points = (3PT makes × 3) + (2PT makes × 2)
historical_PPG = total_points / number_of_games_in_dataset
```
This gives each player's actual scoring rate from the playoff shot data.

**Simulated ePPG (from Monte Carlo):**
```
ePPG = total_simulated_points_across_all_games / N_games
```
Where `N_games = 50`. For each of the 50 simulated games, every shot attempt is resolved as make or miss by the GB model, and points are accumulated. The total is divided by 50 to get the per-game average.

### Per-shot point value
Each shot's point contribution depends on its zone classification under the current court config:
- Shot in a **3PT zone** (outside the arc): worth 3 points if made
- Shot in a **2PT zone** (inside the arc): worth 2 points if made
- Missed shots: 0 points

Because the zone classification changes with the court configuration, the same shot location can be worth 2 points on one court and 3 points on another. This means ePPG is directly sensitive to court geometry — moving the arc inward converts some 2PT attempts into 3PT attempts, potentially increasing a player's ePPG if they shoot well from that range.

### Shot attempt volume
The number of attempts per player per game is sampled from a Poisson distribution:

```
expected_attempts = usage_rate × estimated_minutes × possessions_per_minute
n_attempts ~ Poisson(expected_attempts)
```

- `usage_rate`: fraction of team possessions the player uses (from historical data)
- `estimated_minutes`: known playoff minutes per game (hardwired from StatMuse/BBRef)
- `possessions_per_minute`: team-level pace derived from the dataset, scaled to match real NBA FGA rates (~88 FGA/game)

The Poisson distribution naturally introduces game-to-game variability in shot volume, which is realistic — a player doesn't take the exact same number of shots every game.

### Why ePPG matters
ePPG lets you evaluate not just whether a court config improves 3PT%, but whether it actually translates to more points scored. A player might shoot a higher 3PT% on a modified court but take fewer 3PT attempts (because the arc is farther out and they avoid it), resulting in lower ePPG. The per-player ePPG breakdown in the output makes this tradeoff visible.

---

## 8. Supporting Statistical Methods

### Usage Rate Weighting
Each player's share of their team's shot attempts is computed from historical data as their **usage rate** — the fraction of team possessions they use. During simulation, players are selected for each possession by sampling proportional to these rates, ensuring Curry takes more shots than Bogut, consistent with real game patterns.

### Zone-Based Shooting Percentages
For each player, historical shooting percentages are computed per zone (paint, mid-range, corner 3, above-the-break 3). When a player has 5+ attempts in a zone, their zone-specific % is used. With 1–4 attempts, a fallback to overall 2PT% or 3PT% is used. With 0 attempts, the player never shoots from that zone in simulation. This prevents small-sample noise from distorting results.

### Geometric Zone Classification
Every shot location (x, y) is classified into a zone purely by geometry given the current `CourtConfig`. The distance from the basket is computed as `sqrt(x² + y²)`, and compared against the arc radius. Corner 3s are identified by checking whether the shot is within the corner region (|x| near the sideline, y below the arc-baseline intersection). This reclassification happens fresh for every court configuration, so the same historical shot location can be a 2PT on one court and a 3PT on another.

### Defensive Rating Adjustment
Team defensive ratings are computed from the data and used to scale the opposing team's shot make probabilities. A team facing a stronger defense will have slightly lower make probabilities, adding realism to the simulation beyond just individual player stats.

---

## Summary Table

| Method | Category | Purpose in Project |
|---|---|---|
| Kernel Density Estimation (KDE) | Statistical / Non-parametric | Sample realistic shot locations per player |
| Feedforward Neural Network (FFNN) | Deep Learning | Spatial P(make) surface; importance-weight KDE candidates |
| Player Embeddings | Deep Learning | Encode per-player shooting identity into the NN |
| XGBoost (Gradient Boosted Trees) | Ensemble ML | Per-shot make/miss prediction during simulation |
| Monte Carlo Simulation | Stochastic Simulation | Estimate 3PT% under each court configuration (50 games) |
| Exhaustive Grid Search | Combinatorial Optimization | Find optimal arc/baseline dimensions across all valid configs |
| ePPG (Expected Points Per Game) | Simulation Output Metric | Per-player scoring output under each court configuration |
| Usage Rate Weighting | Sports Statistics | Realistic player shot selection during simulation |
| Zone % with Sparse Fallback | Sports Statistics | Per-player zone shooting accuracy |
| Geometric Zone Classification | Computational Geometry | Reclassify shots under any court configuration |
| Defensive Rating Adjustment | Sports Statistics | Account for opponent defensive strength |
