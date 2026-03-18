"""
Global configuration for the NBA Court Re-Engineering Optimizer.
"""

# --- Data paths (override these to point to your actual files) ---
PRIMARY_DATA_PATH = "warriors_cavs_playoff_shots_MASTER_2014_2024.csv"
SECONDARY_DATA_PATH = "final_nn_input_full_grid.csv"

# --- Court geometry constants ---
BASKET_TO_BASELINE = 5.25  # feet: distance from basket center to baseline

# --- Grid search ranges ---
ARC_RADII = [round(23.75 + 0.25 * i, 2) for i in range(11)]       # 23.75 to 26.25 ft (11 steps)
BASELINE_WIDTHS = [round(50.00 + 0.25 * i, 2) for i in range(21)] # 50.00 to 55.00 ft (21 steps)
# 11 arc × 21 baseline = 231 configs total

# --- Standard NBA court (baseline config) ---
STANDARD_ARC_RADIUS = 23.75
STANDARD_BASELINE_WIDTH = 50.0

# --- Simulation ---
N_GAMES = 25           # games to simulate per Court_Config
KDE_CANDIDATES = 20    # candidate shot locations sampled per shot attempt
NN_TEMPERATURE = 0.3   # flattens NN location bias (1.0=full NN, 0.0=pure KDE)

# --- Model training ---
TRAIN_VAL_SPLIT = 0.8
NN_BATCH_SIZE = 256
NN_MAX_EPOCHS = 100
NN_PATIENCE = 10
GB_N_ESTIMATORS = 500
GB_MAX_DEPTH = 6
GB_LEARNING_RATE = 0.05
GB_SUBSAMPLE = 0.8

# --- Zone attempt thresholds (sparse zone logic) ---
ZONE_ATTEMPT_THRESHOLD = 5  # min attempts to use player's own zone %

# --- Known playoff minutes per game (sourced from StatMuse / Basketball Reference) ---
# Used to override the 24-min fallback in StatCalculator._compute_estimated_minutes
KNOWN_PLAYOFF_MINUTES: dict = {
    # Warriors
    "Stephen Curry":     37.2,
    "Klay Thompson":     36.2,
    "Draymond Green":    37.3,
    "Harrison Barnes":   28.0,
    "Andre Iguodala":    27.0,
    "Shaun Livingston":  18.0,
    "Andrew Bogut":      22.0,
    "Leandro Barbosa":   12.0,
    "Ian Clark":         12.0,
    "Festus Ezeli":      14.0,
    # Cavaliers
    "LeBron James":      41.3,
    "Kyrie Irving":      37.4,
    "Kevin Love":        34.5,
    "JR Smith":          34.8,
    "J.R. Smith":        34.8,
    "Tristan Thompson":  28.8,
    "Iman Shumpert":     22.0,
    "Matthew Dellavedova": 20.0,
    "Richard Jefferson": 18.0,
    "Channing Frye":     16.0,
    "Timofey Mozgov":    14.0,
}
