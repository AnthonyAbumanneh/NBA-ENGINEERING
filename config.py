"""
Global configuration for the NBA Court Re-Engineering Optimizer.
"""

# --- Dataset selection ---
# Set ACTIVE_DATASET to switch between available datasets:
#   "warriors_cavs"   → warriors_cavs_playoff_shots_MASTER_2014_2024.csv
#   "thunder_pacers"  → Thunder_Finals_Shots_2025.csv + Pacers_Playoffs_2025.csv
ACTIVE_DATASET = "thunder_pacers"

# --- Data paths ---
SECONDARY_DATA_PATH = "final_nn_input_full_grid.csv"

if ACTIVE_DATASET == "warriors_cavs":
    PRIMARY_DATA_PATH_TEAM1 = "warriors_cavs_playoff_shots_MASTER_2014_2024.csv"
    PRIMARY_DATA_PATH_TEAM2 = None
    THUNDER_TEAM_ID = 1610612744   # Golden State Warriors
    PACERS_TEAM_ID  = 1610612739   # Cleveland Cavaliers
    TEAM1_LABEL = "Warriors"
    TEAM2_LABEL = "Cavaliers"
    KNOWN_PLAYOFF_MINUTES: dict = {
        "Stephen Curry":       36.5,
        "Klay Thompson":       35.8,
        "Draymond Green":      35.2,
        "Kevin Durant":        36.0,
        "Andre Iguodala":      26.4,
        "Shaun Livingston":    18.5,
        "Harrison Barnes":     28.0,
        "Zaza Pachulia":       18.0,
        "David West":          14.0,
        "JaVale McGee":        12.0,
        "Nick Young":          14.0,
        "Jordan Bell":         12.0,
        "LeBron James":        41.5,
        "Kyrie Irving":        36.8,
        "Kevin Love":          32.5,
        "J.R. Smith":          30.2,
        "Tristan Thompson":    28.5,
        "Iman Shumpert":       22.0,
        "Matthew Dellavedova": 20.0,
        "Richard Jefferson":   20.0,
        "Kyle Korver":         18.0,
        "George Hill":         28.0,
        "Jordan Clarkson":     20.0,
        "Larry Nance Jr.":     16.0,
        "Rodney Hood":         18.0,
    }
else:  # thunder_pacers
    PRIMARY_DATA_PATH_TEAM1 = "Thunder_Finals_Shots_2025.csv"
    PRIMARY_DATA_PATH_TEAM2 = "Pacers_Playoffs_2025.csv"
    THUNDER_TEAM_ID = 1610612760   # Oklahoma City Thunder
    PACERS_TEAM_ID  = 1610612754   # Indiana Pacers
    TEAM1_LABEL = "Thunder"
    TEAM2_LABEL = "Pacers"
    KNOWN_PLAYOFF_MINUTES: dict = {
        "Shai Gilgeous-Alexander": 37.5,
        "Jalen Williams":          35.8,
        "Chet Holmgren":           32.4,
        "Luguentz Dort":           30.2,
        "Isaiah Joe":              22.1,
        "Aaron Wiggins":           20.5,
        "Cason Wallace":           18.3,
        "Jaylin Williams":         14.0,
        "Kenrich Williams":        16.0,
        "Tyrese Haliburton":       36.9,
        "Pascal Siakam":           36.2,
        "Myles Turner":            30.5,
        "Andrew Nembhard":         29.8,
        "Bennedict Mathurin":      25.4,
        "Aaron Nesmith":           24.1,
        "Obi Toppin":              18.7,
        "T.J. McConnell":          17.2,
        "Ben Sheppard":            14.5,
    }

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
NN_TEMPERATURE = 1.0   # flattens NN location bias (1.0=full KDE, 0.0=pure NN)

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


