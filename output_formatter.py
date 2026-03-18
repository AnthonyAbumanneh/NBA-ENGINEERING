"""
Output formatter for the NBA Court Re-Engineering Optimizer.

Prints the historical baseline, top-5 rankings, and combined-optimal detail.
"""

from typing import Dict, List, Optional, Tuple


def print_results(
    historical_warriors_3pt: float,
    historical_cavs_3pt: float,
    cavs_top5: List[Tuple],
    warriors_top5: List[Tuple],
    combined_top5: List[Tuple],
    combined_optimal_cfg: Optional[Tuple[float, float]],
    combined_optimal_result: dict,
    player_names: Optional[Dict] = None,
) -> None:
    """
    Print the full optimizer output to stdout.

    Parameters
    ----------
    historical_warriors_3pt : float
        Warriors 3PT% from the warriors_cavs_2014 dataset.
    historical_cavs_3pt : float
        Cavs 3PT% from the warriors_cavs_2014 dataset.
    cavs_top5 : list of (rank, (arc_r, bw), result_dict)
    warriors_top5 : list of (rank, (arc_r, bw), result_dict)
    combined_top5 : list of (rank, (arc_r, bw), result_dict)
    combined_optimal_cfg : (arc_radius, baseline_width) or None
    combined_optimal_result : dict
    player_names : dict, optional
        {player_id: display_name}
    """
    _print_historical_baseline(historical_warriors_3pt, historical_cavs_3pt)
    _print_top5_section("Cavs-Optimal", cavs_top5, "cavs_3pt_pct")
    _print_top5_section("Warriors-Optimal", warriors_top5, "warriors_3pt_pct")
    _print_top5_section("Combined-Optimal", combined_top5, "combined_3pt_pct")
    _print_best_court(combined_optimal_cfg, combined_optimal_result)
    _print_per_player_stats(combined_optimal_result, player_names)


def _print_historical_baseline(warriors_3pt: float, cavs_3pt: float) -> None:
    print()
    print("=" * 60)
    print("  Historical Baseline (Standard NBA Court)")
    print("=" * 60)
    print(f"  Warriors 3PT%:  {warriors_3pt * 100:.1f}%")
    print(f"  Cavs 3PT%:      {cavs_3pt * 100:.1f}%")
    print()


def _print_top5_section(label: str, top5: List[Tuple], pct_key: str) -> None:
    print("-" * 60)
    print(f"  {label}: Top 5")
    print("-" * 60)
    if not top5:
        print("  (no results)")
        print()
        return

    for rank, (arc_r, bw), result in top5:
        pct = result.get(pct_key, 0.0) * 100
        c3 = "Yes" if result.get("corner3_eliminated", False) else "No"
        print(
            f"  Rank {rank}: Arc {arc_r:.2f} ft | Baseline {bw:.2f} ft | "
            f"{pct_key.replace('_', ' ').title()}: {pct:.1f}% | "
            f"Corner 3 eliminated: {c3}"
        )
    print()


def _print_best_court(
    cfg: Optional[Tuple[float, float]],
    result: dict,
) -> None:
    print("=" * 60)
    print("  Best Court (Combined-Optimal #1)")
    print("=" * 60)
    if cfg is None:
        print("  (no result available)")
        print()
        return

    arc_r, bw = cfg
    cavs_pct = result.get("cavs_3pt_pct", 0.0) * 100
    warriors_pct = result.get("warriors_3pt_pct", 0.0) * 100
    combined_pct = result.get("combined_3pt_pct", 0.0) * 100
    c3 = "Yes" if result.get("corner3_eliminated", False) else "No"

    print(f"  Arc radius:          {arc_r:.2f} ft")
    print(f"  Baseline width:      {bw:.2f} ft")
    print(f"  Cavs 3PT%:           {cavs_pct:.1f}%")
    print(f"  Warriors 3PT%:       {warriors_pct:.1f}%")
    print(f"  Combined 3PT%:       {combined_pct:.1f}%")
    print(f"  Corner 3 eliminated: {c3}")
    print()


def _print_per_player_stats(
    result: dict,
    player_names: Optional[Dict] = None,
) -> None:
    per_player = result.get("per_player", {})
    if not per_player:
        return

    player_names = player_names or {}
    print("-" * 60)
    print("  Per-Player Stats (Combined-Optimal Court)")
    print("-" * 60)
    print(f"  {'Player':<30} {'PPG':>6} {'3PT%':>7} {'3PA/G':>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*7}")

    for pid, stats in sorted(per_player.items(), key=lambda kv: str(kv[0])):
        name = player_names.get(pid, str(pid))
        ppg = stats.get("ppg", 0.0)
        three_pct = stats.get("three_pt_pct", 0.0) * 100
        three_pa = stats.get("three_pa_per_game", 0.0)
        print(f"  {name:<30} {ppg:>6.1f} {three_pct:>6.1f}% {three_pa:>7.1f}")
    print()
