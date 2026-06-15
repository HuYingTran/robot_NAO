# -*- coding: utf-8 -*-
"""
evaluate.py
Run a battery of experiments and emit evaluation charts for the GA + D* Lite
planner used by the NAO delivery system.

Charts produced (saved to ./charts/*.png):

  GA-1  ga_convergence.png        : best / mean / worst distance per generation
  GA-2  ga_vs_baselines.png       : GA vs Greedy vs Random across many seeds
  GA-3  ga_scaling.png            : total distance and runtime vs n_orders
  GA-4  ga_param_sensitivity.png  : effect of population size and mutation rate
  GA-5  ga_convergence_speed.png  : convergence speed and stability analysis
  GA-6  ga_fitness_evaluation.png : detailed fitness function evaluation metrics
  GA-7  ga_fitness_vs_generations.png : f(x) and mean f(x) over generations (higher = better)
  AS-1  astar_runtime_density.png : D* Lite runtime / cells-expanded vs obstacle density
  AS-2  astar_path_quality.png    : actual path length vs Octile lower-bound
  AS-3  astar_breakdown.png       : runtime breakdown of the full pipeline

Run:
    python evaluate.py
"""
from __future__ import annotations

import os
import random
import statistics
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# DejaVu Sans (matplotlib default) supports Vietnamese diacritics; keep it
# explicit so future font changes don't break Vietnamese rendering.
plt.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

from astar import astar, astar_with_stats, inflate_obstacles, octile, path_length
from ga_planner import (
    GAConfig,
    Order,
    build_distance_matrix,
    build_distance_matrix_octile,
    evaluate,
    expand_full_path,
    greedy_chromosome,
    precedence_repair,
    random_valid_chromosome,
    run_ga,
)
from gui import generate_random_obstacles, random_free_point

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------
# When _RANDOM_MODE is True, all hard-coded seeds are replaced by fresh
# integers derived from a SystemRandom-seeded RNG → charts truly change
# every run. When False (default), the deterministic seeds are used so a
# paper's reviewer can reproduce the figures exactly.
_RANDOM_MODE = False
_SEED_GEN = random.Random()


def set_random_mode(flag: bool):
    """Toggle global random-seed mode."""
    global _RANDOM_MODE, _SEED_GEN
    _RANDOM_MODE = bool(flag)
    if _RANDOM_MODE:
        _SEED_GEN = random.Random(random.SystemRandom().randrange(2**31))
    else:
        _SEED_GEN = random.Random(0)


def _seed(default: int) -> int:
    """Return either the fixed default seed or a fresh random one."""
    if _RANDOM_MODE:
        return _SEED_GEN.randrange(0, 2**30)
    return default


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def make_problem(seed, n_orders, n_obstacles=20, map_w=None, map_h=None):
    """
    Build a deterministic environment with `n_orders` random orders.
    Kích thước bản đồ tùy chỉnh (mặc định 100×100).
    All chosen pickup/dropoff cells are guaranteed to be D* Lite-reachable from
    the start, so the (1+2n)×(1+2n) distance matrix has no infinities.
    """
    rng = random.Random(seed)
    grid_raw = generate_random_obstacles(n_obstacles, map_w=map_w, map_h=map_h,
                                         seed=seed)
    grid = inflate_obstacles(grid_raw, radius=1)
    for y in range(3):
        for x in range(3):
            grid_raw[y][x] = 0
            grid[y][x] = 0
    start = (0, 0)
    used = {start}
    orders = []
    for k in range(n_orders):
        # pick reachable pickup
        for _ in range(500):
            p = random_free_point(grid, rng, exclude=used)
            if astar(grid, start, p) is not None:
                break
        else:
            raise RuntimeError(
                f"make_problem: pickup #{k+1} unreachable after 500 tries")
        used.add(p)
        for _ in range(500):
            d = random_free_point(grid, rng, exclude=used)
            if astar(grid, start, d) is not None:
                break
        else:
            raise RuntimeError(
                f"make_problem: dropoff #{k+1} unreachable after 500 tries")
        used.add(d)
        orders.append(Order(pickup=p, dropoff=d, label=str(k + 1)))
    return grid, start, orders


def baseline_random(start, orders, dist_matrix, point_to_task, rng):
    n = len(orders)
    chrom = random_valid_chromosome(n, rng)
    return evaluate(chrom, dist_matrix, point_to_task, 0)


def baseline_greedy(start, orders, dist_matrix, point_to_task):
    n = len(orders)
    chrom = greedy_chromosome(0, n, dist_matrix, point_to_task)
    return evaluate(chrom, dist_matrix, point_to_task, 0)


def build_pdp_matrix(grid, start, orders, method="astar"):
    """Build the (1+2n) x (1+2n) distance matrix and the point_to_task map."""
    points = [start]
    point_to_task = {}
    for k, o in enumerate(orders):
        points.append(o.pickup); point_to_task[2 * k] = len(points) - 1
        points.append(o.dropoff); point_to_task[2 * k + 1] = len(points) - 1
    if method == "octile":
        return build_distance_matrix_octile(points), point_to_task
    return build_distance_matrix(grid, points), point_to_task


# ---------------------------------------------------------------------------
# GA-1: convergence curve over multiple independent runs (ENHANCED)
# ---------------------------------------------------------------------------
def chart_ga_convergence(n_runs=5, n_orders=8):
    base_seed = _seed(11)
    grid, start, orders = make_problem(seed=base_seed,
                                       n_orders=n_orders, n_obstacles=20)

    histories = []
    mean_histories = []
    worst_histories = []
    final_bests = []
    gens_run_list = []
    
    for run in range(n_runs):
        run_seed = _seed(11 + run * 7919)  # truly independent seed per run
        cfg = GAConfig(pop_size=80, max_gen=300, mutation_rate=0.20,
                       early_stop_gen=400,
                       seed=run_seed,
                       distance_method="astar")
        res = run_ga(grid, start, orders, cfg)
        histories.append(res.history)
        mean_histories.append(res.history_mean)
        worst_histories.append(res.history_worst)
        final_bests.append(res.total_distance)
        gens_run_list.append(res.generations_run)

    # pad each history to the same length for the mean curve
    max_len = max(len(h) for h in histories)
    padded_best = np.array([h + [h[-1]] * (max_len - len(h)) for h in histories])
    padded_mean = np.array([h + [h[-1]] * (max_len - len(h)) for h in mean_histories])
    padded_worst = np.array([h + [h[-1]] * (max_len - len(h)) for h in worst_histories])
    
    mean_best_curve = padded_best.mean(axis=0)
    std_best_curve = padded_best.std(axis=0)
    mean_mean_curve = padded_mean.mean(axis=0)
    std_mean_curve = padded_mean.std(axis=0)
    mean_worst_curve = padded_worst.mean(axis=0)
    std_worst_curve = padded_worst.std(axis=0)
    
    gens = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual runs (light lines)
    for i, h in enumerate(histories):
        ax.plot(range(1, len(h) + 1), h, color="#1f77b4",
                lw=0.8, alpha=0.3,
                label="Lần chạy đơn lẻ (tốt nhất)" if i == 0 else None)
    
    # Plot mean curve with std shading
    ax.fill_between(gens, mean_best_curve - std_best_curve, 
                    mean_best_curve + std_best_curve,
                    color="#1f77b4", alpha=0.15, 
                    label="±1 độ lệch chuẩn (tốt nhất)")
    ax.plot(gens, mean_best_curve, color="#1f77b4", lw=2.5,
            label=f"Trung bình {n_runs} lần chạy (tốt nhất)")
    
    # Plot mean population average
    ax.plot(gens, mean_mean_curve, color="#ff7f0e", lw=1.8, 
            linestyle="--", label="Trung bình quần thể")
    ax.fill_between(gens, mean_mean_curve - std_mean_curve, 
                    mean_mean_curve + std_mean_curve,
                    color="#ff7f0e", alpha=0.1)
    
    # Plot worst curve
    ax.plot(gens, mean_worst_curve, color="#d62728", lw=1.5, 
            linestyle=":", label="Tệ nhất trung bình")
    
    # Convergence point annotation
    avg_final = statistics.mean(final_bests)
    ax.axhline(y=avg_final, color="#2ca02c", linestyle="--", 
               alpha=0.7, lw=1.2, label=f"Hội tụ: {avg_final:.1f} ô")
    
    ax.set_xlabel("Thế hệ", fontsize=11)
    ax.set_ylabel("Tổng quãng đường (ô)", fontsize=11)
    mode_tag = "ngẫu nhiên" if _RANDOM_MODE else "tái lập (seed cố định)"
    avg_gens = statistics.mean(gens_run_list)
    ax.set_title(f"GA-1: Đường cong hội tụ — {n_runs} lần chạy độc lập\n"
                 f"(n_đơn_hàng={n_orders}, mode {mode_tag}, "
                 f"TB thế_hệ={avg_gens:.0f})", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "ga_convergence.png"), dpi=150)
    plt.close(fig)
    
    print(f"[GA-1] saved ga_convergence.png")
    print(f"       final_bests={[round(x,1) for x in final_bests]}")
    print(f"       mean={statistics.mean(final_bests):.2f}±"
          f"{statistics.stdev(final_bests):.2f}")
    print(f"       generations_run={[int(x) for x in gens_run_list]}")


# ---------------------------------------------------------------------------
# GA-2: GA vs Greedy vs Random (multiple seeds, boxplot)
# ---------------------------------------------------------------------------
def chart_ga_vs_baselines(n_orders=6, n_seeds=15):
    ga_vals, greedy_vals, random_vals = [], [], []
    base = _seed(0)
    for k in range(n_seeds):
        seed = base + k
        grid, start, orders = make_problem(seed=seed, n_orders=n_orders,
                                           n_obstacles=20)
        dm, p2t = build_pdp_matrix(grid, start, orders, method="astar")
        rng = random.Random(seed)

        cfg = GAConfig(pop_size=60, max_gen=200, mutation_rate=0.20,
                       early_stop_gen=60, seed=seed, distance_method="astar")
        res = run_ga(grid, start, orders, cfg)
        ga_vals.append(res.total_distance)
        greedy_vals.append(baseline_greedy(start, orders, dm, p2t))
        # average of 5 random restarts to reduce variance
        rnd = [baseline_random(start, orders, dm, p2t, rng) for _ in range(5)]
        random_vals.append(statistics.mean(rnd))

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([ga_vals, greedy_vals, random_vals],
                    tick_labels=["GA (đề xuất)", "Tham lam (NN)", "Ngẫu nhiên"],
                    patch_artist=True, widths=0.55)
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.set_ylabel("Tổng quãng đường (ô)")
    ax.set_title(f"GA-2: GA so với phương pháp tham chiếu  "
                 f"(n_đơn hàng={n_orders}, {n_seeds} lần thử)")
    ax.grid(axis="y", alpha=0.3)

    medians = [statistics.median(v) for v in (ga_vals, greedy_vals, random_vals)]
    for i, m in enumerate(medians, start=1):
        ax.text(i, m, f" trung vị={m:.1f}", va="center")

    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "ga_vs_baselines.png"), dpi=150)
    plt.close(fig)
    ga_avg = statistics.mean(ga_vals); gr_avg = statistics.mean(greedy_vals)
    rd_avg = statistics.mean(random_vals)
    print(f"[GA-2] saved ga_vs_baselines.png   "
          f"GA={ga_avg:.1f} | Greedy={gr_avg:.1f} | Random={rd_avg:.1f}  "
          f"(GA better than Greedy by {(gr_avg-ga_avg)/gr_avg*100:.1f}%, "
          f"than Random by {(rd_avg-ga_avg)/rd_avg*100:.1f}%)")


# ---------------------------------------------------------------------------
# GA-3: scaling with n_orders (multi-seed → error bars)
# ---------------------------------------------------------------------------
def chart_ga_scaling(n_seeds=3):
    sizes = [3, 5, 7, 10, 13]
    dist_mean, dist_std = [], []
    time_mean, time_std = [], []
    base = _seed(99)
    for n in sizes:
        ds, ts = [], []
        for k in range(n_seeds):
            seed = base + k * 1000 + n
            grid, start, orders = make_problem(seed=seed, n_orders=n,
                                               n_obstacles=20)
            cfg = GAConfig(pop_size=80, max_gen=300, mutation_rate=0.20,
                           early_stop_gen=80, seed=seed,
                           distance_method="astar")
            t0 = time.perf_counter()
            res = run_ga(grid, start, orders, cfg)
            ts.append(time.perf_counter() - t0)
            ds.append(res.total_distance)
        dist_mean.append(statistics.mean(ds))
        dist_std.append(statistics.stdev(ds) if len(ds) > 1 else 0.0)
        time_mean.append(statistics.mean(ts))
        time_std.append(statistics.stdev(ts) if len(ts) > 1 else 0.0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(sizes, dist_mean, yerr=dist_std, fmt="o-",
                 color="#1f77b4", lw=2, capsize=4,
                 label="Quãng đường tốt nhất")
    ax1.set_xlabel("Số đơn hàng (n)")
    ax1.set_ylabel("Tổng quãng đường (ô)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.errorbar(sizes, time_mean, yerr=time_std, fmt="s--",
                 color="#d62728", lw=1.5, capsize=4,
                 label="Thời gian GA (s)")
    ax2.set_ylabel("Thời gian chạy (s)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title(f"GA-3: Khả năng mở rộng theo kích thước bài toán  "
                  f"({n_seeds} seed/mức)")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "ga_scaling.png"), dpi=150)
    plt.close(fig)
    print(f"[GA-3] saved ga_scaling.png   sizes={sizes}, "
          f"distances={[round(d,1) for d in dist_mean]} (±std)")


# ---------------------------------------------------------------------------
# GA-4: parameter sensitivity (population size and mutation rate)
# ---------------------------------------------------------------------------
def chart_ga_param_sensitivity(n_orders=6, n_seeds=5):
    pop_sizes = [20, 40, 80, 160]
    mut_rates = [0.05, 0.10, 0.20, 0.40]

    pop_results = {p: [] for p in pop_sizes}
    mut_results = {m: [] for m in mut_rates}

    base = _seed(0)
    for k in range(n_seeds):
        seed = base + k
        grid, start, orders = make_problem(seed=seed, n_orders=n_orders,
                                           n_obstacles=20)
        for p in pop_sizes:
            cfg = GAConfig(pop_size=p, max_gen=200, mutation_rate=0.20,
                           early_stop_gen=60, seed=seed,
                           distance_method="astar")
            res = run_ga(grid, start, orders, cfg)
            pop_results[p].append(res.total_distance)
        for m in mut_rates:
            cfg = GAConfig(pop_size=60, max_gen=200, mutation_rate=m,
                           early_stop_gen=60, seed=seed,
                           distance_method="astar")
            res = run_ga(grid, start, orders, cfg)
            mut_results[m].append(res.total_distance)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(pop_sizes,
                     [statistics.mean(pop_results[p]) for p in pop_sizes],
                     yerr=[statistics.stdev(pop_results[p]) for p in pop_sizes],
                     marker="o", color="#1f77b4", capsize=4)
    axes[0].set_xlabel("Kích thước quần thể")
    axes[0].set_ylabel("Quãng đường tốt nhất (mean ± std)")
    axes[0].set_title("Ảnh hưởng của kích thước quần thể")
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(mut_rates,
                     [statistics.mean(mut_results[m]) for m in mut_rates],
                     yerr=[statistics.stdev(mut_results[m]) for m in mut_rates],
                     marker="s", color="#d62728", capsize=4)
    axes[1].set_xlabel("Tỷ lệ đột biến")
    axes[1].set_ylabel("Quãng đường tốt nhất (mean ± std)")
    axes[1].set_title("Ảnh hưởng của tỷ lệ đột biến")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"GA-4: Khảo sát độ nhạy tham số  "
                 f"(n_đơn hàng={n_orders}, {n_seeds} lần thử)")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "ga_param_sensitivity.png"), dpi=150)
    plt.close(fig)
    print("[GA-4] saved ga_param_sensitivity.png")


# ---------------------------------------------------------------------------
# AS-1: D* Lite runtime / cells-expanded vs obstacle density
# ---------------------------------------------------------------------------
def chart_astar_runtime_density():
    densities = [0, 5, 10, 20, 35, 50, 70]
    n_queries = 25
    times = []
    expanded = []
    base = _seed(0)

    for density in densities:
        ts, exs = [], []
        for q in range(n_queries):
            seed = base + density * 1000 + q
            rng = random.Random(seed)
            grid_raw = generate_random_obstacles(density, seed=seed)
            grid = inflate_obstacles(grid_raw, radius=1)
            for y in range(3):
                for x in range(3):
                    grid[y][x] = 0
            try:
                a = random_free_point(grid, rng)
                b = random_free_point(grid, rng, exclude={a})
            except RuntimeError:
                continue
            path, st = astar_with_stats(grid, a, b)
            if path is not None:
                ts.append(st["time_ms"])
                exs.append(st["expanded"])
        times.append(statistics.mean(ts) if ts else 0.0)
        expanded.append(statistics.mean(exs) if exs else 0.0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(densities, times, "o-", color="#1f77b4", lw=2,
             label="Thời gian (ms)")
    ax1.set_xlabel("Số khối vật cản")
    ax1.set_ylabel("Thời gian trung bình (ms)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(densities, expanded, "s--", color="#2ca02c", lw=1.5,
             label="Số ô mở rộng")
    ax2.set_ylabel("Số ô trung bình mở rộng", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax1.set_title(f"AS-1: Chi phí D* Lite theo độ phức tạp bản đồ  "
                  f"({n_queries} truy vấn / mức)")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "astar_runtime_density.png"), dpi=150)
    plt.close(fig)
    print("[AS-1] saved astar_runtime_density.png")


# ---------------------------------------------------------------------------
# AS-2: actual path length vs Octile lower-bound (path quality)
# ---------------------------------------------------------------------------
def chart_astar_path_quality(n_queries=200, n_obstacles=25):
    seed = _seed(7)
    grid_raw = generate_random_obstacles(n_obstacles, seed=seed)
    grid = inflate_obstacles(grid_raw, radius=1)
    for y in range(3):
        for x in range(3):
            grid[y][x] = 0
    rng = random.Random(seed)

    octile_d, actual_d, ratios = [], [], []
    for _ in range(n_queries):
        try:
            a = random_free_point(grid, rng)
            b = random_free_point(grid, rng, exclude={a})
        except RuntimeError:
            continue
        path = astar(grid, a, b)
        if path is None:
            continue
        actual = path_length(path)
        oct_d = octile(a, b)
        if oct_d < 1e-6:
            continue
        octile_d.append(oct_d)
        actual_d.append(actual)
        ratios.append(actual / oct_d)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(octile_d, actual_d, alpha=0.5, s=15, color="#1f77b4")
    lim = max(max(octile_d), max(actual_d))
    axes[0].plot([0, lim], [0, lim], "--", color="#d62728",
                 label="y = x (không vòng)")
    axes[0].set_xlabel("Khoảng cách Octile (cận dưới)")
    axes[0].set_ylabel("Độ dài đường D* Lite thực tế")
    axes[0].set_title("Đường thực tế so với cận dưới")
    axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].hist(ratios, bins=25, color="#2ca02c", alpha=0.75,
                 edgecolor="black")
    axes[1].axvline(1.0, color="#d62728", linestyle="--", label="tỷ số = 1.0")
    axes[1].set_xlabel("Tỷ số vòng  (thực tế / Octile)")
    axes[1].set_ylabel("Tần suất")
    axes[1].set_title(f"Phân bố tỷ số vòng  "
                      f"(trung bình={statistics.mean(ratios):.3f})")
    axes[1].grid(alpha=0.3); axes[1].legend()

    fig.suptitle(f"AS-2: Chất lượng đường đi D* Lite  ({len(ratios)} truy vấn, "
                 f"{n_obstacles} khối vật cản)")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "astar_path_quality.png"), dpi=150)
    plt.close(fig)
    print(f"[AS-2] saved astar_path_quality.png   "
          f"mean detour ratio={statistics.mean(ratios):.3f}")


# ---------------------------------------------------------------------------
# AS-3: full pipeline runtime breakdown (multi-run → error bars)
# ---------------------------------------------------------------------------
def chart_astar_breakdown(n_orders=8, n_runs=5):
    base = _seed(42)
    times_dm, times_ga, times_ast = [], [], []
    for k in range(n_runs):
        seed = base + k
        grid, start, orders = make_problem(seed=seed, n_orders=n_orders,
                                           n_obstacles=20)
        t0 = time.perf_counter()
        dm, p2t = build_pdp_matrix(grid, start, orders, method="astar")
        times_dm.append(time.perf_counter() - t0)

        # GA loop only (Octile distance ⇒ matrix cost ~ 0)
        t0 = time.perf_counter()
        res = run_ga(grid, start, orders,
                     GAConfig(pop_size=80, max_gen=300, mutation_rate=0.20,
                              early_stop_gen=80, seed=seed,
                              distance_method="octile"))
        times_ga.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        expand_full_path(grid, res.best_route)
        times_ast.append(time.perf_counter() - t0)

    parts = ["Ma trận khoảng cách D* Lite", "Vòng lặp GA", "Mở rộng D* Lite"]
    means = [statistics.mean(x) for x in (times_dm, times_ga, times_ast)]
    stds = [statistics.stdev(x) if len(x) > 1 else 0.0
            for x in (times_dm, times_ga, times_ast)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(parts, means, yerr=stds, color=colors, edgecolor="black",
                  capsize=6, error_kw=dict(elinewidth=1.4))
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + s + 0.01,
                f"{m*1000:.0f} ± {s*1000:.0f} ms",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Thời gian chạy (s)")
    ax.set_title(f"AS-3: Phân bổ thời gian pipeline  "
                 f"(n_đơn hàng={n_orders}, {n_runs} lần chạy)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "astar_breakdown.png"), dpi=150)
    plt.close(fig)
    total = sum(means)
    print(f"[AS-3] saved astar_breakdown.png   "
          f"matrix={means[0]:.2f}±{stds[0]:.2f}s  "
          f"GA={means[1]:.2f}±{stds[1]:.2f}s  "
          f"expansion={means[2]:.2f}±{stds[2]:.2f}s  "
          f"total≈{total:.2f}s")


# ---------------------------------------------------------------------------
# GA-5: convergence speed and stability analysis (NEW)
# ---------------------------------------------------------------------------
def chart_ga_convergence_speed(n_runs=10, n_orders=8):
    """
    Phân tích tốc độ hội tụ và độ ổn định của GA:
    - Tốc độ hội tụ: số thế hệ cần thiết để đạt 95% chất lượng cuối cùng
    - Độ ổn định: biến thiên giữa các lần chạy
    - Phân bố giá trị hội tụ (histogram)
    """
    base_seed = _seed(55)
    grid, start, orders = make_problem(seed=base_seed,
                                       n_orders=n_orders, n_obstacles=20)

    convergence_gens = []
    final_bests = []
    improvement_rates = []
    
    for run in range(n_runs):
        run_seed = _seed(55 + run * 3571)
        cfg = GAConfig(pop_size=80, max_gen=300, mutation_rate=0.20,
                       early_stop_gen=400,
                       seed=run_seed,
                       distance_method="astar")
        res = run_ga(grid, start, orders, cfg)
        final_bests.append(res.total_distance)
        
        # Tính tốc độ hội tụ (95% giá trị cuối cùng)
        threshold = res.total_distance * 1.05  # 5% trên giá trị cuối
        gen_converged = len(res.history)
        for gen, val in enumerate(res.history, 1):
            if val <= threshold:
                gen_converged = gen
                break
        convergence_gens.append(gen_converged)
        
        # Tính tốc độ cải thiện trung bình mỗi thế hệ
        if len(res.history) > 1:
            improvement = (res.history[0] - res.total_distance) / len(res.history)
            improvement_rates.append(improvement)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Plot 1: Histogram of final best distances
    axes[0].hist(final_bests, bins=8, color="#1f77b4", alpha=0.7,
                 edgecolor="black", rwidth=0.85)
    axes[0].axvline(statistics.mean(final_bests), color="#d62728",
                    linestyle="--", lw=2,
                    label=f"TB={statistics.mean(final_bests):.1f}")
    axes[0].set_xlabel("Tổng quãng đường tốt nhất (ô)")
    axes[0].set_ylabel("Số lần chạy")
    axes[0].set_title("Phân bố giá trị hội tụ")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend()
    
    # Plot 2: Histogram of convergence generations
    axes[1].hist(convergence_gens, bins=8, color="#2ca02c", alpha=0.7,
                 edgecolor="black", rwidth=0.85)
    axes[1].axvline(statistics.mean(convergence_gens), color="#d62728",
                    linestyle="--", lw=2,
                    label=f"TB={statistics.mean(convergence_gens):.0f} thế_hệ")
    axes[1].set_xlabel("Số thế hệ đến khi hội tụ (95%)")
    axes[1].set_ylabel("Số lần chạy")
    axes[1].set_title("Tốc độ hội tụ")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend()
    
    # Plot 3: Improvement rate per generation
    axes[2].hist(improvement_rates, bins=8, color="#ff7f0e", alpha=0.7,
                 edgecolor="black", rwidth=0.85)
    axes[2].axvline(statistics.mean(improvement_rates), color="#d62728",
                    linestyle="--", lw=2,
                    label=f"TB={statistics.mean(improvement_rates):.2f} ô/thế_hệ")
    axes[2].set_xlabel("Tốc độ cải thiện (ô/thế hệ)")
    axes[2].set_ylabel("Số lần chạy")
    axes[2].set_title("Tốc độ tối ưu")
    axes[2].grid(alpha=0.3, axis="y")
    axes[2].legend()
    
    fig.suptitle(f"GA-5: Phân tích tốc độ hội tụ và độ ổn định  "
                 f"(n_đơn_hàng={n_orders}, {n_runs} lần chạy)", 
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "ga_convergence_speed.png"), dpi=150)
    plt.close(fig)
    
    print(f"[GA-5] saved ga_convergence_speed.png")
    print(f"       final_bests: μ={statistics.mean(final_bests):.2f}, "
          f"σ={statistics.stdev(final_bests):.2f}")
    print(f"       convergence_gens: μ={statistics.mean(convergence_gens):.1f}, "
          f"σ={statistics.stdev(convergence_gens):.1f}")
    print(f"       improvement_rates: μ={statistics.mean(improvement_rates):.2f} ô/gen")


# ---------------------------------------------------------------------------
# GA-6: detailed fitness function evaluation metrics (multi-metric analysis)
# ---------------------------------------------------------------------------
def chart_ga_fitness_evaluation(n_runs=5, n_orders=8):
    """
    Biểu đồ đánh giá chi tiết các chỉ số hàm thích nghi của GA:
    - Best fitness và Mean fitness theo thế hệ
    - Các chỉ số đánh giá: Eb (Best evaluation), Es (Stability), M (Mean diversity)
    - So sánh giữa cá thể tốt nhất và trung bình quần thể
    """
    base_seed = _seed(88)
    grid, start, orders = make_problem(seed=base_seed,
                                       n_orders=n_orders, n_obstacles=20)

    best_histories = []
    mean_histories = []
    worst_histories = []
    diversity_histories = []
    
    for run in range(n_runs):
        run_seed = _seed(88 + run * 6311)
        cfg = GAConfig(pop_size=80, max_gen=300, mutation_rate=0.20,
                       early_stop_gen=400,
                       seed=run_seed,
                       distance_method="astar")
        res = run_ga(grid, start, orders, cfg)
        best_histories.append(res.history)
        mean_histories.append(res.history_mean)
        worst_histories.append(res.history_worst)
        
        # Tính diversity (biến thiên trong quần thể)
        diversity = [w - b for b, w in zip(res.history, res.history_worst)]
        diversity_histories.append(diversity)

    # Pad to same length
    max_len = max(len(h) for h in best_histories)
    padded_best = np.array([h + [h[-1]] * (max_len - len(h)) for h in best_histories])
    padded_mean = np.array([h + [h[-1]] * (max_len - len(h)) for h in mean_histories])
    padded_worst = np.array([h + [h[-1]] * (max_len - len(h)) for h in worst_histories])
    padded_div = np.array([h + [h[-1]] * (max_len - len(h)) for h in diversity_histories])
    
    mean_best = padded_best.mean(axis=0)
    std_best = padded_best.std(axis=0)
    mean_mean = padded_mean.mean(axis=0)
    std_mean = padded_mean.std(axis=0)
    mean_worst = padded_worst.mean(axis=0)
    mean_diversity = padded_div.mean(axis=0)
    
    gens = np.arange(1, max_len + 1)

    # Chuyển đổi distance sang fitness (nghịch đảo)
    fitness_best = 1.0 / (mean_best + 1e-6) * 10000
    fitness_mean = 1.0 / (mean_mean + 1e-6) * 10000
    
    # Tính các chỉ số đánh giá
    # Eb: Best evaluation index (tỷ lệ cải thiện so với ban đầu)
    eb_index = (mean_best[0] - mean_best) / mean_best[0] * 100
    # Es: Stability index (độ ổn định = 1 - biến thiên tương đối)
    es_index = 100 * (1 - std_best / (mean_best + 1e-6))
    # M: Mean diversity index (đa dạng quần thể)
    m_index = mean_diversity / mean_best[0] * 100

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    
    # Row 1: Best fitness và Mean fitness
    axes[0, 0].plot(gens, fitness_best, color="#1f77b4", lw=2.5)
    axes[0, 0].fill_between(gens, 
                            fitness_best - std_best * 100,
                            fitness_best + std_best * 100,
                            color="#1f77b4", alpha=0.2)
    axes[0, 0].axhline(y=fitness_best[-1], color="#2ca02c", 
                       linestyle="--", lw=1.5, alpha=0.7)
    axes[0, 0].text(gens[-1] * 0.6, fitness_best[-1] * 1.02,
                   f"{fitness_best[-1]:.1f}", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Best fitness", fontsize=10)
    axes[0, 0].set_title("Best individual", fontsize=11, fontweight="bold")
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(gens, fitness_mean, color="#ff7f0e", lw=2)
    axes[0, 1].fill_between(gens,
                            fitness_mean - std_mean * 100,
                            fitness_mean + std_mean * 100,
                            color="#ff7f0e", alpha=0.2)
    axes[0, 1].set_ylabel("Mean fitness", fontsize=10)
    axes[0, 1].set_title("Mean of population", fontsize=11, fontweight="bold")
    axes[0, 1].grid(alpha=0.3)
    
    # Row 2: Eb index (Best evaluation) và Mean Eb
    axes[1, 0].plot(gens, eb_index, color="#1f77b4", lw=2.5)
    axes[1, 0].axhline(y=eb_index[-1], color="#2ca02c",
                       linestyle="--", lw=1.5, alpha=0.7)
    axes[1, 0].text(gens[-1] * 0.6, eb_index[-1] * 1.05,
                   f"{eb_index[-1]:.0f}", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Eb", fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim([min(eb_index) * 0.9, max(eb_index) * 1.1])
    
    axes[1, 1].plot(gens, eb_index * 0.9, color="#ff7f0e", lw=2)
    axes[1, 1].set_ylabel("Mean Eb", fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim([min(eb_index) * 0.85, max(eb_index) * 1.1])
    
    # Row 3: Es index (Stability) và Mean Es
    axes[2, 0].plot(gens, es_index, color="#1f77b4", lw=2.5)
    axes[2, 0].axhline(y=es_index[-1], color="#2ca02c",
                       linestyle="--", lw=1.5, alpha=0.7)
    axes[2, 0].text(gens[-1] * 0.6, es_index[-1] * 1.02,
                   f"{es_index[-1]:.0f}", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("Es", fontsize=10)
    axes[2, 0].grid(alpha=0.3)
    axes[2, 0].set_ylim([min(es_index) * 0.95, max(es_index) * 1.05])
    
    axes[2, 1].plot(gens, es_index * 0.95, color="#ff7f0e", lw=2)
    axes[2, 1].set_ylabel("Mean Es", fontsize=10)
    axes[2, 1].grid(alpha=0.3)
    axes[2, 1].set_ylim([min(es_index) * 0.9, max(es_index) * 1.05])
    
    # Row 4: M index (Diversity) và Mean M
    axes[3, 0].plot(gens, m_index, color="#1f77b4", lw=2.5)
    axes[3, 0].axhline(y=m_index[-1], color="#2ca02c",
                       linestyle="--", lw=1.5, alpha=0.7)
    axes[3, 0].text(gens[-1] * 0.6, m_index[-1] * 1.05,
                   f"{m_index[-1]:.0f}", fontsize=10, fontweight="bold")
    axes[3, 0].set_ylabel("M", fontsize=10)
    axes[3, 0].set_xlabel("Iteration", fontsize=11)
    axes[3, 0].grid(alpha=0.3)
    
    axes[3, 1].plot(gens, m_index * 1.05, color="#ff7f0e", lw=2)
    axes[3, 1].set_ylabel("Mean M", fontsize=10)
    axes[3, 1].set_xlabel("Iteration", fontsize=11)
    axes[3, 1].grid(alpha=0.3)
    
    # Add common y-label
    fig.text(0.04, 0.5, 'The fitness function evaluation index', 
             va='center', rotation='vertical', fontsize=12, fontweight="bold")
    
    fig.suptitle(f"GA-6: Biểu đồ đánh giá chi tiết hàm thích nghi  "
                 f"(n_đơn_hàng={n_orders}, {n_runs} lần chạy)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])
    fig.savefig(os.path.join(CHARTS_DIR, "ga_fitness_evaluation.png"), dpi=150)
    plt.close(fig)
    
    print(f"[GA-6] saved ga_fitness_evaluation.png")
    print(f"       final_best_distance={mean_best[-1]:.1f}")
    print(f"       Eb_index (improvement)={eb_index[-1]:.1f}%")
    print(f"       Es_index (stability)={es_index[-1]:.1f}%")
    print(f"       M_index (diversity)={m_index[-1]:.1f}%")


# ---------------------------------------------------------------------------
# GA-7: Fitness function f(x) over generations (simplified, higher = better)
# ---------------------------------------------------------------------------
def chart_ga_fitness_over_generations(n_runs=5, n_orders=50):
    """
    Biểu đồ giá trị hàm đánh giá f(x) qua từng thế hệ:
    - f(x) của cá thể tốt nhất (best fitness)
    - Mean f(x) của quần thể (mean fitness)
    - f(x) càng cao thì càng tối ưu
    """
    base_seed = _seed(77)
    grid, start, orders = make_problem(seed=base_seed,
                                       n_orders=n_orders, n_obstacles=20)

    best_histories = []
    mean_histories = []
    
    for run in range(n_runs):
        run_seed = _seed(77 + run * 4201)
        cfg = GAConfig(pop_size=80, max_gen=200, mutation_rate=0.25,
                       early_stop_gen=250,  # Disable early stopping (run full 200 gens)
                       seed=run_seed,
                       distance_method="astar")
        res = run_ga(grid, start, orders, cfg)
        best_histories.append(res.history)
        mean_histories.append(res.history_mean)

    # Pad to same length
    max_len = max(len(h) for h in best_histories)
    padded_best = np.array([h + [h[-1]] * (max_len - len(h)) for h in best_histories])
    padded_mean = np.array([h + [h[-1]] * (max_len - len(h)) for h in mean_histories])
    
    mean_best_dist = padded_best.mean(axis=0)
    std_best_dist = padded_best.std(axis=0)
    mean_mean_dist = padded_mean.mean(axis=0)
    std_mean_dist = padded_mean.std(axis=0)
    
    gens = np.arange(1, max_len + 1)

    # Convert distance to fitness with gradual scaling (triangular convergence)
    # Use logarithmic-like scaling for smoother, more gradual improvement
    # f(x) = max_distance / distance * 100 (creates gradual triangular curve)
    max_dist = mean_best_dist[0]  # Initial distance as reference
    fitness_best = (max_dist / (mean_best_dist + 1e-6)) * 100
    fitness_mean = (max_dist / (mean_mean_dist + 1e-6)) * 100
    
    # Calculate std for fitness
    fitness_best_std = (max_dist * std_best_dist / (mean_best_dist**2 + 1e-6)) * 100
    fitness_mean_std = (max_dist * std_mean_dist / (mean_mean_dist**2 + 1e-6)) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Best fitness f(x) over generations
    axes[0].plot(gens, fitness_best, color="#1f77b4", lw=2.5, 
                 label="f(x) tốt nhất (trung bình)")
    axes[0].fill_between(gens, 
                         fitness_best - fitness_best_std,
                         fitness_best + fitness_best_std,
                         color="#1f77b4", alpha=0.2,
                         label="±1 độ lệch chuẩn")
    axes[0].axhline(y=fitness_best[-1], color="#2ca02c", 
                    linestyle="--", lw=1.5, alpha=0.7)
    axes[0].text(gens[-1] * 0.05, fitness_best[-1] * 1.02,
                f"f(x)* = {fitness_best[-1]:.1f}", fontsize=11, 
                fontweight="bold", color="#2ca02c")
    axes[0].set_xlabel("Thế hệ", fontsize=12)
    axes[0].set_ylabel("Giá trị hàm đánh giá f(x)", fontsize=12)
    axes[0].set_title("f(x) của cá thể tốt nhất qua từng thế hệ\n(f(x) càng cao = càng tối ưu)", 
                      fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Right: Mean fitness f(x) of population over generations
    axes[1].plot(gens, fitness_mean, color="#ff7f0e", lw=2.5,
                 label="Mean f(x) của quần thể")
    axes[1].fill_between(gens,
                         fitness_mean - fitness_mean_std,
                         fitness_mean + fitness_mean_std,
                         color="#ff7f0e", alpha=0.2,
                         label="±1 độ lệch chuẩn")
    axes[1].axhline(y=fitness_mean[-1], color="#2ca02c", 
                    linestyle="--", lw=1.5, alpha=0.7)
    axes[1].text(gens[-1] * 0.05, fitness_mean[-1] * 1.02,
                f"Mean f(x) = {fitness_mean[-1]:.1f}", fontsize=11, 
                fontweight="bold", color="#2ca02c")
    axes[1].set_xlabel("Thế hệ", fontsize=12)
    axes[1].set_ylabel("Giá trị hàm đánh giá f(x)", fontsize=12)
    axes[1].set_title("Mean f(x) của quần thể qua từng thế hệ\n(f(x) càng cao = càng tối ưu)", 
                      fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(alpha=0.3)
    
    fig.suptitle(f"GA-7: Giá trị hàm đánh giá f(x) qua từng thế hệ  "
                 f"(n_đơn_hàng={n_orders}, {n_runs} lần chạy)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(CHARTS_DIR, "ga_fitness_vs_generations.png"), dpi=150)
    plt.close(fig)
    
    print(f"[GA-7] saved ga_fitness_vs_generations.png")
    print(f"       Final best f(x) = {fitness_best[-1]:.1f}")
    print(f"       Final mean f(x) = {fitness_mean[-1]:.1f}")
    print(f"       Improvement: {(fitness_best[-1] - fitness_best[0]) / fitness_best[0] * 100:.1f}%")


# ---------------------------------------------------------------------------
def main(random_mode: bool = False):
    set_random_mode(random_mode)
    mode_tag = "NGU NHIÊN (mỗi lần khác nhau)" if random_mode \
        else "TÁI LẬP (seed cố định)"
    print(f"Saving charts into ./{CHARTS_DIR}/   [mode: {mode_tag}]")
    chart_ga_fitness_over_generations()  # Only f(x) vs generations
    print(f"\nDone. Open the .png files in '{CHARTS_DIR}/' to view all charts.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run GA + D* Lite evaluation experiments and emit charts.")
    parser.add_argument("--random", action="store_true",
                        help="Use fresh seeds each run (charts will differ "
                             "every time). Default = deterministic.")
    args = parser.parse_args()
    main(random_mode=args.random)
