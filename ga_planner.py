# -*- coding: utf-8 -*-
"""
ga_planner.py
Genetic Algorithm for the Pickup-and-Delivery Problem (PDP) for the NAO robot.

Chromosome encoding:
    A permutation of 2n task indices.
    Each order o_k has 2 tasks:
        - Pickup task:  index 2k   (notation "P{k}")
        - Delivery task: index 2k+1 (notation "D{k}")
    Constraint: in every valid chromosome, the pickup task of
    order k must appear BEFORE the delivery task of order k.

Genetic operators:
    - Tournament selection (k=3)
    - OX-PR crossover (Order Crossover + Precedence Repair)
    - Swap mutation with constraint repair
    - Elitism (top 5% of population preserved)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from astar import astar, octile, path_length, Point


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """An order: picked up at `pickup`, delivered to `dropoff`."""
    pickup: Point       # (x, y) coordinates of pickup location
    dropoff: Point      # (x, y) coordinates of delivery location
    label: str = ""     # Display label (e.g., "1", "2", …)


@dataclass
class GAConfig:
    """Configuration parameters for the genetic algorithm."""
    pop_size: int = 100            # Population size
    max_gen: int = 500             # Maximum number of generations
    crossover_rate: float = 0.85   # Crossover probability
    mutation_rate: float = 0.15    # Mutation probability
    tournament_k: int = 3          # Number of contenders per tournament
    elitism_ratio: float = 0.05    # Fraction of population preserved as elites
    early_stop_gen: int = 80       # Early stop if no improvement for N generations
    seed: Optional[int] = None     # Random seed (for reproducibility)
    distance_method: str = "astar" # "astar" (obstacle-aware) or "octile" (fast)


@dataclass
class GAResult:
    """Result returned from the GA algorithm."""
    best_sequence: List[int]       # Best chromosome (sequence of 2n task indices)
    best_route: List[Point]        # Route: [start, task_1, task_2, …, task_2n]
    total_distance: float          # Total distance (cells)
    generations_run: int           # Actual number of generations executed
    history: List[float] = field(default_factory=list)         # Best per generation
    history_mean: List[float] = field(default_factory=list)    # Mean per generation
    history_worst: List[float] = field(default_factory=list)   # Worst per generation


# ---------------------------------------------------------------------------
# Chromosome encoding helpers
# ---------------------------------------------------------------------------

def task_is_pickup(task_idx: int) -> bool:
    """Check if task_idx is a pickup task (even index = pickup)."""
    return task_idx % 2 == 0


def task_order_id(task_idx: int) -> int:
    """Return the order index corresponding to task_idx."""
    return task_idx // 2


def task_point(task_idx: int, orders: List[Order]) -> Point:
    """Return the (x, y) coordinates of task_idx."""
    o = orders[task_order_id(task_idx)]
    return o.pickup if task_is_pickup(task_idx) else o.dropoff


def chromosome_to_label(chrom: List[int], orders: List[Order]) -> str:
    """Convert chromosome to a readable string, e.g., 'P1 -> D1 -> P2 -> D2'."""
    parts = []
    for t in chrom:
        k = task_order_id(t)
        tag = "P" if task_is_pickup(t) else "D"
        label = orders[k].label or str(k + 1)
        parts.append(f"{tag}{label}")
    return " -> ".join(parts)


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------

def build_distance_matrix(
    grid: List[List[int]],
    points: List[Point],
) -> List[List[float]]:
    """
    Precompute A* distances between all pairs of points.

    Returns an NxN matrix (N = len(points)).
    points[0] is usually the starting position (Start).
    Distance = total A* path cost (√2 for diagonal, 1 for straight).
    If no path exists → value = inf.
    """
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            path = astar(grid, points[i], points[j])
            d = path_length(path)       # inf if path is None
            dist[i][j] = d
            dist[j][i] = d              # Symmetric matrix
    return dist


def build_distance_matrix_octile(points: List[Point]) -> List[List[float]]:
    """
    Fast distance matrix using Octile heuristic (ignores obstacles).

    Used in Step 1 (GA-only): GA optimizes task order based on Octile
    distance (lower bound), then Step 2 (A*) computes the actual obstacle-free path.
    """
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = octile(points[i], points[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist


# ---------------------------------------------------------------------------
# Precedence repair
# ---------------------------------------------------------------------------

def precedence_repair(chrom: List[int]) -> List[int]:
    """
    Ensure that for every order k, pickup (2k) comes before delivery (2k+1).

    If a violation is found → swap the two tasks in the chromosome.
    This step is mandatory after every crossover / mutation operation.
    """
    chrom = chrom[:]                        # copy to avoid modifying original
    pos = {t: i for i, t in enumerate(chrom)}  # current position of each task
    n_orders = len(chrom) // 2
    for k in range(n_orders):
        p_idx = pos[2 * k]       # pickup position
        d_idx = pos[2 * k + 1]   # delivery position
        if d_idx < p_idx:        # violation → swap
            chrom[p_idx], chrom[d_idx] = chrom[d_idx], chrom[p_idx]
            pos[2 * k] = d_idx
            pos[2 * k + 1] = p_idx
    return chrom


# ---------------------------------------------------------------------------
# Population initialization
# ---------------------------------------------------------------------------

def random_valid_chromosome(n_orders: int, rng: random.Random) -> List[int]:
    """
    Create a random valid chromosome:
    random permutation of 2n tasks, then repair constraints.
    """
    chrom = list(range(2 * n_orders))
    rng.shuffle(chrom)
    return precedence_repair(chrom)


def greedy_chromosome(
    start_idx: int,
    n_orders: int,
    dist_matrix: List[List[float]],
    point_to_task: dict,
) -> List[int]:
    """
    Build a greedy (Nearest Neighbor) chromosome respecting constraints:
      - From the current position, pick the unvisited task with the shortest distance.
      - Only allow delivery if the corresponding pickup has already been completed.
    """
    visited_tasks: set = set()       # Set of already-scheduled tasks
    picked_orders: set = set()       # Set of orders with pickup completed
    chrom: List[int] = []            # Chromosome being built
    current_point_idx = start_idx    # Current position (index in distance matrix)
    total_tasks = 2 * n_orders

    while len(chrom) < total_tasks:
        best_task = -1
        best_d = float("inf")
        for t in range(total_tasks):
            if t in visited_tasks:
                continue
            k = task_order_id(t)
            # Only allow delivery if pickup has been completed
            if not task_is_pickup(t) and k not in picked_orders:
                continue
            target_idx = point_to_task[t]
            d = dist_matrix[current_point_idx][target_idx]
            if d < best_d:
                best_d = d
                best_task = t
        if best_task < 0:
            # Fallback (should not happen if graph is connected)
            remaining = [t for t in range(total_tasks) if t not in visited_tasks]
            best_task = remaining[0]
        chrom.append(best_task)
        visited_tasks.add(best_task)
        if task_is_pickup(best_task):
            picked_orders.add(task_order_id(best_task))
        current_point_idx = point_to_task[best_task]
    return chrom


# ---------------------------------------------------------------------------
# Fitness / Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    chrom: List[int],
    dist_matrix: List[List[float]],
    point_to_task: dict,
    start_idx: int = 0,
) -> float:
    """
    Compute total travel distance: start → task_1 → task_2 → … → task_2n.

    Uses the precomputed distance matrix for O(1) lookup per pair.
    Lower value = better chromosome (minimization).
    """
    total = 0.0
    prev = start_idx
    for t in chrom:
        nxt = point_to_task[t]
        total += dist_matrix[prev][nxt]
        prev = nxt
    return total


def fitness(distance: float, eps: float = 1e-6) -> float:
    """Convert distance to fitness (further = lower fitness)."""
    return 1.0 / (distance + eps)


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def tournament_select(
    population: List[List[int]],
    distances: List[float],
    k: int,
    rng: random.Random,
) -> List[int]:
    """
    Tournament selection:
    Pick k random individuals, keep the one with the shortest distance.
    """
    contenders = rng.sample(range(len(population)), k)
    best = min(contenders, key=lambda i: distances[i])
    return population[best][:]


def order_crossover(parent1: List[int], parent2: List[int],
                    rng: random.Random) -> List[int]:
    """
    Order Crossover (OX):
      1. Copy a random segment [a, b] from parent1 into the child.
      2. Fill remaining genes in the order they appear in parent2.

    Output does NOT guarantee precedence constraints → call precedence_repair() after.
    """
    size = len(parent1)
    a, b = sorted(rng.sample(range(size), 2))
    child: List[Optional[int]] = [None] * size
    slice_genes = set(parent1[a:b + 1])
    child[a:b + 1] = parent1[a:b + 1]
    # Fill genes not in segment [a,b], following parent2 order
    fill_iter = (g for g in parent2 if g not in slice_genes)
    for i in range(size):
        if child[i] is None:
            child[i] = next(fill_iter)
    return [g for g in child if g is not None]  # type: ignore


def swap_mutation(chrom: List[int], rng: random.Random) -> List[int]:
    """
    Swap mutation:
    Randomly pick 2 positions and swap their values.
    Call precedence_repair() after mutation.
    """
    chrom = chrom[:]
    if len(chrom) < 2:
        return chrom
    i, j = rng.sample(range(len(chrom)), 2)
    chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------

def run_ga(
    grid: List[List[int]],
    start: Point,
    orders: List[Order],
    config: GAConfig = GAConfig(),
    progress_cb: Optional[Callable[[int, float], None]] = None,
) -> GAResult:
    """
    Run the genetic algorithm to find the optimal pickup-delivery task order.

    Process:
      1. Build point list (start + 2n pickup/dropoff) and distance matrix.
      2. Initialize population: 50% greedy variants + 50% random valid.
      3. Evolution loop:
         a. Preserve elites.
         b. Tournament selection → OX crossover → swap mutation → constraint repair.
         c. Record best/mean/worst per generation.
         d. Early stop if no improvement for `early_stop_gen` consecutive generations.
      4. Return best route as a list of (x, y) coordinates.

    Parameters:
        grid        : 2D map (0=free, 1=obstacle), already inflated.
        start       : Starting coordinates (x, y).
        orders      : List of orders.
        config      : GA parameters (see GAConfig).
        progress_cb : Optional callback(gen, best_dist) called each generation.

    Returns:
        GAResult with best chromosome, route, distance, and history.
    """
    rng = random.Random(config.seed)
    n_orders = len(orders)
    if n_orders == 0:
        return GAResult([], [start], 0.0, 0, [])

    # ===== STEP 1: Build point list and distance matrix =====
    points: List[Point] = [start]
    point_to_task: dict = {}     # mapping: task_index → index in `points`
    for k, o in enumerate(orders):
        points.append(o.pickup)
        point_to_task[2 * k] = len(points) - 1
        points.append(o.dropoff)
        point_to_task[2 * k + 1] = len(points) - 1
    start_idx = 0

    # Choose distance computation method
    if config.distance_method == "octile":
        # Fast: Octile only (ignores obstacles) — suitable for Step 1 (GA-only)
        dist_matrix = build_distance_matrix_octile(points)
    else:
        # Accurate: A* on obstacle map
        dist_matrix = build_distance_matrix(grid, points)

    # ===== STEP 2: Initialize population =====
    population: List[List[int]] = []
    n_greedy = config.pop_size // 2     # 50% greedy-based individuals
    n_random = config.pop_size - n_greedy  # 50% random individuals

    # Base greedy individual (nearest neighbor)
    base_greedy = greedy_chromosome(start_idx, n_orders, dist_matrix, point_to_task)
    population.append(base_greedy)
    # Mutated variants from the greedy individual
    for _ in range(n_greedy - 1):
        c = swap_mutation(base_greedy, rng)
        c = precedence_repair(c)
        population.append(c)
    # Random valid individuals
    for _ in range(n_random):
        population.append(random_valid_chromosome(n_orders, rng))

    # Evaluate initial population
    distances = [evaluate(c, dist_matrix, point_to_task, start_idx)
                 for c in population]
    n_elite = max(1, int(config.pop_size * config.elitism_ratio))

    # ===== STEP 3: Evolution loop =====
    history: List[float] = []
    history_mean: List[float] = []
    history_worst: List[float] = []
    best_idx = min(range(len(population)), key=lambda i: distances[i])
    best_dist = distances[best_idx]
    best_chrom = population[best_idx][:]
    stagnation = 0      # Count of consecutive generations without improvement
    gens_run = 0

    for gen in range(config.max_gen):
        gens_run = gen + 1

        # --- Preserve elites ---
        sorted_idx = sorted(range(len(population)), key=lambda i: distances[i])
        new_pop = [population[i][:] for i in sorted_idx[:n_elite]]

        # --- Reproduction ---
        while len(new_pop) < config.pop_size:
            # Select 2 parents via tournament
            p1 = tournament_select(population, distances, config.tournament_k, rng)
            p2 = tournament_select(population, distances, config.tournament_k, rng)
            # OX crossover
            if rng.random() < config.crossover_rate:
                child = order_crossover(p1, p2, rng)
            else:
                child = p1[:]
            # Swap mutation
            if rng.random() < config.mutation_rate:
                child = swap_mutation(child, rng)
            # Repair pickup-before-delivery constraint
            child = precedence_repair(child)
            new_pop.append(child)

        # Replace population and re-evaluate
        population = new_pop
        distances = [evaluate(c, dist_matrix, point_to_task, start_idx)
                     for c in population]

        # --- Record generation statistics ---
        gen_best_idx = min(range(len(population)), key=lambda i: distances[i])
        gen_best = distances[gen_best_idx]
        gen_mean = sum(distances) / len(distances)
        gen_worst = max(distances)
        history.append(gen_best)
        history_mean.append(gen_mean)
        history_worst.append(gen_worst)

        # --- Update global best ---
        if gen_best < best_dist - 1e-9:
            best_dist = gen_best
            best_chrom = population[gen_best_idx][:]
            stagnation = 0
        else:
            stagnation += 1

        # Progress callback (if provided)
        if progress_cb:
            progress_cb(gen, best_dist)

        # --- Early stop if stagnated ---
        if stagnation >= config.early_stop_gen:
            break

    # ===== STEP 4: Build coordinate route =====
    best_route = [start] + [task_point(t, orders) for t in best_chrom]

    return GAResult(
        best_sequence=best_chrom,
        best_route=best_route,
        total_distance=best_dist,
        generations_run=gens_run,
        history=history,
        history_mean=history_mean,
        history_worst=history_worst,
    )


# ---------------------------------------------------------------------------
# Path expansion via A*
# ---------------------------------------------------------------------------

def expand_full_path(grid: List[List[int]], route: List[Point]) -> List[Point]:
    """
    Convert a high-level route (list of waypoints) into a detailed
    cell-by-cell path by calling A* between consecutive waypoint pairs.

    Input:
        grid  : Inflated map (0=free, 1=blocked).
        route : [start, wp1, wp2, …, wp_{2n}] — output from GA.

    Output:
        Continuous list of (x, y) cells the robot passes through (including start and end).
        Each consecutive step is exactly 1 cell (4-directional) or √2 (8-directional).

    Raises:
        RuntimeError if any segment has no A* path.
    """
    full: List[Point] = []
    for i in range(len(route) - 1):
        seg = astar(grid, route[i], route[i + 1])
        if seg is None:
            raise RuntimeError(
                f"No A* path found between {route[i]} and {route[i + 1]}")
        if i == 0:
            full.extend(seg)
        else:
            full.extend(seg[1:])  # Skip first cell (overlaps with end of previous segment)
    return full
