# -*- coding: utf-8 -*-
"""
astar.py — D* Lite pathfinding on 2D grid (8-directional) with dynamic obstacle support.

D* Lite is an incremental search algorithm that efficiently replans when the
environment changes (e.g., dynamic obstacles appear/disappear). It searches
backward from goal to start, maintaining g-values and rhs-values (one-step
lookahead) for each visited node. When edge costs change, only affected
vertices need to be updated, making replanning much faster than a full A* search.

Features:
  • D* Lite core: incremental search with efficient replanning
  • Static obstacles: fixed obstacles on the grid
  • Dynamic obstacles (DynamicObstacle): appear/disappear based on simulation step
  • Replanning: when dynamic obstacles block the path, D* Lite efficiently
    updates the path without re-searching from scratch

Grid convention:
    grid[y][x] = 0  -> free cell
    grid[y][x] = 1  -> static obstacle
"""
from __future__ import annotations

import heapq
import math
import random as _random
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

Point = Tuple[int, int]   # (x, y)

SQRT2 = math.sqrt(2)

# 8-connected neighbors: (dx, dy, cost)
NEIGHBORS_8 = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
]


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def octile(a: Point, b: Point) -> float:
    """Octile distance heuristic for 8-directional grid (admissible & consistent)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (SQRT2 - 1) * min(dx, dy)


def in_bounds(grid: List[List[int]], p: Point) -> bool:
    """Check if point p is within grid bounds."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    return 0 <= p[0] < w and 0 <= p[1] < h


def is_free(grid: List[List[int]], p: Point) -> bool:
    """Check if cell p is free (not a static obstacle) and in bounds."""
    return in_bounds(grid, p) and grid[p[1]][p[0]] == 0


def inflate_obstacles(grid: List[List[int]], radius: int = 1) -> List[List[int]]:
    """Inflate static obstacles by `radius` cells (safety margin for robot)."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    inflated = [row[:] for row in grid]
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 1:
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and (dx != 0 or dy != 0):
                            inflated[ny][nx] = 1
    return inflated


# ===========================================================================
# D* LITE ALGORITHM
# ===========================================================================

class DLitePlanner:
    """
    D* Lite incremental pathfinding on a 2D grid (8-directional movement).

    D* Lite searches backward from goal to start, maintaining:
      - g(s): cost-to-go from state s to goal
      - rhs(s): one-step lookahead of g(s)

    When the environment changes (obstacles appear/disappear), only affected
    vertices are updated, making replanning much faster than re-running A*.

    Usage:
        planner = DLitePlanner(grid, start, goal)
        path = planner.plan()

        # Later, when obstacles change:
        planner.set_start(new_pos)
        path = planner.replan()
    """

    INF = float('inf')

    def __init__(self, grid: List[List[int]], start: Point, goal: Point,
                 dynamic_mgr: 'DynamicObstacleManager' = None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.dynamic_mgr = dynamic_mgr

        # D* Lite state
        self.g: dict[Point, float] = {}
        self.rhs: dict[Point, float] = {}
        self._U: list = []          # priority queue (OPEN list)
        self._U_set: set = set()    # track membership in U
        self._U_counter: dict = {}  # track insertion counter per node (for stale detection)
        self._counter = 0           # tie-breaker for heap
        self.km = 0.0               # key modifier (accumulates on start changes)
        self.last_stats: dict = {}  # stats from last computation
        self._opened_count = 0      # count of U insertions

        # Build dynamic-aware free-cell check
        if dynamic_mgr is not None:
            self._blocked = dynamic_mgr.get_blocked_cells()
        else:
            self._blocked = set()

        # Initialize: g and rhs are implicitly INF (via dict.get default)
        self.rhs[goal] = 0.0
        self._insert(goal, self._key(goal))

        # Initial computation
        self.last_stats = self._compute_shortest_path()

    def _is_free(self, p: Point) -> bool:
        """Check if cell is free considering both static and dynamic obstacles."""
        return is_free(self.grid, p) and p not in self._blocked

    def _key(self, s: Point) -> Tuple[float, float]:
        """
        Compute priority key for state s.
        k(s) = [min(g(s), rhs(s)) + h(start, s) + km,  min(g(s), rhs(s))]
        """
        g_val = self.g.get(s, self.INF)
        rhs_val = self.rhs.get(s, self.INF)
        v = min(g_val, rhs_val)
        return (v + octile(self.start, s) + self.km, v)

    def _insert(self, s: Point, key: Tuple[float, float]):
        """Insert state s into priority queue U."""
        self._counter += 1
        heapq.heappush(self._U, (key, self._counter, s))
        self._U_set.add(s)
        self._U_counter[s] = self._counter  # track latest insertion counter

    def _remove(self, s: Point):
        """Mark state s as removed from U (lazy deletion)."""
        self._U_set.discard(s)
        self._U_counter.pop(s, None)

    def _in_U(self, s: Point) -> bool:
        """Check if state s is currently in U."""
        return s in self._U_set

    def _rhs(self, s: Point) -> float:
        """
        Compute rhs(s) = min over successors s' of [c(s, s') + g(s')].
        For goal node, rhs = 0.
        If s itself is blocked, rhs = INF (cannot traverse blocked cells).
        """
        if s == self.goal:
            return 0.0
        if not self._is_free(s):
            return self.INF
        sx, sy = s
        best = self.INF
        for dx, dy, step_cost in NEIGHBORS_8:
            sp = (sx + dx, sy + dy)  # successor s'
            if not self._is_free(sp):
                continue
            # Corner-cut check for diagonal moves
            if dx != 0 and dy != 0:
                if not self._is_free((sx + dx, sy)) or not self._is_free((sx, sy + dy)):
                    continue
            val = step_cost + self.g.get(sp, self.INF)
            if val < best:
                best = val
        return best

    def _update_vertex(self, u: Point):
        """
        Update vertex u: recompute rhs(u) and re-insert into U if inconsistent.
        Also updates predecessors of u (neighbors that have u as successor).
        """
        if u != self.goal:
            self.rhs[u] = self._rhs(u)

        # Remove from U if present
        if self._in_U(u):
            self._remove(u)

        # If inconsistent, insert into U
        g_val = self.g.get(u, self.INF)
        rhs_val = self.rhs.get(u, self.INF)
        if g_val != rhs_val:
            self._insert(u, self._key(u))
            self._opened_count += 1

    def _compute_shortest_path(self) -> dict:
        """
        Main D* Lite loop. Processes the priority queue until the start
        state is locally consistent and its key is <= top of queue.

        Returns stats dict with expanded/opened/time_ms.
        """
        import time
        t0 = time.perf_counter()
        stats = {"expanded": 0, "opened": 0, "time_ms": 0.0}
        self._opened_count = 0  # reset insertion counter

        while self._U:
            # Clean stale entries (lazy deletion):
            # Skip entries that are not in U_set, or have a non-matching
            # counter (superseded by a later re-insertion)
            while self._U:
                top_key, top_counter, top_s = self._U[0]
                if top_s in self._U_set and self._U_counter.get(top_s) == top_counter:
                    break
                heapq.heappop(self._U)
            else:
                break

            top_key, top_counter, top_s = self._U[0]
            u = top_s
            k_old = top_key

            # Check termination: start is consistent and key(start) <= k_top
            g_start = self.g.get(self.start, self.INF)
            rhs_start = self.rhs.get(self.start, self.INF)
            if g_start == rhs_start and self._key(self.start) <= k_old:
                break

            heapq.heappop(self._U)
            self._remove(u)

            stats["expanded"] += 1

            g_u = self.g.get(u, self.INF)
            rhs_u = self.rhs.get(u, self.INF)

            if k_old < self._key(u):
                # Key increased — re-insert with updated key
                self._insert(u, self._key(u))
                stats["opened"] += 1

            elif g_u > rhs_u:
                # Locally overconsistent: lower g to rhs, expand
                self.g[u] = rhs_u
                self._update_vertex(u)
                # Update all predecessors (neighbors of u)
                ux, uy = u
                for dx, dy, step_cost in NEIGHBORS_8:
                    s = (ux - dx, uy - dy)  # predecessor
                    if not self._is_free(s):
                        continue
                    # Corner-cut check: move from s to u
                    if dx != 0 and dy != 0:
                        if not self._is_free((s[0] + dx, s[1])) or \
                           not self._is_free((s[0], s[1] + dy)):
                            continue
                    self._update_vertex(s)

            else:
                # Locally underconsistent: raise g to INF, propagate
                self.g[u] = self.INF
                self._update_vertex(u)
                ux, uy = u
                for dx, dy, step_cost in NEIGHBORS_8:
                    s = (ux - dx, uy - dy)  # predecessor
                    if not self._is_free(s):
                        continue
                    if dx != 0 and dy != 0:
                        if not self._is_free((s[0] + dx, s[1])) or \
                           not self._is_free((s[0], s[1] + dy)):
                            continue
                    self._update_vertex(s)

        stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
        stats["opened"] = self._opened_count
        return stats

    def get_path(self) -> Optional[List[Point]]:
        """
        Extract the current best path from start to goal using g-values.

        Starting from start, repeatedly select the neighbor s' that minimizes
        c(current, s') + g(s'), until reaching the goal.

        Returns path as list of Points, or None if no path exists.
        """
        if not self._is_free(self.start) or not self._is_free(self.goal):
            return None
        if self.start == self.goal:
            return [self.start]
        if self.g.get(self.start, self.INF) == self.INF:
            return None

        path = [self.start]
        current = self.start
        visited = {current}  # prevent infinite loops

        while current != self.goal:
            cx, cy = current
            best_cost = self.INF
            best_next = None

            for dx, dy, step_cost in NEIGHBORS_8:
                s = (cx + dx, cy + dy)
                if not self._is_free(s) or s in visited:
                    continue
                if dx != 0 and dy != 0:
                    if not self._is_free((cx + dx, cy)) or \
                       not self._is_free((cx, cy + dy)):
                        continue
                val = step_cost + self.g.get(s, self.INF)
                if val < best_cost:
                    best_cost = val
                    best_next = s

            if best_next is None:
                return None  # stuck — no path to goal
            visited.add(best_next)
            path.append(best_next)
            current = best_next

        return path

    def set_start(self, new_start: Point):
        """
        Update the start position (e.g., after the robot moves).
        Adjusts km and re-queues the old start for re-evaluation.
        """
        if new_start != self.start:
            g_start = self.g.get(self.start, self.INF)
            rhs_start = self.rhs.get(self.start, self.INF)
            self.km += min(g_start, rhs_start) + octile(self.start, new_start)
            old_start = self.start
            self.start = new_start
            # Reset g of old start to force re-exploration if needed
            self.g[old_start] = self.INF
            self._update_vertex(old_start)

    def update_obstacles(self):
        """
        Call when dynamic obstacles have changed. Refreshes the blocked
        cell set and updates all affected vertices, then rebuilds the
        priority queue to eliminate stale entries.
        """
        if self.dynamic_mgr is not None:
            new_blocked = self.dynamic_mgr.get_blocked_cells()
        else:
            new_blocked = set()

        # Find cells whose blocked status changed
        changed = self._blocked.symmetric_difference(new_blocked)
        self._blocked = new_blocked

        # Collect all affected cells: changed cells + their neighbors
        affected = set()
        for p in changed:
            affected.add(p)
            px, py = p
            for dx, dy, _ in NEIGHBORS_8:
                affected.add((px - dx, py - dy))  # predecessors
                affected.add((px + dx, py + dy))  # successors

        # For newly blocked cells, invalidate g immediately
        for s in changed:
            if not self._is_free(s) and s != self.goal:
                self.g[s] = self.INF

        # Update all affected vertices
        for s in affected:
            self._update_vertex(s)

        # Rebuild priority queue from scratch to eliminate all stale entries
        self._U = []
        self._U_set = set()
        self._U_counter = {}
        self._counter = 0
        for s in list(self.rhs.keys()) + list(self.g.keys()):
            g_val = self.g.get(s, self.INF)
            rhs_val = self.rhs.get(s, self.INF)
            if g_val != rhs_val and s not in self._U_set:
                self._insert(s, self._key(s))

    def replan(self) -> Optional[List[Point]]:
        """
        Replan after environment changes. Re-runs ComputeShortestPath
        incrementally. If incremental replan fails to find a valid path
        (due to uncomputed g-values in detour regions), falls back to
        full recomputation.

        Returns the new path or None if no path exists.
        """
        self._compute_shortest_path()
        path = self.get_path()
        if path is not None:
            return path

        # Incremental replan failed — fall back to full recomputation.
        # Reset state and rebuild from scratch.
        self.g = {}
        self.rhs = {}
        self._U = []
        self._U_set = set()
        self._U_counter = {}
        self._counter = 0
        self._opened_count = 0
        self.km = 0.0

        # Re-initialize D* Lite
        self.rhs[self.goal] = 0.0
        self._insert(self.goal, self._key(self.goal))
        self.last_stats = self._compute_shortest_path()
        return self.get_path()


# ===========================================================================
# BACKWARD-COMPATIBLE WRAPPER FUNCTIONS (using D* Lite internally)
# ===========================================================================

def astar(grid: List[List[int]], start: Point, goal: Point) -> Optional[List[Point]]:
    """
    Find shortest path from start to goal on static grid using D* Lite.
    Returns list of Points (including start and goal) or None if no path.
    """
    path, _ = astar_with_stats(grid, start, goal)
    return path


def astar_with_stats(
    grid: List[List[int]], start: Point, goal: Point
) -> Tuple[Optional[List[Point]], dict]:
    """
    D* Lite pathfinding with statistics: expanded, opened, time_ms.
    """
    import time
    t0 = time.perf_counter()
    stats = {"expanded": 0, "opened": 0, "time_ms": 0.0}

    if not is_free(grid, start) or not is_free(grid, goal):
        stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
        return None, stats
    if start == goal:
        stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
        return [start], stats

    planner = DLitePlanner(grid, start, goal)
    path = planner.get_path()

    # Get stats from planner's internal computation
    stats["expanded"] = planner.last_stats.get("expanded", 0)
    stats["opened"] = planner.last_stats.get("opened", 0)
    stats["time_ms"] = planner.last_stats.get("time_ms", 0.0)

    return path, stats


def astar_dynamic(grid: List[List[int]], start: Point, goal: Point,
                  dynamic_mgr: 'DynamicObstacleManager' = None) -> Optional[List[Point]]:
    """
    D* Lite pathfinding from start to goal, considering both static and
    dynamic obstacles. If dynamic_mgr=None, behaves like regular D* Lite.
    """
    path, _ = astar_dynamic_with_stats(grid, start, goal, dynamic_mgr)
    return path


def astar_dynamic_with_stats(
    grid: List[List[int]], start: Point, goal: Point,
    dynamic_mgr: 'DynamicObstacleManager' = None
) -> Tuple[Optional[List[Point]], dict]:
    """
    D* Lite with dynamic obstacles, returns (path, stats).
    Blocked cells = static obstacles OR active dynamic obstacles.
    """
    import time
    t0 = time.perf_counter()
    stats = {"expanded": 0, "opened": 0, "time_ms": 0.0}

    blocked = dynamic_mgr.get_blocked_cells() if dynamic_mgr else set()

    def _is_free_dyn(p: Point) -> bool:
        """Check if cell is free (static + dynamic)."""
        if not is_free(grid, p):
            return False
        if p in blocked:
            return False
        return True

    if not _is_free_dyn(start) or not _is_free_dyn(goal):
        stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
        return None, stats
    if start == goal:
        stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
        return [start], stats

    planner = DLitePlanner(grid, start, goal, dynamic_mgr)
    path = planner.get_path()

    # Get stats from planner's internal computation
    stats["expanded"] = planner.last_stats.get("expanded", 0)
    stats["opened"] = planner.last_stats.get("opened", 0)
    stats["time_ms"] = planner.last_stats.get("time_ms", 0.0)

    return path, stats


def path_length(path: Optional[List[Point]]) -> float:
    """Compute total geometric length of a path. Returns inf if path=None."""
    if path is None or len(path) < 2:
        return float('inf') if path is None else 0.0
    total = 0.0
    for i in range(1, len(path)):
        ax, ay = path[i - 1]
        bx, by = path[i]
        dx, dy = abs(ax - bx), abs(ay - by)
        total += SQRT2 if (dx == 1 and dy == 1) else 1.0
    return total


# ===========================================================================
# DYNAMIC OBSTACLES
# ===========================================================================

@dataclass
class DynamicObstacle:
    """
    A dynamic obstacle on the grid.
    - position: coordinates (x, y)
    - appear_step: simulation step when it appears
    - duration: number of steps it persists (-1 = permanent)
    - active: current state
    """
    position: Point
    appear_step: int = 0
    duration: int = -1
    active: bool = False

    def is_active_at(self, step: int) -> bool:
        """Check if obstacle is active at given simulation step."""
        if step < self.appear_step:
            return False
        if self.duration == -1:
            return True
        return step < self.appear_step + self.duration


@dataclass
class DynamicObstacleManager:
    """
    Manages a collection of dynamic obstacles.
    - Add/remove obstacles
    - Update state based on simulation step
    - Check blocked cells
    - Spawn random obstacles during simulation
    """
    obstacles: List[DynamicObstacle] = field(default_factory=list)
    _blocked_cells: Set[Point] = field(default_factory=set)

    def add(self, obstacle: DynamicObstacle):
        """Add a dynamic obstacle."""
        self.obstacles.append(obstacle)

    def clear(self):
        """Remove all dynamic obstacles."""
        self.obstacles.clear()
        self._blocked_cells.clear()

    def update(self, current_step: int):
        """Update all obstacle states based on current simulation step."""
        self._blocked_cells.clear()
        for obs in self.obstacles:
            obs.active = obs.is_active_at(current_step)
            if obs.active:
                self._blocked_cells.add(obs.position)

    def is_blocked(self, point: Point) -> bool:
        """Check if cell (x, y) is blocked by a dynamic obstacle."""
        return point in self._blocked_cells

    def get_blocked_cells(self) -> Set[Point]:
        """Return the set of cells currently blocked by dynamic obstacles."""
        return self._blocked_cells.copy()

    def get_active_obstacles(self) -> List[DynamicObstacle]:
        """Return list of currently active obstacles."""
        return [obs for obs in self.obstacles if obs.active]

    def spawn_random(self, grid: List[List[int]], current_step: int,
                     count: int = 1, duration: int = 30,
                     avoid_cells: Set[Point] = None,
                     seed: int = None):
        """
        Spawn random dynamic obstacles on free cells.
        - grid: static grid
        - current_step: step at which obstacles appear (immediate)
        - duration: number of steps they persist
        - avoid_cells: cells to avoid (robot position, pickup/dropoff)
        """
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        rng = _random.Random(seed)
        avoid = avoid_cells or set()
        spawned = []
        for _ in range(count):
            for _try in range(200):
                x = rng.randint(0, w - 1)
                y = rng.randint(0, h - 1)
                p = (x, y)
                if (grid[y][x] == 0 and p not in self._blocked_cells
                        and p not in avoid):
                    obs = DynamicObstacle(
                        position=p,
                        appear_step=current_step,
                        duration=duration,
                        active=True
                    )
                    self.obstacles.append(obs)
                    self._blocked_cells.add(p)
                    spawned.append(obs)
                    break
        return spawned

    def spawn_on_path(self, path: List[Point], current_index: int,
                      current_step: int, duration: int = 50,
                      avoid_cells: Set[Point] = None,
                      min_ahead: int = 3, max_ahead: int = 8):
        """
        Spawn a dynamic obstacle directly on the path ahead of the robot.
        Forces the robot to replan its route.

        Args:
            path: current planned path
            current_index: robot's current position index on path
            current_step: current simulation step
            duration: number of steps obstacle persists
            avoid_cells: cells to avoid (waypoint pickup/dropoff)
            min_ahead: minimum distance ahead to place obstacle
            max_ahead: maximum distance ahead to place obstacle

        Returns:
            The spawned DynamicObstacle, or None if no valid position found.
        """
        avoid = avoid_cells or set()
        start_idx = current_index + min_ahead
        end_idx = min(current_index + max_ahead, len(path))

        candidates = []
        for i in range(start_idx, end_idx):
            p = path[i]
            if p not in self._blocked_cells and p not in avoid:
                candidates.append(p)

        if not candidates:
            return None

        chosen = _random.choice(candidates)
        obs = DynamicObstacle(
            position=chosen,
            appear_step=current_step,
            duration=duration,
            active=True
        )
        self.obstacles.append(obs)
        self._blocked_cells.add(chosen)
        return obs


# ===========================================================================
# REPLANNING — Replan path when encountering dynamic obstacles
# ===========================================================================

def check_path_blocked(path: List[Point], current_index: int,
                       dynamic_mgr: DynamicObstacleManager,
                       look_ahead: int = 10) -> Optional[int]:
    """
    Check if the path ahead is blocked by dynamic obstacles.
    - path: full planned path
    - current_index: robot's current position index
    - look_ahead: number of cells to check ahead

    Returns: index of first blocked cell (None if path is clear ahead).
    """
    if dynamic_mgr is None:
        return None
    end_check = min(current_index + look_ahead, len(path))
    for i in range(current_index + 1, end_check):
        if dynamic_mgr.is_blocked(path[i]):
            return i
    return None


def replan_segment(grid: List[List[int]], current_pos: Point, goal: Point,
                   dynamic_mgr: DynamicObstacleManager) -> Optional[List[Point]]:
    """
    Replan path segment from current position to goal, avoiding dynamic obstacles.
    Uses D* Lite for efficient incremental search.
    """
    return astar_dynamic(grid, current_pos, goal, dynamic_mgr)


def replan_full_path(grid: List[List[int]], current_pos: Point,
                     remaining_waypoints: List[Point],
                     dynamic_mgr: DynamicObstacleManager) -> Optional[List[Point]]:
    """
    Replan the full path from current position through remaining waypoints.
    Uses D* Lite for each segment, avoiding dynamic obstacles.

    Args:
        grid: static obstacle grid
        current_pos: robot's current position
        remaining_waypoints: list of waypoints still to visit
        dynamic_mgr: dynamic obstacle manager

    Returns:
        Complete path or None if any segment has no solution.
    """
    if not remaining_waypoints:
        return [current_pos]

    full_path = []
    prev = current_pos

    for wp in remaining_waypoints:
        segment = astar_dynamic(grid, prev, wp, dynamic_mgr)
        if segment is None:
            return None
        if full_path:
            segment = segment[1:]  # skip duplicate start point
        full_path.extend(segment)
        prev = wp

    return full_path


# ===========================================================================
# PATH SMOOTHING — Convert sharp grid turns into smooth Bezier curves
# ===========================================================================

def smooth_path(path: List[Point], radius: float = 0.4,
                samples: int = 8) -> List[Tuple[float, float]]:
    """
    Convert a grid path with sharp corners into a smooth curve path.

    At each interior corner, replaces the sharp turn with a quadratic Bezier
    curve.  Straight segments are left unchanged.

    Args:
        path: list of integer grid points [(x, y), ...]
        radius: how far before/after each corner to start rounding (0 < r < 0.5)
        samples: number of curve sample points per corner

    Returns:
        List of (float, float) points suitable for matplotlib plotting.
    """
    if len(path) < 2:
        return [(float(p[0]), float(p[1])) for p in path]

    r = min(max(radius, 0.01), 0.49)
    result: List[Tuple[float, float]] = []

    # Always include the first point
    result.append((float(path[0][0]), float(path[0][1])))

    for i in range(1, len(path) - 1):
        prev = path[i - 1]
        curr = path[i]
        nxt = path[i + 1]

        # Direction vectors
        dx1 = float(curr[0] - prev[0])
        dy1 = float(curr[1] - prev[1])
        dx2 = float(nxt[0] - curr[0])
        dy2 = float(nxt[1] - curr[1])

        # Cross product → detect corners (0 = collinear)
        cross = abs(dx1 * dy2 - dy1 * dx2)

        if cross < 0.01:
            # Straight line — keep original point
            result.append((float(curr[0]), float(curr[1])))
        else:
            # Corner — insert quadratic Bezier curve
            # P0: entry point (radius before corner)
            p0x = curr[0] - r * dx1
            p0y = curr[1] - r * dy1
            # P1: control point (the corner itself)
            p1x = float(curr[0])
            p1y = float(curr[1])
            # P2: exit point (radius after corner)
            p2x = curr[0] + r * dx2
            p2y = curr[1] + r * dy2

            # Sample the quadratic Bezier: B(t) = (1-t)^2*P0 + 2t(1-t)*P1 + t^2*P2
            for j in range(1, samples + 1):
                t = j / samples
                t1 = 1.0 - t
                bx = t1 * t1 * p0x + 2.0 * t1 * t * p1x + t * t * p2x
                by = t1 * t1 * p0y + 2.0 * t1 * t * p1y + t * t * p2y
                result.append((bx, by))

    # Always include the last point
    result.append((float(path[-1][0]), float(path[-1][1])))

    return result
