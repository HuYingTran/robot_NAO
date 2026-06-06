# -*- coding: utf-8 -*-
"""
astar.py — A* pathfinding on a 2D grid (8-directional) with dynamic obstacle support.

Features:
  • Basic (static) A*: finds the shortest path on a grid with fixed obstacles.
  • Dynamic obstacles (DynamicObstacle): appear/disappear over time (simulation steps).
  • Replanning: when the robot detects a dynamic obstacle blocking the path ahead,
    it automatically recalculates the route from its current position.

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
# BASIC UTILITY FUNCTIONS
# ===========================================================================

def octile(a: Point, b: Point) -> float:
    """Octile distance heuristic for 8-connected grids (admissible & consistent)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (SQRT2 - 1) * min(dx, dy)


def in_bounds(grid: List[List[int]], p: Point) -> bool:
    """Check if point p is within grid bounds."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    return 0 <= p[0] < w and 0 <= p[1] < h


def is_free(grid: List[List[int]], p: Point) -> bool:
    """Check if cell p is free (not a static obstacle)."""
    return in_bounds(grid, p) and grid[p[1]][p[0]] == 0


def inflate_obstacles(grid: List[List[int]], radius: int = 1) -> List[List[int]]:
    """Inflate static obstacles by `radius` cells around them (safety margin for NAO)."""
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
# BASIC A* (STATIC OBSTACLES)
# ===========================================================================

def astar(grid: List[List[int]], start: Point, goal: Point) -> Optional[List[Point]]:
    """
    Run A* from start -> goal on a static grid.
    Returns a list of points (including start and goal) or None if no path exists.
    """
    path, _ = astar_with_stats(grid, start, goal)
    return path


def astar_with_stats(
    grid: List[List[int]], start: Point, goal: Point
) -> Tuple[Optional[List[Point]], dict]:
    """
    Static A* with stats: expanded (closed set), opened, time_ms.
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

    open_heap: List[Tuple[float, int, Point]] = []
    counter = 0
    heapq.heappush(open_heap, (octile(start, goal), counter, start))
    stats["opened"] += 1

    came_from: dict[Point, Point] = {}
    g_score: dict[Point, float] = {start: 0.0}
    closed: set[Point] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
            return _reconstruct(came_from, current), stats
        closed.add(current)
        stats["expanded"] += 1

        cx, cy = current
        for dx, dy, step_cost in NEIGHBORS_8:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            if neighbor in closed or not is_free(grid, neighbor):
                continue
            if dx != 0 and dy != 0:
                if not is_free(grid, (cx + dx, cy)) or not is_free(grid, (cx, cy + dy)):
                    continue
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + octile(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))
                stats["opened"] += 1
    stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
    return None, stats


def _reconstruct(came_from: dict[Point, Point], current: Point) -> List[Point]:
    """Reconstruct path from goal back to start."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_length(path: Optional[List[Point]]) -> float:
    """Compute total geometric length of the path. Returns Inf if path=None."""
    if path is None or len(path) < 2:
        return float('inf') if path is None else 0.0
    total = 0.0
    for i in range(1, len(path)):
        ax, ay = path[i - 1]
        bx, by = path[i]
        dx, dy = abs(ax - bx), abs(ay - by)
        total += SQRT2 if (dx == 1 and dy == 1) else 1.0
    return total


def is_line_of_sight_clear(grid: List[List[int]], p1: Point, p2: Point) -> bool:
    """
    Check if a straight line between p1 and p2 is obstacle-free.
    Uses Bresenham's line algorithm for grid-based line-of-sight.
    """
    x0, y0 = p1
    x1, y1 = p2
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        if not is_free(grid, (x, y)):
            return False
        if (x, y) == (x1, y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True


def smooth_path(grid: List[List[int]], path: List[Point]) -> List[Point]:
    """
    Smooth an A* path by removing unnecessary waypoints.
    Uses greedy shortcutting: if two non-consecutive waypoints have clear
    line-of-sight, skip all intermediate waypoints.
    
    This reduces the number of direction changes, making robot movement
    more fluid and reducing inertia effects.
    
    Args:
        grid: static obstacle grid
        path: original A* path (must include start and goal)
    
    Returns:
        Smoothed path with fewer waypoints but same start and goal.
    """
    if len(path) <= 2:
        return path[:]
    
    smoothed = [path[0]]  # Start with the first point
    current_idx = 0
    
    while current_idx < len(path) - 1:
        # Try to find the farthest point visible from current_idx
        best_idx = current_idx + 1
        for test_idx in range(current_idx + 2, len(path)):
            if is_line_of_sight_clear(grid, path[current_idx], path[test_idx]):
                best_idx = test_idx
        smoothed.append(path[best_idx])
        current_idx = best_idx
    
    return smoothed


# ===========================================================================
# DYNAMIC OBSTACLES
# ===========================================================================

@dataclass
class DynamicObstacle:
    """
    A dynamic obstacle on the grid.
    - position: (x, y) coordinates
    - appear_step: simulation step when it first appears
    - duration: number of steps it stays active (-1 = permanent)
    - active: current state
    """
    position: Point
    appear_step: int = 0
    duration: int = -1
    active: bool = False

    def is_active_at(self, step: int) -> bool:
        """Check if obstacle is active at the given step."""
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
    - Update state per simulation step
    - Check if a cell is blocked
    - Randomly spawn obstacles during simulation
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
        """Update the state of all obstacles for the current step."""
        self._blocked_cells.clear()
        for obs in self.obstacles:
            obs.active = obs.is_active_at(current_step)
            if obs.active:
                self._blocked_cells.add(obs.position)

    def is_blocked(self, point: Point) -> bool:
        """Check if cell (x, y) is blocked by a dynamic obstacle."""
        return point in self._blocked_cells

    def get_blocked_cells(self) -> Set[Point]:
        """Return the set of currently blocked cells."""
        return self._blocked_cells.copy()

    def get_active_obstacles(self) -> List[DynamicObstacle]:
        """Return a list of currently active obstacles."""
        return [obs for obs in self.obstacles if obs.active]

    def spawn_random(self, grid: List[List[int]], current_step: int,
                     count: int = 1, duration: int = 30,
                     avoid_cells: Set[Point] = None,
                     seed: int = None):
        """
        Randomly spawn `count` dynamic obstacles on free cells.
        - grid: static grid
        - current_step: current step (obstacle appears immediately)
        - duration: number of steps the obstacle persists
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
        Spawn a dynamic obstacle DIRECTLY on the path ahead of the robot.
        Forces the robot to recalculate its route.

        Args:
            path: current path
            current_index: robot's current position index on the path
            current_step: current simulation step
            duration: number of steps the obstacle persists
            avoid_cells: cells to avoid (pickup/dropoff waypoints)
            min_ahead: minimum distance ahead to place obstacle
            max_ahead: maximum distance ahead to place obstacle

        Returns:
            The spawned DynamicObstacle, or None if no suitable position found.
        """
        avoid = avoid_cells or set()
        # Find cells on the path ahead (in range min_ahead..max_ahead)
        start_idx = current_index + min_ahead
        end_idx = min(current_index + max_ahead, len(path))

        # Candidate list: path cells that are not blocked and not in avoid set
        candidates = []
        for i in range(start_idx, end_idx):
            p = path[i]
            if p not in self._blocked_cells and p not in avoid:
                candidates.append(p)

        if not candidates:
            return None

        # Randomly pick one cell from candidates
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
# A* WITH DYNAMIC OBSTACLES
# ===========================================================================

def astar_dynamic(grid: List[List[int]], start: Point, goal: Point,
                  dynamic_mgr: DynamicObstacleManager = None) -> Optional[List[Point]]:
    """
    A* from start -> goal, considering both static and dynamic obstacles.
    If dynamic_mgr=None, behaves like standard A*.
    """
    path, _ = astar_dynamic_with_stats(grid, start, goal, dynamic_mgr)
    return path


def astar_dynamic_with_stats(
    grid: List[List[int]], start: Point, goal: Point,
    dynamic_mgr: DynamicObstacleManager = None
) -> Tuple[Optional[List[Point]], dict]:
    """
    A* with dynamic obstacles, returns (path, stats).
    Blocked cell = static obstacle OR currently active dynamic obstacle.
    """
    import time
    t0 = time.perf_counter()
    stats = {"expanded": 0, "opened": 0, "time_ms": 0.0}

    def _is_free_dyn(p: Point) -> bool:
        """Check if cell is free (static + dynamic)."""
        if not is_free(grid, p):
            return False
        if dynamic_mgr and dynamic_mgr.is_blocked(p):
            return False
        return True

    if not _is_free_dyn(start) or not _is_free_dyn(goal):
        stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
        return None, stats
    if start == goal:
        stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
        return [start], stats

    open_heap: List[Tuple[float, int, Point]] = []
    counter = 0
    heapq.heappush(open_heap, (octile(start, goal), counter, start))
    stats["opened"] += 1

    came_from: dict[Point, Point] = {}
    g_score: dict[Point, float] = {start: 0.0}
    closed: set[Point] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
            return _reconstruct(came_from, current), stats
        closed.add(current)
        stats["expanded"] += 1

        cx, cy = current
        for dx, dy, step_cost in NEIGHBORS_8:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            if neighbor in closed or not _is_free_dyn(neighbor):
                continue
            # Check corner-cutting for diagonal moves
            if dx != 0 and dy != 0:
                if not _is_free_dyn((cx + dx, cy)) or not _is_free_dyn((cx, cy + dy)):
                    continue
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + octile(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))
                stats["opened"] += 1
    stats["time_ms"] = (time.perf_counter() - t0) * 1000.0
    return None, stats


# ===========================================================================
# REPLANNING — Recalculate path when hitting dynamic obstacles
# ===========================================================================

def check_path_blocked(path: List[Point], current_index: int,
                       dynamic_mgr: DynamicObstacleManager,
                       look_ahead: int = 10) -> Optional[int]:
    """
    Check if the path ahead is blocked by dynamic obstacles.
    - path: full path
    - current_index: robot's current position index
    - look_ahead: number of cells to check ahead

    Returns: the first blocked index (None if path ahead is clear).
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
    Recalculate path segment from current position -> goal, avoiding dynamic obstacles.
    """
    return astar_dynamic(grid, current_pos, goal, dynamic_mgr)


def replan_full_path(grid: List[List[int]], current_pos: Point,
                     remaining_waypoints: List[Point],
                     dynamic_mgr: DynamicObstacleManager) -> Optional[List[Point]]:
    """
    Recalculate the full path from current position through remaining waypoints.
    Concatenates A* segments (avoiding dynamic obstacles).

    Args:
        grid: static obstacle grid
        current_pos: robot's current position
        remaining_waypoints: list of waypoints not yet visited
        dynamic_mgr: dynamic obstacle manager

    Returns:
        Full path or None if no path can be found.
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
            segment = segment[1:]  # Skip duplicate start point
        full_path.extend(segment)
        prev = wp

    return full_path
