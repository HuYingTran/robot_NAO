# -*- coding: utf-8 -*-
"""
gui.py  —  GUI Control Center for NAO Robot
All operations revolve around this interface:
  • Problem configuration (map size, orders, obstacles)
  • Run GA (Step 1) and A* (Step 2)
  • Simulation (robot animation on the grid map)
  • Real NAO execution (NAOqi / qi-framework, with FAKE mode)
  • Export evaluation charts for GA & A*

Run:
    python gui.py
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from astar import (
    inflate_obstacles, path_length, smooth_path,
    DynamicObstacle, DynamicObstacleManager,
    astar_dynamic, check_path_blocked, replan_full_path,
)
from ga_planner import (
    GAConfig,
    Order,
    expand_full_path,
    run_ga,
    task_is_pickup,
    task_order_id,
)
from nao_controller import NaoController

# Default map size (configurable in GUI)
MAP_W_DEFAULT = 100
MAP_H_DEFAULT = 100

PALETTE = [
    "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e",
    "#17becf", "#bcbd22", "#8c564b", "#e377c2", "#7f7f7f",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194",
]


# ---------------------------------------------------------------------------
# Random environment helpers
# ---------------------------------------------------------------------------
def generate_random_obstacles(n_blocks, map_w=None, map_h=None,
                              min_size=4, max_size=15, seed=None):
    """
    Generate n_blocks line-shaped obstacles (1 cell thick) on a map_w × map_h grid.
    Each obstacle is a random horizontal (height=1) or vertical (width=1) segment.
    """
    w = map_w or MAP_W_DEFAULT
    h = map_h or MAP_H_DEFAULT
    rng = random.Random(seed)
    grid = [[0] * w for _ in range(h)]
    for _ in range(n_blocks):
        length = rng.randint(min_size, max_size)  # Obstacle segment length
        if length >= w or length >= h:
            length = min(w, h) - 1
        if rng.random() < 0.5:
            # Horizontal segment: height=1, width=length
            bx = rng.randint(0, w - length)
            by = rng.randint(0, h - 1)
            for x in range(bx, bx + length):
                grid[by][x] = 1
        else:
            # Vertical segment: width=1, height=length
            bx = rng.randint(0, w - 1)
            by = rng.randint(0, h - length)
            for y in range(by, by + length):
                grid[y][bx] = 1
    return grid


def random_free_point(grid_inflated, rng, exclude=None, max_tries=5000):
    """Find a random free cell on the grid (infers size from grid dimensions)."""
    exclude = exclude or set()
    h = len(grid_inflated)
    w = len(grid_inflated[0]) if h > 0 else 0
    for _ in range(max_tries):
        x = rng.randint(0, w - 1)
        y = rng.randint(0, h - 1)
        if (x, y) not in exclude and grid_inflated[y][x] == 0:
            return (x, y)
    raise RuntimeError("Cannot find a free cell.")


def color_for(idx: int) -> str:
    return PALETTE[idx % len(PALETTE)]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("NAO Delivery — GA + A* | Control Center")
        self.root.geometry("1280x880")

        # ---- environment / planning state ----
        self.map_w = MAP_W_DEFAULT
        self.map_h = MAP_H_DEFAULT
        self.grid_raw = None
        self.grid_inflated = None
        self.start = (0, 0)
        self.orders = []
        self.unmatched_pickups = []
        self.unmatched_dropoffs = []
        self.result = None
        self.full_path = None
        self.actual_distance = 0.0

        # ---- simulation state ----
        self.sim_running = False
        self.sim_paused = False
        self.sim_index = 0
        self.sim_after_id = None
        self._robot_dot = None
        self._sim_msg = None  # text artist

        # ---- dynamic obstacles state ----
        self.dyn_mgr = DynamicObstacleManager()
        self._dyn_scatter = None  # matplotlib scatter artist
        self._replan_count = 0    # replan count during simulation

        # ---- NAO state ----
        self.nao = NaoController()
        self.nao_running = False
        self._nao_stop_flag = False

        self._build_ui()
        self._draw_empty()

    # ------------------------------------------------------------ UI build
    def _build_ui(self):
        # left: notebook with tabs + status bar at the bottom
        left = ttk.Frame(self.root, padding=6)
        left.pack(side=tk.LEFT, fill=tk.Y)

        self.nb = ttk.Notebook(left, width=320)
        self.nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tab_problem = ttk.Frame(self.nb, padding=10)
        self.tab_sim     = ttk.Frame(self.nb, padding=10)
        self.tab_nao     = ttk.Frame(self.nb, padding=10)
        self.tab_eval    = ttk.Frame(self.nb, padding=10)
        self.nb.add(self.tab_problem, text="📋 Problem")
        self.nb.add(self.tab_sim,     text="▶ Simulation")
        self.nb.add(self.tab_nao,     text="🤖 Real NAO")
        self.nb.add(self.tab_eval,    text="📊 Evaluation")

        self._build_tab_problem(self.tab_problem)
        self._build_tab_sim(self.tab_sim)
        self._build_tab_nao(self.tab_nao)
        self._build_tab_eval(self.tab_eval)

        ttk.Separator(left).pack(fill=tk.X, pady=4)
        ttk.Label(left, text="STATUS",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.status_var = tk.StringVar(value="Ready. Generate an environment.")
        ttk.Label(left, textvariable=self.status_var,
                  foreground="#1f4f8b", wraplength=300,
                  font=("Segoe UI", 10),
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 4))

        # right: matplotlib canvas
        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.fig = Figure(figsize=(8.5, 8.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        NavigationToolbar2Tk(self.canvas, right).update()

    # -------- Tab 1: Problem ----------
    def _build_tab_problem(self, p):
        ttk.Label(p, text="ENVIRONMENT",
                  font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)

        # --- Map size ---
        frm_map = ttk.Frame(p)
        frm_map.pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(frm_map, text="Map size:  ", font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self.var_map_w = tk.IntVar(value=MAP_W_DEFAULT)
        ttk.Spinbox(frm_map, from_=10, to=500, textvariable=self.var_map_w,
                    width=5).pack(side=tk.LEFT)
        ttk.Label(frm_map, text=" × ", font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self.var_map_h = tk.IntVar(value=MAP_H_DEFAULT)
        ttk.Spinbox(frm_map, from_=10, to=500, textvariable=self.var_map_h,
                    width=5).pack(side=tk.LEFT)

        ttk.Label(p, text="Pickup points:", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(8, 0))
        self.var_pickups = tk.IntVar(value=5)
        ttk.Spinbox(p, from_=1, to=50, textvariable=self.var_pickups,
                    width=10).pack(anchor=tk.W)

        ttk.Label(p, text="Dropoff points:", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(6, 0))
        self.var_dropoffs = tk.IntVar(value=5)
        ttk.Spinbox(p, from_=1, to=50, textvariable=self.var_dropoffs,
                    width=10).pack(anchor=tk.W)

        ttk.Label(p, text="Obstacle walls:", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(6, 0))
        self.var_obstacles = tk.IntVar(value=20)
        ttk.Spinbox(p, from_=0, to=120, textvariable=self.var_obstacles,
                    width=10).pack(anchor=tk.W)

        ttk.Label(p, text="Seed:", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(6, 0))
        self.var_seed = tk.IntVar(value=42)
        ttk.Entry(p, textvariable=self.var_seed, width=12).pack(anchor=tk.W)

        ttk.Separator(p).pack(fill=tk.X, pady=8)

        ttk.Label(p, text="GA PARAMETERS",
                  font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(p, text="Population:", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(4, 0))
        self.var_pop = tk.IntVar(value=80)
        ttk.Spinbox(p, from_=10, to=500, textvariable=self.var_pop,
                    width=10).pack(anchor=tk.W)
        ttk.Label(p, text="Max generations:", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(6, 0))
        self.var_gen = tk.IntVar(value=200)
        ttk.Spinbox(p, from_=10, to=2000, textvariable=self.var_gen,
                    width=10).pack(anchor=tk.W)

        ttk.Separator(p).pack(fill=tk.X, pady=8)

        ttk.Button(p, text="🌍  Generate Environment",
                   command=self.action_generate).pack(fill=tk.X, pady=2)
        self.ga_btn = ttk.Button(p, text="① Step 1: Run GA",
                                 command=self.action_run_ga)
        self.ga_btn.pack(fill=tk.X, pady=2)
        self.astar_btn = ttk.Button(p, text="② Step 2: Run A*",
                                    command=self.action_run_astar,
                                    state=tk.DISABLED)
        self.astar_btn.pack(fill=tk.X, pady=2)
        ttk.Button(p, text="🧹  Reset",
                   command=self.action_reset).pack(fill=tk.X, pady=2)

    # -------- Tab 2: Simulation ----------
    def _build_tab_sim(self, p):
        ttk.Label(p, text="ROBOT SIMULATION",
                  font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(p, text="Robot moves along the A* path on the map.\n"
                          "Step 2 (A*) must be completed first.",
                  foreground="#555", justify=tk.LEFT, font=("Segoe UI", 10),
                  wraplength=290).pack(anchor=tk.W, pady=(2, 8))

        ttk.Label(p, text="Speed (cells/sec):", font=("Segoe UI", 10)).pack(anchor=tk.W)
        self.var_sim_speed = tk.DoubleVar(value=20.0)
        ttk.Scale(p, from_=2.0, to=80.0, orient=tk.HORIZONTAL,
                  variable=self.var_sim_speed, length=260
                  ).pack(anchor=tk.W, pady=(2, 6))
        self.sim_speed_lbl = ttk.Label(p, text="20.0 cells/s", font=("Segoe UI", 10))
        self.sim_speed_lbl.pack(anchor=tk.W)
        self.var_sim_speed.trace_add(
            "write",
            lambda *_: self.sim_speed_lbl.configure(
                text=f"{self.var_sim_speed.get():.1f} cells/s"))

        ttk.Separator(p).pack(fill=tk.X, pady=8)

        self.sim_play_btn = ttk.Button(p, text="▶  Play",
                                       command=self.action_sim_play)
        self.sim_play_btn.pack(fill=tk.X, pady=2)
        self.sim_pause_btn = ttk.Button(p, text="⏸  Pause / Resume",
                                        command=self.action_sim_pause,
                                        state=tk.DISABLED)
        self.sim_pause_btn.pack(fill=tk.X, pady=2)
        self.sim_stop_btn = ttk.Button(p, text="■  Stop",
                                       command=self.action_sim_stop,
                                       state=tk.DISABLED)
        self.sim_stop_btn.pack(fill=tk.X, pady=2)

        ttk.Separator(p).pack(fill=tk.X, pady=8)
        ttk.Label(p, text="Progress:", font=("Segoe UI", 10)).pack(anchor=tk.W)
        self.sim_progress = ttk.Progressbar(p, length=260, mode="determinate")
        self.sim_progress.pack(anchor=tk.W, pady=(2, 4))
        self.sim_msg_var = tk.StringVar(value="(waiting for Play)")
        ttk.Label(p, textvariable=self.sim_msg_var,
                  foreground="#1f4f8b", font=("Segoe UI", 10),
                  wraplength=290, justify=tk.LEFT).pack(anchor=tk.W)

        # ---- Dynamic Obstacles ----
        ttk.Separator(p).pack(fill=tk.X, pady=8)
        ttk.Label(p, text="DYNAMIC OBSTACLES",
                  font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(p, text="Obstacles appear on the path during simulation.\n"
                          "Robot automatically replans when blocked.",
                  foreground="#555", justify=tk.LEFT, font=("Segoe UI", 10),
                  wraplength=290).pack(anchor=tk.W, pady=(2, 6))

        self.var_dyn_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(p, text="Enable dynamic obstacles",
                        variable=self.var_dyn_enabled).pack(anchor=tk.W)

        ttk.Label(p, text="Frequency (spawn 1 every N steps):", font=("Segoe UI", 10)).pack(
            anchor=tk.W, pady=(4, 0))
        self.var_dyn_freq = tk.IntVar(value=30)
        ttk.Spinbox(p, from_=5, to=200, textvariable=self.var_dyn_freq,
                    width=10).pack(anchor=tk.W)

        ttk.Label(p, text="Duration (steps):", font=("Segoe UI", 10)).pack(
            anchor=tk.W, pady=(4, 0))
        self.var_dyn_duration = tk.IntVar(value=50)
        ttk.Spinbox(p, from_=10, to=500, textvariable=self.var_dyn_duration,
                    width=10).pack(anchor=tk.W)

        ttk.Label(p, text="Look-ahead (cells):", font=("Segoe UI", 10)).pack(
            anchor=tk.W, pady=(4, 0))
        self.var_dyn_lookahead = tk.IntVar(value=10)
        ttk.Spinbox(p, from_=3, to=50, textvariable=self.var_dyn_lookahead,
                    width=10).pack(anchor=tk.W)

    # -------- Tab 3: Real NAO ----------
    def _build_tab_nao(self, p):
        ttk.Label(p, text="REAL NAO CONTROL",
                  font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(p, text="Connect via NAOqi/qi-framework.\n"
                          "If SDK not installed → FAKE mode (print only).",
                  foreground="#555", justify=tk.LEFT, font=("Segoe UI", 10),
                  wraplength=290).pack(anchor=tk.W, pady=(2, 8))

        ttk.Label(p, text="NAO IP:", font=("Segoe UI", 10)).pack(anchor=tk.W)
        self.var_nao_ip = tk.StringVar(value="127.0.0.1")
        ttk.Entry(p, textvariable=self.var_nao_ip, width=20
                  ).pack(anchor=tk.W, pady=(2, 6))

        ttk.Label(p, text="Port:", font=("Segoe UI", 10)).pack(anchor=tk.W)
        self.var_nao_port = tk.IntVar(value=9559)
        ttk.Entry(p, textvariable=self.var_nao_port, width=10
                  ).pack(anchor=tk.W, pady=(2, 6))

        ttk.Label(p, text="Cell size (meters):", font=("Segoe UI", 10)).pack(anchor=tk.W)
        self.var_cell_m = tk.DoubleVar(value=0.10)
        ttk.Spinbox(p, from_=0.02, to=1.0, increment=0.01,
                    textvariable=self.var_cell_m, width=10
                    ).pack(anchor=tk.W, pady=(2, 6))

        self.var_nao_fake = tk.BooleanVar(value=True)
        ttk.Checkbutton(p, text="Force FAKE mode (print only)",
                        variable=self.var_nao_fake
                        ).pack(anchor=tk.W, pady=(4, 6))

        ttk.Separator(p).pack(fill=tk.X, pady=6)

        self.nao_connect_btn = ttk.Button(p, text="🔌  Connect NAO",
                                          command=self.action_nao_connect)
        self.nao_connect_btn.pack(fill=tk.X, pady=2)
        self.nao_run_btn = ttk.Button(p, text="🚶  Run A* Path",
                                      command=self.action_nao_run,
                                      state=tk.DISABLED)
        self.nao_run_btn.pack(fill=tk.X, pady=2)
        self.nao_stop_btn = ttk.Button(p, text="🛑  Emergency Stop",
                                       command=self.action_nao_stop,
                                       state=tk.DISABLED)
        self.nao_stop_btn.pack(fill=tk.X, pady=2)
        self.nao_disconnect_btn = ttk.Button(p, text="❌  Disconnect",
                                             command=self.action_nao_disconnect,
                                             state=tk.DISABLED)
        self.nao_disconnect_btn.pack(fill=tk.X, pady=2)

        ttk.Separator(p).pack(fill=tk.X, pady=6)
        ttk.Label(p, text="NAO Log:", font=("Segoe UI", 10)).pack(anchor=tk.W)
        self.nao_log = scrolledtext.ScrolledText(p, height=8, width=38,
                                                 font=("Consolas", 9))
        self.nao_log.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

    # -------- Tab 4: Evaluation ----------
    def _build_tab_eval(self, p):
        ttk.Label(p, text="EVALUATION CHARTS",
                  font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(p, text="Run experiments and export 7 charts "
                          "for GA and A* (saved in ./charts/).",
                  foreground="#555", justify=tk.LEFT, font=("Segoe UI", 10),
                  wraplength=290).pack(anchor=tk.W, pady=(2, 8))

        self.var_eval_random = tk.BooleanVar(value=False)
        ttk.Checkbutton(p, text="Random mode (non-reproducible)",
                        variable=self.var_eval_random).pack(anchor=tk.W, pady=(0, 6))

        self.eval_btn = ttk.Button(p, text="📊  Export All Charts",
                                   command=self.action_eval_run)
        self.eval_btn.pack(fill=tk.X, pady=2)
        ttk.Button(p, text="📁  Open Charts Folder",
                   command=self.action_eval_open_folder).pack(fill=tk.X, pady=2)

        ttk.Separator(p).pack(fill=tk.X, pady=8)
        ttk.Label(p, text="Chart list:", font=("Segoe UI", 10)).pack(anchor=tk.W)
        self.eval_list = tk.Listbox(p, height=10, font=("Consolas", 9))
        self.eval_list.pack(fill=tk.BOTH, expand=True, pady=(2, 4))
        self.eval_list.bind("<Double-Button-1>", self._on_eval_open_chart)
        ttk.Button(p, text="View Selected Chart",
                   command=self._on_eval_open_chart_btn
                   ).pack(fill=tk.X, pady=(2, 0))
        self._refresh_chart_list()

    # =========================================================== Drawing
    def _draw_empty(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.map_w)
        self.ax.set_ylim(self.map_h, 0)
        self.ax.set_aspect("equal")
        self._apply_grid_ticks(self.ax)
        self.ax.set_title(
            f"Map {self.map_w}×{self.map_h} (grid) — no data yet")
        self.ax.text(self.map_w / 2, self.map_h / 2,
                     "Click \"Generate Environment\"",
                     ha="center", va="center", fontsize=14, color="#888")
        self._robot_dot = None
        self._sim_msg = None
        self.canvas.draw_idle()

    def _apply_grid_ticks(self, ax):
        w, h = self.map_w, self.map_h
        # Major ticks: every 10 cells
        major_step = 10 if w >= 20 else max(1, w // 5)
        ax.set_xticks(range(0, w + 1, major_step))
        ax.set_yticks(range(0, h + 1, major_step))
        # Minor ticks: every 1 cell (only drawn for maps ≤ 150)
        if w <= 150 and h <= 150:
            ax.set_xticks(range(0, w + 1, 1), minor=True)
            ax.set_yticks(range(0, h + 1, 1), minor=True)
            ax.grid(which="minor", color="#dddddd", linewidth=0.3, alpha=0.6)
        ax.grid(which="major", color="#888888", linewidth=0.6, alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=8)

    def _draw_environment(self):
        self.ax.clear()
        h, w = self.map_h, self.map_w

        img = np.ones((h, w, 3), dtype=float)
        for y in range(h):
            for x in range(w):
                if self.grid_raw[y][x] == 1:
                    img[y, x] = (0.15, 0.15, 0.15)
                elif self.grid_inflated[y][x] == 1:
                    img[y, x] = (0.85, 0.85, 0.85)
        self.ax.imshow(img, extent=(0, w, h, 0), interpolation="nearest",
                       zorder=0)
        self.ax.set_axisbelow(False)

        sx, sy = self.start
        self.ax.scatter([sx + 0.5], [sy + 0.5], s=240, c="gold",
                        edgecolors="black", marker="*", zorder=6,
                        label="Start")

        for i, o in enumerate(self.orders):
            c = color_for(i)
            self.ax.scatter([o.pickup[0] + 0.5], [o.pickup[1] + 0.5],
                            s=70, c=c, edgecolors="black", marker="o",
                            zorder=5)
            self.ax.scatter([o.dropoff[0] + 0.5], [o.dropoff[1] + 0.5],
                            s=70, c=c, edgecolors="black", marker="s",
                            zorder=5)
            self.ax.text(o.pickup[0] + 0.5, o.pickup[1] - 1.6,
                         f"P{o.label}", ha="center", fontsize=7,
                         color=c, fontweight="bold")
            self.ax.text(o.dropoff[0] + 0.5, o.dropoff[1] - 1.6,
                         f"D{o.label}", ha="center", fontsize=7,
                         color=c, fontweight="bold")
        for p in self.unmatched_pickups:
            self.ax.scatter([p[0] + 0.5], [p[1] + 0.5], s=50, c="#cccccc",
                            edgecolors="black", marker="o", zorder=4)
        for d in self.unmatched_dropoffs:
            self.ax.scatter([d[0] + 0.5], [d[1] + 0.5], s=50, c="#cccccc",
                            edgecolors="black", marker="s", zorder=4)

        if self.full_path:
            xs = [p[0] + 0.5 for p in self.full_path]
            ys = [p[1] + 0.5 for p in self.full_path]
            self.ax.plot(xs, ys, "-", color="#2ca02c", lw=1.8, alpha=0.9,
                         label="A* path", zorder=3)
            for idx, (x, y) in enumerate(self.result.best_route[1:], start=1):
                self.ax.text(x + 0.5, y + 0.5, str(idx),
                             ha="center", va="center", fontsize=6.5,
                             color="white", fontweight="bold", zorder=7)
        elif self.result is not None:
            xs = [p[0] + 0.5 for p in self.result.best_route]
            ys = [p[1] + 0.5 for p in self.result.best_route]
            self.ax.plot(xs, ys, "--", color="#ff7f0e", lw=1.6, alpha=0.9,
                         label="GA waypoint order", zorder=3)
            for idx, (x, y) in enumerate(self.result.best_route[1:], start=1):
                self.ax.text(x + 0.5, y + 0.5, str(idx),
                             ha="center", va="center", fontsize=6.5,
                             color="white", fontweight="bold", zorder=7)

        self.ax.set_xlim(0, w); self.ax.set_ylim(h, 0)
        self.ax.set_aspect("equal")
        self._apply_grid_ticks(self.ax)
        title = (f"Map {self.map_w}×{self.map_h} | {len(self.orders)} orders | "
                 f"○ pickup, □ dropoff")
        if self.full_path is not None:
            title += (f"\n[Step 2 ✓] Actual distance (A*): "
                      f"{self.actual_distance:.2f} cells"
                      f", {len(self.full_path)} steps")
        elif self.result is not None:
            title += (f"\n[Step 1 ✓] GA done — reference distance: "
                      f"{self.result.total_distance:.2f} cells  (A* not run yet)")
        self.ax.set_title(title, fontsize=11)

        # add (empty) robot dot artist for animation
        self._robot_dot, = self.ax.plot([], [], "o", color="#d62728",
                                        markersize=10, zorder=8,
                                        markeredgecolor="black")
        self._sim_msg = self.ax.text(0, 0, "", color="#d62728", fontsize=10,
                                     fontweight="bold", zorder=9)
        self.canvas.draw_idle()

    # ============================================================ Actions
    def action_generate(self):
        n_pick = max(1, self.var_pickups.get())
        n_drop = max(1, self.var_dropoffs.get())
        n_obs = max(0, self.var_obstacles.get())
        seed = self.var_seed.get()
        # Read map size from GUI
        self.map_w = max(10, self.var_map_w.get())
        self.map_h = max(10, self.var_map_h.get())
        n_orders = min(n_pick, n_drop)
        if n_pick != n_drop:
            messagebox.showwarning("Warning",
                f"Pickup count ({n_pick}) differs from dropoff count ({n_drop}). "
                f"System can only pair {n_orders} orders.")
        rng = random.Random(seed)
        self.grid_raw = generate_random_obstacles(
            n_obs, map_w=self.map_w, map_h=self.map_h, seed=seed)
        self.grid_inflated = inflate_obstacles(self.grid_raw, radius=1)
        for y in range(3):
            for x in range(3):
                self.grid_raw[y][x] = 0
                self.grid_inflated[y][x] = 0
        self.start = (0, 0)
        try:
            used = {self.start}
            pickups = []
            for _ in range(n_pick):
                p = random_free_point(self.grid_inflated, rng, exclude=used)
                used.add(p); pickups.append(p)
            dropoffs = []
            for _ in range(n_drop):
                d = random_free_point(self.grid_inflated, rng, exclude=used)
                used.add(d); dropoffs.append(d)
        except RuntimeError as e:
            messagebox.showerror("Error", f"Cannot place enough points: {e}"); return
        self.orders = [Order(pickup=pickups[i], dropoff=dropoffs[i],
                             label=str(i + 1)) for i in range(n_orders)]
        self.unmatched_pickups = pickups[n_orders:]
        self.unmatched_dropoffs = dropoffs[n_orders:]
        self.result = None; self.full_path = None; self.actual_distance = 0.0
        self.astar_btn.configure(state=tk.DISABLED)
        self._update_sim_buttons()
        self._update_nao_run_button()
        self.status_var.set(
            f"Environment created {self.map_w}×{self.map_h}:\n"
            f"  • {n_orders} orders\n"
            f"  • {n_obs} obstacle walls\n  • Start = (0, 0)\n\n"
            f"→ Go to Problem tab to run GA, "
            f"or Simulation after A* path is ready.")
        self._draw_environment()

    # ---- Step 1: GA ----
    def action_run_ga(self):
        if not self.orders:
            messagebox.showinfo("Info", "Please generate an environment first."); return
        self._stop_simulation()
        self.ga_btn.configure(state=tk.DISABLED)
        self.astar_btn.configure(state=tk.DISABLED)
        self.status_var.set("⏳ Step 1: GA running (Octile)...")
        threading.Thread(target=self._run_ga_thread, daemon=True).start()

    def _run_ga_thread(self):
        try:
            cfg = GAConfig(pop_size=self.var_pop.get(),
                           max_gen=self.var_gen.get(),
                           crossover_rate=0.85, mutation_rate=0.20,
                           tournament_k=3, elitism_ratio=0.05,
                           early_stop_gen=80, seed=self.var_seed.get(),
                           distance_method="octile")
            self.result = run_ga(self.grid_inflated, self.start,
                                 self.orders, cfg)
            self.full_path = None; self.actual_distance = 0.0
            self.root.after(0, self._on_ga_done)
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self._on_error(err, step="GA"))

    def _on_ga_done(self):
        self.ga_btn.configure(state=tk.NORMAL)
        self.astar_btn.configure(state=tk.NORMAL)
        order_str = " → ".join(
            f"{'P' if task_is_pickup(t) else 'D'}"
            f"{self.orders[task_order_id(t)].label}"
            for t in self.result.best_sequence)
        self.status_var.set(
            f"✓ Step 1 (GA) complete\n"
            f"  • Octile distance: {self.result.total_distance:.2f} cells\n"
            f"  • Generations: {self.result.generations_run}\n"
            f"  • Order: {order_str}\n→ Step 2: run A*.")
        self._update_sim_buttons(); self._update_nao_run_button()
        self._draw_environment()

    # ---- Step 2: A* ----
    def action_run_astar(self):
        if self.result is None:
            messagebox.showinfo("Info", "Please run Step 1 (GA) first."); return
        self._stop_simulation()
        self.ga_btn.configure(state=tk.DISABLED)
        self.astar_btn.configure(state=tk.DISABLED)
        self.status_var.set("⏳ Step 2: A* pathfinding around obstacles...")
        threading.Thread(target=self._run_astar_thread, daemon=True).start()

    def _run_astar_thread(self):
        try:
            # Expand full path through all waypoints
            self.full_path = expand_full_path(self.grid_inflated,
                                              self.result.best_route)
            # Smooth path to reduce unnecessary waypoints and inertia
            original_len = len(self.full_path)
            self.full_path = smooth_path(self.grid_inflated, self.full_path)
            smoothed_count = original_len - len(self.full_path)
            # Recalculate distance after smoothing
            self.actual_distance = path_length(self.full_path)
            # Store smoothing stats for display
            self._smoothed_waypoints = smoothed_count
            self.root.after(0, self._on_astar_done)
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self._on_error(err, step="A*"))

    def _on_astar_done(self):
        self.ga_btn.configure(state=tk.NORMAL)
        self.astar_btn.configure(state=tk.NORMAL)
        gain = self.actual_distance - self.result.total_distance
        sign = "+" if gain >= 0 else ""
        smoothed = getattr(self, '_smoothed_waypoints', 0)
        self.status_var.set(
            f"✓ Step 2 (A*) complete\n"
            f"  • Actual distance: {self.actual_distance:.2f} cells\n"
            f"  • Path steps: {len(self.full_path)}\n"
            f"  • Smoothed: removed {smoothed} waypoints\n"
            f"  • Δ vs GA: {sign}{gain:.2f} cells\n\n"
            f"→ Ready to SIMULATE or RUN on real NAO.")
        self._update_sim_buttons(); self._update_nao_run_button()
        self._draw_environment()

    def _on_error(self, msg, step="GA"):
        self.ga_btn.configure(state=tk.NORMAL)
        if self.result is not None:
            self.astar_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Error in {step}: {msg}")
        messagebox.showerror(f"Error running {step}", msg)

    def action_reset(self):
        self._stop_simulation()
        self.grid_raw = None; self.grid_inflated = None
        self.orders = []; self.unmatched_pickups = []; self.unmatched_dropoffs = []
        self.result = None; self.full_path = None; self.actual_distance = 0.0
        self.astar_btn.configure(state=tk.DISABLED)
        self._update_sim_buttons(); self._update_nao_run_button()
        self._draw_empty()
        self.status_var.set("Reset complete.")

    # ===================================================== Simulation
    def _update_sim_buttons(self):
        ready = self.full_path is not None
        self.sim_play_btn.configure(state=tk.NORMAL if ready else tk.DISABLED)
        if not ready:
            self.sim_pause_btn.configure(state=tk.DISABLED)
            self.sim_stop_btn.configure(state=tk.DISABLED)

    def action_sim_play(self):
        if not self.full_path:
            messagebox.showinfo("Info", "No A* path available."); return
        if self.sim_running and self.sim_paused:
            self.sim_paused = False
            self.sim_msg_var.set("Running...")
            self._sim_step()
            return
        # fresh start
        self.sim_running = True; self.sim_paused = False
        self.sim_index = 0
        self._replan_count = 0
        self.dyn_mgr.clear()
        self._sim_full_path = list(self.full_path)  # copy to modify during replan
        self.sim_progress.configure(maximum=len(self._sim_full_path) - 1, value=0)
        self.sim_msg_var.set("Running...")
        self.sim_pause_btn.configure(state=tk.NORMAL)
        self.sim_stop_btn.configure(state=tk.NORMAL)
        self._waypoint_set = {wp: idx for idx, wp
                              in enumerate(self.result.best_route)}
        # Remaining waypoints for replanning
        self._remaining_waypoints = list(self.result.best_route[1:])
        self._sim_step()

    def action_sim_pause(self):
        if not self.sim_running:
            return
        self.sim_paused = not self.sim_paused
        if self.sim_paused:
            self.sim_msg_var.set("⏸ Paused.")
            if self.sim_after_id is not None:
                self.root.after_cancel(self.sim_after_id)
                self.sim_after_id = None
        else:
            self.sim_msg_var.set("Running...")
            self._sim_step()

    def action_sim_stop(self):
        self._stop_simulation()
        self.sim_msg_var.set("■ Stopped.")

    def _stop_simulation(self):
        self.sim_running = False; self.sim_paused = False
        if self.sim_after_id is not None:
            try: self.root.after_cancel(self.sim_after_id)
            except Exception: pass
            self.sim_after_id = None
        self.sim_pause_btn.configure(state=tk.DISABLED)
        self.sim_stop_btn.configure(state=tk.DISABLED)
        if self._robot_dot is not None:
            self._robot_dot.set_data([], [])
        if self._sim_msg is not None:
            self._sim_msg.set_text("")
        # Remove dynamic obstacles from map
        if self._dyn_scatter is not None:
            self._dyn_scatter.remove()
            self._dyn_scatter = None
        self.dyn_mgr.clear()
        self.canvas.draw_idle()

    def _sim_step(self):
        """
        Each simulation step:
        1. Update dynamic obstacles (spawn new + expire old)
        2. Check if the path ahead is blocked
        3. If blocked → replan (recalculate path)
        4. Move robot to the next cell
        """
        if not self.sim_running or self.sim_paused:
            return
        if self.sim_index >= len(self._sim_full_path):
            self.sim_msg_var.set(
                f"✓ Simulation complete. Replanned: {self._replan_count} time(s).")
            self._stop_simulation()
            return

        # --- 1) Update dynamic obstacles ---
        if self.var_dyn_enabled.get():
            self.dyn_mgr.update(self.sim_index)
            freq = max(5, self.var_dyn_freq.get())
            duration = max(10, self.var_dyn_duration.get())
            look_ahead = max(3, self.var_dyn_lookahead.get())
            # Spawn new obstacle ON THE PATH at given frequency
            if self.sim_index > 0 and self.sim_index % freq == 0:
                # Avoid placing obstacle on waypoints (pickup/dropoff)
                avoid = {self.start}
                for o in self.orders:
                    avoid.add(o.pickup)
                    avoid.add(o.dropoff)
                # Place obstacle directly on path ahead of robot
                self.dyn_mgr.spawn_on_path(
                    self._sim_full_path, self.sim_index,
                    current_step=self.sim_index,
                    duration=duration,
                    avoid_cells=avoid,
                    min_ahead=3, max_ahead=look_ahead
                )

            # --- 2) Check if path ahead is blocked ---
            blocked_idx = check_path_blocked(
                self._sim_full_path, self.sim_index, self.dyn_mgr, look_ahead)

            if blocked_idx is not None:
                # --- 3) Replan: recalculate path from current position ---
                current_pos = self._sim_full_path[self.sim_index]
                # Find next unvisited waypoint
                self._update_remaining_waypoints(current_pos)
                new_path = replan_full_path(
                    self.grid_inflated, current_pos,
                    self._remaining_waypoints, self.dyn_mgr)
                if new_path is not None:
                    self._replan_count += 1
                    # Smooth the new path to reduce waypoints
                    original_len = len(new_path)
                    new_path = smooth_path(self.grid_inflated, new_path)
                    # Merge already-traveled portion + smoothed new path
                    self._sim_full_path = (
                        self._sim_full_path[:self.sim_index] + new_path)
                    self.sim_progress.configure(
                        maximum=len(self._sim_full_path) - 1)
                    smoothed_count = original_len - len(new_path)
                    self.sim_msg_var.set(
                        f"⚠ Dynamic obstacle! Replanned "
                        f"(#{self._replan_count}, -{smoothed_count} waypoints)")
                    # Redraw new path on map
                    self._redraw_path()
                # If no path found → keep going, obstacle may expire

            # Draw dynamic obstacles on map
            self._draw_dynamic_obstacles()

        # --- 4) Move robot ---
        x, y = self._sim_full_path[self.sim_index]
        if self._robot_dot is not None:
            self._robot_dot.set_data([x + 0.5], [y + 0.5])
        # waypoint event?
        if (x, y) in self._waypoint_set and self.sim_index > 0:
            wp_idx = self._waypoint_set[(x, y)]
            if 1 <= wp_idx <= 2 * len(self.orders):
                t = self.result.best_sequence[wp_idx - 1]
                label = self.orders[task_order_id(t)].label
                kind = "📦 Pickup" if task_is_pickup(t) else "🏁 Deliver"
                msg = f"{kind} order {label}"
                if self._sim_msg is not None:
                    self._sim_msg.set_position((x + 1.5, y + 1.5))
                    self._sim_msg.set_text(msg)
                self.sim_msg_var.set(f"{msg}  (step {self.sim_index}/"
                                     f"{len(self._sim_full_path) - 1})")
                # Update remaining_waypoints when reaching a waypoint
                if (x, y) in self._remaining_waypoints:
                    self._remaining_waypoints.remove((x, y))
        self.sim_progress.configure(value=self.sim_index)
        self.canvas.draw_idle()
        self.sim_index += 1
        speed = max(1.0, self.var_sim_speed.get())
        delay_ms = max(15, int(1000.0 / speed))
        self.sim_after_id = self.root.after(delay_ms, self._sim_step)

    def _update_remaining_waypoints(self, current_pos):
        """Remove already-visited waypoints from the remaining list."""
        while (self._remaining_waypoints and
               self._remaining_waypoints[0] == current_pos):
            self._remaining_waypoints.pop(0)

    def _draw_dynamic_obstacles(self):
        """Draw/update dynamic obstacles (red X markers) on the map."""
        if self._dyn_scatter is not None:
            self._dyn_scatter.remove()
            self._dyn_scatter = None
        blocked = self.dyn_mgr.get_blocked_cells()
        if blocked:
            xs = [p[0] + 0.5 for p in blocked]
            ys = [p[1] + 0.5 for p in blocked]
            self._dyn_scatter = self.ax.scatter(
                xs, ys, s=60, c="red", marker="x",
                linewidths=2, zorder=7, label="Dynamic obstacle")

    def _redraw_path(self):
        """Redraw the path (after replan) on the map."""
        # Remove old path and redraw
        for line in self.ax.lines[:]:
            if hasattr(line, 'get_label') and line.get_label() == "A* path":
                line.remove()
        # Draw new path
        xs = [p[0] + 0.5 for p in self._sim_full_path]
        ys = [p[1] + 0.5 for p in self._sim_full_path]
        self.ax.plot(xs, ys, "-", color="#2ca02c", lw=1.8, alpha=0.9,
                     label="A* path", zorder=3)

    # ===================================================== NAO real
    def _log_nao(self, msg: str):
        self.nao_log.insert(tk.END, msg + "\n")
        self.nao_log.see(tk.END)

    def _update_nao_run_button(self):
        ready = self.nao.connected and self.full_path is not None
        self.nao_run_btn.configure(state=tk.NORMAL if ready else tk.DISABLED)

    def action_nao_connect(self):
        ip = self.var_nao_ip.get().strip() or "127.0.0.1"
        port = int(self.var_nao_port.get())
        self.nao = NaoController(cell_size_m=float(self.var_cell_m.get()),
                                 fake=bool(self.var_nao_fake.get()))
        ok, msg = self.nao.connect(ip, port)
        self._log_nao(f"[connect] {msg}")
        if ok:
            self.nao_connect_btn.configure(state=tk.DISABLED)
            self.nao_disconnect_btn.configure(state=tk.NORMAL)
            self._update_nao_run_button()
            self.nao.stand_init()
            self._log_nao(f"[sdk] {self.nao.sdk_name}")
        else:
            messagebox.showerror("NAO", msg)

    def action_nao_disconnect(self):
        if self.nao_running:
            self._nao_stop_flag = True
            self._log_nao(f"[stop] requesting stop before disconnect...")
            return
        self.nao.disconnect()
        self._log_nao("[disconnect] disconnected.")
        self.nao_connect_btn.configure(state=tk.NORMAL)
        self.nao_disconnect_btn.configure(state=tk.DISABLED)
        self.nao_run_btn.configure(state=tk.DISABLED)
        self.nao_stop_btn.configure(state=tk.DISABLED)

    def action_nao_run(self):
        if not (self.nao.connected and self.full_path):
            messagebox.showinfo("NAO", "Not ready (connect + A* required)."); return
        if self.nao_running: return
        self.nao_running = True; self._nao_stop_flag = False
        self.nao.cell_size_m = float(self.var_cell_m.get())
        self.nao_run_btn.configure(state=tk.DISABLED)
        self.nao_stop_btn.configure(state=tk.NORMAL)
        self.nao_disconnect_btn.configure(state=tk.DISABLED)
        self._waypoint_set = {wp: idx for idx, wp
                              in enumerate(self.result.best_route)}
        self._log_nao(f"[run] following {len(self.full_path)} cells "
                      f"({self.nao.cell_size_m:.2f} m/cell)")
        threading.Thread(target=self._nao_run_thread, daemon=True).start()

    def _nao_run_thread(self):
        try:
            self.nao.say("Starting delivery")

            def on_progress(i, n):
                # i is the index in path[1:], so cell = full_path[i]
                cell = self.full_path[i]
                self.root.after(0, self._on_nao_step, i, n, cell)

            self.nao.follow_path_cells(
                self.full_path,
                on_progress=on_progress,
                stop_flag=lambda: self._nao_stop_flag)
            self.nao.say("Delivery complete")
            self.root.after(0, self._on_nao_done, False)
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self._on_nao_done(True, err))

    def _on_nao_step(self, i, n, cell):
        # update on-canvas robot dot too
        if self._robot_dot is not None:
            self._robot_dot.set_data([cell[0] + 0.5], [cell[1] + 0.5])
            self.canvas.draw_idle()
        if cell in self._waypoint_set and i > 0:
            wp_idx = self._waypoint_set[cell]
            if 1 <= wp_idx <= 2 * len(self.orders):
                t = self.result.best_sequence[wp_idx - 1]
                label = self.orders[task_order_id(t)].label
                if task_is_pickup(t):
                    self.nao.say(f"Picked up order {label}")
                    self._log_nao(f"  ► P{label}  (cell {cell})")
                else:
                    self.nao.say(f"Delivered order {label}")
                    self._log_nao(f"  ► D{label}  (cell {cell})")

    def _on_nao_done(self, error: bool, msg: str = ""):
        self.nao_running = False
        self.nao_run_btn.configure(state=tk.NORMAL if self.nao.connected
                                   else tk.DISABLED)
        self.nao_stop_btn.configure(state=tk.DISABLED)
        self.nao_disconnect_btn.configure(state=tk.NORMAL if self.nao.connected
                                          else tk.DISABLED)
        if error:
            self._log_nao(f"[error] {msg}")
            messagebox.showerror("NAO", msg)
        else:
            self._log_nao("[done] journey complete.")

    def action_nao_stop(self):
        self._nao_stop_flag = True
        self._log_nao("[stop] stopping at next waypoint...")

    # ===================================================== Evaluate
    def action_eval_run(self):
        self.eval_btn.configure(state=tk.DISABLED)
        rand = bool(self.var_eval_random.get())
        tag = "random" if rand else "reproducible"
        self.status_var.set(f"⏳ Generating evaluation charts ({tag}, ~30s)...")
        threading.Thread(target=self._eval_run_thread,
                         args=(rand,), daemon=True).start()

    def _eval_run_thread(self, random_mode: bool):
        try:
            import importlib, evaluate
            importlib.reload(evaluate)
            evaluate.main(random_mode=random_mode)
            self.root.after(0, self._on_eval_done, None)
        except Exception as e:
            err = str(e)
            self.root.after(0, self._on_eval_done, err)

    def _on_eval_done(self, err):
        self.eval_btn.configure(state=tk.NORMAL)
        if err:
            self.status_var.set(f"Chart export error: {err}")
            messagebox.showerror("Evaluation", err); return
        self.status_var.set("✓ Charts exported to ./charts/")
        self._refresh_chart_list()

    def _refresh_chart_list(self):
        self.eval_list.delete(0, tk.END)
        if not os.path.isdir("charts"): return
        for f in sorted(os.listdir("charts")):
            if f.lower().endswith(".png"):
                self.eval_list.insert(tk.END, f)

    def action_eval_open_folder(self):
        path = os.path.abspath("charts")
        os.makedirs(path, exist_ok=True)
        self._open_path(path)

    def _on_eval_open_chart(self, _ev=None):
        self._on_eval_open_chart_btn()

    def _on_eval_open_chart_btn(self):
        sel = self.eval_list.curselection()
        if not sel:
            messagebox.showinfo("Evaluation", "Please select a chart."); return
        f = self.eval_list.get(sel[0])
        self._open_path(os.path.abspath(os.path.join("charts", f)))

    @staticmethod
    def _open_path(path: str):
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Cannot open", str(e))


def main():
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
