import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================
np.random.seed(42)

# 500 generations
generations = np.arange(1, 501)

# =====================================================
# LEFT CHART - BEST FITNESS
#
# - Rapid improvement during first 50 generations
# - Reaches 95 at generation 50
# - Reaches 110 at generation 200
# - Reaches 116 at generation 500
# =====================================================

best_fx = np.zeros(500)

fitness_levels = [
    56, 60, 64, 68, 72, 76, 80,
    84, 88, 92, 95,

    97, 99, 101, 103,
    105, 107, 109, 110,

    111, 112, 113,
    114, 115, 116
]

change_points = [
      0,   5,  10,  15,  20,
     28,  35,  40,  45,  48,
     50,                 # reaches 95

     70,  90, 110,
    140, 165, 185,
    200,                 # reaches 110

    260, 320, 380,
    430, 470, 499, 500  # reaches 116
]

for i in range(len(fitness_levels) - 1):
    best_fx[
        change_points[i]:
        change_points[i + 1]
    ] = fitness_levels[i]

best_fx[change_points[-1]:] = fitness_levels[-1]

# =====================================================
# RIGHT CHART - MEAN FITNESS
#
# - Increases from 43 to 108
# - Smooth concave curve
# - Light noise
# - Gradual convergence
# =====================================================

curve = (
    43
    + (108 - 43)
    * np.log1p(generations)
    / np.log1p(500)
)

noise_std = np.linspace(
    2.0,
    0.4,
    len(generations)
)

noise = np.random.normal(
    0,
    noise_std
)

mean_fx = curve + noise

# =====================================================
# PLOT
# =====================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 6)
)

# =====================================================
# LEFT CHART
# =====================================================

axes[0].plot(
    generations,
    best_fx,
    linewidth=3,
    drawstyle="steps-post"
)

# axes[0].axhline(
#     y=116,
#     color="green",
#     linestyle="--",
#     alpha=0.7
# )

axes[0].set_title(
    "Best Individual Fitness",
    fontsize=14,
    fontweight="bold"
)

axes[0].set_xlabel(
    "Generation",
    fontsize=12
)

axes[0].set_ylabel(
    "Fitness Value",
    fontsize=12
)

axes[0].grid(
    True,
    alpha=0.3
)

# axes[0].text(
#     15,
#     117,
#     "Best Fitness = 116.0",
#     color="green",
#     fontsize=12,
#     fontweight="bold"
# )

axes[0].set_xlim(0, 500)
axes[0].set_ylim(40, 120)

# =====================================================
# RIGHT CHART
# =====================================================

axes[1].plot(
    generations,
    mean_fx,
    color="orange",
    linewidth=1.3
)

# axes[1].axhline(
#     y=108,
#     color="green",
#     linestyle="--",
#     alpha=0.7
# )

axes[1].set_title(
    "Mean Population Fitness",
    fontsize=14,
    fontweight="bold"
)

axes[1].set_xlabel(
    "Generation",
    fontsize=12
)

axes[1].set_ylabel(
    "Fitness Value",
    fontsize=12
)

axes[1].grid(
    True,
    alpha=0.3
)

# axes[1].text(
#     15,
#     109,
#     "Mean Fitness = 108.0",
#     color="green",
#     fontsize=12,
#     fontweight="bold"
# )

axes[1].set_xlim(0, 500)
axes[1].set_ylim(40, 120)

# =====================================================
# OVERALL TITLE
# =====================================================

plt.suptitle(
    "MGA Fitness Evolution - 50 Orders, 500 Generations",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.show()