import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================

np.random.seed(42)

generations = np.arange(1, 501)

# =====================================================
# f1 : MAIN OBJECTIVE
# Similar to overall fitness
# =====================================================

best_f1 = np.zeros(500)

levels_f1 = [
    56, 60, 64, 68, 72, 76, 80,
    84, 88, 92, 95,

    97, 99, 101, 103,
    105, 107, 109, 110,

    111, 112, 113,
    114, 115, 116
]

points_f1 = [
      0,   5,  10,  15,  20,
     28,  35,  40,  45,  48,
     50,

     70,  90, 110,
    140, 165, 185,
    200,

    260, 320, 380,
    430, 470, 499, 500
]

for i in range(len(levels_f1) - 1):
    best_f1[
        points_f1[i]:
        points_f1[i + 1]
    ] = levels_f1[i]

best_f1[points_f1[-1]:] = levels_f1[-1]

curve_f1 = (
    43
    + (108 - 43)
    * np.log1p(generations)
    / np.log1p(500)
)

mean_f1 = curve_f1 + np.random.normal(
    0,
    np.linspace(2.0, 0.4, 500)
)

# =====================================================
# f2 : HIGHLY VARIABLE COMPONENT
# Strong fluctuations
# =====================================================

best_f2 = np.zeros(500)

levels_f2 = [
    20, 28, 25, 35, 32,
    42, 38, 48, 45,
    55, 52, 60, 58,
    67, 65, 72, 70,
    76, 75, 80
]

points_f2 = [
      0,  10,  25,  40,  60,
     80, 100, 120, 145,
    170, 200, 230, 260,
    300, 340, 380, 420,
    450, 480, 499
]

for i in range(len(levels_f2) - 1):
    best_f2[
        points_f2[i]:
        points_f2[i + 1]
    ] = levels_f2[i]

best_f2[points_f2[-1]:] = levels_f2[-1]

mean_f2 = (
    15
    + 55 * np.sqrt(generations / 500)
)

mean_f2 += np.random.normal(
    0,
    np.linspace(5, 2, 500)
)

# =====================================================
# f3 : PHASE-BASED IMPROVEMENT
# Large jumps and long plateaus
# =====================================================

best_f3 = np.zeros(500)

levels_f3 = [
    40, 41, 42, 44, 45,
    50, 55,
    40, 30, 20,
    35, 50, 60, 70,
    71, 72,
    78, 85, 86,
    90, 95, 96,
    98, 100
]

points_f3 = [
      0,  15,  30,  50,  70,
     90, 110,
    125, 140, 150,
    170, 190, 210, 230,
    245, 260,
    290, 320, 370,
    400, 430, 470,
    485, 499
]

for i in range(len(levels_f3) - 1):
    best_f3[
        points_f3[i]:
        points_f3[i + 1]
    ] = levels_f3[i]

best_f3[points_f3[-1]:] = levels_f3[-1]

mean_f3 = (
    25
    + 65 * (
        1 - np.exp(-generations / 180)
    )
)

mean_f3 += np.random.normal(
    0,
    np.linspace(2.5, 0.8, 500)
)

# =====================================================
# PLOT
# =====================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 9)
)

# =====================================================
# ROW 1 : f1
# =====================================================

axes[0, 0].plot(
    generations,
    best_f1,
    linewidth=3,
    drawstyle="steps-post"
)

axes[0, 0].set_title(
    "Best individual",
    fontweight="bold"
)

axes[0, 0].set_ylabel(
    "Tms"
)
axes[0, 1].set_ylabel(
    "Mean Tms"
)
axes[0, 1].set_title(
    "Mean of population",
    fontweight="bold"
)
axes[0, 0].grid(
    True,
    alpha=0.3
)

axes[0, 0].set_xlim(0, 500)

axes[0, 1].plot(
    generations,
    mean_f1,
    color="orange",
    linewidth=1.3
)



axes[0, 1].grid(
    True,
    alpha=0.3
)

axes[0, 1].set_xlim(0, 500)

# =====================================================
# ROW 2 : f2
# =====================================================

axes[1, 0].plot(
    generations,
    best_f2,
    linewidth=3,
    drawstyle="steps-post"
)

# axes[1, 0].set_title(
#     "Best Fitness - Component f₂",
#     fontweight="bold"
# )

axes[1, 0].set_ylabel(
    "Es"
)

axes[1, 1].set_ylabel(
    "Mean Es"
)

axes[1, 0].grid(
    True,
    alpha=0.3
)

axes[1, 0].set_xlim(0, 500)

axes[1, 1].plot(
    generations,
    mean_f2,
    color="orange",
    linewidth=1.3
)

# axes[1, 1].set_title(
#     "Mean of population",
#     fontweight="bold"
# )

axes[1, 1].grid(
    True,
    alpha=0.3
)

axes[1, 1].set_xlim(0, 500)

# =====================================================
# ROW 3 : f3
# =====================================================

axes[2, 0].plot(
    generations,
    best_f3,
    linewidth=3,
    drawstyle="steps-post"
)

# axes[2, 0].set_title(
#     "Best Fitness - Component f₃",
#     fontweight="bold"
# )

axes[2, 0].set_ylabel(
    "Ds"
)

axes[2, 1].set_ylabel(
    "Ds"
)

axes[2, 0].set_xlabel(
    "Generation"
)

axes[2, 0].grid(
    True,
    alpha=0.3
)

axes[2, 0].set_xlim(0, 500)

axes[2, 1].plot(
    generations,
    mean_f3,
    color="orange",
    linewidth=1.3
)

# axes[2, 1].set_title(
#     "Mean Fitness - Component f₃",
#     fontweight="bold"
# )

axes[2, 1].set_xlabel(
    "Generation"
)

axes[2, 1].grid(
    True,
    alpha=0.3
)

axes[2, 1].set_xlim(0, 500)

# =====================================================
# SAME SCALE FOR ALL CHARTS
# =====================================================

for ax in axes.flat:
    ax.set_ylim(0, 125)

# =====================================================
# OVERALL TITLE
# =====================================================

plt.suptitle(
    r"Component Fitness Evolution ($f(x)=0.9f_1+0.12f_2+0.36f_3$)",
    fontsize=18,
    fontweight="bold"
)
plt.subplots_adjust(
    hspace=0.55,   # khoảng cách dọc giữa các hàng
    wspace=0.30    # khoảng cách ngang giữa các cột
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(
    hspace=0.55,
    wspace=0.30
)

plt.show()