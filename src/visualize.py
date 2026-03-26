import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import os

from environment import STUDY_CASES

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# Couleurs des cellules
CELL_COLORS = {
    'S': '#4CAF50',  # Depart
    'F': '#E3F2FD',  # Glace
    'H': '#D32F2F',  # Trou
    'G': '#FFD700',  # Objectif
}

ACTION_ARROWS = {0: '←', 1: '↓', 2: '→', 3: '↑'}
ACTION_DELTAS = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}


def visualize_grid(case_name, save_path=None, ax=None):
    case = STUDY_CASES[case_name]
    grid = case["map"]
    size = case["size"]

    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for i in range(size):
        for j in range(size):
            cell = grid[i][j]
            color = CELL_COLORS[cell]
            rect = plt.Rectangle((j, i), 1, 1,
                                 facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, cell,
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if cell == 'H' else 'black')

    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect('equal')
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_title(f"{case_name}: {case['desc']}", fontsize=14, fontweight='bold')
    ax.grid(False)

    if save_path and show:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()


def visualize_all_cases(save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, case_name in enumerate(STUDY_CASES):
        visualize_grid(case_name, ax=axes[idx])

    legend_elements = [
        mpatches.Patch(facecolor=CELL_COLORS['S'], edgecolor='black', label='Départ (S)'),
        mpatches.Patch(facecolor=CELL_COLORS['F'], edgecolor='black', label='Gelé (F)'),
        mpatches.Patch(facecolor=CELL_COLORS['H'], edgecolor='black', label='Trou (H)'),
        mpatches.Patch(facecolor=CELL_COLORS['G'], edgecolor='black', label='Objectif (G)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=12, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Cas d'étude Frozen Lake", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(results, metric, title, ylabel, save_path=None):
    cases = list(STUDY_CASES.keys())
    strategies = list(results.keys())
    colors = sns.color_palette("husl", len(strategies))

    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4))
    if len(cases) == 1:
        axes = [axes]

    for idx, case_name in enumerate(cases):
        ax = axes[idx]
        for s_idx, strategy in enumerate(strategies):
            data = results[strategy][case_name]
            episodes = data["episodes"]
            mean_vals = data[metric]
            std_vals = data.get(f"{metric}_std", None)

            ax.plot(episodes, mean_vals, label=strategy,
                    color=colors[s_idx], linewidth=2)
            if std_vals is not None:
                ax.fill_between(episodes,
                                np.array(mean_vals) - np.array(std_vals),
                                np.array(mean_vals) + np.array(std_vals),
                                alpha=0.2, color=colors[s_idx])

        ax.set_title(STUDY_CASES[case_name]["desc"], fontsize=12)
        ax.set_xlabel("Pas d'entraînement")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_policy(Q_values, case_name, strategy_name, save_path=None, ax=None):
    case = STUDY_CASES[case_name]
    grid = case["map"]
    size = case["size"]

    if Q_values.ndim == 2:
        best_actions = np.argmax(Q_values, axis=1)
    else:
        best_actions = Q_values.astype(int)

    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    for i in range(size):
        for j in range(size):
            cell = grid[i][j]
            color = CELL_COLORS[cell]
            rect = plt.Rectangle((j, i), 1, 1,
                                 facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)

            state = i * size + j
            if cell in ('F', 'S'):
                action = best_actions[state]
                arrow = ACTION_ARROWS[action]
                ax.text(j + 0.5, i + 0.5, arrow,
                        ha='center', va='center', fontsize=20, fontweight='bold',
                        color='#1565C0')
            else:
                label = 'H' if cell == 'H' else 'G'
                ax.text(j + 0.5, i + 0.5, label,
                        ha='center', va='center', fontsize=16, fontweight='bold',
                        color='white' if cell == 'H' else 'black')

    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect('equal')
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_title(f"{strategy_name} - {case_name}", fontsize=13, fontweight='bold')
    ax.grid(False)

    if save_path and show:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.close()


def compare_policies(policies, case_name, save_path=None):
    n = len(policies)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for idx, (strategy_name, Q_vals) in enumerate(policies.items()):
        visualize_policy(Q_vals, case_name, strategy_name, ax=axes[idx])

    plt.suptitle(f"Comparaison des politiques - {case_name}", fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    os.makedirs("results/figures", exist_ok=True)
    visualize_all_cases(save_path="results/figures/study_cases.png")
