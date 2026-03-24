import gymnasium as gym
import numpy as np
from typing import Optional

# Grilles personnalisees pour les 3 cas d'etude

# Cas 1 : Simple 4x4 (1 trou, 6.25% de danger)
CASE_1_MAP = [
    "SFFF",
    "FHFF",
    "FFFF",
    "FFFG"
]

# Cas 2 : Modere 8x8 (10 trous, 15.6% de danger)
CASE_2_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG"
]

# Cas 3 : Difficile 8x8 (17 trous, 26.6% de danger)
CASE_3_MAP = [
    "SFFHFFFF",
    "FHFHFFHF",
    "FFFHHFHF",
    "HFFHFFFF",
    "FHFFFHHF",
    "FFHFFFHF",
    "FHFHFHFF",
    "FFFFFFFG"
]

STUDY_CASES = {
    "case_1": {
        "desc": "Simple 4x4 - 1 hole, low danger",
        "map": CASE_1_MAP,
        "size": 4
    },
    "case_2": {
        "desc": "Medium 8x8 - 10 holes, moderate danger",
        "map": CASE_2_MAP,
        "size": 8
    },
    "case_3": {
        "desc": "Hard 8x8 - 17 holes, high danger",
        "map": CASE_3_MAP,
        "size": 8
    }
}


def count_holes(desc):
    return sum(row.count('H') for row in desc)


def create_env(case_name, is_slippery=True, render_mode=None):
    case = STUDY_CASES[case_name]
    env = gym.make(
        "FrozenLake-v1",
        desc=case["map"],
        is_slippery=is_slippery,
        render_mode=render_mode
    )
    return env


def print_env_info(case_name):
    case = STUDY_CASES[case_name]
    n_holes = count_holes(case["map"])
    total_cells = case["size"] ** 2
    danger_ratio = n_holes / total_cells

    print(f"Cas: {case_name} | {case['desc']}")
    print(f"  Grille: {case['size']}x{case['size']}, {n_holes} trous, danger: {danger_ratio:.2%}")
    for row in case["map"]:
        print(f"  {row}")

    env = create_env(case_name, is_slippery=True)
    print(f"  Etats: {env.observation_space}, Actions: {env.action_space}")

    # Exemple de transitions depuis l'etat 0, action droite
    transitions = env.unwrapped.P[0][2]
    print(f"  Transitions (etat 0, droite):")
    for prob, next_state, reward, terminated in transitions:
        print(f"    P={prob:.4f} -> etat {next_state}, r={reward}, fin={terminated}")
    env.close()


if __name__ == "__main__":
    for case_name in STUDY_CASES:
        print_env_info(case_name)
        print()

    # Comparaison deterministe vs stochastique
    for slippery in [False, True]:
        mode = "Stochastique" if slippery else "Deterministe"
        env = create_env("case_1", is_slippery=slippery)
        print(f"{mode} (is_slippery={slippery}):")
        transitions = env.unwrapped.P[0][2]
        for prob, next_state, reward, terminated in transitions:
            print(f"  P={prob:.4f} -> etat {next_state}")
        env.close()
