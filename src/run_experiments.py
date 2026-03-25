import os
import json
import numpy as np

from environment import STUDY_CASES
from train_baselines import (
    train_dqn, train_ppo, get_policy_from_model,
    DQN_PARAMS, PPO_PARAMS
)
from visualize import (
    visualize_all_cases, plot_training_curves,
    compare_policies, visualize_policy
)

N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 1024]
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
TABLES_DIR = "results/tables"


def ensure_dirs():
    for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)


def aggregate_results(all_runs):
    min_len = min(len(r["eval_timesteps"]) for r in all_runs)
    timesteps = all_runs[0]["eval_timesteps"][:min_len]

    metrics = {}
    for key in ["eval_mean_rewards", "eval_success_rates", "eval_hole_rates",
                "eval_mean_lengths"]:
        values = np.array([r[key][:min_len] for r in all_runs])
        metrics[key + "_mean"] = values.mean(axis=0).tolist()
        metrics[key + "_std"] = values.std(axis=0).tolist()

    min_train_len = min(len(r["train_hole_rates"]) for r in all_runs)
    train_vals = np.array([r["train_hole_rates"][:min_train_len] for r in all_runs])
    metrics["train_hole_rates_mean"] = train_vals.mean(axis=0).tolist()
    metrics["train_hole_rates_std"] = train_vals.std(axis=0).tolist()
    metrics["train_timesteps"] = all_runs[0]["train_timesteps_log"][:min_train_len]
    metrics["eval_timesteps"] = [int(t) for t in timesteps]
    return metrics


def run_all_experiments():
    ensure_dirs()

    strategies = {"DQN": train_dqn, "PPO": train_ppo}
    all_results = {}
    all_policies = {}

    for strategy_name, train_fn in strategies.items():
        all_results[strategy_name] = {}
        all_policies[strategy_name] = {}

        for case_name in STUDY_CASES:
            print(f"[{strategy_name}] {case_name} ({N_SEEDS} seeds)")

            runs = []
            last_model = None

            for i, seed in enumerate(SEEDS):
                print(f"  seed {seed} ({i+1}/{N_SEEDS})...", end=" ", flush=True)
                model, results = train_fn(case_name, seed=seed, verbose=0)
                runs.append(results)
                last_model = model

                sr = results["eval_success_rates"][-1] if results["eval_success_rates"] else 0
                hr = results["eval_hole_rates"][-1] if results["eval_hole_rates"] else 0
                print(f"succes={sr:.2%}, chutes={hr:.2%}")

            agg = aggregate_results(runs)
            all_results[strategy_name][case_name] = agg

            policy = get_policy_from_model(last_model, case_name)
            all_policies[strategy_name][case_name] = policy.tolist()

    # Sauvegarde des resultats
    with open(os.path.join(RESULTS_DIR, "baseline_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "baseline_policies.json"), "w") as f:
        json.dump(all_policies, f, indent=2)

    return all_results, all_policies


def generate_figures(all_results, all_policies):
    ensure_dirs()

    visualize_all_cases(save_path=os.path.join(FIGURES_DIR, "study_cases.png"))

    # Taux de succes
    formatted = {}
    for strategy in all_results:
        formatted[strategy] = {}
        for case in all_results[strategy]:
            data = all_results[strategy][case]
            formatted[strategy][case] = {
                "episodes": data["eval_timesteps"],
                "success_rate": data["eval_success_rates_mean"],
                "success_rate_std": data["eval_success_rates_std"],
            }
    plot_training_curves(formatted, "success_rate", "Success Rate During Training",
                         "Success Rate",
                         save_path=os.path.join(FIGURES_DIR, "success_rate.png"))

    # Taux de chutes (evaluation)
    formatted_holes = {}
    for strategy in all_results:
        formatted_holes[strategy] = {}
        for case in all_results[strategy]:
            data = all_results[strategy][case]
            formatted_holes[strategy][case] = {
                "episodes": data["eval_timesteps"],
                "hole_rate": data["eval_hole_rates_mean"],
                "hole_rate_std": data["eval_hole_rates_std"],
            }
    plot_training_curves(formatted_holes, "hole_rate",
                         "Hole Fall Rate During Training (Evaluation)",
                         "Hole Fall Rate",
                         save_path=os.path.join(FIGURES_DIR, "hole_rate.png"))

    # Recompense moyenne
    formatted_rewards = {}
    for strategy in all_results:
        formatted_rewards[strategy] = {}
        for case in all_results[strategy]:
            data = all_results[strategy][case]
            formatted_rewards[strategy][case] = {
                "episodes": data["eval_timesteps"],
                "reward": data["eval_mean_rewards_mean"],
                "reward_std": data["eval_mean_rewards_std"],
            }
    plot_training_curves(formatted_rewards, "reward", "Mean Reward During Training",
                         "Mean Reward",
                         save_path=os.path.join(FIGURES_DIR, "mean_reward.png"))

    # Taux de chutes cumulatif (entrainement)
    formatted_train_holes = {}
    for strategy in all_results:
        formatted_train_holes[strategy] = {}
        for case in all_results[strategy]:
            data = all_results[strategy][case]
            formatted_train_holes[strategy][case] = {
                "episodes": data["train_timesteps"],
                "train_hole_rate": data["train_hole_rates_mean"],
                "train_hole_rate_std": data["train_hole_rates_std"],
            }
    plot_training_curves(formatted_train_holes, "train_hole_rate",
                         "Cumulative Hole Fall Rate During Training",
                         "Cumulative Hole Fall Rate",
                         save_path=os.path.join(FIGURES_DIR, "train_hole_rate.png"))

    # Politiques
    for case_name in STUDY_CASES:
        policies = {}
        for strategy in all_policies:
            policies[strategy] = np.array(all_policies[strategy][case_name])
        compare_policies(policies, case_name,
                         save_path=os.path.join(FIGURES_DIR, f"policy_{case_name}.png"))


def generate_tables(all_results):
    ensure_dirs()

    # Table texte
    header = f"{'Strategie':<10} {'Cas':<10} {'Succes':>15} {'Chutes':>15} {'Recompense':>15}"
    lines = [header, "-" * len(header)]

    for strategy in all_results:
        for case in all_results[strategy]:
            data = all_results[strategy][case]
            sr_mean = data["eval_success_rates_mean"][-1]
            sr_std = data["eval_success_rates_std"][-1]
            hr_mean = data["eval_hole_rates_mean"][-1]
            hr_std = data["eval_hole_rates_std"][-1]
            rw_mean = data["eval_mean_rewards_mean"][-1]
            rw_std = data["eval_mean_rewards_std"][-1]

            lines.append(
                f"{strategy:<10} {case:<10} "
                f"{sr_mean:>6.2%} +/- {sr_std:.2%} "
                f"{hr_mean:>6.2%} +/- {hr_std:.2%} "
                f"{rw_mean:>6.3f} +/- {rw_std:.3f}"
            )

    table = "\n".join(lines)
    print(f"\n{table}")

    with open(os.path.join(TABLES_DIR, "summary_baselines.txt"), "w") as f:
        f.write(table)

    # Table LaTeX
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Performance comparison of baseline strategies (mean $\pm$ std over 5 seeds).}",
        r"\label{tab:baseline_results}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Strategy & Case & Success Rate & Hole Rate & Mean Reward \\",
        r"\midrule",
    ]

    for strategy in all_results:
        for case in all_results[strategy]:
            data = all_results[strategy][case]
            sr_mean = data["eval_success_rates_mean"][-1] * 100
            sr_std = data["eval_success_rates_std"][-1] * 100
            hr_mean = data["eval_hole_rates_mean"][-1] * 100
            hr_std = data["eval_hole_rates_std"][-1] * 100
            rw_mean = data["eval_mean_rewards_mean"][-1]
            rw_std = data["eval_mean_rewards_std"][-1]

            case_label = case.replace("_", " ").title()
            latex_lines.append(
                f"{strategy} & {case_label} & "
                f"${sr_mean:.1f} \\pm {sr_std:.1f}\\%$ & "
                f"${hr_mean:.1f} \\pm {hr_std:.1f}\\%$ & "
                f"${rw_mean:.3f} \\pm {rw_std:.3f}$ \\\\"
            )
        latex_lines.append(r"\midrule")

    latex_lines[-1] = r"\bottomrule"
    latex_lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(os.path.join(TABLES_DIR, "baseline_results.tex"), "w") as f:
        f.write("\n".join(latex_lines))


if __name__ == "__main__":
    all_results, all_policies = run_all_experiments()
    generate_figures(all_results, all_policies)
    generate_tables(all_results)

