import os
import json
import numpy as np

from environment import STUDY_CASES
from train_baselines import (
    train_dqn, train_ppo, get_policy_from_model,
    DQN_PARAMS, PPO_PARAMS
)
from train_safe_rl import (
    train_safe_ql, train_standard_ql,
    SAFE_QL_PARAMS, STD_QL_PARAMS
)
from visualize import (
    visualize_all_cases, plot_training_curves,
    compare_policies
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
    for key in ["eval_mean_rewards", "eval_success_rates", "eval_hole_rates"]:
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


def run_sb3_experiments():
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

    return all_results, all_policies


def run_ql_experiments():
    all_results = {}
    all_policies = {}

    for strategy_name, train_fn in [("Q-Learning", train_standard_ql),
                                      ("Safe Q-Learning", train_safe_ql)]:
        all_results[strategy_name] = {}
        all_policies[strategy_name] = {}

        for case_name in STUDY_CASES:
            print(f"[{strategy_name}] {case_name} ({N_SEEDS} seeds)")

            runs = []
            last_agent = None

            for i, seed in enumerate(SEEDS):
                print(f"  seed {seed} ({i+1}/{N_SEEDS})...", end=" ", flush=True)
                agent, results = train_fn(case_name, seed=seed)
                runs.append(results)
                last_agent = agent
                sr = results["eval_success_rates"][-1] if results["eval_success_rates"] else 0
                hr = results["eval_hole_rates"][-1] if results["eval_hole_rates"] else 0
                thr = results["train_hole_rates"][-1] if results["train_hole_rates"] else 0
                print(f"succes={sr:.2%}, chutes={hr:.2%}, chutes_train={thr:.2%}")

            agg = aggregate_results(runs)
            all_results[strategy_name][case_name] = agg
            policy = last_agent.get_policy()
            all_policies[strategy_name][case_name] = policy.tolist()

    return all_results, all_policies


def generate_all_figures(all_results, all_policies):
    ensure_dirs()

    visualize_all_cases(save_path=os.path.join(FIGURES_DIR, "study_cases.png"))

    strategies = list(all_results.keys())

    for metric_key, metric_label, title_base, filename in [
        ("eval_success_rates", "success_rate", "Taux de succès durant l'entraînement", "success_rate"),
        ("eval_hole_rates", "hole_rate", "Taux de chutes (évaluation) durant l'entraînement", "hole_rate"),
        ("eval_mean_rewards", "reward", "Récompense moyenne durant l'entraînement", "mean_reward"),
        ("train_hole_rates", "train_hole_rate", "Taux de chutes cumulatif durant l'entraînement", "train_hole_rate"),
    ]:
        formatted = {}
        for strategy in strategies:
            formatted[strategy] = {}
            for case in STUDY_CASES:
                data = all_results[strategy][case]
                ts_key = "train_timesteps" if metric_key == "train_hole_rates" else "eval_timesteps"
                formatted[strategy][case] = {
                    "episodes": data[ts_key],
                    metric_label: data[metric_key + "_mean"],
                    metric_label + "_std": data[metric_key + "_std"],
                }

        ylabel_map = {
            "success_rate": "Taux de succès",
            "hole_rate": "Taux de chutes",
            "reward": "Récompense moyenne",
            "train_hole_rate": "Taux de chutes cumulatif",
        }
        plot_training_curves(
            formatted, metric_label,
            title_base,
            ylabel_map.get(metric_label, metric_label),
            save_path=os.path.join(FIGURES_DIR, f"all_{filename}.png")
        )

    # Comparaison des politiques
    for case_name in STUDY_CASES:
        policies = {}
        for strategy in all_policies:
            policies[strategy] = np.array(all_policies[strategy][case_name])
        compare_policies(policies, case_name,
                         save_path=os.path.join(FIGURES_DIR, f"all_policy_{case_name}.png"))


def generate_all_tables(all_results):
    ensure_dirs()

    # Table LaTeX
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Performance comparison of all strategies (mean $\pm$ std over 5 seeds).}",
        r"\label{tab:all_results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Strategy & Case & Success Rate & Hole Rate (Eval) & Train Hole Rate & Mean Reward \\",
        r"\midrule",
    ]

    for strategy in all_results:
        for case in all_results[strategy]:
            data = all_results[strategy][case]
            sr_mean = data["eval_success_rates_mean"][-1] * 100
            sr_std = data["eval_success_rates_std"][-1] * 100
            hr_mean = data["eval_hole_rates_mean"][-1] * 100
            hr_std = data["eval_hole_rates_std"][-1] * 100
            thr_mean = data["train_hole_rates_mean"][-1] * 100
            thr_std = data["train_hole_rates_std"][-1] * 100
            rw_mean = data["eval_mean_rewards_mean"][-1]
            rw_std = data["eval_mean_rewards_std"][-1]

            case_label = case.replace("_", " ").title()
            latex_lines.append(
                f"{strategy} & {case_label} & "
                f"${sr_mean:.1f} \\pm {sr_std:.1f}\\%$ & "
                f"${hr_mean:.1f} \\pm {hr_std:.1f}\\%$ & "
                f"${thr_mean:.1f} \\pm {thr_std:.1f}\\%$ & "
                f"${rw_mean:.3f} \\pm {rw_std:.3f}$ \\\\"
            )
        latex_lines.append(r"\midrule")

    latex_lines[-1] = r"\bottomrule"
    latex_lines.extend([
        r"\end{tabular}}",
        r"\end{table}",
    ])

    with open(os.path.join(TABLES_DIR, "all_results.tex"), "w") as f:
        f.write("\n".join(latex_lines))

    print(f"\n{'Strategie':<20} {'Cas':<10} {'Succes':>10} {'Chutes(E)':>10} "
          f"{'Chutes(T)':>10} {'Recomp.':>10}")
    for strategy in all_results:
        for case in all_results[strategy]:
            data = all_results[strategy][case]
            sr = data["eval_success_rates_mean"][-1]
            hr = data["eval_hole_rates_mean"][-1]
            thr = data["train_hole_rates_mean"][-1]
            rw = data["eval_mean_rewards_mean"][-1]
            print(f"{strategy:<20} {case:<10} {sr:>9.2%} {hr:>9.2%} "
                  f"{thr:>9.2%} {rw:>9.3f}")


if __name__ == "__main__":
    # Phase 1 : DQN + PPO
    print("Phase 1 : DQN + PPO")
    sb3_results, sb3_policies = run_sb3_experiments()

    # Phase 2 : Q-Learning + Safe Q-Learning
    print("Phase 2 : Q-Learning + Safe Q-Learning")
    ql_results, ql_policies = run_ql_experiments()

    # Fusion des resultats
    all_results = {**sb3_results, **ql_results}
    all_policies = {**sb3_policies, **ql_policies}

    # Sauvegarde
    ensure_dirs()
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "all_policies.json"), "w") as f:
        json.dump(all_policies, f, indent=2)

    # Generation des sorties
    generate_all_figures(all_results, all_policies)
    generate_all_tables(all_results)
