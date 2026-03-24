import os
import sys
import json
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment import create_env, STUDY_CASES


class SafetyMetricsCallback(BaseCallback):
    """Callback pour suivre les metriques de securite pendant l'entrainement."""

    def __init__(self, eval_env, case_name, eval_freq=1000, n_eval_episodes=20,
                 verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.case_name = case_name
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        # Metriques d'evaluation
        self.eval_timesteps = []
        self.eval_mean_rewards = []
        self.eval_std_rewards = []
        self.eval_success_rates = []
        self.eval_success_std = []
        self.eval_hole_rates = []
        self.eval_hole_std = []
        self.eval_mean_lengths = []

        # Suivi des chutes pendant l'entrainement
        self.train_hole_count = 0
        self.train_episode_count = 0
        self.train_hole_rates = []
        self.train_timesteps_log = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if dones is not None:
            for i, done in enumerate(dones):
                if done:
                    self.train_episode_count += 1
                    if infos and i < len(infos):
                        reward = self.locals.get("rewards", [0])[i] if "rewards" in self.locals else 0
                        terminated = infos[i].get("TimeLimit.truncated", False) == False
                        if terminated and reward == 0:
                            self.train_hole_count += 1

        # Evaluation periodique
        if self.n_calls % self.eval_freq == 0:
            rewards, successes, hole_falls, lengths = self._evaluate()

            self.eval_timesteps.append(self.num_timesteps)
            self.eval_mean_rewards.append(np.mean(rewards))
            self.eval_std_rewards.append(np.std(rewards))
            self.eval_success_rates.append(np.mean(successes))
            self.eval_success_std.append(np.std(successes))
            self.eval_hole_rates.append(np.mean(hole_falls))
            self.eval_hole_std.append(np.std(hole_falls))
            self.eval_mean_lengths.append(np.mean(lengths))

            if self.train_episode_count > 0:
                rate = self.train_hole_count / self.train_episode_count
            else:
                rate = 0.0
            self.train_hole_rates.append(rate)
            self.train_timesteps_log.append(self.num_timesteps)

        return True

    def _evaluate(self):
        rewards = []
        successes = []
        hole_falls = []
        lengths = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
                total_reward += reward
                steps += 1
                done = terminated or truncated

            rewards.append(total_reward)
            successes.append(1.0 if total_reward > 0 else 0.0)
            hole_falls.append(1.0 if terminated and total_reward == 0 else 0.0)
            lengths.append(steps)

        return rewards, successes, hole_falls, lengths

    def get_results(self):
        return {
            "eval_timesteps": self.eval_timesteps,
            "eval_mean_rewards": self.eval_mean_rewards,
            "eval_std_rewards": self.eval_std_rewards,
            "eval_success_rates": self.eval_success_rates,
            "eval_success_std": self.eval_success_std,
            "eval_hole_rates": self.eval_hole_rates,
            "eval_hole_std": self.eval_hole_std,
            "eval_mean_lengths": self.eval_mean_lengths,
            "train_hole_rates": self.train_hole_rates,
            "train_timesteps_log": self.train_timesteps_log,
        }


# Hyper-parametres DQN par cas
DQN_PARAMS = {
    "case_1": {
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "learning_starts": 500,
        "batch_size": 64,
        "gamma": 0.99,
        "target_update_interval": 250,
        "exploration_fraction": 0.5,
        "exploration_final_eps": 0.05,
        "total_timesteps": 50000,
        "eval_freq": 1000,
    },
    "case_2": {
        "learning_rate": 5e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 128,
        "gamma": 0.99,
        "target_update_interval": 500,
        "exploration_fraction": 0.5,
        "exploration_final_eps": 0.05,
        "total_timesteps": 100000,
        "eval_freq": 2000,
    },
    "case_3": {
        "learning_rate": 5e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 128,
        "gamma": 0.99,
        "target_update_interval": 500,
        "exploration_fraction": 0.6,
        "exploration_final_eps": 0.05,
        "total_timesteps": 150000,
        "eval_freq": 3000,
    },
}

# Hyper-parametres PPO par cas
PPO_PARAMS = {
    "case_1": {
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "total_timesteps": 50000,
        "eval_freq": 1000,
    },
    "case_2": {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "total_timesteps": 100000,
        "eval_freq": 2000,
    },
    "case_3": {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "total_timesteps": 150000,
        "eval_freq": 3000,
    },
}


def train_dqn(case_name, seed=0, verbose=0):
    params = DQN_PARAMS[case_name]

    env = create_env(case_name, is_slippery=True)
    eval_env = create_env(case_name, is_slippery=True)

    model = DQN(
        "MlpPolicy", env,
        learning_rate=params["learning_rate"],
        buffer_size=params["buffer_size"],
        learning_starts=params["learning_starts"],
        batch_size=params["batch_size"],
        gamma=params["gamma"],
        target_update_interval=params["target_update_interval"],
        exploration_fraction=params["exploration_fraction"],
        exploration_final_eps=params["exploration_final_eps"],
        seed=seed, verbose=verbose,
    )

    callback = SafetyMetricsCallback(
        eval_env=eval_env, case_name=case_name,
        eval_freq=params["eval_freq"], n_eval_episodes=20,
    )

    model.learn(total_timesteps=params["total_timesteps"], callback=callback)

    results = callback.get_results()
    env.close()
    eval_env.close()
    return model, results


def train_ppo(case_name, seed=0, verbose=0):
    params = PPO_PARAMS[case_name]

    env = create_env(case_name, is_slippery=True)
    eval_env = create_env(case_name, is_slippery=True)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        n_epochs=params["n_epochs"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        ent_coef=params["ent_coef"],
        seed=seed, verbose=verbose,
    )

    callback = SafetyMetricsCallback(
        eval_env=eval_env, case_name=case_name,
        eval_freq=params["eval_freq"], n_eval_episodes=20,
    )

    model.learn(total_timesteps=params["total_timesteps"], callback=callback)

    results = callback.get_results()
    env.close()
    eval_env.close()
    return model, results


def get_policy_from_model(model, case_name):
    case = STUDY_CASES[case_name]
    n_states = case["size"] ** 2
    policy = np.zeros(n_states, dtype=int)

    for state in range(n_states):
        obs = np.array([state])
        action, _ = model.predict(obs, deterministic=True)
        policy[state] = action

    return policy


if __name__ == "__main__":
    print("Test DQN sur case_1...")
    model_dqn, results_dqn = train_dqn("case_1", seed=42)
    print(f"  Succes: {results_dqn['eval_success_rates'][-1]:.2%}")
    print(f"  Chutes: {results_dqn['eval_hole_rates'][-1]:.2%}")

    print("Test PPO sur case_1...")
    model_ppo, results_ppo = train_ppo("case_1", seed=42)
    print(f"  Succes: {results_ppo['eval_success_rates'][-1]:.2%}")
    print(f"  Chutes: {results_ppo['eval_hole_rates'][-1]:.2%}")
