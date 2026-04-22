import os
import json
import numpy as np

from environment import create_env, STUDY_CASES


class SafeQLearning:
    """Q-Learning securitaire avec double table Q (recompense + cout)."""

    def __init__(self, n_states, n_actions=4, lr=0.1, lr_cost=0.2,
                 gamma=0.99, gamma_cost=0.5, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=0.5, cost_threshold=0.3,
                 env=None, case_name=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.lr_cost = lr_cost
        self.gamma = gamma
        self.gamma_cost = gamma_cost
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.cost_threshold = cost_threshold

        # Q_r : recompense esperee, Q_c : probabilite de danger
        self.Q_r = np.zeros((n_states, n_actions))
        self.Q_c = np.zeros((n_states, n_actions))

        self.epsilon = epsilon_start

        # Initialisation de Q_c avec le modele de transition connu
        if env is not None and case_name is not None:
            self._init_immediate_cost(env, case_name)

    def _init_immediate_cost(self, env, case_name):
        """Initialise Q_c avec la probabilite immediate de tomber dans un trou."""
        case = STUDY_CASES[case_name]
        grid = case["map"]
        size = case["size"]

        hole_states = set()
        for i in range(size):
            for j in range(size):
                if grid[i][j] == 'H':
                    hole_states.add(i * size + j)

        P = env.unwrapped.P
        for state in range(self.n_states):
            for action in range(self.n_actions):
                if state in hole_states:
                    self.Q_c[state, action] = 1.0
                    continue
                cost = sum(prob for prob, ns, r, d in P[state][action]
                           if ns in hole_states)
                self.Q_c[state, action] = cost

    def get_safe_actions(self, state):
        """Retourne les actions dont le cout est sous le seuil."""
        safe = [a for a in range(self.n_actions)
                if self.Q_c[state, a] < self.cost_threshold]
        if not safe:
            # Fallback : choisir les actions les moins couteuses
            min_cost = np.min(self.Q_c[state])
            safe = [a for a in range(self.n_actions)
                    if self.Q_c[state, a] <= min_cost + 0.1]
        return safe

    def select_action(self, state):
        safe_actions = self.get_safe_actions(state)
        if np.random.random() < self.epsilon:
            return np.random.choice(safe_actions)
        else:
            q_vals = self.Q_r[state, safe_actions]
            best_idx = np.argmax(q_vals)
            return safe_actions[best_idx]

    def update(self, state, action, reward, next_state, done, cost):
        # Mise a jour Q_r (recompense)
        if done:
            target_r = reward
        else:
            target_r = reward + self.gamma * np.max(self.Q_r[next_state])
        self.Q_r[state, action] += self.lr * (target_r - self.Q_r[state, action])

        # Mise a jour Q_c (cout, gamma reduit pour danger immediat)
        if done:
            target_c = cost
        else:
            target_c = cost + self.gamma_cost * np.max(self.Q_c[next_state])
        self.Q_c[state, action] += self.lr_cost * (target_c - self.Q_c[state, action])

    def decay_epsilon(self, step, total_steps):
        decay_steps = int(total_steps * self.epsilon_decay)
        if step < decay_steps:
            self.epsilon = self.epsilon_start - \
                (self.epsilon_start - self.epsilon_end) * (step / decay_steps)
        else:
            self.epsilon = self.epsilon_end

    def get_policy(self):
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            safe_actions = self.get_safe_actions(s)
            q_vals = self.Q_r[s, safe_actions]
            policy[s] = safe_actions[np.argmax(q_vals)]
        return policy

    def get_unconstrained_policy(self):
        return np.argmax(self.Q_r, axis=1)


class StandardQLearning:
    """Q-Learning standard sans mecanisme de securite (baseline)."""

    def __init__(self, n_states, n_actions=4, lr=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done, cost=None):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def decay_epsilon(self, step, total_steps):
        decay_steps = int(total_steps * self.epsilon_decay)
        if step < decay_steps:
            self.epsilon = self.epsilon_start - \
                (self.epsilon_start - self.epsilon_end) * (step / decay_steps)
        else:
            self.epsilon = self.epsilon_end

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


# Hyper-parametres Safe Q-Learning par cas
SAFE_QL_PARAMS = {
    "case_1": {
        "lr": 0.1, "lr_cost": 0.2, "gamma": 0.99,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.5,
        "cost_threshold": 0.3,
        "total_episodes": 20000, "eval_freq": 500,
        "n_eval_episodes": 50, "max_steps": 100,
    },
    "case_2": {
        "lr": 0.1, "lr_cost": 0.15, "gamma": 0.99,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.5,
        "cost_threshold": 0.35,
        "total_episodes": 50000, "eval_freq": 1000,
        "n_eval_episodes": 50, "max_steps": 200,
    },
    "case_3": {
        "lr": 0.1, "lr_cost": 0.15, "gamma": 0.99,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.6,
        "cost_threshold": 0.4,
        "total_episodes": 80000, "eval_freq": 2000,
        "n_eval_episodes": 50, "max_steps": 200,
    },
}

# Hyper-parametres Q-Learning standard par cas
STD_QL_PARAMS = {
    "case_1": {
        "lr": 0.1, "gamma": 0.99, "epsilon_start": 1.0,
        "epsilon_end": 0.05, "epsilon_decay": 0.5,
        "total_episodes": 20000, "eval_freq": 500,
        "n_eval_episodes": 50, "max_steps": 100,
    },
    "case_2": {
        "lr": 0.1, "gamma": 0.99, "epsilon_start": 1.0,
        "epsilon_end": 0.05, "epsilon_decay": 0.5,
        "total_episodes": 50000, "eval_freq": 1000,
        "n_eval_episodes": 50, "max_steps": 200,
    },
    "case_3": {
        "lr": 0.1, "gamma": 0.99, "epsilon_start": 1.0,
        "epsilon_end": 0.05, "epsilon_decay": 0.6,
        "total_episodes": 80000, "eval_freq": 2000,
        "n_eval_episodes": 50, "max_steps": 200,
    },
}


def evaluate_agent(agent, env, n_episodes=50, max_steps=200):
    rewards = []
    successes = []
    hole_falls = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = int(obs)
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            if isinstance(agent, SafeQLearning):
                safe_actions = agent.get_safe_actions(state)
                q_vals = agent.Q_r[state, safe_actions]
                action = safe_actions[np.argmax(q_vals)]
            else:
                action = np.argmax(agent.Q[state])

            obs, reward, terminated, truncated, info = env.step(action)
            state = int(obs)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        successes.append(1.0 if total_reward > 0 else 0.0)
        hole_falls.append(1.0 if terminated and total_reward == 0 else 0.0)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean(successes),
        "std_success": np.std(successes),
        "hole_rate": np.mean(hole_falls),
        "std_hole": np.std(hole_falls),
    }


def train_safe_ql(case_name, seed=0):
    np.random.seed(seed)
    params = SAFE_QL_PARAMS[case_name]
    case = STUDY_CASES[case_name]
    n_states = case["size"] ** 2

    env = create_env(case_name, is_slippery=True)
    eval_env = create_env(case_name, is_slippery=True)

    agent = SafeQLearning(
        n_states=n_states,
        lr=params["lr"], lr_cost=params["lr_cost"],
        gamma=params["gamma"],
        gamma_cost=params.get("gamma_cost", 0.5),
        epsilon_start=params["epsilon_start"],
        epsilon_end=params["epsilon_end"],
        epsilon_decay=params["epsilon_decay"],
        cost_threshold=params["cost_threshold"],
        env=env, case_name=case_name,
    )

    eval_timesteps = []
    eval_rewards = []
    eval_success = []
    eval_holes = []
    train_hole_count = 0
    train_episode_count = 0
    train_hole_rates = []
    train_timesteps = []
    total_steps = 0

    for episode in range(params["total_episodes"]):
        obs, _ = env.reset()
        state = int(obs)
        done = False
        steps = 0

        while not done and steps < params["max_steps"]:
            action = agent.select_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = int(obs)
            done = terminated or truncated
            steps += 1
            total_steps += 1

            cost = 1.0 if (terminated and reward == 0.0) else 0.0
            agent.update(state, action, reward, next_state, terminated, cost)
            agent.decay_epsilon(total_steps,
                                params["total_episodes"] * params["max_steps"] * 0.3)
            state = next_state

        train_episode_count += 1
        if terminated and reward == 0.0:
            train_hole_count += 1

        # Evaluation periodique
        if (episode + 1) % params["eval_freq"] == 0:
            eval_result = evaluate_agent(agent, eval_env,
                                         params["n_eval_episodes"],
                                         params["max_steps"])
            eval_timesteps.append(total_steps)
            eval_rewards.append(eval_result["mean_reward"])
            eval_success.append(eval_result["success_rate"])
            eval_holes.append(eval_result["hole_rate"])

            train_rate = train_hole_count / train_episode_count if train_episode_count > 0 else 0
            train_hole_rates.append(train_rate)
            train_timesteps.append(total_steps)

    env.close()
    eval_env.close()

    return agent, {
        "eval_timesteps": eval_timesteps,
        "eval_mean_rewards": eval_rewards,
        "eval_success_rates": eval_success,
        "eval_hole_rates": eval_holes,
        "train_hole_rates": train_hole_rates,
        "train_timesteps_log": train_timesteps,
    }


def train_standard_ql(case_name, seed=0):
    np.random.seed(seed)
    params = STD_QL_PARAMS[case_name]
    case = STUDY_CASES[case_name]
    n_states = case["size"] ** 2

    env = create_env(case_name, is_slippery=True)
    eval_env = create_env(case_name, is_slippery=True)

    agent = StandardQLearning(
        n_states=n_states, lr=params["lr"], gamma=params["gamma"],
        epsilon_start=params["epsilon_start"],
        epsilon_end=params["epsilon_end"],
        epsilon_decay=params["epsilon_decay"],
    )

    eval_timesteps = []
    eval_rewards = []
    eval_success = []
    eval_holes = []
    train_hole_count = 0
    train_episode_count = 0
    train_hole_rates = []
    train_timesteps = []
    total_steps = 0

    for episode in range(params["total_episodes"]):
        obs, _ = env.reset()
        state = int(obs)
        done = False
        steps = 0

        while not done and steps < params["max_steps"]:
            action = agent.select_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = int(obs)
            done = terminated or truncated
            steps += 1
            total_steps += 1

            agent.update(state, action, reward, next_state, terminated)
            agent.decay_epsilon(total_steps,
                                params["total_episodes"] * params["max_steps"] * 0.3)
            state = next_state

        train_episode_count += 1
        if terminated and reward == 0.0:
            train_hole_count += 1

        if (episode + 1) % params["eval_freq"] == 0:
            eval_result = evaluate_agent(agent, eval_env,
                                         params["n_eval_episodes"],
                                         params["max_steps"])
            eval_timesteps.append(total_steps)
            eval_rewards.append(eval_result["mean_reward"])
            eval_success.append(eval_result["success_rate"])
            eval_holes.append(eval_result["hole_rate"])

            train_rate = train_hole_count / train_episode_count if train_episode_count > 0 else 0
            train_hole_rates.append(train_rate)
            train_timesteps.append(total_steps)

    env.close()
    eval_env.close()

    return agent, {
        "eval_timesteps": eval_timesteps,
        "eval_mean_rewards": eval_rewards,
        "eval_success_rates": eval_success,
        "eval_hole_rates": eval_holes,
        "train_hole_rates": train_hole_rates,
        "train_timesteps_log": train_timesteps,
    }


if __name__ == "__main__":
    for case in ["case_1", "case_2", "case_3"]:

        agent_std, res_std = train_standard_ql(case, seed=42)
        print(f"Q-Learning   | Succes: {res_std['eval_success_rates'][-1]:.2%}, "
              f"Chutes eval: {res_std['eval_hole_rates'][-1]:.2%}, "
              f"Chutes train: {res_std['train_hole_rates'][-1]:.2%}")

        agent_safe, res_safe = train_safe_ql(case, seed=42)
        print(f"Safe Q-Learn | Succes: {res_safe['eval_success_rates'][-1]:.2%}, "
              f"Chutes eval: {res_safe['eval_hole_rates'][-1]:.2%}, "
              f"Chutes train: {res_safe['train_hole_rates'][-1]:.2%}")
