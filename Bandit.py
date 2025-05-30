import numpy as np

class MultiArmedBandit:
    def __init__(self, num_machines, num_levers, std_dev, num_actions):
        self.num_machines = num_machines
        self.num_levers = num_levers
        self.std_dev = std_dev
        self.num_actions = num_actions
        self.machines = [
            [np.random.normal(np.random.uniform(0, 1), std_dev) for _ in range(num_levers)]
            for _ in range(num_machines)
        ]

    def pull_lever(self, machine_idx, lever_idx):
        return np.random.normal(self.machines[machine_idx][lever_idx], self.std_dev)


def epsilon_greedy(bandit, epsilon, num_actions):
    num_machines = bandit.num_machines
    num_levers = bandit.num_levers
    rewards = np.zeros((num_machines, num_levers))
    counts = np.zeros((num_machines, num_levers))
    total_rewards = []

    for _ in range(num_actions):
        if np.random.rand() < epsilon:
            machine_idx = np.random.randint(0, num_machines)
            lever_idx = np.random.randint(0, num_levers)
        else:
            machine_idx, lever_idx = np.unravel_index(
                np.argmax(rewards / (counts + 1e-5)), (num_machines, num_levers)
            )

        reward = bandit.pull_lever(machine_idx, lever_idx)
        counts[machine_idx][lever_idx] += 1
        rewards[machine_idx][lever_idx] += reward
        total_rewards.append(reward)

    return rewards, counts, total_rewards


def softmax(bandit, tau, num_actions):
    num_machines = bandit.num_machines
    num_levers = bandit.num_levers
    rewards = np.zeros((num_machines, num_levers))
    counts = np.zeros((num_machines, num_levers))
    total_rewards = []

    for _ in range(num_actions):
        probabilities = np.exp(rewards / (counts + 1e-5) / tau)
        probabilities /= probabilities.sum()
        flattened_prob = probabilities.flatten()
        choice = np.random.choice(len(flattened_prob), p=flattened_prob)
        machine_idx, lever_idx = divmod(choice, num_levers)

        reward = bandit.pull_lever(machine_idx, lever_idx)
        counts[machine_idx][lever_idx] += 1
        rewards[machine_idx][lever_idx] += reward
        total_rewards.append(reward)

    return rewards, counts, total_rewards


num_machines = 3
num_levers = 4
std_dev = 1.0
num_actions = 1000
epsilon = 0.1
tau = 0.1

bandit = MultiArmedBandit(num_machines, num_levers, std_dev, num_actions)

rewards_egreedy, counts_egreedy, total_rewards_egreedy = epsilon_greedy(bandit, epsilon, num_actions)

rewards_softmax, counts_softmax, total_rewards_softmax = softmax(bandit, tau, num_actions)

print("Epsilon-Greedy Rewards:")
print(rewards_egreedy)
print("Counts:")
print(counts_egreedy)

print("Softmax Rewards:")
print(rewards_softmax)
print("Counts:")
print(counts_softmax)

#Οι τιμές αναπαριστούν το άθροισμα ανταμοιβών για κάθε μοχλό (lever) σε κάθε μηχανή (machine) 