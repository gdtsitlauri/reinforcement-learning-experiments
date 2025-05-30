import numpy as np

class Gridworld:
    def __init__(self):
        self.states = list(range(1, 15)) 
        self.actions = ["up", "down", "left", "right"] 
        self.grid_size = (4, 4)  
        self.reward = -1  
        self.terminal_states = [0, 15]  

    def step(self, state, action):
        row, col = divmod(state - 1, 4)

        if action == "up":
            new_row, new_col = max(row - 1, 0), col
        elif action == "down":
            new_row, new_col = min(row + 1, 3), col
        elif action == "left":
            new_row, new_col = row, max(col - 1, 0)
        elif action == "right":
            new_row, new_col = row, min(col + 1, 3)
        else:
            raise ValueError("Invalid action")

        new_state = new_row * 4 + new_col + 1
        if new_state in self.terminal_states:
            return state, self.reward  
        return new_state, self.reward

def policy_evaluation_two_arrays(env, policy, theta=1e-4, discount=1.0):
    V = np.zeros(16)
    while True:
        delta = 0
        new_V = V.copy()
        for state in env.states:
            value = 0
            for action in env.actions:
                next_state, reward = env.step(state, action)
                value += 1 / len(env.actions) * (reward + discount * V[next_state - 1])
            new_V[state - 1] = value
            delta = max(delta, abs(V[state - 1] - value))
        V = new_V
        if delta < theta:
            break
    return V

def policy_evaluation_one_array(env, policy, theta=1e-4, discount=1.0):
    V = np.zeros(16)
    while True:
        delta = 0
        for state in env.states:
            value = 0
            for action in env.actions:
                next_state, reward = env.step(state, action)
                value += 1 / len(env.actions) * (reward + discount * V[next_state - 1])
            delta = max(delta, abs(V[state - 1] - value))
            V[state - 1] = value
        if delta < theta:
            break
    return V

def find_optimal_policy(env, V, discount=1.0):
    policy = {}
    for state in env.states:
        action_values = {}
        for action in env.actions:
            next_state, reward = env.step(state, action)
            action_values[action] = reward + discount * V[next_state - 1]
        best_action_value = max(action_values.values())
        best_actions = [action for action, value in action_values.items() if value == best_action_value]
        policy[state] = best_actions
    return policy

env = Gridworld()
policy = {state: env.actions for state in env.states}  

V_two_arrays = policy_evaluation_two_arrays(env, policy)
print("Value Function (Two Arrays):")
print(V_two_arrays.reshape(env.grid_size))

V_one_array = policy_evaluation_one_array(env, policy)
print("Value Function (One Array):")
print(V_one_array.reshape(env.grid_size))

optimal_policy = find_optimal_policy(env, V_two_arrays)
print("Optimal Policy:")
for state, actions in optimal_policy.items():
    print(f"State {state}: {actions}")
