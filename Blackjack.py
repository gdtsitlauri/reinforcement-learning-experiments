import numpy as np
import random

class Blackjack:
    def __init__(self):
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  
        self.player_sum = 0
        self.dealer_sum = 0
        self.player_usable_ace = False
        self.terminal = False

    def draw_card(self):
        return random.choice(self.deck)

    def start_game(self):
        self.player_sum = 0
        self.dealer_sum = 0
        self.player_usable_ace = False
        self.terminal = False

        player_cards = [self.draw_card() for _ in range(2)]
        dealer_cards = [self.draw_card() for _ in range(2)]

        self.dealer_sum = dealer_cards[0]  
        self.player_sum = sum(player_cards)

        if 1 in player_cards and self.player_sum <= 11:
            self.player_sum += 10
            self.player_usable_ace = True

        return self.get_state(), dealer_cards[1]  

    def get_state(self):
        return (self.player_sum, self.dealer_sum, self.player_usable_ace)

    def step(self, action):
        if action == "hit":
            card = self.draw_card()
            self.player_sum += card
            if self.player_usable_ace and self.player_sum > 21:
                self.player_sum -= 10
                self.player_usable_ace = False

            if self.player_sum > 21:
                self.terminal = True
                return self.get_state(), -1, True  

        elif action == "stick":
            while self.dealer_sum < 17:
                card = self.draw_card()
                self.dealer_sum += card
                if self.dealer_sum > 21:
                    return self.get_state(), 1, True 
            self.terminal = True
            if self.player_sum > self.dealer_sum:
                return self.get_state(), 1, True  
            elif self.player_sum < self.dealer_sum:
                return self.get_state(), -1, True  
            else:
                return self.get_state(), 0, True  

        return self.get_state(), 0, False  

def monte_carlo_es(episodes=100000):
    Q = {}
    returns = {}
    policy = {}

    for player_sum in range(12, 22):
        for dealer_card in range(1, 11):
            for usable_ace in [True, False]:
                Q[(player_sum, dealer_card, usable_ace)] = {"hit": 0, "stick": 0}
                returns[(player_sum, dealer_card, usable_ace)] = {"hit": [], "stick": []}
                policy[(player_sum, dealer_card, usable_ace)] = "stick" if player_sum >= 20 else "hit"

    env = Blackjack()

    for _ in range(episodes):
        state = (random.randint(12, 21), random.randint(1, 10), random.choice([True, False]))
        if state not in Q:
            continue

        env.player_sum, env.dealer_sum, env.player_usable_ace = state
        env.terminal = False

        episode = []
        while not env.terminal:
            action = random.choice(["hit", "stick"])
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = reward  
            if (state, action) not in visited and state in Q:
                returns[state][action].append(G)
                Q[state][action] = np.mean(returns[state][action])
                policy[state] = max(Q[state], key=Q[state].get)
                visited.add((state, action))

    return policy, Q

policy, Q = monte_carlo_es()

print("Optimal Policy:")
for state, action in policy.items():
    print(f"State {state}: Action {action}")
