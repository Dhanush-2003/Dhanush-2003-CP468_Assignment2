import numpy as np
import random

# Define the environment
class TrafficGrid:
    def __init__(self, size=3):
        self.size = size
        self.state_space = size * size
        self.action_space = 4  
        self.goal_state = (size - 1, size - 1)
        self.traffic_states = [(1, 1), (2, 1)] 

    def get_reward(self, current_state):
        if current_state == self.goal_state:
            return 100  
        if current_state in self.traffic_states:
            return -1  
        return 0  

    def is_terminal_state(self, state):
        return state == self.goal_state

    def state_to_index(self, state):
        return state[0] * self.size + state[1]

    def index_to_state(self, index):
        return (index // self.size, index % self.size)

    def step(self, current_state, action):
        
        # Define action effects
        action_effects = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        new_state = (current_state[0] + action_effects[action][0],
                     current_state[1] + action_effects[action][1])

        new_state = (max(0, min(new_state[0], self.size - 1)),
                     max(0, min(new_state[1], self.size - 1)))

        reward = self.get_reward(new_state)
        return new_state, reward

# Initialize parameters
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001
num_episodes = 1000

# Initialize Q-table
env = TrafficGrid()
Q = np.zeros((env.state_space, env.action_space))

# Q-learning algorithm
for episode in range(num_episodes):
    state = (0, 0)  
    done = False

    while not done:
        if random.uniform(0, 1) < exploration_rate:
            action = random.randint(0, env.action_space - 1)  
        else:
            state_index = env.state_to_index(state)
            action = np.argmax(Q[state_index])  

        new_state, reward = env.step(state, action)
        new_state_index = env.state_to_index(new_state)
        state_index = env.state_to_index(state)

        # Update Q-table
        Q[state_index, action] = Q[state_index, action] + learning_rate * (reward + discount_factor * np.max(Q[new_state_index]) - Q[state_index, action])

        state = new_state
        done = env.is_terminal_state(state)

    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

# Display learned Q-values
print("Q-table:")
print(Q)
