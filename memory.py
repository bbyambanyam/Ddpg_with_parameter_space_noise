import numpy as np
from collections import deque
import random

# experience-uudiig hadgalah buffer

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=self.size)

    # experience-g buffer-t nemeh

    def add(self, state, action, reward, next_state):
        exp = (state, action, reward, next_state)
        self.buffer.append(exp)

    # Randor-oor size-toonii experience-g awah

    def sample_exp(self, size):
        batch = []
        size = min(size, len(self.buffer))
        batch = random.sample(self.buffer, size)
        
        states = np.float32([arr[0] for arr in batch])
        actions = np.float32([arr[1] for arr in batch])
        rewards = np.float32([arr[2] for arr in batch])
        next_states = np.float32([arr[3] for arr in batch])
        
        return states, actions, rewards, next_states