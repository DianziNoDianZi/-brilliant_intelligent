import numpy as np


class EpisodicMemory:
    """Ring buffer with priority-weighted sampling.

    Priority = prediction_error + abs(reward) + epsilon (the δ + r formula).
    """

    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done, prediction_error):
        priority = prediction_error + abs(reward) + 1e-8

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities[len(self.buffer) - 1] = priority
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, alpha=0.6):
        n = len(self.buffer)
        if n < batch_size:
            batch_size = n

        probs = self.priorities[:n] ** alpha
        probs /= probs.sum() + 1e-8

        indices = np.random.choice(n, batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def sample_all(self, alpha=0.6):
        """Sample the entire buffer based on priorities (for background replay)."""
        return self.sample(len(self.buffer), alpha)

    def __len__(self):
        return len(self.buffer)
