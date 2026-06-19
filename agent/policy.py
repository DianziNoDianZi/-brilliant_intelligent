import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCritic(nn.Module):
    """Actor-Critic policy network for discrete action spaces."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.encoder(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def act(self, state, device='cpu', deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits, value = self.forward(state)
            dist = torch.distributions.Categorical(logits=logits)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def evaluate(self, state, action):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value
