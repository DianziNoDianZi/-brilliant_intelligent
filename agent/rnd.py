import torch
import torch.nn as nn
import torch.nn.functional as F


class RNDTarget(nn.Module):
    """Fixed random network that defines 'novelty' of states.

    Never trained — outputs are the ground-truth random features.
    """

    def __init__(self, state_dim, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        # Initialize with small random weights and freeze
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, state):
        return self.net(state)


class RNDPredictor(nn.Module):
    """Trainable predictor that learns to match the target network.

    Prediction error = novelty bonus. High error = novel state.
    """

    def __init__(self, state_dim, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, state):
        return self.net(state)


class RND:
    """Random Network Distillation for exploration bonus.

    Provides a persistent novelty signal: the predictor network
    learns to match a fixed random target. Novel states have high
    prediction error, driving exploration.
    """

    def __init__(self, state_dim, feature_dim=128, lr=1e-3, device='cpu'):
        self.device = device
        self.target = RNDTarget(state_dim, feature_dim).to(device)
        self.predictor = RNDPredictor(state_dim, feature_dim).to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def compute_bonus(self, state):
        """Returns curiosity bonus (MSE between target and predictor)."""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            target_feat = self.target(s)
        pred_feat = self.predictor(s)
        error = (pred_feat - target_feat).pow(2).mean().item()
        return error

    def update(self, states):
        """One training step for the predictor network."""
        if isinstance(states, torch.Tensor):
            s = states
        else:
            s = torch.FloatTensor(states).to(self.device)

        if s.dim() == 1:
            s = s.unsqueeze(0)

        with torch.no_grad():
            target_feat = self.target(s)

        pred_feat = self.predictor(s)
        loss = F.mse_loss(pred_feat, target_feat)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
