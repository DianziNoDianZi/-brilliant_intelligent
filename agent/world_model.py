import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """Predicts state delta from (state, action)."""

    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.action_embed = nn.Linear(action_dim, hidden_dim // 2)
        self.lstm = nn.LSTM(
            state_dim + hidden_dim // 2, hidden_dim, num_layers,
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action, hidden=None):
        if state.dim() == 2:
            state = state.unsqueeze(1)
            action = action.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        a_emb = F.relu(self.action_embed(action))
        lstm_input = torch.cat([state, a_emb], dim=-1)

        if hidden is not None:
            lstm_out, (h, c) = self.lstm(lstm_input, hidden)
        else:
            lstm_out, (h, c) = self.lstm(lstm_input)

        pred_delta = self.output(lstm_out)

        if squeeze:
            pred_delta = pred_delta.squeeze(1)

        return pred_delta, (h, c)

    def predict(self, state, action, hidden=None):
        with torch.no_grad():
            return self.forward(state, action, hidden)

    def compute_loss(self, pred_delta, target, state):
        actual_delta = target - state
        return F.mse_loss(pred_delta, actual_delta)


class EnsembleWorldModel(nn.Module):
    """Ensemble of WorldModels for uncertainty-aware prediction.

    Curiosity reward = disagreement (variance) across ensemble members.
    High variance = epistemic uncertainty = knowledge gap = explore.
    """

    def __init__(self, state_dim, action_dim, ensemble_size=3, **kwargs):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList([
            WorldModel(state_dim, action_dim, **kwargs)
            for _ in range(ensemble_size)
        ])

    def forward(self, state, action, member_idx=None):
        """Forward one or all ensemble members.

        Returns:
            If member_idx is not None: (pred_delta, hidden) for that member.
            If member_idx is None:
                pred_deltas: list of tensors, one per member
                stats: dict with 'mean', 'var', 'stds'
        """
        if member_idx is not None:
            return self.models[member_idx](state, action)

        pred_deltas = []
        for m in self.models:
            pd, _ = m(state, action)
            pred_deltas.append(pd)

        stacked = torch.stack(pred_deltas)  # (E, B, state_dim)
        mean = stacked.mean(dim=0)
        var = stacked.var(dim=0, unbiased=False)

        return pred_deltas, {
            'mean': mean,
            'var': var.mean().item(),       # scalar: avg var across dims
            'stds': var.sqrt().mean().item(),
        }

    def predict(self, state, action, member_idx=None):
        with torch.no_grad():
            return self.forward(state, action, member_idx)

    def compute_loss(self, pred_deltas, target, state):
        """Compute MSE loss for each ensemble member.

        pred_deltas: list of tensors from forward()
        Returns: list of scalar losses, one per member
        """
        actual_delta = target - state
        losses = []
        for pd in pred_deltas:
            losses.append(F.mse_loss(pd, actual_delta))
        return losses
