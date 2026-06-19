import torch
import numpy as np

from environment.grid_world import NUM_ACTIONS


def background_replay(memory, world_model, wm_optimizer, config, device, steps=10):
    """Sample high-priority experiences from memory and train world model."""
    if len(memory) < config.BATCH_SIZE:
        return 0.0

    total_loss = 0.0
    for _ in range(steps):
        states, actions, _, next_states, _ = memory.sample(config.BATCH_SIZE)

        s = torch.FloatTensor(states).to(device)
        a = torch.FloatTensor(np.eye(NUM_ACTIONS)[actions]).to(device)
        ns = torch.FloatTensor(next_states).to(device)

        pred, _ = world_model(s, a)
        loss = world_model.compute_loss(pred, ns, state=s)

        wm_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
        wm_optimizer.step()

        total_loss += loss.item()

    return total_loss / steps
