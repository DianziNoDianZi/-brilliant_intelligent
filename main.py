"""Phase 1 → 2 过渡入口。

OBS_MODE='vector':   Phase 1 原逻辑（单世界模型 + IdentityDecomposer）
OBS_MODE='pixels':   视觉输入 + 系综世界模型 + EnsembleDecomposer
"""

import torch
import numpy as np

import config
from environment.grid_world import GridWorld, NUM_ACTIONS
from agent.policy import ActorCritic
from agent.memory import EpisodicMemory
from agent.intrinsic_reward import IntrinsicReward, IdentityDecomposer, EnsembleDecomposer
from training.train_loop import run_training


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Mode: {config.OBS_MODE}")

    visual_mode = config.OBS_MODE == 'pixels'

    env = GridWorld(
        grid_size=config.GRID_SIZE,
        wall_density=config.WALL_DENSITY,
        interactive_density=config.INTERACTIVE_DENSITY,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        obs_mode=config.OBS_MODE,
    )

    if visual_mode:
        state_dim = config.FEATURE_DIM
    else:
        state_dim = env.observation_space.shape[0]

    # --- Policy (same for both modes) ---
    policy = ActorCritic(state_dim, NUM_ACTIONS).to(device)

    # --- World model & encoder ---
    if visual_mode:
        from agent.visual_encoder import VisualEncoder
        from agent.world_model import EnsembleWorldModel

        encoder = VisualEncoder(feature_dim=config.FEATURE_DIM).to(device)
        world_model = EnsembleWorldModel(
            state_dim, NUM_ACTIONS,
            ensemble_size=config.ENSEMBLE_SIZE,
            hidden_dim=config.WM_HIDDEN_DIM,
        ).to(device)

        intrinsic_reward = IntrinsicReward(
            decomposer=EnsembleDecomposer(scale=1.0)
        )

        # Optimizers: separate optimizers for each ensemble member
        wm_optimizer = [
            torch.optim.Adam(m.parameters(), lr=config.WM_LR)
            for m in world_model.models
        ]
        encoder_optimizer = torch.optim.Adam(
            encoder.parameters(), lr=config.ENCODER_LR)
        # Combine encoder + policy into one optimizer
        policy_optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(encoder.parameters()),
            lr=config.POLICY_LR,
        )
    else:
        from agent.world_model import WorldModel

        encoder = None
        world_model = WorldModel(state_dim, NUM_ACTIONS,
                                 hidden_dim=config.WM_HIDDEN_DIM).to(device)
        intrinsic_reward = IntrinsicReward(decomposer=IdentityDecomposer())
        wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=config.WM_LR)
        policy_optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.POLICY_LR)

    memory = EpisodicMemory(capacity=config.MEMORY_CAPACITY)

    # --- Visualization ---
    renderer = None
    if config.RENDER:
        from visualization.renderer import Phase1Renderer
        renderer = Phase1Renderer(env, config)

    # --- Run ---
    print(f"Grid: {config.GRID_SIZE}x{config.GRID_SIZE}, "
          f"{config.NUM_EPISODES} episodes, "
          f"{config.MAX_STEPS_PER_EPISODE} steps/ep")
    print(f"World model: {'ensemble x' + str(config.ENSEMBLE_SIZE) if visual_mode else 'single'}")
    print("-" * 60)

    history = run_training(
        env=env, world_model=world_model, policy=policy, memory=memory,
        intrinsic_reward=intrinsic_reward,
        wm_optimizer=wm_optimizer, policy_optimizer=policy_optimizer,
        config=config, device=device, encoder=encoder, renderer=renderer,
        replay_interval=config.REPLAY_INTERVAL,
        replay_steps=config.REPLAY_STEPS,
    )

    print("-" * 60)
    print("Training complete!")
    print(f"Episodes: {len(history['episode'])}")
    print(f"Final interactions/ep: {history['total_interactions'][-1]}")
    print(f"Max interactions/ep: {max(history['total_interactions'])}")

    if renderer:
        print("Close the plot window to exit.")
        renderer.fig.canvas.mpl_connect('close_event', lambda _: exit(0))
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show(block=True)


if __name__ == '__main__':
    main()
