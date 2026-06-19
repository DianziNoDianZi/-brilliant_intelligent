import torch
import torch.nn.functional as F
import numpy as np

from environment.grid_world import NUM_ACTIONS


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if t == len(rewards) - 1 or dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (0.0 if dones[t] else gae)
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return np.array(returns, dtype=np.float32), np.array(advantages, dtype=np.float32)


def _update_policy(trajectory, policy, policy_optimizer, config, device):
    states_traj = np.array([t[0] for t in trajectory], dtype=np.float32)
    actions_traj = np.array([t[1] for t in trajectory], dtype=np.int64)
    rewards_traj = np.array([t[2] for t in trajectory], dtype=np.float32)
    dones_traj = np.array([t[3] for t in trajectory], dtype=np.float32)
    values_traj = np.array([t[5] for t in trajectory], dtype=np.float32)

    returns, advantages = compute_gae(rewards_traj, values_traj, dones_traj, config.GAMMA)
    adv_torch = torch.FloatTensor(advantages).to(device)
    adv_torch = (adv_torch - adv_torch.mean()) / (adv_torch.std() + 1e-8)

    s_torch = torch.FloatTensor(states_traj).to(device)
    a_torch = torch.LongTensor(actions_traj).to(device)
    ret_torch = torch.FloatTensor(returns).to(device)

    log_probs, entropies, values = policy.evaluate(s_torch, a_torch)
    log_probs = torch.clamp(log_probs, min=-20.0)

    actor_loss = -(log_probs * adv_torch).mean()
    critic_loss = F.mse_loss(values.squeeze(-1), ret_torch)
    entropy_loss = -entropies.mean() * config.ENTROPY_COEFF
    policy_loss = actor_loss + critic_loss + entropy_loss

    if torch.isfinite(policy_loss):
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy_optimizer.step()

    return policy_loss.item() if torch.isfinite(policy_loss) else float('nan')


def _to_tensor(x, device):
    return torch.FloatTensor(x).to(device)


def _act(policy, state, device):
    """Helper: policy.act that handles both numpy and already-tensor states."""
    return policy.act(state, device=device)


def train_episode(env, world_model, policy, memory, intrinsic_reward,
                  wm_optimizer, policy_optimizer, config, device,
                  encoder=None, renderer=None):
    """Run one episode. Supports both vector and pixel modes."""
    obs, _ = env.reset()
    visual_mode = encoder is not None

    metrics = {
        'rewards': [], 'errors': [], 'rnd_bonus': [],
        'interactions': [], 'actions': [], 'policy_loss': 0.0,
    }
    trajectory = []
    seg_start = 0

    for step in range(config.MAX_STEPS_PER_EPISODE):
        # --- Encode observation ---
        if visual_mode:
            img_t = torch.ByteTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
            state_vec = encoder(img_t).squeeze(0).detach().cpu().numpy()
        else:
            state_vec = obs

        # --- Act ---
        action, log_prob, value = policy.act(state_vec, device=device)
        next_obs, _, terminated, truncated, _ = env.step(action)

        # --- Encode next observation ---
        if visual_mode:
            nimg_t = torch.ByteTensor(next_obs).permute(2, 0, 1).unsqueeze(0).to(device)
            next_vec = encoder(nimg_t).squeeze(0).detach().cpu().numpy()
        else:
            next_vec = next_obs

        # --- World model prediction & curiosity reward ---
        s_t = _to_tensor(state_vec, device).unsqueeze(0)
        ns_t = _to_tensor(next_vec, device).unsqueeze(0)
        a_t = torch.FloatTensor(np.eye(NUM_ACTIONS)[action]).unsqueeze(0).to(device)

        # Detach features for world model (encoder trained only by policy gradient)
        s_detached = s_t.detach()
        ns_detached = ns_t.detach()

        if hasattr(world_model, 'ensemble_size'):  # Ensemble mode
            pred_deltas, stats = world_model.predict(s_detached, a_t)
            reward, components = intrinsic_reward.compute(
                ensemble_variance=stats['var']
            )
            wm_error = stats['var']  # log ensemble disagreement
        else:  # Single world model mode
            pred_delta, _ = world_model.predict(s_detached, a_t)
            wm_error = world_model.compute_loss(pred_delta, ns_detached,
                                                 s_detached).item()
            reward, components = intrinsic_reward.compute(
                prediction_error=wm_error
            )

        reward = min(reward * config.REWARD_SCALE, config.REWARD_MAX)

        # --- Store in memory (always feature vectors, not pixels) ---
        memory.push(state_vec, action, reward, next_vec,
                    terminated or truncated, wm_error)
        trajectory.append((state_vec, action, reward, terminated or truncated,
                          log_prob, value))

        metrics['rewards'].append(reward)
        metrics['errors'].append(wm_error)
        metrics['actions'].append(action)
        total_ints = env.get_interaction_count()
        metrics['interactions'].append(total_ints)

        # --- Train world model (ensemble: train each member) ---
        if len(memory) >= config.BATCH_SIZE:
            batch = memory.sample(config.BATCH_SIZE)
            s_b = _to_tensor(batch[0], device)
            a_b = torch.FloatTensor(np.eye(NUM_ACTIONS)[batch[1]]).to(device)
            ns_b = _to_tensor(batch[3], device)

            if hasattr(world_model, 'ensemble_size'):
                preds, _ = world_model(s_b, a_b)
                losses = world_model.compute_loss(preds, ns_b, s_b)
                for i, loss in enumerate(losses):
                    if torch.isfinite(loss):
                        wm_optimizer[i].zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            world_model.models[i].parameters(), 1.0)
                        wm_optimizer[i].step()
            else:
                pred_b, _ = world_model(s_b, a_b)
                loss = world_model.compute_loss(pred_b, ns_b, s_b)
                wm_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
                wm_optimizer.step()

        # --- Policy update ---
        if (step + 1) % config.TRAIN_POLICY_EVERY == 0 and step > 0:
            seg = trajectory[seg_start:step + 1]
            if len(seg) >= 4:
                pl = _update_policy(seg, policy, policy_optimizer, config, device)
                metrics['policy_loss'] += pl
            seg_start = step + 1

        obs = next_obs

        if renderer:
            renderer.add_metric(reward, wm_error, total_ints)
            renderer.update(env)

        if terminated or truncated:
            break

    if seg_start < len(trajectory):
        seg = trajectory[seg_start:]
        if len(seg) >= 4:
            pl = _update_policy(seg, policy, policy_optimizer, config, device)
            metrics['policy_loss'] += pl

    metrics['total_interactions'] = total_ints
    metrics['steps'] = len(trajectory)
    return metrics


def _train_world_model_replay(memory, world_model, wm_optimizers, config, device):
    """Single batch world model update (handles both single and ensemble)."""
    if len(memory) < config.BATCH_SIZE:
        return
    batch = memory.sample(config.BATCH_SIZE)
    s_b = torch.FloatTensor(batch[0]).to(device)
    a_b = torch.FloatTensor(np.eye(NUM_ACTIONS)[batch[1]]).to(device)
    ns_b = torch.FloatTensor(batch[3]).to(device)

    if hasattr(world_model, 'ensemble_size'):
        preds, _ = world_model(s_b, a_b)
        losses = world_model.compute_loss(preds, ns_b, s_b)
        for i, loss in enumerate(losses):
            if torch.isfinite(loss):
                wm_optimizers[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(world_model.models[i].parameters(), 1.0)
                wm_optimizers[i].step()
    else:
        pred_b, _ = world_model(s_b, a_b)
        loss = world_model.compute_loss(pred_b, ns_b, s_b)
        wm_optimizers.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
        wm_optimizers.step()


def run_training(env, world_model, policy, memory, intrinsic_reward,
                 wm_optimizer, policy_optimizer, config, device,
                 encoder=None, renderer=None,
                 replay_interval=0, replay_steps=10):
    """Full training loop. vector/pixels mode determined by encoder being set."""
    history = {
        'episode': [],
        'avg_reward': [],
        'avg_error': [],
        'total_interactions': [],
        'steps': [],
        'policy_loss': [],
    }

    for episode in range(1, config.NUM_EPISODES + 1):
        metrics = train_episode(
            env, world_model, policy, memory, intrinsic_reward,
            wm_optimizer, policy_optimizer, config, device,
            encoder=encoder, renderer=renderer,
        )

        if replay_interval > 0 and episode % replay_interval == 0:
            for _ in range(replay_steps):
                _train_world_model_replay(
                    memory, world_model, wm_optimizer, config, device)

        avg_reward = np.mean(metrics['rewards']) if metrics['rewards'] else 0.0
        avg_error = np.mean(metrics['errors']) if metrics['errors'] else 0.0

        history['episode'].append(episode)
        history['avg_reward'].append(avg_reward)
        history['avg_error'].append(avg_error)
        history['total_interactions'].append(metrics['total_interactions'])
        history['steps'].append(metrics['steps'])
        history['policy_loss'].append(metrics['policy_loss'])

        if episode % 50 == 0 or episode == 1:
            mode_tag = 'VIS' if encoder else 'VEC'
            print(
                f"[{mode_tag}] Ep {episode:4d} | "
                f"avg_r={avg_reward:+.4f} | avg_err={avg_error:.5f} "
                f"| interactions={metrics['total_interactions']:2d}"
            )

        if renderer:
            renderer.update_plot(history)

    return history
