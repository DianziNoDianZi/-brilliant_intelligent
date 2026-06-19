#!/usr/bin/env python3
"""Phase 2 入口 — 语言指令驱动桌面操作 + 内模拟。

用法:
  python main_phase2.py "计算 1+1"                         # 预演
  python main_phase2.py "计算 1+1" --execute               # 真实执行
  python main_phase2.py "计算 1+1" --train                 # 带内模拟+在线训练
  python main_phase2.py "..." --app notepad --execute
"""

from __future__ import annotations
import argparse
import sys
import torch
import numpy as np

from agent.wsg import WorldStateGraph, SubGoal
from agent.planner import Planner
from agent.validator import auto_fix_plan, validate_result
from agent.executor import Executor, dry_run
from environment.desktop_env import DesktopEnv


def build_parser():
    p = argparse.ArgumentParser(description="Phase 2 — Desktop Agent")
    p.add_argument("instruction", type=str, help='e.g. "press 1+1"')
    p.add_argument("--app", default="calculator",
                   choices=["calculator", "notepad"])
    p.add_argument("--execute", action="store_true",
                   help="Execute for real (default: dry-run)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--llm", default="qwen2.5:7b")
    p.add_argument("--train", action="store_true",
                   help="Enable inner simulation + world model training")
    p.add_argument("--epochs", type=int, default=3,
                   help="Training epochs per experience (default: 3)")
    p.add_argument("--dual", action="store_true",
                   help="Dual-loop mode: fast loop caches skills, avoids LLM for repeated tasks")
    return p


def _setup_world_model(state_dim, action_dim, device):
    """Create ensemble world model and simulator."""
    from agent.world_model import EnsembleWorldModel
    from agent.simulator import Simulator

    wm = EnsembleWorldModel(
        state_dim, action_dim,
        ensemble_size=3,
        hidden_dim=32,
    ).to(device)
    simulator = Simulator(wm, device=device, confidence_threshold=0.5)
    memory = []  # list of (state_vec, action_onehot, next_state_vec)
    return wm, simulator, memory


def _train_world_model(memory, wm, wm_optimizers, device, epochs=3):
    """Train ensemble world model from collected experiences."""
    if len(memory) < 2:
        return 0.0

    total_loss = 0.0
    count = 0
    for _ in range(epochs):
        indices = np.random.choice(len(memory), min(len(memory), 32),
                                   replace=False)
        for idx in indices:
            s, a, ns = memory[idx]
            s_t = torch.FloatTensor(s).unsqueeze(0).to(device)
            a_t = torch.FloatTensor(a).unsqueeze(0).to(device)
            ns_t = torch.FloatTensor(ns).unsqueeze(0).to(device)

            preds, _ = wm(s_t, a_t)
            losses = wm.compute_loss(preds, ns_t, s_t)

            for i, loss in enumerate(losses):
                if torch.isfinite(loss):
                    wm_optimizers[i].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        wm.models[i].parameters(), 1.0)
                    wm_optimizers[i].step()
                    total_loss += loss.item()
                    count += 1

    return total_loss / max(count, 1)


def main():
    parser = build_parser()
    args = parser.parse_args()
    dry_run_mode = not args.execute
    train_mode = args.train

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f"  Phase 2 Desktop Agent")
    print(f"  Instruction: {args.instruction}")
    print(f"  App: {args.app}")
    print(f"  Mode: {'DRY-RUN' if dry_run_mode else 'EXECUTE'}"
          f"{' + TRAIN' if train_mode else ''}")
    print(f"  Device: {device}")
    print("=" * 60)

    # 1. Environment
    print("\n[1] Initializing desktop environment...")
    env = DesktopEnv(app_name=args.app)
    wsg = env.reset()
    print(f"  Detected {len(wsg.entities)} UI elements")

    # 2. World model (train mode only)
    wm = None
    simulator = None
    memory = None
    wm_optimizers = None

    if train_mode:
        from agent.wsg_encoder import STATE_DIM, ACTION_FEAT_DIM
        from environment.grid_world import NUM_ACTIONS

        print("\n[1b] Initializing ensemble world model...")
        wm, simulator, memory = _setup_world_model(
            STATE_DIM, NUM_ACTIONS, device)
        wm_optimizers = [
            torch.optim.Adam(m.parameters(), lr=1e-3)
            for m in wm.models
        ]
        print(f"  Ensemble size: {wm.ensemble_size}")
        print(f"  State dim: {STATE_DIM}, Action dim: {NUM_ACTIONS}")

    # 3. Fast/Slow loop
    from agent.intent import IntentVector, detect_task_type
    from agent.fast_loop import FastLoop
    from agent.skill_lib import SkillLibrary

    fast_loop = FastLoop(skill_lib=SkillLibrary()) if args.dual else None
    used_fast_path = False

    if args.dual:
        intent = IntentVector.from_values(
            values=[],  # filled by planner or fast loop
            instruction=args.instruction,
            task_type=detect_task_type(args.instruction),
        )
        use_fast, fast_values, _ = fast_loop.decide(intent)
        if use_fast and fast_values:
            from agent.planner import _values_to_plan
            plan = _values_to_plan(fast_values, wsg)
            used_fast_path = True
            print(f"\n[2] Fast loop (cached skill) → {fast_values}")

    # Create planner for both paths (needed for validation)
    planner = Planner(model=args.llm)

    if not used_fast_path:
        print(f"\n[2] Calling LLM ({args.llm})...")
        plan = planner.plan(args.instruction, wsg)

    if plan is None:
        print("[FAIL] No valid plan")
        env.close()
        sys.exit(1)

    print(f"  Initial plan: {len(plan)} steps ({'fast' if used_fast_path else 'slow'} path)")

    # 4. Validate & auto-fix
    print(f"\n[3] Validating plan...")

    def llm_fix_fn(wsg: WorldStateGraph, error_msg: str):
        return planner.fix_plan(args.instruction, wsg, error_msg)

    result = auto_fix_plan(plan, wsg, llm_fix_fn, max_retries=2)
    final_plan = result.fixed_plan or plan

    if not result.valid:
        print("[FAIL] Plan validation failed after retries:")
        for err in result.errors:
            print(f"  - {err}")
        dry_run(final_plan, wsg)
        env.close()
        sys.exit(1)

    print(f"  Plan valid ({len(final_plan)} steps)")

    # 5. Simulate (train mode)
    if train_mode and simulator:
        print(f"\n[4] Inner simulation...")
        sim_result = simulator.rollout(final_plan, wsg)
        simulator.print_rollout_report(sim_result)

    # 6. Execute
    print(f"\n[5] {'Dry-run' if dry_run_mode else 'Executing'}...")

    if dry_run_mode:
        dry_run(final_plan, wsg)

        if train_mode and simulator:
            print(f"\n  Simulation report:")
            print(f"  {sim_result.simulated_count} simulated, "
                  f"{sim_result.executed_count} would need real execution")
            print(f"  Avg confidence: {sim_result.avg_confidence:.1%}")

        print("\n  Use --execute to run for real")
    else:
        executor = Executor()

        if train_mode and simulator:
            # Execute with simulation-guided confidence
            _execute_with_simulation(
                executor, final_plan, wsg, env, simulator,
                memory, wm, wm_optimizers, device, args.epochs)
        else:
            # Plain execution
            success, step_results = executor.execute_plan(final_plan, wsg, env)
            _report_execution_result(success, step_results, final_plan, env, wsg)

            # Dual-loop: record success for skill compilation
            if success and args.dual and fast_loop:
                from agent.planner import _values_to_plan
                vals = plan_to_values(final_plan, wsg)
                if vals:
                    intent = IntentVector.from_values(
                        vals, args.instruction,
                        detect_task_type(args.instruction))
                    fast_loop.record_success(intent, vals)
                    print(f"\n  [SKILL] Recorded + trained. {fast_loop}")

    if args.dual and fast_loop:
        print(f"\n  Fast loop stats: {fast_loop}")

    env.close()
    print("\nDone.")


def plan_to_values(plan, wsg):
    """Extract button value or type sequence from a validated plan."""
    values = []
    for step in plan:
        if step.action == 'type' and step.value:
            values.append(f'type:{step.value}')
        else:
            e = wsg.get_entity_by_id(step.target_id)
            if e and e.text:
                cn_to_en = {'零':'0','一':'1','二':'2','三':'3','四':'4',
                            '五':'5','六':'6','七':'7','八':'8','九':'9',
                            '加':'+','减':'-','乘以':'*','除以':'/','等于':'='}
                v = cn_to_en.get(e.text.strip(), e.text.strip())
                values.append(v)
    return values if values else None


def _execute_with_simulation(executor, plan, wsg, env, simulator,
                              memory, wm, optimizers, device, epochs):
    """Execute plan with inner simulation rollout.

    The simulator decides per-step: simulate (high confidence) or
    execute in reality (low confidence). Real executions produce
    training data that improves the world model over time.
    """
    from agent.wsg_encoder import encode_wsg

    training_data = []

    def real_exec_fn(step, current_wsg):
        """Execute a step in reality, collect training data."""
        # Remap target_id by matching entity text in current WSG
        orig_e = wsg.get_entity_by_id(step.target_id)
        if orig_e:
            new_e = current_wsg.get_entity_by_text(orig_e.text)
            if new_e:
                step.target_id = new_e.id
        wsgbefore_vec = encode_wsg(current_wsg)
        ok = executor.execute_step(step, current_wsg, env)
        if ok:
            new_wsg = env.reset()
            wsgafter_vec = encode_wsg(new_wsg)

            act_onehot = np.zeros(5, dtype=np.float32)
            act_types = ['click', 'type', 'tab', 'enter', 'wait']
            if step.action in act_types:
                act_onehot[act_types.index(step.action)] = 1.0
            training_data.append((wsgbefore_vec, act_onehot, wsgafter_vec))

            # Online training after each real execution
            for s, a, ns in training_data[-3:]:
                s_t = torch.FloatTensor(s).unsqueeze(0).to(device)
                a_t = torch.FloatTensor(a).unsqueeze(0).to(device)
                ns_t = torch.FloatTensor(ns).unsqueeze(0).to(device)

                if hasattr(wm, 'ensemble_size'):
                    preds, _ = wm(s_t, a_t)
                    losses = wm.compute_loss(preds, ns_t, s_t)
                    for i, loss in enumerate(losses):
                        if torch.isfinite(loss):
                            optimizers[i].zero_grad()
                            loss.backward()
                            optimizers[i].step()
                else:
                    pd, _ = wm(s_t, a_t)
                    loss = wm.compute_loss(pd, ns_t, s_t)
                    optimizers.zero_grad()
                    loss.backward()
                    optimizers.step()

            simulator.record_training()
            return True, new_wsg
        return False, current_wsg

    # Multi-step rollout
    result = simulator.rollout(plan, wsg, real_exec_fn=real_exec_fn)
    simulator.print_rollout_report(result)

    memory.extend(training_data)
    print(f"  [TRAIN] Collected {len(training_data)} experiences, "
          f"total memory: {len(memory)}")

    if result.executed_count > 0:
        conf_trend = result.avg_confidence
        print(f"  [TRAIN] World model avg confidence: {conf_trend:.2f}")
    else:
        print(f"  [TRAIN] No real executions needed (full simulation)")


def _report_execution_result(success, step_results, plan, env, wsg):
    if success:
        print("\n  All steps executed successfully")
        wsg_after = env.reset()
        v = validate_result(wsg, wsg_after, plan)
        if v.valid:
            print("  Result validation: OK")
        else:
            print(f"  Result issues: {v.errors}")
    else:
        print(f"\n  Failed at step {len(step_results)}/{len(plan)}")


def _report_training_progress(conf_before, conf_after, memory_size):
    print(f"\n  [TRAINING REPORT]")
    print(f"    Experiences collected: {memory_size}")
    print(f"    World model confidence trend: "
          f"{conf_before:.2f} -> {conf_after:.2f}")


if __name__ == '__main__':
    main()
