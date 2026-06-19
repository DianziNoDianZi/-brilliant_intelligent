"""Inner simulation — multi-step rollout with confidence-based execution.

High confidence steps are simulated entirely in the world model.
Low confidence steps trigger real execution, then simulation continues.

This is the core of "内模拟": imagine the outcome, verify only when uncertain.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np
from typing import Optional, Callable

from agent.wsg import WorldStateGraph, SubGoal
from agent.wsg_encoder import encode_wsg, encode_action, STATE_DIM


@dataclass
class RolloutStep:
    step: int
    action_desc: str
    confidence: float
    is_simulated: bool       # True = pure simulation, False = real execution
    predicted_changes: list
    real_execution_result: bool = True


@dataclass
class RolloutResult:
    steps: list[RolloutStep]
    total_steps: int
    simulated_count: int     # steps done in imagination only
    executed_count: int      # steps that needed real verification
    avg_confidence: float
    full_simulation: bool    # True if ALL steps were simulated


class Simulator:
    """Inner simulation engine with multi-step rollout.

    Given a plan of N actions, simulates each step through the ensemble
    world model. Steps with confidence below threshold are flagged for
    real execution.

    After each real execution, the mental WSG is updated and simulation
    continues for the remaining steps.
    """

    MIN_TRAINING_STEPS = 10  # minimum training steps before simulation is trusted

    def __init__(self, ensemble_world_model, device='cpu',
                 confidence_threshold=0.5):
        self.model = ensemble_world_model
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._training_steps = 0

    @property
    def _can_simulate(self) -> bool:
        """Simulation is only trusted after enough training."""
        return self._training_steps >= self.MIN_TRAINING_STEPS

    def rollout(self, plan: list[SubGoal], wsg: WorldStateGraph,
                real_exec_fn: Callable = None) -> RolloutResult:
        """Multi-step rollout with mixed simulation/execution.

        Args:
            plan: list of SubGoal actions
            wsg: current world state
            real_exec_fn: optional function to execute a step in reality.
                          Called as real_exec_fn(step, wsg) -> (success, new_wsg)
                          If None, all steps are simulated.

        Returns:
            RolloutResult with per-step decisions and final outcome
        """
        if not plan:
            return RolloutResult([], 0, 0, 0, 0.0, full_simulation=True)

        steps = []
        simulated = 0
        executed = 0
        current_wsg = wsg

        for step in plan:
            step_result = self._simulate_step(step, current_wsg)
            confidence = step_result.confidence

            # Decision: simulate or execute?
            needs_real = confidence < self.confidence_threshold

            if needs_real and real_exec_fn is not None:
                # Execute in reality
                success, new_wsg = real_exec_fn(step, current_wsg)
                if success:
                    current_wsg = new_wsg
                    executed += 1
                    # Add training data for world model
                    self._add_training_experience(current_wsg, step, new_wsg)

                steps.append(RolloutStep(
                    step=step.step,
                    action_desc=step.description,
                    confidence=confidence,
                    is_simulated=False,
                    predicted_changes=step_result.predicted_changes,
                    real_execution_result=success,
                ))
            else:
                # Simulate: update mental WSG
                if step_result.predicted_changes:
                    self._update_mental_wsg(current_wsg, step_result.predicted_changes)
                simulated += 1
                steps.append(RolloutStep(
                    step=step.step,
                    action_desc=step.description,
                    confidence=confidence,
                    is_simulated=True,
                    predicted_changes=step_result.predicted_changes,
                ))

        confidences = [s.confidence for s in steps]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return RolloutResult(
            steps=steps,
            total_steps=len(plan),
            simulated_count=simulated,
            executed_count=executed,
            avg_confidence=avg_conf,
            full_simulation=(executed == 0),
        )

    def _simulate_step(self, step: SubGoal, wsg: WorldStateGraph):
        """Single-step simulation: predict delta, compute confidence."""
        if not hasattr(self.model, 'ensemble_size') or not self._can_simulate:
            return _Prediction(confidence=0.0, predicted_changes=[])

        state_vec = encode_wsg(wsg)
        s_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)

        act_types = ['click', 'type', 'tab', 'enter', 'wait']
        a_onehot = np.zeros(len(act_types), dtype=np.float32)
        if step.action in act_types:
            a_onehot[act_types.index(step.action)] = 1.0
        a_t = torch.FloatTensor(a_onehot).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_deltas, stats = self.model.predict(s_t, a_t)

        # Confidence from ensemble variance
        confidence = float(1.0 / (1.0 + stats['var'] * 10.0))
        confidence = max(0.0, min(1.0, confidence))

        changes = []
        if pred_deltas:
            mean_delta = torch.stack(pred_deltas).mean(dim=0)
            changes = self._decode_changes(
                mean_delta.squeeze(0).cpu().numpy(), wsg)

        return _Prediction(confidence=confidence, predicted_changes=changes)

    def _decode_changes(self, pred_delta, wsg):
        from agent.wsg_encoder import ENTITY_FEAT_DIM, MAX_ENTITIES
        changes = []
        for i, entity in enumerate(wsg.entities[:MAX_ENTITIES]):
            offset = i * ENTITY_FEAT_DIM
            mag = float(np.mean(np.abs(pred_delta[offset:offset + ENTITY_FEAT_DIM])))
            if mag > 0.03:
                changes.append({
                    'entity_id': entity.id,
                    'text': entity.text,
                    'type': entity.type,
                    'change_magnitude': round(mag, 4),
                })
        return changes

    def _update_mental_wsg(self, wsg, changes):
        """Update mental WSG (simplified — marks predicted changes)."""
        pass

    def _add_training_experience(self, wsg_before, action, wsg_after):
        """Collect real execution data (called externally via training loop)."""
        pass  # Handled by the training loop

    def print_rollout_report(self, result: RolloutResult):
        """Print a human-readable simulation report."""
        print(f"\n  [SIMULATION] {result.total_steps} steps: "
              f"{result.simulated_count} simulated + {result.executed_count} executed")
        print(f"  [SIMULATION] Avg confidence: {result.avg_confidence:.2f}")
        print(f"  [SIMULATION] Mode: {'FULL SIM' if result.full_simulation else 'MIXED'}")

        for s in result.steps:
            mode = "SIM" if s.is_simulated else "EXEC"
            conf = f"conf={s.confidence:.2f}"
            print(f"    Step {s.step}: [{mode}] {s.action_desc} ({conf})")
            if s.predicted_changes[:2]:
                for c in s.predicted_changes[:2]:
                    print(f"       ~ {c.get('text','')} mag={c.get('change_magnitude',0):.3f}")


    def record_training(self, n_steps: int = 1):
        """Record that the world model has been trained on n more steps."""
        self._training_steps += n_steps


class _Prediction:
    """Internal prediction result."""
    def __init__(self, confidence=0.0, predicted_changes=None):
        self.confidence = confidence
        self.predicted_changes = predicted_changes or []
