"""Plan validation and auto-correction, execution result validation.

validate_plan: check entity IDs exist, action types are legal
auto_fix_plan: retry LLM with error message (max 2 times)
validate_result: compare WSG before/after to detect execution failure
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable

from agent.wsg import WorldStateGraph, SubGoal, action_is_valid_for_entity


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = None
    fixed_plan: list[SubGoal] = None

    def __bool__(self):
        return self.valid

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def validate_plan(plan: list[SubGoal], wsg: WorldStateGraph) -> ValidationResult:
    """Validate a plan against the current WSG.

    Checks:
    - Each step's target_id exists in WSG entities
    - Action type is valid for the target entity's type
    """
    if not plan:
        return ValidationResult(valid=False, errors=['Plan is empty'])

    errors = []

    for step in plan:
        # Key actions (like alt+tab) don't need entity lookup
        if step.action == 'key':
            continue

        entity = wsg.get_entity_by_id(step.target_id)
        if entity is None:
            available_ids = [e.id for e in wsg.entities]
            errors.append(
                f"Step {step.step}: entity_id={step.target_id} not found. "
                f"Available IDs: {available_ids}"
            )
            continue

        if not action_is_valid_for_entity(step.action, entity):
            errors.append(
                f"Step {step.step}: action '{step.action}' is not valid "
                f"for entity type '{entity.type}' "
                f"(entity_id={step.target_id}, text='{entity.text}')"
            )

    return ValidationResult(valid=len(errors) == 0, errors=errors, fixed_plan=plan)


def auto_fix_plan(plan: list[SubGoal], wsg: WorldStateGraph,
                  llm_fn: Callable, max_retries=2) -> ValidationResult:
    """Validate plan and auto-correct via LLM retry if needed.

    llm_fn: function that takes (wsg, instruction_with_error) -> new_plan
    Returns: ValidationResult with the final (possibly corrected) plan.
    """
    result = validate_plan(plan, wsg)
    if result.valid:
        return result

    for attempt in range(max_retries):
        error_text = "\n".join(result.errors)
        fix_instruction = (
            f"Previous plan has errors that need fixing:\n{error_text}\n"
            f"Please regenerate the plan, correcting the above issues."
        )
        new_plan = llm_fn(wsg, fix_instruction)
        if not new_plan:
            break
        result = validate_plan(new_plan, wsg)
        if result.valid:
            return result

    return result


def validate_result(wsg_before: WorldStateGraph, wsg_after: WorldStateGraph,
                    plan: list[SubGoal] = None) -> ValidationResult:
    """Post-execution validation: compare WSG states to detect failure.

    Heuristics:
    - If same screenshot -> nothing changed -> execution likely failed
    - If critical entities disappeared -> unexpected state
    """
    if wsg_before.screenshot is not None and wsg_after.screenshot is not None:
        if wsg_before.screenshot.shape == wsg_after.screenshot.shape:
            same = (wsg_before.screenshot == wsg_after.screenshot).mean()
            if same > 0.98:
                return ValidationResult(
                    valid=False,
                    errors=['Screenshot unchanged, execution may have failed']
                )

    if len(wsg_after.entities) == 0:
        return ValidationResult(
            valid=False,
            errors=['No UI elements detected after execution']
        )

    return ValidationResult(valid=True)
