"""Action executor — dry_run (print) and real execution (PyAutoGUI).

Coordinate mapping: screenshot bbox -> window offset -> screen coords.
"""

from __future__ import annotations
from typing import Optional
import time

from agent.wsg import WorldStateGraph, SubGoal


ACTION_GLYPH = {
    'click': '[CLICK]',
    'type': '[TYPE] ',
    'tab': '[TAB]  ',
    'enter': '[ENTER]',
    'wait': '[WAIT] ',
}


def dry_run(plan: list[SubGoal], wsg: WorldStateGraph) -> None:
    """Print human-readable operation description without touching the desktop."""

    if not plan:
        print("[Dry-Run] Empty plan, no operations")
        return

    print("=" * 60)
    print("  [DRY-RUN] Operation Plan")
    print("=" * 60)

    for step in plan:
        entity = wsg.get_entity_by_id(step.target_id)
        entity_info = f"'{entity.text}' ({entity.type})" if entity else f"ID={step.target_id}"

        cx, cy = entity.center if entity else (0, 0)
        screen_x, screen_y = wsg.screenshot_to_screen(cx, cy) if entity else (0, 0)

        glyph = ACTION_GLYPH.get(step.action, '[????] ')
        desc = step.description or f"{step.action} {entity_info}"
        coord_str = f" screenshot({cx},{cy}) -> screen({screen_x},{screen_y})" if entity else ""
        val_str = f"  value='{step.value}'" if step.value else ""

        print(f"  {glyph} Step {step.step}: {desc}")
        print(f"     Target: {entity_info}{coord_str}{val_str}")

    print("=" * 60)
    print("  [DRY-RUN] No real actions executed. Use --execute to run.")


class Executor:
    """Executes sub-goals via PyAutoGUI with coordinate mapping."""

    def __init__(self):
        import pyautogui
        self.pyautogui = pyautogui
        self.pyautogui.FAILSAFE = True
        self.pyautogui.PAUSE = 0.15
        print("[SAFETY] FAILSAFE enabled - move mouse to any screen corner to emergency stop")

    def execute_step(self, step: SubGoal, wsg: WorldStateGraph,
                     env=None) -> bool:
        """Execute a single sub-goal. Returns success."""
        entity = wsg.get_entity_by_id(step.target_id)
        if entity is None:
            # Try to find by the step description text
            for e in wsg.entities:
                if e.text and e.text in step.description:
                    entity = e
                    step.target_id = e.id
                    break
        if entity is None:
            print(f"  [ERROR] entity_id={step.target_id} not found in WSG")
            return False

        cx, cy = entity.center
        sx, sy = wsg.screenshot_to_screen(cx, cy)
        glyph = ACTION_GLYPH.get(step.action, '[????] ')
        print(f"  {glyph} Step {step.step}: {step.description} "
              f"-> screen({sx},{sy})")

        try:
            if step.action == 'click':
                self.pyautogui.click(sx, sy)
            elif step.action == 'type':
                if step.value:
                    self.pyautogui.click(sx, sy)
                    time.sleep(0.1)
                    self.pyautogui.typewrite(step.value, interval=0.05)
                else:
                    self.pyautogui.click(sx, sy)
            elif step.action == 'tab':
                self.pyautogui.press('tab')
            elif step.action == 'enter':
                self.pyautogui.press('enter')
            elif step.action == 'key':
                key_str = step.value.lower().replace('+', '+')
                if '+' in key_str:
                    mods = key_str.split('+')
                    self.pyautogui.hotkey(*mods)
                else:
                    self.pyautogui.press(key_str)
            elif step.action == 'wait':
                time.sleep(float(step.value or 0.5))
            else:
                print(f"  [ERROR] Unknown action: {step.action}")
                return False
        except self.pyautogui.FailSafeException:
            print("  [STOP] FAILSAFE triggered, stopped")
            return False
        except Exception as e:
            print(f"  [ERROR] Execution failed: {e}")
            return False

        time.sleep(0.3)
        return True

    def execute_plan(self, plan: list[SubGoal], wsg: WorldStateGraph,
                     env=None) -> tuple[bool, list[bool]]:
        """Execute all steps in a plan. Returns (all_ok, per_step_results).

        When env is provided, refreshes WSG after each step to keep
        entity IDs valid.
        """
        results = []
        current_wsg = wsg
        for step in plan:
            ok = self.execute_step(step, current_wsg, env)
            results.append(ok)
            if not ok:
                print(f"  [FAIL] Step {step.step} failed, aborting")
                return False, results
            # Refresh WSG after each successful step
            if env is not None:
                current_wsg = env.reset()
                # Remap remaining steps' target_ids
                for remaining in plan[step.step:]:
                    orig = wsg.get_entity_by_id(remaining.target_id)
                    if orig:
                        new_e = current_wsg.get_entity_by_text(orig.text)
                        if new_e:
                            remaining.target_id = new_e.id
        return True, results
