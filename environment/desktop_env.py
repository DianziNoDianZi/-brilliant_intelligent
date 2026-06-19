"""Desktop environment — screenshot, window detection, WSG generation, actions.

Uses Windows UIAutomation API for element detection (no model downloads).
Falls back to EasyOCR if UIA fails.
"""

from __future__ import annotations
import time
from typing import Optional
import numpy as np
from PIL import ImageGrab

from agent.wsg import WorldStateGraph, WSGEntity

APP_WINDOWS = {
    'calculator': ['计算器', 'Calculator'],
    'notepad': ['记事本', 'Notepad', '无标题 - 记事本'],
    'wordpad': ['WordPad', '写字板'],
}


class DesktopEnv:
    """Windows desktop environment wrapper using UIAutomation."""

    def __init__(self, app_name='calculator'):
        self.app_name = app_name.lower()
        self.window = None
        self._uia_offset = (0, 0)  # (offset_x, offset_y) for coordinate mapping
        self._init_window()
        print("[DesktopEnv] Initialized")

    def _init_window(self):
        import pygetwindow as gw
        app_lower = self.app_name.lower()
        for w in gw.getAllWindows():
            if not w.title.strip():
                continue
            wt = w.title.lower()
            if app_lower in wt or ('calc' in wt or '计算' in wt):
                if app_lower == 'calculator' or app_lower in wt or 'calc' in wt:
                    self.window = w
                    break
            if app_lower == 'notepad' and ('notepad' in wt or '记事本' in wt):
                self.window = w
                break
            if app_lower == 'wordpad' and ('wordpad' in wt or '写字板' in wt):
                self.window = w
                break
        if self.window:
            safe_title = self.window.title.encode('utf-8', errors='replace').decode('utf-8')
            print(f"[DesktopEnv] Found window: '{safe_title}'")
            try:
                self.window.activate()
            except Exception:
                pass
        else:
            print(f"[DesktopEnv] Window '{self.app_name}' not found")

    def _find_uia_window(self, control):
        """Find the top-level window matching our target app."""
        import uiautomation as uia
        titles = APP_WINDOWS.get(self.app_name,
                                 [self.app_name.capitalize()])

        def match(ctrl):
            try:
                name = ctrl.Name
                return any(t in name for t in titles)
            except Exception:
                return False

        root = uia.GetRootControl()
        # Search for the window (depth 1)
        for child in root.GetChildren():
            if match(child):
                return child
        return None

    def _detect_elements_uia(self) -> list[WSGEntity]:
        """Detect UI elements using UIAutomation API."""
        import uiautomation as uia
        entities = []
        next_id = 1

        ctrl = self._find_uia_window(uia)
        if ctrl is None:
            return entities

        # Store UIA window offset for coordinate mapping
        try:
            win_rect = ctrl.BoundingRectangle
            win_x, win_y = win_rect.left, win_rect.top
            self._uia_offset = (win_x, win_y)
        except Exception:
            win_x, win_y = self._uia_offset

        def walk(control, depth=0):
            nonlocal next_id
            if depth > 5:
                return
            try:
                name = control.Name
                rect = control.BoundingRectangle
                ctrl_type = control.ControlTypeName

                if name and rect.right > rect.left and rect.bottom > rect.top:
                    # Map to entity type
                    # Use a meaningful name for unnamed input areas
                    if not name.strip() and ('edit' in ctrl_type.lower() or 'document' in ctrl_type.lower()):
                        name = 'text_area'
                    if 'button' in ctrl_type.lower():
                        etype = 'button'
                    elif 'edit' in ctrl_type.lower() or 'document' in ctrl_type.lower():
                        etype = 'input'
                    elif 'text' in ctrl_type.lower() or 'static' in ctrl_type.lower():
                        etype = 'text'
                    elif 'checkbox' in ctrl_type.lower() or 'radio' in ctrl_type.lower():
                        etype = 'button'
                    else:
                        etype = 'text'

                    # Convert to screenshot coordinates
                    x1 = rect.left - win_x
                    y1 = rect.top - win_y
                    x2 = rect.right - win_x
                    y2 = rect.bottom - win_y

                    entities.append(WSGEntity(
                        id=next_id,
                        type=etype,
                        text=name.strip(),
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        properties={'ctrl_type': ctrl_type},
                    ))
                    next_id += 1
            except Exception:
                pass

            try:
                for child in control.GetChildren():
                    walk(child, depth + 1)
            except Exception:
                pass

        walk(ctrl)
        return entities

    def _capture_window(self) -> tuple[np.ndarray, int, int]:
        """Capture screenshot. Returns (image, offset_x, offset_y)."""
        import uiautomation as uia
        ctrl = self._find_uia_window(uia)
        if ctrl is not None:
            try:
                ctrl.SetFocus()
                time.sleep(0.2)
                rect = ctrl.BoundingRectangle
                left, top = rect.left, rect.top
                right, bottom = rect.right, rect.bottom
                if right > left and bottom > top:
                    img = ImageGrab.grab(bbox=(left, top, right, bottom))
                    return np.array(img), left, top
            except Exception:
                pass
        if self.window:
            try:
                self.window.activate()
                time.sleep(0.2)
                left, top = self.window.left, self.window.top
                right, bottom = self.window.right, self.window.bottom
                if right > left and bottom > top:
                    img = ImageGrab.grab(bbox=(left, top, right, bottom))
                    return np.array(img), left, top
            except Exception:
                pass
        img = ImageGrab.grab()
        return np.array(img), 0, 0

    def reset(self) -> WorldStateGraph:
        """Capture state and return WSG via UIAutomation detection."""
        img, ox, oy = self._capture_window()
        # Try UIA first, fall back to empty WSG
        entities = self._detect_elements_uia()
        if not entities:
            # If the app window wasn't found, still return an empty WSG
            print("[WARN] No UI elements detected via UIAutomation")

        wsg = WorldStateGraph(entities, screenshot=img,
                              window_offset=(ox, oy))
        wsg.compute_spatial_relations()
        print(f"  WSG: {len(wsg.entities)} entities, "
              f"{len(wsg.relations)} relations")
        return wsg

    def step(self, action: str, **kwargs) -> WorldStateGraph:
        import pyautogui
        try:
            if action == 'click':
                pyautogui.click(kwargs.get('x', 0), kwargs.get('y', 0))
            elif action == 'type':
                pyautogui.typewrite(kwargs.get('text', ''), interval=0.05)
            elif action == 'tab':
                pyautogui.press('tab')
            elif action == 'enter':
                pyautogui.press('enter')
            elif action == 'wait':
                time.sleep(kwargs.get('seconds', 0.5))
        except pyautogui.FailSafeException:
            print("[STOP] FAILSAFE triggered")
            raise
        except Exception as e:
            print(f"[ERROR] Action failed: {e}")

        time.sleep(0.3)
        return self.reset()

    def close(self):
        pass
