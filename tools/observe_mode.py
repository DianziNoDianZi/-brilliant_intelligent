"""被动观察模式 — 记录用户操作 + WSG 变化。

运行方式：
  /d/briliant_env/Scripts/python tools/observe_mode.py

然后在文件管理器（或其他应用）中正常操作。
每次点击或按键，系统会自动记录 WSG 变化。
按 Ctrl+C 停止观察。
"""

import sys, os, time, json, signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from agent.wsg_encoder import encode_wsg
from environment.desktop_env import DesktopEnv


LOG_PATH = "D:/briliant_intelligent/data/observation_log.jsonl"
POLL_INTERVAL = 0.5  # seconds between WSG snapshots


def main():
    print("=" * 55)
    print("  被动观察模式")
    print("=" * 55)
    print(f"  日志: {LOG_PATH}")
    print(f"  在文件管理器中操作，系统自动记录 WSG 变化")
    print(f"  按 Ctrl+C 停止")
    print("-" * 55)

    env = DesktopEnv(app_name='calculator')  # dummy init to get UIA access
    running = True

    def stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    prev_wsg = None
    prev_wsg_vec = None
    last_action = "initial"
    record_count = 0

    while running:
        try:
            # Capture current WSG from the active window
            # UIAutomation detects the foreground window automatically
            wsg = env.reset()
            current_vec = encode_wsg(wsg)

            if prev_wsg_vec is not None:
                # Detect significant change
                diff = np.mean(np.abs(current_vec - prev_wsg_vec))
                if diff > 0.01:  # WSG changed significantly
                    record = {
                        "timestamp": time.time(),
                        "action": last_action,
                        "entity_count_before": len(prev_wsg.entities),
                        "entity_count_after": len(wsg.entities),
                        "change_magnitude": round(float(diff), 4),
                        "app": detect_active_app(wsg),
                    }
                    with open(LOG_PATH, "a") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    record_count += 1
                    print(f"  [{record_count:3d}] {record['app']:15s} "
                          f"变化={record['change_magnitude']:.4f} "
                          f"实体={record['entity_count_before']}->{record['entity_count_after']}")

            prev_wsg = wsg
            prev_wsg_vec = current_vec
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  [WARN] {e}")
            time.sleep(1)

    print(f"\n{'=' * 55}")
    print(f"  观察结束")
    print(f"  记录: {record_count} 次 WSG 变化")
    print(f"  日志: {LOG_PATH}")
    print(f"{'=' * 55}")


def detect_active_app(wsg) -> str:
    """Try to identify the active application from WSG content."""
    titles = set(e.text.lower() for e in wsg.entities if e.text)
    all_text = ' '.join(titles)
    if 'explorer' in all_text or '资源管理器' in all_text or '此电脑' in all_text:
        return 'explorer'
    if 'calc' in all_text or '计算器' in all_text:
        return 'calculator'
    if 'notepad' in all_text or '记事本' in all_text:
        return 'notepad'
    return 'unknown'


if __name__ == '__main__':
    main()
