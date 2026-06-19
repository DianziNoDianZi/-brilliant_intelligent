"""跨应用迁移验证。

用 L3 抽象模板将技能迁移到第三个应用。
测试流程：
1. 打开目标应用（如 WordPad、浏览器搜索框等）
2. 系统检测 WSG，匹配 L3 抽象模板
3. 通过模板找到关联的已知技能（记事本 type_text）
4. 适配到新应用的 UI 坐标
5. 执行并验证
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.skill_lib import SkillLibrary
from agent.abstract_templates import AbstractLibrary
from agent.intent import IntentVector
from environment.desktop_env import DesktopEnv


def try_migrate(app_name: str, instruction: str):
    """尝试将已知技能迁移到新应用。

    1. 检测新应用的 WSG
    2. 匹配 L3 抽象模板
    3. 找到关联的 L2 技能
    4. 适配执行
    """
    print(f"[MIGRATE] Target: {app_name}")
    print(f"[MIGRATE] Task: {instruction}")

    # 1. Load L2 + L3
    lib = SkillLibrary()
    al = AbstractLibrary()
    al.extract_all(lib)

    # 2. Detect new app
    import traceback
    try:
        env = DesktopEnv(app_name=app_name)
        wsg = env.reset()
    except Exception as e:
        err_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
        print(f"[FAIL] Cannot open {app_name}: {err_msg}")
        traceback.print_exc()
        return

    print(f"  WSG: {len(wsg.entities)} entities")

    # 3. Match L3 templates based on WSG content
    detected_tags = []
    for e in wsg.entities:
        if e.type == 'input':
            detected_tags.append('text_input')
        if 'edit' in e.text.lower() or 'input' in e.text.lower():
            detected_tags.append('text_input')
    detected_tags = list(set(detected_tags))

    if not detected_tags:
        # Fallback: try the instruction
        if 'type' in instruction.lower():
            detected_tags = ['text_input']

    print(f"  Detected tags: {detected_tags}")

    # 4. Find matching L3 templates
    matches = al.find_matches(detected_tags)
    if not matches:
        print(f"[FAIL] No L3 template matches tags: {detected_tags}")
        env.close()
        return

    for t in matches:
        print(f"  L3 match: {t.template_id} ({t.name}) -> {len(t.concrete_skills)} concrete skills")

    # 5. Try to execute using the best matching skill
    for t in matches:
        for skill_name in t.concrete_skills:
            skill = next((s for s in lib.skills if s.name == skill_name), None)
            if skill and skill.is_compiled:
                print(f"\n  Trying skill: {skill.name}")
                print(f"  Template: {skill.value_template}")
                print(f"  Source app: {skill.app}")

                # Build plan and execute for real
                from agent.executor import Executor
                from agent.planner import _values_to_plan
                from agent.fast_loop import FastLoop

                fl = FastLoop(skill_lib=lib)
                predicted = fl._parse_instruction(instruction)
                if predicted:
                    plan = _values_to_plan(predicted, wsg)
                    if plan:
                        executor = Executor()
                        ok, _ = executor.execute_plan(plan, wsg, env=env)
                        if ok:
                            print(f"  [EXEC] Success on {app_name}!")
                        else:
                            print(f"  [EXEC] Failed on {app_name}")
                break
        break  # Try only the first matched template

    env.close()


if __name__ == '__main__':
    app = sys.argv[1] if len(sys.argv) > 1 else 'notepad'
    instr = sys.argv[2] if len(sys.argv) > 2 else 'type Hello'
    try_migrate(app, instr)
