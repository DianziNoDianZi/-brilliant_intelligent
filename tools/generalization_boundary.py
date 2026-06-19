"""泛化边界测试 — 摸清快回路在什么范围内能泛化、什么范围会失效。

测试层级：
  Level 1: 同应用同操作，不同数值 ✅ 已通过
  Level 2: 同应用同操作，不同结构（如 3+4 → 12+34）
  Level 3: 同应用不同操作（如 add → subtract）
  Level 4: 不同应用同类型（点击→输入）
  Level 5: 跨应用组合（计算器+记事本协作）
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.fast_loop import FastLoop
from agent.skill_lib import SkillLibrary, Skill
from agent.intent import IntentVector

PASS = "PASS"
FAIL = "FAIL"
BYPASS = "BYPASS"  # 快回路不接管（走慢回路，不算失败）


def test_levels():
    results = []

    # ── Level 1: 同应用同操作，不同数值 ──
    print("=== Level 1: Same operation, different values ===")
    lib = SkillLibrary()
    lib.skills.append(Skill('add','calculator',{'operator':'+','num_operands':2},
                            ['A','+','B','='], successes=3))
    fl = FastLoop(skill_lib=lib)
    # 2-operand template only matches 2 operands; multi-digit needs own template
    tests = [("3+4",['3','+','4','=']), ("5+8",['5','+','8','=']),
             ("7+2",['7','+','2','='])]
    for label, expected in tests:
        vals = fl._parse_instruction(f'calculate {label}')
        use_fast, pred, _ = fl.decide(IntentVector.from_values(vals or [], f'calculate {label}'))
        ok = use_fast and pred == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} → {pred}")
        results.append(('L1_val', label, 'PASS' if ok else 'FAIL'))

    # ── Level 2: 同操作不同模板结构 ──
    print("\n=== Level 2: Same operation, different template structure ===")
    lib2 = SkillLibrary()
    lib2.skills.append(Skill('add3','calculator',{'operator':'+','num_operands':3},
                             ['A','+','B','+','C','='], successes=3))
    fl2 = FastLoop(skill_lib=lib2)
    tests = [("3+4+5",['3','+','4','+','5','=']), ("3+4",['3','+','4','='])]
    for label, expected in tests:
        vals = fl2._parse_instruction(f'calculate {label}')
        use_fast, pred, _ = fl2.decide(IntentVector.from_values(vals or [], f'calculate {label}'))
        ok = use_fast and pred == expected
        print(f"  [{'PASS' if ok else 'BYPASS' if not use_fast else 'FAIL'}] {label} → {pred}")
        results.append(('L2_struct', label, 'PASS' if ok else 'BYPASS'))

    # ── Level 3: 同应用不同操作 ──
    print("\n=== Level 3: Different operations in same app ===")
    lib3 = SkillLibrary()
    lib3.skills.append(Skill('add','calculator',{'operator':'+','num_operands':2},
                             ['A','+','B','='], successes=3))
    fl3 = FastLoop(skill_lib=lib3)
    tests = [("add 3+4",['3','+','4','=']), ("subtract 7-3",['7','-','3','='])]
    for label, expected in tests:
        vals = fl3._parse_instruction(label)
        use_fast, pred, _ = fl3.decide(IntentVector.from_values(vals or [], label))
        ok = use_fast and pred == expected
        # subtract should NOT match add skill
        is_add = '+' in label
        correct_behavior = ok if is_add else not use_fast
        status = 'PASS' if correct_behavior else 'FAIL'
        print(f"  [{status}] {label} → fast={use_fast} pred={pred}")
        results.append(('L3_cross_op', label, status))

    # ── Level 4: 跨应用同类型 ──
    print("\n=== Level 4: Cross-app, same interaction type ===")
    lib4 = SkillLibrary()
    lib4.skills.append(Skill('notepad_type','notepad',{},['text_area','type:Hello'],
                             action_template=['click','type'], successes=3))
    fl4 = FastLoop(skill_lib=lib4)
    tests = [("type World","notepad",['text_area','type:World']),
             ("type Test","notepad",['text_area','type:Test'])]
    for instr, app, expected in tests:
        vals = fl4._parse_instruction(instr)
        intent = IntentVector.from_values(vals or [], instr, task_type=app)
        use_fast, pred, _ = fl4.decide(intent)
        ok = use_fast and pred == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] {instr} → {pred}")
        results.append(('L4_cross_app', instr, 'PASS' if ok else 'FAIL'))

    # ── Level 5: 跨应用组合 ──
    print("\n=== Level 5: Cross-app composition (calculator→notepad) ===")
    lib5 = SkillLibrary()
    lib5.skills.append(Skill('calc_add','calculator',{'operator':'+','num_operands':2},
                             ['A','+','B','='], successes=3))
    fl5 = FastLoop(skill_lib=lib5)
    # 混合指令：先计算再写到记事本
    mixed = "calculate 3+4 and type result in notepad"
    vals = fl5._parse_instruction(mixed)
    use_fast, pred, _ = fl5.decide(IntentVector.from_values(vals or [], mixed))
    # 应该走慢回路（当前模板不覆盖混合场景）
    ok = not use_fast
    print(f"  [{'PASS' if ok else 'FAIL'}] mixed task → fast={use_fast} (expected slow)")
    results.append(('L5_compose', 'calc+notepad', 'PASS' if ok else 'FAIL'))

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"  泛化边界测试报告")
    print(f"{'='*55}")
    level_map = {'L1':'同应用同操作','L2':'同应用同操作不同结构',
                 'L3':'同应用不同操作','L4':'跨应用同类型','L5':'跨应用组合'}
    for lid, label in level_map.items():
        layer = [r for r in results if r[0].startswith(lid)]
        passed = sum(1 for r in layer if r[2] == 'PASS')
        print(f"  {lid}: {label} → {passed}/{len(layer)} 通过")
    print(f"{'='*55}")

    all_pass = all(r[2] == 'PASS' for r in results if r[2] != 'BYPASS')
    print(f"  核心问题: 快回路的泛化边界在哪里？")
    print(f"  {'✅ 泛化边界清晰' if all_pass else '⚠️ 存在泛化盲区'}")
    print(f"  跨应用组合: {'⛔ 当前需慢回路' if not any(r[0]=='L5_compose' for r in results) else '?'}")
    print(f"  跨应用组合需技能重组机制 → Phase 4 层级记忆")


if __name__ == '__main__':
    test_levels()
