"""泛化能力验证标准。

测试快回路是否真正学会了"操作模式"而非"记忆答案"。

验证流程：
1. 用单一任务生成技能（如 "type Hello"）
2. 用变体任务测试泛化（如 "type World"）
3. 通过条件：变体任务走快回路且输出正确

跨应用验证：
- 计算器泛化 ✅ 已验证（12+34 模板 → 56+78 成功）
- 记事本泛化 ⬜ 待验证
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.fast_loop import FastLoop
from agent.skill_lib import SkillLibrary, Skill
from agent.intent import IntentVector


def test_calculator_generalization():
    """验证计算器泛化：模板 A+B+C+D= → 任意四位数加法"""
    lib = SkillLibrary()
    # Add the generated skill
    lib.skills.append(Skill(
        'gen_+', 'calculator',
        {'operator': '+', 'num_operands': 4},
        ['A', 'B', '+', 'C', 'D', '='],
        successes=3,
    ))
    fl = FastLoop(skill_lib=lib)

    tests = [
        ("calculate 12+34", ['1','2','+','3','4','=']),
        ("calculate 56+78", ['5','6','+','7','8','=']),
        ("calculate 90+12", ['9','0','+','1','2','=']),
        ("calculate 34+56", ['3','4','+','5','6','=']),
    ]
    print("=== Calculator Generalization (A+B+C+D=) ===")
    all_pass = True
    for instruction, expected in tests:
        vals = fl._parse_instruction(instruction)
        intent = IntentVector.from_values(vals or [], instruction)
        use_fast, predicted, skill_name = fl.decide(intent)
        correct = predicted == expected if use_fast else False
        status = "PASS" if correct else ("SLOW" if not use_fast else "FAIL")
        if status != "PASS":
            all_pass = False
        print(f"  [{status}] {instruction}")
        print(f"         expected={expected}")
        if use_fast:
            print(f"         got={predicted}")
        else:
            print(f"         (slow path - no fast match)")
    print(f"  => {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def test_notepad_generalization():
    """验证记事本泛化：模板 [text_area, type:VALUE] → 不同文本"""
    lib = SkillLibrary()
    # Add a notepad type skill
    lib.skills.append(Skill(
        'notepad_type', 'notepad',
        {},
        ['text_area', 'type:Hello'],
        action_template=['click', 'type'],
        successes=3,
    ))
    fl = FastLoop(skill_lib=lib)

    tests = [
        ("type Hello", ['text_area', 'type:Hello']),
        ("type World", ['text_area', 'type:World']),
        ("type FooBar", ['text_area', 'type:FooBar']),
    ]
    print("\n=== Notepad Generalization (type VARIABLE) ===")
    all_pass = True
    for instruction, expected in tests:
        vals = fl._parse_instruction(instruction)
        intent = IntentVector.from_values(
            vals or [], instruction, task_type='notepad')
        use_fast, predicted, skill_name = fl.decide(intent)
        correct = predicted == expected if use_fast else False
        status = "PASS" if correct else ("SLOW" if not use_fast else "FAIL")
        if status != "PASS":
            all_pass = False
        print(f"  [{status}] {instruction}")
        print(f"         expected={expected}")
        if use_fast:
            print(f"         got={predicted}")
    print(f"  => {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == '__main__':
    calc_ok = test_calculator_generalization()
    note_ok = test_notepad_generalization()
    print(f"\n{'='*50}")
    print(f"  Calculator generalization: {'PASS' if calc_ok else 'FAIL'}")
    print(f"  Notepad generalization:    {'PASS' if note_ok else 'FAIL'}")
    print(f"  Cross-app generalization:  {'CONFIRMED' if calc_ok and note_ok else 'PARTIAL'}")
    print(f"{'='*50}")
