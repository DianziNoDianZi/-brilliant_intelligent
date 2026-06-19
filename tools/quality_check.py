"""质量抽检：5 个典型任务走完整生成管线。

每个任务：LLM提案 → 内模拟验证 → 统计置信度 → 入技能库
生成完成后手动在桌面验证。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.wsg import WorldStateGraph, WSGEntity
from agent.skill_generator import SkillGenerator
from agent.simulator import Simulator
from agent.world_model import EnsembleWorldModel
from agent.wsg_encoder import STATE_DIM, encode_wsg
from agent.skill_lib import SkillLibrary
from agent.planner import Planner
from environment.desktop_env import DesktopEnv
import torch
import numpy as np

TEST_TASKS = [
    ("calc_12+34", "calculate 12+34", "calculator"),
    ("calc_8*7-3", "calculate 8*7-3", "calculator"),
    ("calc_100_div_4", "calculate 100/4", "calculator"),
    ("notepad_type_save", "type 'hello world' and save", "notepad"),
    ("notepad_multi_line", "type two lines: first line and second line", "notepad"),
]


def setup_simulator():
    wm = EnsembleWorldModel(STATE_DIM, 5, ensemble_size=3, hidden_dim=32)
    sim = Simulator(wm, device='cpu', confidence_threshold=0.3)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    dws = WorldStateGraph()
    for i in range(10):
        dws.add_entity(WSGEntity(id=i, type='button', text=str(i), bbox=[0,0,10,10]))
    for _ in range(50):
        s = encode_wsg(dws)
        ns = encode_wsg(dws)
        st = torch.FloatTensor(s).unsqueeze(0)
        nt = torch.FloatTensor(ns).unsqueeze(0)
        at = torch.zeros(1, 5); at[0, np.random.randint(0,5)] = 1
        preds, _ = wm(st, at)
        losses = wm.compute_loss(preds, nt, st)
        for loss in losses:
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward(); opt.step()
    sim._training_steps = 20
    return wm, sim


def main():
    print("=" * 60)
    print("  Quality Check: 5 Tasks")
    print("=" * 60)

    wm, sim = setup_simulator()
    lib = SkillLibrary(storage_path='/d/tmp/quality_results.json')
    planner = Planner(backend='ollama', model='qwen2.5:7b')
    gen = SkillGenerator(planner, sim, lib)

    output = []

    for name, instruction, app in TEST_TASKS:
        print(f"\n{'-' * 50}")
        print(f"  Task: {name}")
        print(f"  Instruction: {instruction}")
        print(f"  App: {app}")
        print(f"{'-' * 50}")

        env = DesktopEnv(app_name=app)
        wsg = env.reset()
        print(f"  WSG: {len(wsg.entities)} entities")

        skill = gen.generate(instruction, wsg)
        env.close()

        entry = {'name': name, 'instruction': instruction, 'app': app}
        if skill:
            entry['skill_name'] = skill.name
            entry['template'] = skill.value_template
            entry['compiled'] = skill.is_compiled
            entry['status'] = 'PASS'
            print(f"  -> [PASS] {skill.name}: {skill.value_template}")
        else:
            entry['status'] = 'FAIL'
            print(f"  -> [FAIL] Rejected: {gen.stats.get('rejected', 0)}")
            entry['rejected_count'] = gen.stats.get('rejected', 0)
        output.append(entry)

    # Summary
    print(f"\n{'=' * 60}")
    passed = sum(1 for o in output if o['status'] == 'PASS')
    print(f"  Results: {passed}/{len(output)} passed")
    for o in output:
        print(f"  {'[OK]' if o['status']=='PASS' else '[FAIL]'} {o['name']}: "
              f"{o.get('template','N/A')}")
    all_pass = passed >= 4
    print(f"\n  Verdict: {'BATCH GENERATION READY' if all_pass else 'FIX BEFORE BATCH'}")
    print(f"{'=' * 60}")

    # Save report
    import json
    with open('/d/tmp/quality_report.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Report saved to /d/tmp/quality_report.json")


if __name__ == '__main__':
    main()
