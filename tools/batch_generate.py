"""批量技能生成脚本。

输入：tasks/calculator_tasks.json（任务描述列表）
流程：LLM生成序列 → 内模拟验证 → 冲突仲裁 → 入库
全程不碰桌面，在想象中完成。
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.skill_generator import SkillGenerator
from agent.simulator import Simulator
from agent.world_model import EnsembleWorldModel
from agent.wsg_encoder import STATE_DIM
from agent.skill_lib import SkillLibrary, Skill
from agent.planner import Planner
from agent.wsg import WorldStateGraph, WSGEntity


def make_dummy_wsg(task_type='calc'):
    """Create a minimal WSG for simulation (only needed for entity count)."""
    wsg = WorldStateGraph()
    if task_type == 'calc':
        for txt, x in [('1',0),('2',30),('3',60),('4',90),('5',120),
                        ('6',150),('7',180),('8',210),('9',240),('0',270),
                        ('+',300),('-',330),('*',360),('/',390),('=',420)]:
            wsg.add_entity(WSGEntity(id=len(wsg.entities)+1, type='button',
                                      text=txt, bbox=[x,0,x+25,25]))
    else:
        wsg.add_entity(WSGEntity(id=1, type='input', text='text_area',
                                  bbox=[0,0,500,300]))
    wsg.compute_spatial_relations()
    return wsg


def batch_generate(task_file: str, output_path: str, ollama_model: str = 'qwen2.5:7b'):
    """Run batch generation."""
    with open(task_file) as f:
        tasks = json.load(f)
    print(f"[BATCH] Loading {len(tasks)} tasks from {task_file}")

    # Setup
    lib = SkillLibrary(storage_path=output_path)
    wm = EnsembleWorldModel(STATE_DIM, 5, ensemble_size=3, hidden_dim=32)
    sim = Simulator(wm, device='cpu', confidence_threshold=0.3)
    sim._training_steps = 15
    planner = Planner(backend='ollama', model=ollama_model)
    gen = SkillGenerator(planner, sim, lib)

    stats = {'generated': 0, 'skipped': 0, 'failed': 0, 'conflict': 0}

    for i, task in enumerate(tasks, 1):
        tid = task['id']
        instruction = task['instruction']
        task_type = task['type']
        print(f"\n[{i}/{len(tasks)}] {tid}: {instruction}")

        # Check if this task ID is already in the library
        existing = [s for s in lib.skills if s.name == tid]
        if existing:
            print(f"  [SKIP] Already exists: {existing[0].value_template}")
            stats['skipped'] += 1
            continue

        # Generate via imagination
        wsg = make_dummy_wsg(task_type)
        try:
            skill = gen.generate(instruction, wsg, task_type=task_type)
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            stats['failed'] += 1
            continue

        if skill is None:
            print(f"  [FAIL] Generation returned None")
            stats['failed'] += 1
            continue

        # Force-add: rename to task ID and save
        skill.name = tid
        skill.task_type = 'calculator' if task_type == 'calc' else 'notepad'
        skill.count = skill.successes  # ensure count matches successes for sorting
        # Remove old skill with same name if exists
        lib.skills = [s for s in lib.skills if s.name != tid]
        lib.skills.append(skill)
        lib._save()
        print(f"  [OK] {skill.name}: {skill.value_template}")
        stats['generated'] += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"  Batch Generation Complete")
    print(f"{'='*50}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  Total skills in library: {len(lib.skills)}")
    print(f"  Compiled: {len(lib.compiled_skills)}")
    print(f"{'='*50}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', default='D:/briliant_intelligent/tasks/calculator_tasks.json')
    parser.add_argument('--output', default='D:/d/tmp/skills.json')
    parser.add_argument('--llm', default='qwen2.5:7b')
    args = parser.parse_args()

    batch_generate(args.tasks, args.output, args.llm)
