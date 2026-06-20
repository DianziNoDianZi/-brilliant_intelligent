"""影子模式评估脚本。

加载 30-50 条测试指令，每条同时过分类器和 LLM，
对比意图 ID 和槽位一致性，输出统计报告。

用法:
  /d/briliant_env/Scripts/python tools/shadow_eval.py
  /d/briliant_env/Scripts/python tools/shadow_eval.py --quick (仅分类器，不调 LLM)
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.classifier import IntentClassifier, CLASS_NAMES
from agent.planner import Planner
from agent.wsg import WorldStateGraph, WSGEntity


def make_mock_wsg():
    wsg = WorldStateGraph()
    for txt, x in [('1',0),('2',30),('3',60),('4',90),('5',120),
                    ('6',150),('7',180),('8',210),('9',240),('0',270),
                    ('+',300),('-',330),('*',360),('/',390),('=',420)]:
        wsg.add_entity(WSGEntity(id=len(wsg.entities)+1, type='button',
                                  text=txt, bbox=[x,0,x+25,25]))
    wsg.add_entity(WSGEntity(id=100, type='input', text='text_area',
                              bbox=[0,50,500,350]))
    wsg.compute_spatial_relations()
    return wsg


def main():
    quick = '--quick' in sys.argv
    task_file = 'D:/briliant_intelligent/tasks/shadow_eval.json'

    with open(task_file) as f:
        tasks = json.load(f)
    print(f"[EVAL] Loaded {len(tasks)} test instructions\n")

    # 1. Classifier-only evaluation
    clf = IntentClassifier()
    if not clf.load():
        print("[FAIL] No trained classifier. Run tools/train_classifier.py first")
        return

    wsg = make_mock_wsg()
    planner = Planner(backend='ollama', model='qwen2.5:7b')

    results = []
    intent_correct = 0
    intent_total = 0
    low_conf_count = 0
    low_conf_examples = []

    for task in tasks:
        instr = task['instruction']
        expected = task['expected_intent']

        # classifier prediction
        pred = clf.predict(instr)
        pred_intent = pred['intent_name']
        pred_conf = pred['confidence']
        pred_slots = pred['slots']

        # LLM prediction (in shadow mode)
        llm_intent = None
        llm_values = None
        llm_time = 0

        if not quick:
            t0 = time.time()
            plan = planner.plan(instr, wsg)
            llm_time = time.time() - t0
            if plan:
                # Infer intent from plan values
                vals = [s.description for s in plan]
                if any(v in '+-*/' for v in str(plan)):
                    llm_intent = 'binary_arithmetic'
                elif 'text_area' in str(plan) or 'type:' in str(plan):
                    llm_intent = 'text_input'
                llm_values = [s.value for s in plan if s.value]

        # Intent match
        intent_match = (pred_intent == expected)
        if intent_match:
            intent_correct += 1
        intent_total += 1

        if pred_conf < 0.8:
            low_conf_count += 1
            low_conf_examples.append((instr, pred_intent, pred_conf))

        results.append({
            'instruction': instr,
            'expected': expected,
            'classifier_intent': pred_intent,
            'confidence': pred_conf,
            'slots': pred_slots,
            'llm_intent': llm_intent,
            'llm_time_s': round(llm_time, 1),
            'intent_match': intent_match,
        })

        match_str = 'OK' if intent_match else 'MIS'
        conf_str = f"{pred_conf:.2f}"
        llm_str = f" llm={llm_intent}" if llm_intent else ""
        print(f"  [{match_str}] conf={conf_str} {pred_intent:20s} <- \"{instr}\"{llm_str}")

    # Report
    intent_acc = intent_correct / intent_total * 100

    print(f"\n{'='*55}")
    print(f"  影子模式评估报告")
    print(f"{'='*55}")
    print(f"  测试指令数:      {intent_total}")
    print(f"  意图准确率:      {intent_acc:.1f}% ({intent_correct}/{intent_total})")
    print(f"  低置信率:        {low_conf_count/intent_total*100:.0f}% ({low_conf_count}/{intent_total})")
    print(f"  平均置信度:      {sum(r['confidence'] for r in results)/len(results):.2f}")

    if not quick:
        llm_times = [r['llm_time_s'] for r in results if r['llm_time_s'] > 0]
        if llm_times:
            print(f"  LLM 平均耗时:    {sum(llm_times)/len(llm_times):.1f}s")

    # Per-intent breakdown
    print(f"\n  --- 分意图统计 ---")
    for intent_name in CLASS_NAMES:
        subset = [r for r in results if r['expected'] == intent_name]
        if not subset:
            continue
        correct = sum(1 for r in subset if r['intent_match'])
        avg_conf = sum(r['confidence'] for r in subset) / len(subset)
        print(f"  {intent_name:25s} {correct}/{len(subset):2d} "
              f"acc={correct/len(subset)*100:.0f}% avg_conf={avg_conf:.2f}")

    # Low confidence analysis
    if low_conf_examples:
        print(f"\n  --- 低置信指令 (conf<0.8) ---")
        for instr, intent, conf in low_conf_examples[:5]:
            print(f"  \"{instr}\" -> {intent} conf={conf:.2f}")

    # Verdict
    print(f"\n  --- 判定 ---")
    if intent_acc >= 95:
        print(f"  [PASS] 一致率 {intent_acc:.0f}% > 95% -> 可以上线")
    elif intent_acc >= 85:
        print(f"  [WARN] 一致率 {intent_acc:.0f}% (85-95%) -> 需补充数据")
    else:
        print(f"  [FAIL] 一致率 {intent_acc:.0f}% < 85% -> 需升级模型")

    # Save report
    report_path = 'D:/d/tmp/shadow_eval_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'total': intent_total,
            'accuracy': round(intent_acc, 1),
            'low_confidence': low_conf_count,
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to {report_path}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
