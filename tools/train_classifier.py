"""训练意图分类器。

从 data/intent_log.jsonl 加载数据，训练 MLP 分类器，评估准确率。

用法:
  /d/briliant_env/Scripts/python tools/train_classifier.py
  /d/briliant_env/Scripts/python tools/train_classifier.py --min-data 30
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from agent.classifier import IntentClassifier, CLASS_NAMES, tokenize
from agent.intent_log import load_log


def main():
    # 加载数据
    records = load_log()
    print(f"[DATA] Loaded {len(records)} records from intent_log.jsonl")

    if len(records) < 5:
        print("[SKIP] Too few records (<5). Run some tasks first.")
        return

    # 提取文本和标签
    texts = []
    labels = []
    for r in records:
        if not r.get('success', True):
            continue
        instr = r.get('instruction', '')
        vals = r.get('plan_values', [])
        if not instr:
            continue
        texts.append(instr)

        # 优先使用 l3_template_id（精确标注），否则从 plan_values 推断
        tid = r.get('l3_template_id', '')
        if tid in ('save_file', 'save_document'):
            labels.append(2)
        elif tid == 'input_then_save':
            labels.append(3)
        elif tid == 'cross_app_chain':
            labels.append(4)
        elif any(v in '+-*/' for v in vals):
            labels.append(0)  # binary_arithmetic
        elif 'text_area' in vals or any(v.startswith('type:') for v in vals):
            if any('save' in v.lower() for v in vals if isinstance(v, str)):
                labels.append(3)  # input_then_save
            else:
                labels.append(1)  # text_input
        else:
            labels.append(0)

    print(f"[DATA] {len(texts)} usable texts")
    label_counts = {CLASS_NAMES[l]: labels.count(l) for l in set(labels)}
    for name, count in sorted(label_counts.items()):
        print(f"  {name}: {count}")

    # 训练
    print(f"\n[TRAIN] Training classifier...")
    clf = IntentClassifier()
    clf.fit(texts, labels, lr=0.01, epochs=100)

    # 评估
    correct = 0
    for t, l in zip(texts, labels):
        pred = clf.predict(t)
        if pred['intent_id'] == l:
            correct += 1

    acc = correct / len(texts)
    print(f"\n[EVAL] Train accuracy: {acc:.2%} ({correct}/{len(texts)})")

    # 演示
    print(f"\n[DEMO] Sample predictions:")
    for instr in ["calculate 3+4", "type Hello", "calculate 5+8 and save",
                   "type World", "type 'test' and save"]:
        pred = clf.predict(instr)
        slots_str = ', '.join(f"{k}={v}" for k, v in pred['slots'].items())
        print(f"  \"{instr}\"")
        print(f"         -> {pred['intent_name']} (conf={pred['confidence']:.2f}) [{slots_str}]")

    print(f"\n[DONE] Model saved to {clf.model_path}")


if __name__ == '__main__':
    main()
