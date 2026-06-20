"""最小集成验证：用多模态世界模型替代 LLM 做低置信 fallback。

流程：
  分类器低置信(如 conf<0.55) → 不是直接调 LLM
  → 先生成候选动作序列 → 多模态世界模型预测每条 → 选误差最低的
  → 如果预测误差低于阈值 → 直接执行（不调 LLM）
  → 否则 fallback 到 LLM

这个脚本独立运行，验证通过后再合入 planner.py。
"""

import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agent.wsg_encoder import encode_wsg, STATE_DIM
from agent.wsg import WorldStateGraph, WSGEntity
from agent.classifier import IntentClassifier, CLASS_NAMES, tokenize, VOCAB_SIZE
from tools.compare_multimodal import SingleModalWM, MultiModalWM, bow_to_ids


def load_models():
    """Load classifier + both world models."""
    clf = IntentClassifier()
    if not clf.load():
        print("[FAIL] No classifier")
        return None, None, None, None

    with open("D:/briliant_intelligent/data/text_vocab.json") as f:
        vocab = json.load(f)

    sm = SingleModalWM()
    mm = MultiModalWM(vocab_size=len(vocab))

    sm_path = "D:/briliant_intelligent/data/single_modal_wm.pth"
    mm_path = "D:/briliant_intelligent/data/multi_modal_wm.pth"
    if not os.path.exists(sm_path) or not os.path.exists(mm_path):
        print("[FAIL] World model weights not found. Run compare_multimodal.py first")
        return None, None, None, None

    sm.load_state_dict(torch.load(sm_path, map_location='cpu'))
    mm.load_state_dict(torch.load(mm_path, map_location='cpu'))
    sm.eval()
    mm.eval()

    return clf, sm, mm, vocab


def make_dummy_wsg() -> WorldStateGraph:
    """Generic WSG for simulation."""
    wsg = WorldStateGraph()
    for txt, x in [('1',0),('2',30),('3',60),('4',90),('5',120),
                    ('6',150),('7',180),('8',210),('9',240),('0',270),
                    ('+',300),('-',330),('*',360),('/',390),('=',420)]:
        wsg.add_entity(WSGEntity(id=len(wsg.entities)+1, type='button',
                                  text=txt, bbox=[x,0,x+25,25]))
    wsg.add_entity(WSGEntity(id=99, type='input', text='text_area',
                              bbox=[0,50,500,350]))
    wsg.compute_spatial_relations()
    return wsg


def generate_candidates(instruction: str) -> list[list[str]]:
    """Generate diverse candidate action sequences from instruction.

    The multi-modal world model evaluates each candidate and picks the best.
    """
    il = instruction.lower()
    candidates = []

    # 1. Calculator candidates (if numbers present)
    numbers = re.findall(r'\d+', instruction)
    ops = re.findall(r'[+\-*/]', instruction)
    if numbers and ops:
        vals = []
        for n in numbers[0]:
            vals.append(n)
        vals.append(ops[0])
        for n in numbers[1:]:
            for d in n:
                vals.append(d)
        vals.append('=')
        candidates.append(vals)
        # Also try with leading clear
        candidates.append(['C'] + vals[:])

    # 2. Text input candidates (type/write/enter)
    text_variants = []
    for keyword in ['type', 'write', 'enter', 'input']:
        if keyword in il or f"'{keyword}" in il or f'"{keyword}' in il:
            text = instruction
            for kw in ['type', 'write', 'enter', 'input']:
                text = re.sub(rf'(?i)^{kw}\s*', '', text)
            text = text.strip('\'" ')
            # Also handle "type 'hello' and save"
            and_pos = re.search(r'\s+and\s+', text)
            if and_pos:
                text1 = text[:and_pos.start()]
                text2 = text[and_pos.end():]
                text_variants.append(text1)
                # Combined type+save
                candidates.append(['text_area', f'type:{text1}', 'file_menu', 'save', 'save_button'])
            else:
                text_variants.append(text)
            break

    for t in text_variants:
        candidates.append(['text_area', f'type:{t}'])

    # 3. Save document candidates
    if 'save' in il:
        candidates.append(['file_menu', 'save', 'save_button'])
        candidates.append(['file_menu', 'save_as', 'filename', 'save_button'])

    # 4. Calculator + notepad cross-app
    if numbers and any(kw in il for kw in ['type', 'write', 'enter', 'save', 'notepad']):
        calc_cands = [c for c in candidates if any(v in '0123456789' for v in c)]
        if calc_cands:
            calc_base = calc_cands[0]
        else:
            calc_base = list(numbers[0]) + ['+'] + list(numbers[-1] if len(numbers) > 1 else numbers[0]) + ['=']
        text_val = text_variants[0] if text_variants else 'result'
        candidates.append(calc_base + ['alt_tab', 'text_area', f'type:{text_val}'])

    return candidates


def evaluate_candidates(candidates, instruction, wsg_before, mm, vocab, device='cpu'):
    """Score each candidate by world model prediction error."""
    feat = torch.FloatTensor(encode_wsg(wsg_before)).unsqueeze(0).to(device)
    tokens = tokenize(instruction)
    text_ids = torch.LongTensor(bow_to_ids(tokens, vocab)).unsqueeze(0).to(device)
    act_types = ['click', 'type', 'tab', 'enter', 'wait']

    results = []
    for vals in candidates:
        action = np.zeros(5, dtype=np.float32)
        if any(v.startswith('type:') for v in vals):
            action[1] = 1.0  # type
        else:
            action[0] = 1.0  # click

        act_t = torch.FloatTensor(action).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = mm(feat, act_t, text_ids)  # multimodal
            error = F.mse_loss(pred, feat).item()  # measure prediction stability

        results.append((error, vals))

    results.sort()  # lowest error first
    return results


def test_fallback(instruction: str, clf, mm, vocab):
    """Test the full fallback chain."""
    print(f"\n{'─'*50}")
    print(f"  Instruction: \"{instruction}\"")

    # Step 1: Try classifier
    pred = clf.predict(instruction)
    print(f"  Step 1 - Classifier: {pred['intent_name']} conf={pred['confidence']:.2f}")

    from agent.classifier import CLASS_THRESHOLDS
    threshold = CLASS_THRESHOLDS.get(pred['intent_name'], 0.8)

    if pred['confidence'] >= threshold:
        print(f"  -> Classifier handles (conf {pred['confidence']:.2f} >= {threshold:.2f})")
        print(f"  [PASS] No world model fallback needed")
        return True

    print(f"  -> Low confidence ({pred['confidence']:.2f} < {threshold:.2f})")

    # Step 2: Generate candidates
    candidates = generate_candidates(instruction)
    if not candidates:
        print(f"  Step 2 - No candidates generated -> LLM fallback")
        return False

    print(f"  Step 2 - {len(candidates)} candidates:")
    for i, c in enumerate(candidates):
        print(f"    {i+1}. {c}")

    # Step 3: Evaluate with world model
    wsg = make_dummy_wsg()
    scored = evaluate_candidates(candidates, instruction, wsg, mm, vocab)

    print(f"  Step 3 - World model evaluation:")
    best_error, best_vals = scored[0]
    for error, vals in scored:
        marker = " ← BEST" if vals == best_vals else ""
        print(f"    error={error:.6f} {vals}{marker}")

    # Threshold: error < 0.005 means confident
    if best_error < 0.005:
        print(f"  -> World model fallback SUCCESS (error={best_error:.6f})")
        print(f"    Selected: {best_vals}")
        return True
    else:
        print(f"  -> World model error too high ({best_error:.6f}) -> LLM fallback")
        return False


def main():
    print("=" * 50)
    print("  多模态世界模型 fallback 验证")
    print("=" * 50)

    clf, sm, mm, vocab = load_models()
    if clf is None:
        return

    test_instructions = [
        "calculate 3+4",          # classifier should handle
        "type Hello",              # classifier should handle
        "save the document",       # classifier low conf -> WM fallback
        "type hello and save",     # classifier low conf -> WM fallback
        "calculate 3+4 and type",  # classifier low conf -> WM fallback
    ]

    successes = 0
    for instr in test_instructions:
        ok = test_fallback(instr, clf, mm, vocab)
        if ok:
            successes += 1

    print(f"\n{'='*50}")
    print(f"  Result: {successes}/{len(test_instructions)} handled without LLM")
    print(f"  LLM calls saved: {successes}/{len(test_instructions)}")
    print(f"{'='*50}")

    # Compare with pure classifier baseline
    clf_only = 0
    for instr in test_instructions:
        pred = clf.predict(instr)
        from agent.classifier import CLASS_THRESHOLDS
        thr = CLASS_THRESHOLDS.get(pred['intent_name'], 0.8)
        if pred['confidence'] >= thr:
            clf_only += 1
    print(f"\n  Classifier alone:     {clf_only}/{len(test_instructions)}")
    print(f"  + WM fallback:        {successes}/{len(test_instructions)}")
    print(f"  Improvement:          +{successes - clf_only} more saved LLM calls")


if __name__ == '__main__':
    main()
