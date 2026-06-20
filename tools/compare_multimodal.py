"""多模态世界模型对比实验。

单模态 (baseline): 输入 WSG + 动作 → 预测 WSG
多模态 (实验组):  输入 WSG + 动作 + 指令文本 → 预测 WSG

如果多模态预测误差显著更低 → 文本信号有帮助 → Phase 5c 可行。
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

from agent.wsg_encoder import STATE_DIM
from agent.classifier import VOCAB_SIZE


# ── 单模态 baseline ──
class SingleModalWM(nn.Module):
    """Input: WSG + action. Output: predicted next WSG."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, STATE_DIM),
        )
    def forward(self, feat, action):
        x = torch.cat([feat, action], dim=-1)
        return feat + self.net(x)  # residual: predict delta


# ── 多模态实验组 ──
class MultiModalWM(nn.Module):
    """Input: WSG + action + text_embedding. Output: predicted next WSG."""
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=32):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.text_encoder = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        total_dim = STATE_DIM + 5 + 16
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, STATE_DIM),
        )

    def forward(self, feat, action, text_ids):
        # Text embedding
        emb = self.text_embed(text_ids)  # (B, T, D)
        text_vec = emb.mean(dim=1)       # (B, D) — average pooling
        text_feat = self.text_encoder(text_vec)  # (B, 16)
        x = torch.cat([feat, action, text_feat], dim=-1)
        return feat + self.fusion(x)


def bow_to_ids(tokens: list[str], vocab: list[str], max_len=10) -> list[int]:
    """Convert token list to vocab IDs (bag of words → order)."""
    ids = []
    for t in tokens[:max_len]:
        if t in vocab:
            ids.append(vocab.index(t) + 1)  # +1 because 0 is padding
        else:
            ids.append(0)
    while len(ids) < max_len:
        ids.append(0)
    return ids


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MULTIMODAL] Device: {device}")

    # Load data
    # Load both JSON array and JSONL formats
    data = []
    with open("D:/briliant_intelligent/data/multimodal_data.json") as f:
        raw = f.read().strip()
        if raw.startswith('['):
            data = json.loads(raw)
        else:
            for line in raw.split('\n'):
                if line.strip():
                    data.append(json.loads(line))
    with open("D:/briliant_intelligent/data/text_vocab.json") as f:
        vocab = json.load(f)

    print(f"[DATA] {len(data)} samples, vocab_size={len(vocab)}")

    # Split
    np.random.shuffle(data)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    eval_data = data[split:]
    print(f"[DATA] Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Build tensors
    def to_tensors(samples):
        feats = torch.FloatTensor([s['feat_before'] for s in samples])
        actions = torch.FloatTensor([s['action_vec'] for s in samples])
        targets = torch.FloatTensor([s['feat_after'] for s in samples])
        text_ids = torch.LongTensor([
            bow_to_ids(s['tokens'], vocab) for s in samples
        ])
        return feats.to(device), actions.to(device), targets.to(device), text_ids.to(device)

    train_feats, train_acts, train_tgt, train_txt = to_tensors(train_data)
    eval_feats, eval_acts, eval_tgt, eval_txt = to_tensors(eval_data)

    # ── Train SingleModal ──
    print(f"\n{'='*50}")
    print("  [BASELINE] Training SingleModalWM...")
    print(f"{'='*50}")
    sm = SingleModalWM().to(device)
    sm_opt = torch.optim.Adam(sm.parameters(), lr=1e-3)
    sm_losses = []
    for epoch in range(200):
        pred = sm(train_feats, train_acts)
        loss = F.mse_loss(pred, train_tgt)
        sm_opt.zero_grad()
        loss.backward()
        sm_opt.step()
        sm_losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                eval_loss = F.mse_loss(sm(eval_feats, eval_acts), eval_tgt).item()
            print(f"  Epoch {epoch+1}: train_loss={loss.item():.6f} eval_loss={eval_loss:.6f}")

    # ── Train MultiModal ──
    print(f"\n{'='*50}")
    print("  [EXPERIMENT] Training MultiModalWM...")
    print(f"{'='*50}")
    mm = MultiModalWM(vocab_size=len(vocab)).to(device)
    mm_opt = torch.optim.Adam(mm.parameters(), lr=1e-3)
    mm_losses = []
    for epoch in range(200):
        pred = mm(train_feats, train_acts, train_txt)
        loss = F.mse_loss(pred, train_tgt)
        mm_opt.zero_grad()
        loss.backward()
        mm_opt.step()
        mm_losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                eval_loss = F.mse_loss(
                    mm(eval_feats, eval_acts, eval_txt), eval_tgt).item()
            print(f"  Epoch {epoch+1}: train_loss={loss.item():.6f} eval_loss={eval_loss:.6f}")

    # ── Compare ──
    with torch.no_grad():
        sm_eval = F.mse_loss(sm(eval_feats, eval_acts), eval_tgt).item()
        mm_eval = F.mse_loss(mm(eval_feats, eval_acts, eval_txt), eval_tgt).item()

    print(f"\n{'='*50}")
    print(f"  [RESULT] Comparison on eval set")
    print(f"{'='*50}")
    print(f"  SingleModal (WSG+action):     MSE = {sm_eval:.6f}")
    print(f"  MultiModal  (WSG+action+text): MSE = {mm_eval:.6f}")
    improvement = (sm_eval - mm_eval) / sm_eval * 100
    print(f"  Improvement: {improvement:+.1f}%")
    if improvement > 0:
        print(f"  [OK] Text signal reduces prediction error -> Phase 5c viable")
    else:
        print(f"  [FAIL] Text signal does not help -> need different approach")
    print(f"{'='*50}")

    # Save models
    torch.save(sm.state_dict(), "D:/briliant_intelligent/data/single_modal_wm.pth")
    torch.save(mm.state_dict(), "D:/briliant_intelligent/data/multi_modal_wm.pth")
    print(f"\n  Models saved to data/")


if __name__ == '__main__':
    main()
