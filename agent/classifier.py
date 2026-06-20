"""极简意图分类器。

MLP on Bag-of-Words，参数量 < 5000，50 条数据即可启动训练。
"""

from __future__ import annotations
import json, os, re, pickle
import numpy as np
from typing import Optional
from collections import Counter


VOCAB_SIZE = 200
CLASS_NAMES = ['binary_arithmetic', 'text_input', 'save_file',
               'input_then_save', 'cross_app_chain']

# 每类独立置信度阈值（上线用，低于阈值则 fallback 到 LLM）
CLASS_THRESHOLDS = {
    'binary_arithmetic': 0.55,
    'text_input': 0.65,
    'save_file': 0.80,        # 数据不足，全 fallback
    'input_then_save': 0.80,  # 数据不足，全 fallback
    'cross_app_chain': 0.80,  # 数据不足，全 fallback
}


class IntentClassifier:
    """Bag-of-Words + MLP 意图分类器。

    Predict(instruction) → {"intent_id": int, "slots": dict, "confidence": float}
    """

    def __init__(self, model_path: str = ""):
        self.model_path = model_path or "D:/briliant_intelligent/data/classifier.pkl"
        self.vocab: list[str] = []
        self.weights: np.ndarray = None  # (VOCAB_SIZE, n_classes)
        self.bias: np.ndarray = None
        self.fitted = False

    def build_vocab(self, texts: list[str], max_words: int = VOCAB_SIZE):
        """从文本列表中构建词表（不依赖预训练词向量）。"""
        all_words = []
        for t in texts:
            all_words.extend(tokenize(t))
        counter = Counter(all_words)
        self.vocab = [w for w, _ in counter.most_common(max_words)]

    def _bow(self, text: str) -> np.ndarray:
        """将文本转为词袋向量。"""
        vec = np.zeros(len(self.vocab) if self.vocab else VOCAB_SIZE, dtype=np.float32)
        for w in tokenize(text):
            if w in self.vocab:
                vec[self.vocab.index(w)] += 1.0
        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def fit(self, texts: list[str], labels: list[int], lr=0.01, epochs=100):
        """训练 MLP 分类器。"""
        self.build_vocab(texts)
        n = len(self.vocab)
        k = len(CLASS_NAMES)  # 固定输出维度，与类别数一致

        self.weights = np.random.randn(n, k) * 0.01
        self.bias = np.zeros(k)

        for epoch in range(epochs):
            loss = 0.0
            correct = 0
            for t, label in zip(texts, labels):
                x = self._bow(t)
                logits = x @ self.weights + self.bias
                probs = softmax(logits)
                loss += -np.log(probs[label] + 1e-10)
                if np.argmax(probs) == label:
                    correct += 1
                # 梯度下降
                grad = probs
                grad[label] -= 1.0
                self.weights -= lr * np.outer(x, grad)
                self.bias -= lr * grad

            if (epoch + 1) % 20 == 0:
                acc = correct / len(texts)
                print(f"  [TRAIN] Epoch {epoch+1}: loss={loss/len(texts):.4f} acc={acc:.3f}")

        self.fitted = True
        self.save()

    def predict(self, instruction: str) -> dict:
        """预测意图。

        Returns: {"intent_id": int, "intent_name": str, "slots": dict, "confidence": float}
        """
        if not self.fitted:
            return {"intent_id": 0, "intent_name": "binary_arithmetic",
                    "slots": {}, "confidence": 0.0}

        x = self._bow(instruction)
        logits = x @ self.weights + self.bias
        probs = softmax(logits)
        intent_id = int(np.argmax(probs))
        confidence = float(probs[intent_id])

        # 槽位填充（基于规则，不学习）
        slots = {}
        numbers = re.findall(r'\d+', instruction)
        for i, n in enumerate(numbers[:9]):
            slots[chr(65 + i)] = n
        type_match = re.search(r'type\s+[\'"]?(.*?)[\'"]?\s*(?:and|$)', instruction, re.I)
        if type_match:
            slots['VALUE'] = type_match.group(1).strip()

        return {
            "intent_id": intent_id,
            "intent_name": CLASS_NAMES[intent_id] if intent_id < len(CLASS_NAMES) else "unknown",
            "slots": slots,
            "confidence": round(confidence, 3),
        }

    def save(self):
        """保存模型参数。"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({'vocab': self.vocab, 'weights': self.weights,
                         'bias': self.bias, 'fitted': self.fitted}, f)

    def load(self):
        """加载模型参数。"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            self.vocab = data['vocab']
            self.weights = data['weights']
            self.bias = data['bias']
            self.fitted = data.get('fitted', True)
            return True
        return False


def tokenize(text: str) -> list[str]:
    """极简 tokenizer：小写 + 拆分。"""
    text = text.lower()
    text = re.sub(r'[^\w\s+\-*/=]', ' ', text)
    tokens = []
    for t in text.split():
        # 将 "3+4" 拆为 "3", "+", "4"
        parts = re.findall(r'\d+|[a-z]+|[+\-*/=]', t)
        tokens.extend(parts)
    return tokens


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
