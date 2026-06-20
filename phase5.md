# Phase 5 — 自主意图理解（轻量版）

## 核心假设验证

用极简分类器替代外部 LLM 的意图理解部分。

先验证一个关键问题：**是否可以不需要外部 LLM，仅通过系统内部已有的指令→模板映射数据，训练一个足够小的模型来理解自然语言指令？**

如果 100 条数据 + 一个 MLP 就能在现有 5 个 L3 模板上达到 >90% 的准确率，说明路径可行，再扩展规模。如果不行，加大模型也没有用。

## 验证方案

### 阶段1：数据积累

当前系统每次成功执行后自动记录：
```
(指令, L3模板ID, 槽位值, 时间戳)
```

改造 `main_phase2.py` 的执行日志，在每次成功执行后追加一条记录到 `data/intent_log.jsonl`。

不需要额外的手工标注——数据是执行的副产品。

### 阶段2：极简分类器

**模型结构：**
- 词袋编码（Bag of Words）：指令分词后映射到 200 维 TF-IDF 向量
- 分类器：单层 MLP（200 → 64 → 5），参数量 < 5000
- 槽位填充：正则表达式从指令中提取数字/文本（不学习，只匹配）

**训练数据：** 积累的 (指令, intent_id) 对，50-100 条即可启动训练。

**评估标准：**
- 5-fold 交叉验证，意图分类准确率 > 90%
- 槽位提取正确率 > 95%（基于规则的槽位填充通常比学习更稳定）

### 阶段3：影子模式

分类器与 LLM 并行运行，记录两者输出的一致性：
```python
# agent/planner.py
def plan(self, instruction, wsg):
    classifier_result = self.classifier.predict(instruction)
    # 同时记录 LLM 结果和分类器结果
    log_consistency(instruction, classifier_result, llm_result)
    
    # 仅当分类器置信度 > 0.9 且已验证准确率 > 95% 时启用
    if self.use_classifier and classifier_result.confidence > 0.9:
        return classifier_result.to_plan(wsg)
    return self.llm_plan(instruction, wsg)
```

### 阶段4：切换条件

当分类器在影子模式下连续 100 次与 LLM 输出一致，切换为分类器优先。

### 阶段5：多模态世界模型（可选）

如果阶段1-4验证通过，再考虑将分类器升级为小型 Transformer，以及世界模型的多模态化。当前不实施。

## 不做的部分

- 多模态世界模型（等分类器验证通过后再评估必要性）
- 教程学习器（依赖多模态世界模型）
- 自主探索模块（依赖教程学习器）

## 改动文件

| 文件 | 改动 |
|------|------|
| `main_phase2.py` | 成功执行后追加 intent_log |
| `agent/classifier.py` | 新增：极简意图分类器 |
| `agent/planner.py` | 新增影子模式 |
| `tools/train_classifier.py` | 新增：训练脚本 |
| `config.py` | 新增分类器相关配置 |

不修改：世界模型、技能库、快回路、内模拟器、WSG 生成。
