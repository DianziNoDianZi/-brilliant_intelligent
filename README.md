# Brilliant Intelligent

好奇心驱动的自主智能体，从网格世界到真实桌面操作。

## 项目概况

```
Phase 1          Phase 1.5       Phase 2           Phase 3            Phase 3.5          Phase 4
网格好奇心闭环    视觉+系综       桌面UI理解         双回路骨架         技能生长+内模拟     层级记忆
                 不确定性分解     LLM规划+执行       快慢回路分离       跨应用泛化         跨应用迁移
```

## 快速开始

### 环境要求

- Python 3.12+（推荐 3.12 以获得 CUDA 支持）
- NVIDIA GPU 6GB+（可选，用于加速）
- Ollama + Qwen-2.5-7B（用于 LLM 规划）

### 安装

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. 安装 Phase 2 桌面依赖
pip install -r requirements_phase2.txt

# 4. 安装 CUDA PyTorch（可选，GPU 加速）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. 启动 Ollama 并拉取模型
ollama pull qwen2.5:7b
ollama serve
```

## 使用指南

### Phase 1 — 网格世界好奇心闭环

```bash
python main.py                    # 默认 500 episode，可视化
```

智能体在网格世界中自主探索，好奇心驱动交互。

### Phase 2 — 桌面理解 + LLM 规划

**计算器：**
```bash
# 预演模式（默认）
python main_phase2.py "calculate 3+4"

# 真实执行
python main_phase2.py "calculate 3+4" --execute

# 快回路模式（3次后技能编译）
python main_phase2.py "calculate 7+4" --execute --dual
```

**记事本：**
```bash
python main_phase2.py "type Hello" --app notepad --execute
python main_phase2.py "type World" --app notepad --dual --execute
```

### Phase 3.5 — 内模拟 + 技能生成

```bash
# 带内模拟训练
python main_phase2.py "calculate 7+4" --execute --train

# 批量生成技能（22个任务，全部在想象中完成）
python tools/batch_generate.py

# 泛化验证
python tools/generalization_test.py
python tools/generalization_boundary.py
```

### Phase 4 — 层级记忆 + 跨应用迁移

```bash
# 跨应用迁移验证（记事本技能自动适配到 WordPad）
python tools/cross_app_migration.py wordpad "type Hello World"
```

## 项目结构

```
briliant_intelligent/
├── main.py                    # Phase 1 入口
├── main_phase2.py             # Phase 2/3/4 入口
├── config.py                  # 超参数
├── environment/
│   ├── grid_world.py          # 网格世界环境
│   └── desktop_env.py         # Windows 桌面环境 (UIAutomation)
├── agent/
│   ├── world_model.py         # 世界模型 + 系综
│   ├── policy.py              # Actor-Critic 策略
│   ├── memory.py              # 经验回放缓冲
│   ├── intrinsic_reward.py    # 好奇心奖励 + 不确定性分解
│   ├── wsg.py                 # 世界状态图谱
│   ├── wsg_encoder.py         # WSG → 特征向量
│   ├── planner.py             # LLM 规划器 (Qwen-2.5)
│   ├── validator.py           # 计划校验 + 执行验证
│   ├── executor.py            # 动作执行 (PyAutoGUI)
│   ├── simulator.py           # 内模拟引擎
│   ├── skill_lib.py           # L2 技能块（55+ 模板，带自动标签）
│   ├── abstract_templates.py  # L3 抽象模板（5 个，跨应用迁移）
│   ├── fast_loop.py           # 快回路 (技能匹配)
│   ├── skill_generator.py     # 内模拟驱动的技能生成
│   ├── skill_recombinator.py  # 技能重组（跨应用组合）
│   ├── intent.py              # 意图向量编码
│   └── visual_frontend.py     # MobileNetV3 视觉前端
├── training/
│   ├── train_loop.py          # 训练循环
│   └── replay.py              # 后台回放
├── tools/
│   ├── quality_check.py       # 质量抽检
│   ├── generalization_test.py # 泛化验证
│   ├── generalization_boundary.py # 泛化边界测试
│   ├── eval_world_model.py    # 世界模型评估
│   ├── collect_wsg_data.py    # WSG 数据采集
│   ├── batch_generate.py      # 批量技能生成
│   └── cross_app_migration.py # 跨应用迁移测试
├── tasks/
│   └── calculator_tasks.json  # 批量生成任务列表
├── designs/
│   └── hierarchical_memory.md # 层级记忆设计文档
├── visualization/
│   └── renderer.py            # 可视化
├── problems.md                # 开发问题记录
├── PHASE_SUMMARY.md           # 阶段总结
└── requirements*.txt          # 依赖清单
```

## 架构亮点

### 双回路系统
- **慢回路**：LLM 规划器负责新任务的推理和计划生成
- **快回路**：技能库匹配 + 模板填充，绕过 LLM 直接执行
- 经过 3 次慢回路积累后，同类任务自动由快回路接管

### 内模拟驱动成长
- LLM 提出候选动作序列
- 系综世界模型在"想象"中验证每一步的可信度
- 高置信序列无需真实执行即可编译为技能
- 实现了"离线学习"——技能可以在想象中生成

### 层级记忆（L1/L2/L3）
- **L1 原始体验**：完整执行轨迹，用于世界模型训练
- **L2 技能块**：55+ 编译技能，自动标记操作类型标签
- **L3 抽象模板**：5 个跨应用通用操作模式，从 L2 自动提取
- 迁移验证：记事本 type_text → WordPad 自动适配

## 验证结果

| 测试 | 结果 |
|------|------|
| 计算器 12+34 → 快回路泛化 56+78 | ✅ |
| 记事本 type Hello → 快回路泛化 type World | ✅ |
| 跨应用迁移：记事本 → WordPad | ✅ |
| 快回路正确率 | 100% |
| 技能模板数量 | 55+（47 编译） |
| 世界模型 WSG 预测 MSE | < 0.01 |
| 内模拟置信度 | 0.93+ |
| L3 抽象模板 | 5 个 |
