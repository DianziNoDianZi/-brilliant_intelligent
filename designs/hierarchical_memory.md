# 层级记忆 — 数据结构设计与实现方案

## 三层结构

```
抽象操作模板层 (Abstract)
  └─ "保存文档" → [点击菜单, 点击保存, 输入文件名, 点击确定]
      "输入文本" → [点击编辑区, 输入文字]
  ↑ 技能重组 + 跨应用抽象
技能块层 (Skill Block)
  └─ 计算器 add       → ['A','+','B','=']
      记事本 type_text → ['text_area','type:VALUE']
  ↑ 批量编译 + 内模拟
原始体验层 (Raw Experience)
  └─ (WSG_before, action, WSG_after, timestamp, success)
      完整执行轨迹（逐帧WSG变化）
```

## 数据模型

### Layer 1: 原始体验 (RawExperience)

```python
@dataclass
class RawExperience:
    id: str
    task_type: str
    instruction: str              # 原始自然语言指令
    trajectory: list[StepRecord]  # 逐帧记录
    wsg_snapshots: list[dict]     # 每一步后的 WSG JSON
    result: str                   # success / failure
    timestamp: float
    app: str                      # calculator / notepad

@dataclass
class StepRecord:
    step: int
    action: str                   # click / type / key
    target_id: int
    screen_coord: tuple
    wsg_before: dict
    wsg_after: dict
    confidence: float             # 内模拟置信度
```

### Layer 2: 技能块 (SkillBlock)

```python
@dataclass
class SkillBlock:
    skill_id: str
    name: str
    task_type: str                # calculator / notepad
    app: str                      # 来源应用
    value_template: list[str]     # ['A','+','B','=']
    action_template: list[str]    # ['click','click','click','click']
    preconditions: dict           # 前置条件（如"计算器窗口打开"）
    postconditions: dict          # 后置条件（如"显示区有结果"）
    version: int
    experience_ids: list[str]     # 来源原始体验 ID
    compiled: bool
    success_rate: float
    abstraction_id: str           # 指向 Layer 3 的抽象模板 ID（如有）
```

### Layer 3: 抽象操作模板 (AbstractTemplate)

```python
@dataclass
class AbstractTemplate:
    template_id: str
    name: str                     # "document_save" / "text_input"
    description: str              # 人类可读描述
    steps: list[AbstractStep]     # 设备无关的操作步骤
    concrete_skills: list[str]    # 各应用下的具体技能 ID 列表
    transfer_count: int           # 成功迁移到新应用的次数

@dataclass
class AbstractStep:
    action_type: str              # click / type / navigate
    target_role: str              # "text_editor" / "menu_item" / "input_field"
    value_var: str                # 变量名（如 VALUE）
```

## 核心流程

### 技能上升通道

```
原始体验 Layer1
  │  batch_generate.py 分析成功轨迹 → 提取模式
  ▼
技能块 Layer2
  │  skill_recombinator.py 匹配跨应用兼容对
  │  + 人工/自动识别"保存"、"输入"等通用模式
  ▼
抽象模板 Layer3
```

### 新应用迁移

```
新应用（如 Word）→ WSG 检测
  → 匹配 Layer3 抽象模板 "text_input"
    → 找到具体实现：记事本 type_text 技能
      → 适配：调整坐标、菜单路径
        → 编译为 Word 特定技能块
```

## 存储结构

```
data/
├── experiences/        # Layer 1: 原始体验
│   ├── exp_001.json
│   └── exp_002.json
├── skills/             # Layer 2: 技能块（现有 skill_lib.py）
│   └── skills.json
└── abstracts/          # Layer 3: 抽象模板
    └── templates.json
```

## 实现优先级

1. **SkillBlock 升级现有 skill_lib** — 添加 preconditions/postconditions/experience_ids 字段
2. **RawExperience 存储** — 成功执行后自动记录 WSG 变化轨迹
3. **AbstractTemplate 定义** — 手工标记 3-5 个通用抽象模板（保存、输入、计算）
4. **迁移验证** — 将抽象模板应用到第三个应用验证跨应用迁移

## 与技能重组的衔接

技能重组（当前实现）→ 生成 compound_skills
层级记忆（下一步）→ 将 compound_skills 自动解析为：
  - Layer2 中的原子技能引用
  - Layer3 中的抽象操作模式
  - 跨应用迁移所需的适配参数
