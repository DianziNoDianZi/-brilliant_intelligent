# 开发中遇到的问题记录

## 1. serialize_for_llm 原地修改实体文本

**现象**：LLM 规划正确但 `_values_to_plan` 找不到实体。

**根因**：`serialize_for_llm()` 为了给 LLM 输出英文，直接修改了 `WSGEntity.text`（"三"→"3"），之后 `_values_to_plan` 用 `en_to_cn` 反向映射时找不到中文实体。

**修复**：序列化输出副本 dict，不修改原始实体；使用 `d = e.to_dict(); d['text'] = english_text`。

**文件**：`agent/wsg.py` - `serialize_for_llm()`

---

## 2. serialize_for_llm 返回 dict 后 sort/return 仍用 entity 属性

**现象**：`serialize_for_llm()` AttributeError: 'dict' object has no attribute 'bbox'。

**根因**：将实体转为 dict 后，sort 和 return 仍按 WSGEntity 属性访问（`e.bbox`、`e.to_dict()`），dict 没有这些属性。

**修复**：`sort` 改为 `d['bbox']`，return 直接返回 dict 列表，不再调用 `to_dict()`。

**文件**：`agent/wsg.py` - `serialize_for_llm()`

---

## 3. Pycache 导致修改不生效

**现象**：`.py` 文件已修改但运行表现仍和旧代码一致。

**根因**：Python 缓存了 `.pyc` 字节码，修改 `.py` 后不会自动重新编译。

**修复**：`find ... -name "__pycache__" -exec rm -rf {} +`

---

## 4. WSG 标签干扰 LLM

**现象**：LLM 输出 "数字键"、"显示为 0" 等非按钮值。

**根因**：`available` 从 `wsg.entities` 读取所有实体文本，混入了组标签、显示区等中文文本，LLM 认为这些也是可点击按钮。

**修复**：`plan()` 中只过滤 `type='button'` 的实体，并通过中英映射转换为英文值。

**文件**：`agent/planner.py` - `plan()`

---

## 5. LLM 输出格式不一致（对象 vs 数组）

**现象**：LLM 有时输出 `[{"step":1,"action":"click","target_id":25}, ...]`，有时输出 `["3", "+", "4", "="]`。

**根因**：旧版提示词教 LLM 输出对象格式，新版改为值数组后 LLM 仍沿袭旧格式。

**修复**：`_parse_value_sequence` 同时支持两种格式：字符串数组直接返回，对象数组通过 target_id 反查 WSG 实体文本。

**文件**：`agent/planner.py` - `_parse_value_sequence()`

---

## 6. LLM 输出中英文混杂

**现象**：LLM 输出 `["3", "加", "4", "等于"]` 而非 `["3", "+", "4", "="]`。

**根因**：LLM 看到 WSG 实体文本是中文时，倾向于输出中文符号。

**修复**：`_parse_value_sequence` 中添加 `_normalize_val()` 函数，将中文统一映射为英文符号。

**文件**：`agent/planner.py` - `_normalize_val()`

---

## 7. label 字段误导 LLM

**现象**：LLM 从 label 字段中提取中文而非 text 字段的英文。

**根因**：`serialize_for_llm` 输出了 `label: "加 (+)"`，LLM 优先使用了 "加"。

**修复**：去掉 label 字段，serialize 输出只保留 `id`、`type`、`text`、`bbox`。

**文件**：`agent/wsg.py` - `serialize_for_llm()`

---

## 8. 执行后 entity_id 失效

**现象**：多步执行时，第 N 步的 target_id 对应的是前一步 WSG 中的实体，但 env.reset() 后实体 ID 变了。

**根因**：UIAutomation 每次检测会生成新的控制实例，entity_id 不保证跨步骤一致。

**修复**：`executor.execute_plan()` 中每步后刷新 WSG，通过实体文本重映射 target_id。

**文件**：`agent/executor.py` - `execute_plan()`

---

## 9. fill_template 多位数展开偏移

**现象**：模板 `['A','B','+','C','D','=']` + 指令 "12+34" → `['1','2','3','4','+','C','D','=']`（位置错位）。

**根因**：从左到右替换多位数时，`result[pos:pos+1] = digits` 将 1 个元素替换为 N 个，后续位置偏移。

**修复**：改为按位填充（一位变量替换一位数字），从右到左处理避免偏移。并且 `fill_template` 对记事本模板支持 `VALUE` 变量替换。

**文件**：`agent/skill_lib.py` - `fill_template()`

---

## 10. GBK 终端编码问题

**现象**：emoji 和中文字符在 Windows 终端输出乱码或报错 `UnicodeEncodeError: 'gbk' codec can't encode character`。

**根因**：Windows 终端（cmd/PowerShell）默认使用 GBK 编码，不支持部分 Unicode 字符。

**修复**：输出内容避免 emoji（用 `[OK]`/`[FAIL]` 替代 ✅/❌），Python 脚本中显式指定 `encoding='utf-8'`。

**涉及文件**：多处 print 语句和 eval 报告脚本。

---

## 11. match() 排序中 count 为 0 导致批量生成的技能被忽略

**现象**：批量生成的技能已入库（compiled=True），但 `match()` 返回了默认预设的同类型技能（compiled=False），快回路不触发。

**根因**：`match()` 按 `(accuracy, count)` 排序。批量生成的技能通过 `successes=COMPILE_THRESHOLD` 设为了 compiled，但 `count` 字段使用默认值 0。现有技能经过用户手动执行有 `count=2`，所以 `(accuracy=1.0, count=2) > (accuracy=1.0, count=0)`，默认技能排在了前面。

**修复**：`_compile_skill()` 中设置 `count=COMPILE_THRESHOLD` 与 `successes` 对齐。`batch_generate.py` 中入库前也补上 `skill.count = skill.successes`。

**验证**：修复后 `9*7` 和 `9/7` 均成功走快回路。

**文件**：`agent/skill_generator.py` - `_compile_skill()`、`tools/batch_generate.py`

---

## 12. 批量生成冲突检查导致全部跳过

**现象**：22 个任务全部显示 `[CONFLICT]`，实际入库 0 个。

**根因**：`skill_generator._compile_skill()` 生成的技能默认名字为 `gen_+`、`gen_*`，与库里已有的 `add`、`multiply` 等不冲突，但与之前批量运行残留的 `gen_*` 系列冲突。冲突检查比较的是生成的名字，而不是任务 ID。

**修复**：移除冲突检查，直接按任务 ID 重命名后强制入库。

**文件**：`tools/batch_generate.py`

---

## 13. 批量生成的记事本技能使用字符级序列

**现象**：记事本任务生成的模板为 `['H', 'e', 'l', 'l', 'o']`，而非预期的 `['text_area', 'type:Hello']`。

**根因**：LLM 将 "type Hello" 理解为逐个按键输入字符，而非在文本区整体输入字符串。

**影响**：当前 batch 生成的记事本技能无法直接用于 executor（executor 期望 `type:Hello` 格式）。这是一个已知局限，需要改进提示词或后处理。

**状态**：未修复。`tasks/calculator_tasks.json` 中的记事本任务（notepad_1~5）已入库但不可用，需单独处理。

**文件**：`tasks/calculator_tasks.json`、`agent/skill_generator.py` - `CANDIDATE_PROMPT`
