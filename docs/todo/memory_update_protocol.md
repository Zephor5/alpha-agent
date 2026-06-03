# 记忆变更协议近端方案

## 目标

当前认知实现中，长期记忆的写入和召回已经走工具路径，但写入接口仍偏向“提交一条新记忆”。这会导致两个直接问题：

- 冲突判断粒度太粗，容易把无关偏好当成同类冲突。
- `correction` 被放在 `kind` 里，实际表达的是“修正操作”，不是记忆类型。

近期目标是建立更直接的模型主导记忆变更模式：

- 模型决定什么时候召回。
- 模型决定什么时候写入。
- 模型决定是新增、强化、替换、合并、修正还是撤回。
- 工具侧负责候选召回、保守校验、可审计落库和不确定时要求明确目标。

核心原则：宁可多存一条待后续合并的记忆，也不要在工具侧隐藏地做深层语义裁决。

## 当前问题

### 1. 写入接口混合了类型和操作

当前 `memory_propose` 使用：

```text
kind = preference | constraint | correction | procedure
```

其中 `preference`、`constraint`、`procedure` 是记忆类型；`correction` 是用户意图或变更操作。它们不在同一层。

目标改成：

```text
operation = append | reinforce | replace | merge | correct | retract
memory.type = preference | constraint | procedure | factual
```

### 2. 冲突检测缺少模型可引用目标

当前 `memory_recall` 返回 compact content，不暴露 belief id、confidence、source 或 evidence。模型即使召回到了相关内容，也无法明确地对某条 belief 发出替换或合并指令。

目标改成：`memory_recall` 返回可引用 handle，至少包括：

```json
{
  "id": "belief:...",
  "content": "...",
  "type": "preference",
  "scope": "counterpart"
}
```

### 3. 写入工具不应自行做深层语义裁决

写入工具可以在没有显式 target 时执行一次候选召回，但只能返回候选和要求模型选择，不能直接判断“这是同一个偏好所以替换”。

## 新工具语义

近期保留工具名 `memory_propose`，但语义改为“记忆变更提案”。实现和测试中按记忆变更协议组织参数、校验和事件。

### 工具参数

建议 schema：

```json
{
  "updates": [
    {
      "operation": "append",
      "targets": [],
      "target_hint": "",
      "memory": {
        "type": "preference",
        "content": "User prefers Chinese answers.",
        "evidence": "User said: 以后都用中文回答我.",
        "scope": "counterpart"
      },
      "reason": "User explicitly stated a stable answer-language preference."
    }
  ]
}
```

字段含义：

- `operation`：对记忆状态的变更操作。
- `targets`：要操作的 active belief ids，默认为空数组。覆盖、合并、强化、撤回类操作必须有明确 target。
- `target_hint`：没有明确 target 时，模型对目标记忆的自然语言描述，默认为空字符串。
- `memory`：新增或替换后的 belief 内容。除 `retract` 外都必须提供。
- `memory.type`：记忆类型，不再包含 `correction`。
- `memory.scope`：`counterpart` 或 `global`，沿用当前作用域。
- `memory.evidence`：用户原始表达或模型可解释的证据。
- `reason`：模型为什么选择这个操作。

### 记忆类型

- `preference`：用户或项目长期偏好，例如回答语言、解释深度、示例倾向。
- `constraint`：必须遵守的长期限制，例如仓库规则、隐私边界、工具禁用规则。
- `procedure`：可复用工作流程，例如修改代码后运行哪些验证命令。
- `factual`：稳定事实，只用于用户、项目或长期上下文明确要求保留的事实；普通会话事实和临时任务状态不写入长期记忆。

### 返回状态

工具返回状态使用以下集合：

- `accepted`：变更已落库。
- `pending_confirmation`：需要用户确认后才能改写 active belief。
- `needs_target_selection`：发现相关候选，但模型需要明确选择 append、replace、merge、correct 或 retract。
- `rejected`：请求无效或不允许。
- `mixed`：一次工具调用中的多条 update 产生了不同状态。

`user_action` 使用现有语义：

- `none`：模型可继续处理。
- `ask_confirmation`：模型应向用户确认。
- `explain_rejection`：模型应解释为什么没有写入。

### Operation 规则

#### append

新增一条长期记忆。

允许条件：

- `memory` 存在。
- `memory.content` 和 `memory.evidence` 非空。
- `targets` 可为空。

工具行为：

- 如果 exact duplicate，转成 `reinforce`。
- 如果内部候选召回发现疑似相关 active beliefs，返回 `needs_target_selection`，不直接覆盖。
- 如果没有明显候选，发出 `memory_proposed` 和 `belief_formed`。

#### reinforce

强化已有记忆，或为已有记忆追加来源。

允许条件：

- 必须有一个或多个 target。
- target 必须存在且 active。

工具行为：

- 发出 `belief_strengthened` 或等价事件。
- 不新建内容相同的 belief。

#### replace

用一条新 belief 替换一条旧 belief。

允许条件：

- 必须有且仅有一个 target。
- target 必须存在且 active。
- `memory` 存在。
- evidence 应表达用户明确改变、覆盖或重新指定。

工具行为：

- 发出 `memory_proposed`。
- 发出 `belief_superseded`，并保留 old/new belief causal chain。
- 如果 evidence 不足，返回 `pending_confirmation`。

#### merge

将多条相关 belief 合并成一条更清晰的 belief。

允许条件：

- 至少两个 targets。
- `memory` 存在。
- reason 应说明合并依据。

工具行为：

- 发出 `memory_proposed`。
- 发出合并后的 `belief_formed` 或 `belief_superseded` 链。
- 旧 belief 标记为 superseded。

#### correct

用户指出旧记忆错误，并提供修正。

允许条件：

- 可以有 target，也可以只有 `target_hint`。
- 必须有 evidence 说明用户在纠正旧记忆。

工具行为：

- 有明确 target 时，发出 `belief_form_pending_confirmation`，payload 中记录 target、候选修正内容和 evidence。
- 没有 target 时，先返回候选；如果没有候选，返回 `pending_confirmation` 并要求用户确认目标。
- `correct` 本身不直接改写 active belief。确认后由模型发出 `replace` 或 `retract`。
- 不再把 `correct` 当作 memory type。

#### retract

撤回、忘记或删除某条记忆。

允许条件：

- 必须有 target。
- evidence 必须表达用户明确要求忘记、删除、不再记住或该记忆无效。

工具行为：

- 发出 `belief_retracted`。
- 无 target 时返回 `needs_target_selection`。

## 两种冲突检测路径

### 路径 A：模型先召回，再发变更指令

推荐路径。

流程：

1. 模型调用 `memory_recall`。
2. `memory_recall` 返回可引用 belief ids。
3. 模型判断相关性，调用新语义的 `memory_propose`。
4. 工具校验 target 和 operation。
5. 工具落库并返回结构化结果。

示例：

```json
{
  "operation": "replace",
  "targets": ["belief:old_python_examples"],
  "memory": {
    "type": "preference",
    "content": "User prefers Rust code examples.",
    "evidence": "User said: 以后代码示例改用 Rust.",
    "scope": "counterpart"
  },
  "reason": "User explicitly changed the previous code-example language preference."
}
```

工具只校验：

- target 是否 active。
- scope 是否匹配当前 counterpart/global。
- operation 是否允许。
- evidence 是否满足覆盖类操作的最低要求。

### 路径 B：模型直接写入，工具内部召回候选

兜底路径。

流程：

1. 模型直接调用 append。
2. 工具用 candidate content、evidence、scope、type 调用当前 belief recall 逻辑。
3. 如果没有候选，正常 append。
4. 如果找到疑似相关项，工具返回 `needs_target_selection`。
5. 模型下一轮选择 append、replace、merge、correct、retract 或向用户确认。

返回示例：

```json
{
  "status": "needs_target_selection",
  "user_action": "none",
  "message_hint": "Related memories were found. Choose append, replace, merge, or ask the user.",
  "candidates": [
    {
      "id": "belief:old_python_examples",
      "content": "User prefers Python code examples.",
      "type": "preference",
      "scope": "counterpart",
      "relation_hint": "possibly_related"
    }
  ]
}
```

工具不得在此路径中直接 replace 或 merge，除非是 exact duplicate 转 reinforce。

候选召回规则：

- 只考虑 active beliefs。
- 优先同 scope、同 `memory.type` 的候选。
- 最多返回 3 条候选，避免把选择负担转嫁给模型。
- exact duplicate 指同 scope、同 `memory.type` 且 normalized content 相同。
- 非 exact candidate 只能触发 `needs_target_selection`，不能触发自动覆盖。

## `memory_recall` 改动

当前 `memory_recall` 输出过于 compact。近期应增加面向模型变更指令的字段：

```json
{
  "results": [
    {
      "id": "belief:...",
      "content": "...",
      "type": "preference",
      "scope": "counterpart",
      "status": "active"
    }
  ]
}
```

可选调试字段可以只进 trace，不暴露给模型：

- score
- reasons
- confidence
- source ids

默认输出仍保持短，但必须让模型能引用 target。

## 事件与投影影响

### 事件

继续使用 event-sourced cognition，不需要新增复杂存储层。

需要支持或明确复用以下事件：

- `memory_proposed`
- `belief_formed`
- `belief_strengthened`
- `belief_superseded`
- `belief_retracted`
- `belief_form_pending_confirmation`

所有变更事件 payload 至少记录：

- `operation`
- `target_belief_ids`
- `reason`
- `evidence`
- `tool_call_id`
- `session_id`
- `turn_id`

形成或替换新 belief 的事件还必须记录：

- `new_belief_id`
- `belief`

撤回类事件不记录 `new_belief_id`。

merge 的落库口径：

- 先形成一条 merged belief。
- 再对每个旧 target 分别发出 `belief_superseded`，指向同一个 merged belief。
- merged belief 的 sources 必须包含所有 target belief refs 和当前 session message ref。

### Projection

`BeliefProjection` 需要保证：

- active belief 可按 id 查询。
- superseded belief 保留 old/new 链。
- retracted belief 不参与 active recall。
- strengthened belief 更新 confidence 或来源信息。

## CLI 与调试

近期复用并增强以下检查路径：

```bash
uv run alpha cognition evidence <belief-id>
uv run alpha debug prompt <message> --trace
```

已有命令可以继续复用，但 trace 输出中应能看见：

- 模型召回了哪些 belief。
- 模型选择了哪个 operation。
- 工具是否返回 `needs_target_selection`。
- 最终形成、替换、合并或撤回了哪些 belief。

## 测试计划

### memory_recall

- 返回 active belief id。
- 不返回 retracted belief。
- counterpart scope 不泄漏其他 counterpart belief。
- global scope 可被当前 counterpart recall。

### append

- 新偏好正常形成 belief。
- exact duplicate 转 reinforce，不新建重复 belief。
- 疑似相关但非 exact duplicate 时返回 `needs_target_selection`。

### replace

- 无 target 时返回 `needs_target_selection` 或 `rejected`，不得执行替换。
- target inactive 时拒绝。
- target active 且 evidence 充分时 supersede。
- superseded old belief 不再参与 active recall。

### merge

- 少于两个 targets 时拒绝。
- 多个 active targets 合并后旧 beliefs superseded。
- merged belief 保留所有 target/source refs。

### correct

- `operation=correct` 不再作为 memory type。
- 有 target 时进入 `pending_confirmation`，不得直接改写 active belief。
- 无 target 时返回候选或 `pending_confirmation`。

### retract

- 无 target 时返回 `needs_target_selection`。
- 有 target 且 evidence 明确时 retracted。
- retracted belief 不进入 profile 或 recall。

## 实施顺序

### 阶段 1：召回可引用

- 修改 `memory_recall` 输出，增加 `id` 和 `status`。
- 补 recall 输出测试。
- 确保输出仍然紧凑。

验收：

- 模型能在后续工具调用中引用 recall 返回的 belief id。

### 阶段 2：拆分 operation 和 memory.type

- 修改 `memory_propose` schema。
- 去掉 `kind=correction`。
- 引入 operation 分支。
- 输出状态扩展为 accepted、pending_confirmation、needs_target_selection、rejected、mixed。

验收：

- preference、constraint、procedure、factual 都作为 `memory.type`。
- correction 只作为 `operation=correct`。

### 阶段 3：实现 target 驱动的变更

- 实现 append、reinforce、replace、merge、correct、retract。
- 覆盖类操作必须要求 target。
- 无 target 且发现候选时返回 `needs_target_selection`。

验收：

- 无关偏好不会互相覆盖。
- 明确 target 的替换可以稳定发生。

### 阶段 4：内部候选召回兜底

- append 时调用现有 recall candidate 逻辑查疑似相关 belief。
- exact duplicate 自动 reinforce。
- 非 exact 相关项返回候选，不自动裁决。

验收：

- 工具能提示模型“你可能要 replace/merge”，但不会自己做语义覆盖。

### 阶段 5：调试与文档同步

- 更新 README 或 cognition status 中关于 memory write/read 的说明。
- 更新 CLI trace 展示。
- 补 evidence trace 对 replacement/merge/retract 的展示。

验收：

- 一条记忆从 recall 到 update 到 projection 的链路可追踪。

## 判断标准

成功标准：

- 模型能看见相关旧记忆。
- 模型能引用旧记忆发明确变更指令。
- 工具不会在不确定时擅自覆盖。
- 所有变更都有 target、evidence 和 causal chain。
- 用户明确修改偏好时，系统能稳定替换旧 belief。
