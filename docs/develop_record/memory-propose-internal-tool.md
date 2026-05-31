# memory_propose 工具写入方案

## 状态

Implemented。当前实现以 `build_tool_registry()` 作为唯一注册入口，tool result 只向 LLM 返回 `"success"` 或 `"failed"`。

## 背景

当前 Reactive tick 已经能把一次对话过程写成可追溯的 cognition event：

- `perceived / judged / decided / acted / received_feedback / revised` 等过程事件。
- 关键事件 payload 已有轻量 contract 校验。
- `perceived` 和 `turn_sources_recorded` 已把 cognition event 与 `session_messages`、tool traces、LLM traces 关联起来。
- `belief_view` 能从 `belief_formed` 等生命周期事件投影出可召回的长期 belief。

但写入侧仍有一个核心缺口：Reactive tick 主要写下“发生了什么过程”，并不会在前台把“这轮对话里哪些内容值得成为长期认知”表达成明确、可审计、可拒绝、可投影的写入意图。完全依赖后台巩固会让重要显性偏好、纠正、稳定约束进入长期认知的时机变慢，也会让写入原因不够清楚。

同时，不应该为每个 tick 再额外调用一次 LLM 做 cognition extraction。那会直接增加延迟和成本，而且多数普通 turn 并不产生值得写入的长期认知，投入产出很差。

## 决策

引入一个由 runtime/cognition 处理的模型可见工具：`memory_propose`。

它不是外部工具，不执行网络、文件或用户可见副作用。它的作用是让本轮主 LLM 在原本回答用户的同一次工具调用机制里，主动提出“候选认知写入”。runtime 接收后只把它记录为候选事件或待确认事件；是否立即形成 `belief_formed`，由 deterministic gate 和现有 cognition policy 决定。

核心原则：

1. 不新增每 tick 必跑的独立 cognition extraction 调用；当模型实际调用 `memory_propose` 时，接受现有 tool loop 带来的一轮普通工具结果后续 LLM 调用。
2. 不用规则理解用户意图；规则只做 schema、来源、风险、重复、冲突校验。
3. 前台只收显性、低争议写入；复杂抽象仍交给后台批处理巩固。
4. 所有候选写入必须带 source refs，不能形成无来源的长期认知。
5. 高风险、冲突、含糊或影响行为较大的内容先进入 proposal pending/review，不直接成为 active belief。

## 工具语义

`memory_propose` 的调用者是主 LLM。它表达的是：

> “基于当前轮输入和已有召回内容，我认为以下内容可能应该写入长期认知。”

它不表达：

- “一定要写入”。
- “已经写入成功”。
- “用户已经确认”。
- “可以覆盖旧 belief”。

工具 description 必须直接告诉模型触发边界：只用于当前用户 turn 中明确表达的长期偏好、稳定约束、可复用流程或直接纠正；不要用于普通事实、临时 session 上下文、猜测或外部工具结果摘要；tool result 只可能是 `"success"` 或 `"failed"`。

工具结果只返回给 LLM 一个机器可读的整体接收状态：

```json
"success"
```

或：

```json
"failed"
```

只有具备完整 Reactive 写入上下文，并且本次 tool call 中所有 proposal 都被 gate 判定为 `accepted`，才返回 `"success"`。只要出现 `pending`、`rejected`、参数非法、无 proposal，或非 Reactive 写入上下文调用，统一返回 `"failed"`。逐项 gate 结果、拒绝原因和 proposal id 只进入 cognition event / audit 链路，不进入 LLM 可见的 tool result。

一次 tool call 可以包含多个 proposals。系统必须为每个 proposal 生成独立
`proposal_id`，并为每个 proposal 单独发一条 `memory_proposed` event；`proposal_id`
只留在 cognition event / audit 链路里，不暴露给 LLM。

## 建议工具参数

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "proposals": {
      "type": "array",
      "minItems": 1,
      "maxItems": 5,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "kind": {
            "type": "string",
            "enum": [
              "preference",
              "constraint",
              "correction",
              "procedure"
            ]
          },
          "content": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
            "description": "自然语言认知内容，必须能独立读懂。"
          },
          "evidence": {
            "type": "string",
            "minLength": 1,
            "maxLength": 300,
            "description": "本轮消息中的证据摘要，不存长原文。"
          },
          "scope": {
            "type": "string",
            "enum": ["counterpart", "global"]
          }
        },
        "required": [
          "kind",
          "content",
          "evidence",
          "scope"
        ]
      }
    }
  },
  "required": ["proposals"]
}
```

第一版不要追求完整认知抽取。建议只允许以下高价值、近实现的类型：

- `preference`：用户明确表达的偏好，例如“以后回答用中文”。
- `constraint`：用户明确要求长期遵守的约束，例如“不要在仓库里写入本机绝对路径”。
- `correction`：用户明确纠正旧认知，例如“不是 A，是 B”；第一版只记录 proposal pending/rejected，不修改旧 belief。
- `procedure`：用户明确要求的稳定做法，例如“以后修改多文件先跑测试”。

第一版不提供 `operation` 和 `target_belief_id` 参数。工具调用默认只表达“提出一个可能的新认知写入”；`strengthen / weaken / supersede / retract` 这些更新类操作需要可靠召回旧 belief，并把候选写入绑定到明确的目标 belief，当前还没有到这个实现阶段。

第一版也不提供 `risk` 参数。`risk` 的目的原本是防止高风险、敏感、影响行为较大的内容直接进入 active belief；但让 LLM 自己标 `low/medium/high` 仍然是不可验证的自报分类。第一版默认按低风险处理，gate 只按结构完整性、来源完整性、作用域和 kind 白名单保守决定 accepted、pending 或 rejected；后续再补系统内部风险策略。

第一版不支持 `scope=session`，也不提供 `ttl` 参数。如果一条信息只在当前 turn 或当前 session 内有用，它应该留在 LLM 上下文、session context 或 context window，不应进入长期 cognition write proposal。

第一版也不把 `about` 暴露成工具参数。当前写入侧还没有稳定的“模型可见 Reference 列表”，让 LLM 填 `about.id` 只会引入伪稳定 id。`about` 仍然是内部 cognition/Belief 字段，但由 runtime 派生，不由模型填写：

- `scope=counterpart`：runtime 派生为当前 `CounterpartRef`，例如当前本地默认用户通常是 `counterpart:main-user`。
- `scope=global`：第一版也可以自动晋升。它表示主体级通用认知，不绑定某个 counterpart；映射到 `Belief.about=[]`，沿用现有 projection 对 global belief 的语义。

`session`、`session_message`、`tool_call`、`llm_call` 不应该成为 `about`。它们分别属于 source refs 或 audit refs。

## Runtime 接入方式

第一版应把 `memory_propose` 作为 runtime 默认注册的模型可见工具，不引入单独的 internal 注册或 internal 标记：

1. `memory_propose` 必须稳定出现在同一 session 的 `tools` 参数中。不要按轮次、阶段、是否可能写入来动态增删工具定义；稳定 tools 参数是保持 provider 前缀缓存稳定的前提之一。
2. `build_tool_registry()` 是唯一 registry 构建入口，直接把 `memory_propose` 与其他默认工具注册到同一个 `ToolRegistry`；渲染给 LLM 的 tools 必须保持注册顺序，不做按名称排序。
3. `memory_propose` 跟现有 tools 放在同一个 `ToolExecutor` 路径里执行；第一版不需要单独 dispatcher。实现上只需要给这个 tool 注入额外 cognition 上下文。
4. `ToolExecutor` 执行 `memory_propose` 时需要一个扩展上下文，至少包含 `tick_id`、当前 user source message id、当前 LLM call/trace id、当前 `decision_event_id`、`EventEmitter` 或等价写入入口，以及 accepted proposal 同步 apply `BeliefProjection` 所需的投影入口。
5. `MemoryProposeTool.run()` 只写 cognition event，不做网络、文件或用户可见副作用。
6. 工具调用仍走现有 LLM/tool loop；当模型选择调用 `memory_propose` 时，接受普通工具结果后续 LLM round。
7. 若模型没有提出任何候选写入，本轮成本与当前工具机制基本一致。

`memory_propose` 的调用和结果必须保留在 LLM 上下文中，按 provider tool protocol 进入后续模型消息，并持久化到 `session_messages`。第一版不新增 session message kind，仍使用现有 assistant/tool message 形态，也不新增 `internal_tool` / `control_message` / `cognition_fact_source` 这类没有实际消费端的标记。

`memory_propose` 自身不能成为 cognition 的事实来源。它是认知体系内部的写入工具，不是用户证据。第一版候选写入的 source refs 只指向当前用户消息；可以记录 `tool_call_id` 作为审计链路，但不能把 `memory_propose` 工具消息当成支持 belief 的 source。

如果非 Reactive 写入上下文里的 LLM 调用了 `memory_propose`，例如 context compression、handover、pre-user maintenance，工具必须返回稳定的 no-op 结果，不写 cognition event。工具定义可以全局稳定存在，但只有带有完整 Reactive 写入上下文的调用才允许产生 `memory_proposed`。no-op result 固定为：

```json
"failed"
```

## ID 与来源边界

第一版把 ID 分成两类：事实来源 ID 和审计关联 ID。

事实来源 ID 可以进入 `source_refs`，并参与后续 `Belief.sources`：

- `session_message`：必要，且第一版必须是当前用户消息。用户消息是前台 proposal 唯一事实证据。
- `session`：可保留，用于定位上下文，但不能单独支撑一个 belief。
- 第一版不把外部工具结果或已有 belief 放入 `memory_propose` 的 source refs；后续有稳定引用基础后再扩展。

审计关联 ID 不进入 `Belief.sources`，只用于 replay、debug 和事件链路：

- `tick_id`：建议作为 `memory_proposed` 和后续 lifecycle event 的必填字段。它表示这条 proposal 属于哪一次 Reactive tick，用来串起 `perceived / judged / acted / revised / memory_proposed / belief_formed`，不是事实证据。
- `tool_call_id`：记录本次 `memory_propose` 调用，证明 proposal 是怎么进入系统的，不是事实证据。
- `llm_call_id / llm_trace_id`：有调试和审计价值，可作为 audit metadata 保留；它们不是事实证据，也不应进入 accepted belief 的 sources。
- `session_id`：运行时定位字段，通常保留；事实支撑仍以具体 source message 为准。

因此，一个 accepted belief 的证据链应该是：

```text
belief_formed
  -> memory_proposed audit chain
  -> source_refs 中的当前用户消息
```

而不是：

```text
belief_formed
  -> memory_propose tool message
```

## 写入事件设计

建议新增一个候选事件，而不是让模型直接发 `belief_formed`：

```text
MEMORY_PROPOSED = "memory_proposed"
```

每个 proposal 发一条 `memory_proposed` event。payload 建议包括：

```json
{
  "tick_id": "tick_...",
  "session_id": "session_...",
  "tool_call_id": "call_...",
  "proposal_id": "proposal_...",
  "proposal": {
    "kind": "preference",
    "content": "用户偏好使用中文讨论认知系统设计。",
    "evidence": "用户连续用中文讨论写入侧方案。",
    "scope": "counterpart"
  },
  "derived_about": [
    {"kind": "counterpart", "id": "counterpart:main-user"}
  ],
  "source_refs": [
    {"kind": "session", "id": "session_..."},
    {"kind": "session_message", "id": "msg_..."}
  ],
  "audit_refs": [
    {"kind": "tool_call", "id": "call_..."},
    {"kind": "llm_call", "id": "llm_..."},
    {"kind": "llm_trace", "id": "trace_..."}
  ],
  "gate": {
    "decision": "accepted|pending|rejected",
    "reason": "low_risk_explicit_preference"
  }
}
```

后续投影/巩固可以根据 gate 结果继续发已有生命周期事件：

- `accepted proposal` → `belief_formed`
- 更新类操作第一版不实现，不从 `memory_propose` 触发 `belief_strengthened / belief_weakened / belief_superseded / belief_retracted`
- `pending` → 后续统一的 review queue 事件；第一版不复用语义更窄的 `belief_form_pending_confirmation`
- `rejected` → 只保留 `memory_proposed` 审计痕迹，不进入 active belief

这样做的原因是：proposal 是模型建议，belief lifecycle 是系统承诺。两者不能混在一个事件里。

accepted proposal 的事件顺序必须固定：

```text
append memory_proposed
  -> append belief_formed with causal_parent = memory_proposed.id
  -> apply belief_formed to BeliefProjection in the same tool execution
```

`memory_proposed` 的 causal parent 使用当前 Reactive 链路中最近的可用事件，第一版建议使用 `decision_event_id`。`belief_formed` 必须在本次 turn 内进入 `belief_view`，让同一轮结束后的 inspection 和下一轮 recall 能看到它。

执行层还需要把 `memory_propose` 产生的 cognition event id 回传到本轮 debug/inspection 链路。当前 Reactive controller 的 `event_ids` 通常只包含固定 stage event；如果 `memory_proposed` / `belief_formed` 在工具执行期间额外产生，但没有合并到本轮 `turn_sources_recorded` 或等价 inspection payload，后续排查会看不到这条写入链。第一版可以用一个明确字段承接，例如：

```json
{
  "tool_cognitive_event_ids": ["event_memory_proposed_...", "event_belief_formed_..."]
}
```

`turn_sources_recorded` 可以继续保留原来的 `reactive_event_ids`，但必须额外记录这些由 tool execution 产生的 cognition event ids，或在生成记录前把它们并入本轮可检查事件列表。不要只依赖 provider tool message id 来追踪认知写入；tool message 不是事实源，也不是 cognition lifecycle event。

## Pending 与确认事件

现有 `belief_form_pending_confirmation` 不是通用 pending queue。它当前服务于一个很窄的场景：当 Reactive tick 解释到当前输入和已召回 belief 存在矛盾，并且 L2 策略 `require_explicit_confirm_on_contradiction` 生效时，`Reviser` 发出这个事件，表示“这次矛盾更新需要显式确认”。

这个事件的语义依赖 `contradict_ids` 和对应策略，不适合承接 `memory_propose` 的高风险、证据不足、纠正候选、全局作用域等 pending 情况。第一版 `memory_propose` 的 pending 可以先只体现在 `memory_proposed.gate.decision = "pending"` 中；如果需要独立队列，再新增通用的 proposal review/pending 事件。

## Gate 策略

Gate 是 deterministic 的，但它不负责“理解内容”，只负责守住写入边界：

- schema 完整：字段、枚举、长度、数量必须合法。
- 来源完整：`source_refs` 必须包含当前用户 `session_message`；`tick_id / tool_call_id / llm_trace` 只作为审计关联。
- 第一版自动晋升只允许 `kind in {preference, constraint, procedure}`；`correction` 不自动晋升。
- 第一版自动晋升允许 `scope in {counterpart, global}`。
- `scope=counterpart` 自动晋升要求 runtime 能派生出非空 `derived_about`；`scope=global` 自动晋升使用空 `derived_about/about`。
- 第一版默认按低风险处理；后续再加入更细的系统内部风险策略。
- 第一版自动晋升要求 `content / evidence / source_refs` 完整，且 `content <= 500` 字符、`evidence <= 300` 字符。
- 其余情况一律 pending 或 rejected，不尝试在 gate 内做复杂语义判断。

第一版 gate 不应尝试判断“这句话真实含义是什么”。语义判断由主 LLM 通过 proposal 表达；gate 只根据 proposal 的结构化字段和当前可验证上下文做保守准入，避免“错写比漏写更危险”。

这里的 `accepted` 不是绕过 proposal 直接写 belief。所有 `memory_propose` 调用都先形成 `memory_proposed` proposal；`accepted` 只表示系统 gate 决定把这个 proposal 在同一条写入链路中自动晋升为 `belief_formed`。`pending/rejected` proposal 则不会形成 active belief。

## 与后台巩固的关系

`memory_propose` 解决的是前台高精度写入，不替代后台巩固。

分工如下：

| 路径 | 触发 | 是否额外 LLM | 适合内容 |
| --- | --- | --- | --- |
| `memory_propose` | 主 LLM 本轮主动调用 | 调用时接受普通工具结果后续 round；不新增每 tick 必跑的独立 extraction call | 显性偏好、明确约束、简单 procedure |
| 后台批处理巩固 | 按事件窗口/阈值异步运行 | 可以用 LLM，但批量、低频、异步 | 跨多轮模式、复杂关系、主题总结、画像更新 |
| raw event retrieval | 使用侧按查询召回原始事件 | 不需要生成式 LLM | 还没抽象成 belief 的原始证据 |

这个组合避免两个极端：

- 每轮都做 LLM extraction，成本高且命中率低。
- 只做后台巩固，显性重要信息进入长期认知太慢。

## Proposal 到 Belief 的最小映射

第一版不新增 `MemoryCandidate`、`MemoryDecision` 这类中间领域结构。实现上只需要一个小的转换函数：

```text
memory_propose payload
  -> memory_proposed event
  -> gate accepted
  -> build_belief_from_memory_proposal()
  -> belief_formed event
```

`build_belief_from_memory_proposal()` 只负责把 accepted proposal 填进现有 `Belief` 模型，不引入新的持久层抽象。

建议映射如下：

| Belief 字段 | 来源 |
| --- | --- |
| `subject` | 当前 `SubjectProjection.current()` |
| `about` | runtime 根据 proposal.`scope` 派生；`counterpart` 使用当前 CounterpartRef，`global` 使用空列表 |
| `object` | 由 `kind + scope + derived_about` 生成稳定对象名，例如 `preference:counterpart:main-user` 或 `procedure:global` |
| `content` | proposal.`content` |
| `cognitive_type` | `preference -> PREFERENCE`，`constraint/procedure -> PROCEDURAL` |
| `structure` | 第一版用 `None`，不做结构化 claim |
| `sources` | 只从当前用户 `session_message` source ref 生成，不包含 `memory_propose` tool message、`tool_call_id`、`llm_call_id` |
| `confidence` | 不接受 LLM 自报分值；accepted foreground proposal 使用系统固定初始值，后续由 belief lifecycle 调整 |
| `applicability` | 由 `scope / derived_about` 生成最小适用范围；第一版没有 `ttl` 参数 |
| `value_profile` | 空 profile，沿用投影侧已有派生逻辑 |
| `relations` | 空列表 |
| `formed_in` | 当前 context window 的 situation，缺省时用当前 thread/situation ref |
| `holder_role` | 当前主体默认 role |
| `action_orientation` | 第一版空列表 |
| `update_policy` | 保守默认值，例如 conflict 时走 pending/review，不自动覆盖 |
| `status` | `active` |
| `held_since` | 当前事件时间 |
| `derivation` | 可记录 `memory_proposed` event id 和 gate reason |

这块不要先补复杂实体归一、claim parser、画像模型或候选表。第一版目标是把显性、低风险、可追溯的 proposal 变成现有 `belief_formed`，让 projection 和 recall 能跑通。

## 第一版实现切片

1. 新增 `memory_propose` tool 定义，并通过 `build_tool_registry()` 这个唯一入口注册。
2. 新增 `memory_proposed` 事件 kind 与 payload contract。
3. 扩展 Reactive completion 调用链，把 `tick_id`、当前用户 source refs、当前 user source message id、当前 `decision_event_id`、LLM trace 信息、投影 apply 入口传给 `memory_propose`。
4. 实现 `MemoryProposeTool.run()`：校验参数、补齐当前用户消息 source ref、写入每条 `memory_proposed`；工具消息保留在 LLM 上下文，但不作为 cognition fact source。
5. 实现最小 deterministic gate：accepted / pending / rejected。
6. accepted proposal 先支持 `scope in {counterpart, global}` 且 `kind in {preference, constraint, procedure}` 自动晋升为 `belief_formed`，并通过注入的投影入口同步 apply 到 `BeliefProjection`。
7. 实现 `build_belief_from_memory_proposal()`，直接映射到现有 `Belief`，不新增候选/决策中间结构。
8. 把 `memory_proposed` / `belief_formed` 等 tool execution 产生的 cognition event ids 回传到本轮 debug/inspection，并让 `turn_sources_recorded` 或等价记录能看到它们。
9. pending 不复用 `belief_form_pending_confirmation`；第一版可只保留在 `memory_proposed.gate`，如需要独立队列再新增通用 proposal review/pending 事件。
10. 更新 renderer/system prompt：告诉模型只有在用户明确表达长期偏好、稳定约束、可复用流程，或明确纠正旧认知时才调用；其中纠正类第一版只进入 proposal pending/rejected。
11. 增加 CLI inspection 或测试 helper，能看到 proposal、gate 结果和最终 belief。

## 验收标准

- 普通问答通常不触发 `memory_propose`；如果非 Reactive 写入上下文误触发，必须返回 `"failed"` 且不写 cognition event。
- 用户明确说“以后都用中文回答我”时，产生 `memory_proposed`，gate accepted，并形成可召回的 preference belief。
- 一个 tool call 包含多个 proposals 时，每个 proposal 都有独立 `memory_proposed` event；tool result 只在全部 proposal accepted 时返回 `"success"`，否则返回 `"failed"`。
- accepted proposal 在同一 tool execution 内产生 `belief_formed`，并同步进入 `belief_view`。
- 本轮 inspection 能同时看到 provider tool message id、`memory_proposed` event id、accepted 后的 `belief_formed` event id；三者边界清楚，tool message 不被当作事实源。
- 用户纠正旧 belief 时，第一版不执行 supersede/retract；只记录 proposal pending/rejected，不改旧 belief。
- 每个 accepted belief 都能追溯到 source message、tick、tool call、LLM trace；其中 `memory_propose` tool call、LLM call、LLM trace 只作为审计链路，不作为 belief 的事实来源。
- 所有新增事件都有 payload contract 和投影/worker 消费测试。

## 暂不做

- 不做通用“从对话里抽取一切知识”的语义抽取器。
- 不做每 tick 必跑的独立 LLM cognition extraction。
- 不做复杂知识图谱、实体归一、跨项目全局画像。
- 不把 `memory_propose` 作为用户可直接调用的外部工具。
- 不让模型直接写 `belief_formed`；模型只能提出 proposal。
