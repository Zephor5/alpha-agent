# Phase 02 — Reactive 单 tick 跑通

**Status:** pending
**Depends on:** Phase 01
**Scope:** L
**Design ref:** `cognition_from_scratch.md` §4；README 三条架构不变量

## 0. 目标

把 9 步 Reactive Loop（Perceive → Attend → Interpret → Judge → Decide → Act
→ Feedback → Reflect → Revise）作为**纯函数 stage**实现，并由
`CognitiveController.reactive_tick(stimulus, thread_id)` 串起来。一轮 tick
跑完后，事件日志里能看到完整 9 类事件的因果链。

这条 Reactive 循环天生跑在 LoopCoordinator 与 Counterpart 路由之上：
`AlphaAgent.respond()` 在进入循环前先 `coordinator.try_acquire(LoopPriority.
REACTIVE, ...)`——拿不到锁不阻塞、直接 busy 回复——并把 `source_metadata`
推成 CounterpartRef（必要时触发 `counterpart_first_observed`）放进 Stimulus
/ Perception。这两件事不是额外模块，它们和 9 步 stage 一同构成一次 tick 的
完整入口。

这一阶段把 Phase 00 留下的"最简对话"升级成**真正的认知循环**——但所有
projection 仍是 stub（in-memory dict），Phase 03 / 04 才会用 SQLite 投影替
换。

## 1. 范围

### 1.1 In scope

- 9 个 stage 模块，每个一份纯函数 + 一个事件 emit。
- `CognitiveController.reactive_tick(...)`。
- 最简 stub projection：
  - `SubjectProjection.current()` → 从 EventLog replay 最近一个
    `self_model_updated` 事件，或返回 default Subject（id=SUBJECT_SELF）。
  - `BeliefProjection.recall(...)` → 暂时返回空列表；Phase 03 实现。
  - `ProcedureProjection.match(...)` → 暂时返回空列表；Phase 06 学到第一个
    Procedure 之前都为空。
  - `ContextWindowProjection.get(...)` → 暂时返回"最近 K 条 perceived 事件
    包成 foreground"，无 background；Phase 04 正式实现。
- `Effector` 调 LLM——这是这一阶段唯一调外部世界的地方。LLM call 走现有
  `alpha_agent/llm/` 抽象。
- 替换 `AlphaAgent.respond()` 让它走 `CognitiveController.reactive_tick`，但
  保持对外签名不变。
- 单 tick 集成测试：given user stimulus → 验证 event log 形态。

### 1.2 Out of scope

- 真正的 BeliefProjection（Phase 03）。
- 真正的 ContextWindowProjection（Phase 04）。
- Reflector 规则（Phase 05；本阶段 reflect 只发空 Reflection 占位事件）。
- Consolidation Loop（Phase 06）。
- ValueLens 冲突解决（Phase 07；Judger 这一阶段直接采用所有 Judgment，不解
  冲突）。
- Renderer 解耦（Phase 09；这一阶段 Effector 内部直接拼最简 prompt）。
- Drive Loop（Phase 10；本阶段只接受外部 stimulus）。

## 2. 任务清单

### 2.1 Stage 模块

- [ ] `cognition/stages/perceive.py`：`Perceiver.perceive(stimulus, subject,
  situation) -> Perception`，emit `perceived`。**Perception.from_counterpart
  从 Stimulus.source 直接继承**；Situation.social.present_counterparts 由
  perceive 阶段初始化（v1 简化：仅 `[stimulus.source]` 若非 None）。
- [ ] `cognition/stages/attend.py`：`Attender.focus(perception, subject) ->
  AttentionFocus`，emit `attended`。
- [ ] `cognition/stages/interpret.py`：`Interpreter.interpret(focus, window,
  recalled, subject) -> Interpretation`，emit `interpreted`。
- [ ] `cognition/stages/judge.py`：`Judger.judge(interpretation, value_lens)
  -> list[Judgment]`，emit `judged`。
- [ ] `cognition/stages/decide.py`：`Decider.decide(judgments, procedures,
  subject, window) -> Decision`，emit `decided`。
- [ ] `cognition/stages/effector.py`：`Effector.execute(decision, window) ->
  Outcome`，emit `acted`。LLM call 落在这里。
- [ ] `cognition/stages/feedback.py`：`FeedbackReader.compare(decision,
  outcome) -> Feedback`，emit `received_feedback`。
- [ ] `cognition/stages/reflect.py`：`ReflectorL1.audit(...) ->
  list[Reflection]`，emit `reflected`。这一阶段返回空列表，Phase 05 加规则。
- [ ] `cognition/stages/revise.py`：`Reviser.derive(...) -> list[Revision]`，
  emit `revised` 与 `belief_formed` / `belief_superseded` 等下游事件。这一
  阶段只支持 `belief_formed`，不解冲突。

每个 stage 是 callable，签名见 §3.1。所有 stage 都接受 `EventEmitter` 注入
而不是自己创建 EventLog——便于测试。

### 2.2 Controller

- [ ] `cognition/controller.py`：`CognitiveController.reactive_tick(...)`。
  - 内部按顺序调 stage。
  - 每步 `emit` 后把 reference 传给下一步。
  - 一轮 tick 共享一个 `tick_id`（uuid），写进每个 event 的 `payload` 里方便
    日志聚合。

### 2.3 Stub Projections

- [ ] `cognition/projections/subject.py`：`SubjectProjection`。先用 replay
  事件流的方式重建 Subject 字段；找不到就返回 `Subject.default()`（id 固
  定为 `SUBJECT_SELF`）。
- [ ] `cognition/projections/belief.py`：`BeliefProjection` stub——`recall`
  永远返回空列表；`status` 字段 `"stub"`，便于 inspection 区分。Phase 03 替
  换实现。
- [ ] `cognition/projections/procedure.py`：`ProcedureProjection` stub——
  `match` 永远返回空。
- [ ] `cognition/projections/context_window.py`：`ContextWindowProjection`
  stub——`get(thread_id, subject, at)` 把最近 K 个 `perceived` 事件拼成
  `ContextWindow.foreground`，`background=None`，`recalled=[]`。Phase 04 替
  换正式实现。

### 2.4 接入 AlphaAgent

`AlphaAgent.respond()` 同时承担三件事：尝试拿 Reactive 锁、解析 Counterpart、
驱动 reactive_tick。**重点：锁拿不到时不阻塞**——立刻返回 busy 响应，且这
一次 stimulus 不写任何 cognitive event、不写 conversation_messages。

- [ ] `runtime/agent.py`：`AlphaAgent.respond` 改为：

```python
def respond(self, user_message, session_id, source_metadata=None):
    # 1. 先 try_acquire——锁忙时直接 busy reply，不走 9 步链。
    req = LoopAcquireRequest(
        loop_name="reactive",
        priority=LoopPriority.REACTIVE,
        max_chunk_duration=timedelta(seconds=120),
    )
    try:
        ctx = self.coordinator.try_acquire(req)
        ctx.__enter__()
    except LockBusy as busy:
        # 这一次请求不进入主体的认知——不 emit perceived，
        # 不写 conversation_messages，CounterpartRef 也不解析（避免
        # counterpart_first_observed 在被拒绝的请求上 fire）。
        return AgentTurnResult(
            response=self._compose_busy_message(busy),
            session_id=session_id,
            debug={"busy": True, "holder": busy.holder, "since": busy.since},
        )

    try:
        # 2. 把 source_metadata 推成 CounterpartRef（必要时 emit
        #    counterpart_first_observed）。
        counterpart_ref = self.counterpart_router.upsert_from_source_metadata(
            source_metadata,
            emitter=self.emitter,
        )

        # 3. 装配 Stimulus，附带 CounterpartRef 与 thread_id。
        # ThreadId 在 Phase 01 已就位；StimulusRouter（集中路由策略）
        # 在 Phase 04 落地，本阶段直接用 ThreadId.from_session 即可。
        stimulus = Stimulus(
            kind="user_message",
            source=counterpart_ref,
            payload=user_message,
            thread_id=ThreadId.from_session(session_id, source_metadata),
            received_at=utc_now(),
        )

        # 4. 跑认知循环。
        result = self.cognitive_controller.reactive_tick(
            stimulus=stimulus,
            thread_id=stimulus.thread_id,
        )
    finally:
        ctx.__exit__(None, None, None)

    return AgentTurnResult(
        response=result.response_text,
        session_id=session_id,
        debug=result.debug,
    )
```

约定：

- `_compose_busy_message(busy)`：返回一条对外可读的系统提示，例如
  `"Agent is currently consolidating memory (started 12s ago); please retry
  in ~30s."` 这一文本对 caller 透明，CLI / Gateway 直接显示。
- **busy 路径完全不写**任何 event log 与 conversation_messages——从主体视
  角，被拒的请求"未曾发生"。这是不变量 2 的硬要求。
- `subject_id` 不再是参数——系统只有一个 Subject（`SUBJECT_SELF`），controller
  里固定。
- StimulusRouter 在 Phase 04 落地（集中路由 self_signal / clock_tick 等
  非会话来源）；Phase 02 只有 user_message 这一类，直接用
  `ThreadId.from_session(...)`。

- [ ] `runtime/counterpart_router.py`（新）：`CounterpartRouter`，封装
  source_metadata → CounterpartRef 的映射逻辑：
  - platform + user_id → 唯一 CounterpartId（稳定 hash）。
  - 已在 counterpart_view 的直接返回；首次见 emit
    `counterpart_first_observed`、role 推断（默认 "user"）。
  - 后续 phase 可扩展：trust_level 从 token / 平台信号推断、role 从权限推断。
- [ ] 删除 Phase 00 留下的最简 prompt 拼接逻辑——它现在被 stage 替代。
- [ ] `runtime/prompt_builder.py`：标记为 deprecated；Phase 09 整合到
  Renderer。Effector 内部暂时直接拼 chat messages。

### 2.5 测试

- [ ] `tests/cognition/test_reactive_tick_happy_path.py`：
  - 给一条用户 stimulus → tick 跑完 → event log 里有 9 类事件且因果链正确。
  - LLM 用 mock provider。
- [ ] `tests/cognition/test_reactive_tick_tool_call.py`：
  - Decision = use_tool → Effector 触发 tool registry → tool result 作为
    feedback。
- [ ] `tests/cognition/test_controller_emits_tick_id.py`：
  - 一轮 tick 内所有事件 payload 含同一个 `tick_id`。
- [ ] `tests/cognition/test_stub_projections.py`：
  - ContextWindow stub 能取出最近 K 个 perceived。
  - Belief/Procedure stub 返回空。
- [ ] `tests/cognition/test_counterpart_router.py`：
  - 首次见某 (platform, user_id) → emit `counterpart_first_observed`。
  - 再次见 → 不重复 emit；返回同一 CounterpartRef。
  - Perception.from_counterpart == Stimulus.source。
- [ ] `tests/cognition/test_reactive_busy_response.py`：
  - 模拟低优先级 holder 持锁 → `respond()` 立即返回 busy AgentTurnResult，
    不阻塞、不写任何 cognitive event、不写 conversation_messages。
  - 返回文本以 "Agent is currently" 开头，含 holder 与 since 字段信息。
  - debug 含 `busy=True / holder / since`。
- [ ] `tests/cognition/test_reactive_busy_no_counterpart_observation.py`：
  - 模拟第一次见的新 (platform, user_id) 在 busy 状态下被拒 →
    `counterpart_first_observed` 事件**不**被 emit；下次锁空闲时第一条消
    息才触发 first_observed。
- [ ] `tests/cognition/test_reactive_no_preemption.py`：
  - 低优先级 holder 持锁、Reactive try_acquire 反复失败 → coordinator 不
    发任何 cancel/interrupt 信号到 holder。
  - 仅当 holder 主动 yield 后，下一次 Reactive try_acquire 才有机会成功。

### 2.6 文档

- [ ] AGENTS.md 项目导航更新：`cognition/stages/`、`cognition/controller.py`、
  `cognition/projections/`。
- [ ] 在 `docs/develop_record/` 留一篇 dev note，记录本阶段对 stage 边界的
  实际选择（例如 Attend 与 Interpret 在第一版是否合并）。

## 3. 接口契约（草案）

### 3.1 Stage 签名

`Stimulus` 已在 Phase 01 §3.1 定义（带 `source: CounterpartRef | None`、
`thread_id`、`received_at`），本阶段直接使用，不再重复声明。本节只列 Phase
02 新增的中间数据类。

```python
@dataclass(frozen=True)
class AttentionFocus:
    entities: list[EntityRef]
    salient_claims: list[NLStatement]
    value_signals: dict[ValueKind, float]

@dataclass(frozen=True)
class Interpretation:
    stance: Literal["consistent", "contradicting", "novel", "ambiguous"]
    supporting_beliefs: list[BeliefRef]
    contradicting_beliefs: list[BeliefRef]
    novel_claims: list[NLStatement]
    ambiguity_notes: list[str]

@dataclass(frozen=True)
class Outcome:
    text: str | None
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    raw_llm_response: Any

@dataclass(frozen=True)
class Feedback:
    matched_expected: bool
    surprises: list[str]
    affected_belief_ids: list[BeliefId]
```

每个 stage 是协议：

```python
class Stage(Protocol[I, O]):
    def __call__(self, *args, emitter: EventEmitter, **kwargs) -> O: ...
```

### 3.2 Controller

```python
@dataclass(frozen=True)
class LoopResult:
    response_text: str
    decision: Decision
    reflections: list[Reflection]
    debug: dict[str, Any]

class CognitiveController:
    def __init__(
        self,
        event_log: EventLog,
        projections: ProjectionRegistry,
        llm: LLMProvider,
        tools: ToolRegistry,
        clock: Callable[[], Instant] = utc_now,
    ): ...

    def reactive_tick(
        self,
        stimulus: Stimulus,
        thread_id: ThreadId,
    ) -> LoopResult: ...
```

系统只有一个 Subject（`SUBJECT_SELF`），controller 内部固定，所以 `reactive_
tick` 不再接受 `subject_id` 参数。`ProjectionRegistry` 是后续阶段的扩展点——
本阶段注册 stub，Phase 03/04 注册真实版。

### 3.3 事件 payload 模板

```python
# perceived
{"tick_id": ..., "stimulus_kind": ..., "payload_digest": "..."}

# attended
{"tick_id": ..., "focused_entity_count": N}

# interpreted
{"tick_id": ..., "stance": "...", "support_ids": [...],
 "contradict_ids": [...]}

# judged
{"tick_id": ..., "judgment_count": N}

# decided
{"tick_id": ..., "action": "...", "expected_feedback": "..."}

# acted
{"tick_id": ..., "outcome_text_len": N, "tool_call_count": N}

# received_feedback
{"tick_id": ..., "matched_expected": bool, "surprises": [...]}

# reflected
{"tick_id": ..., "reflection_count": N}  # 本阶段恒为 0

# revised
{"tick_id": ..., "revisions": [...]}
```

## 4. 文件清单

### 4.1 新增

```text
src/alpha_agent/cognition/controller.py
src/alpha_agent/cognition/stages/__init__.py
src/alpha_agent/cognition/stages/perceive.py
src/alpha_agent/cognition/stages/attend.py
src/alpha_agent/cognition/stages/interpret.py
src/alpha_agent/cognition/stages/judge.py
src/alpha_agent/cognition/stages/decide.py
src/alpha_agent/cognition/stages/effector.py
src/alpha_agent/cognition/stages/feedback.py
src/alpha_agent/cognition/stages/reflect.py
src/alpha_agent/cognition/stages/revise.py
src/alpha_agent/cognition/projections/subject.py
src/alpha_agent/cognition/projections/belief.py
src/alpha_agent/cognition/projections/procedure.py
src/alpha_agent/cognition/projections/context_window.py
# EventEmitter 已在 Phase 01 引入，本阶段直接使用
src/alpha_agent/runtime/counterpart_router.py   # source_metadata → CounterpartRef
tests/cognition/test_reactive_tick_happy_path.py
tests/cognition/test_reactive_tick_tool_call.py
tests/cognition/test_controller_emits_tick_id.py
tests/cognition/test_stub_projections.py
tests/cognition/test_counterpart_router.py
tests/cognition/test_reactive_busy_response.py
tests/cognition/test_reactive_busy_no_counterpart_observation.py
tests/cognition/test_reactive_no_preemption.py
```

### 4.2 修改

```text
src/alpha_agent/runtime/agent.py           respond() 接 CognitiveController
src/alpha_agent/runtime/prompt_builder.py  标记 deprecated
src/alpha_agent/cli.py                     debug 子命令展示 cognition trace
AGENTS.md                                  项目导航补 stages/ projections/
```

### 4.3 删除

```text
src/alpha_agent/runtime/prompt_builder.py   （Phase 09 删；本阶段先 deprecate）
```

## 5. 验收标准

- [ ] `uv run pytest tests/cognition/ -q` 全绿。
- [ ] `alpha chat` 跑一轮对话，`alpha debug prompt --trace` 能列出本轮 9 类
  事件及其因果父事件。
- [ ] 同一 session 中连续 5 轮对话，每轮 event log 多 9 条，无丢失。
- [ ] 工具调用路径走通：用户发"用 X 工具做 Y"，能看到 `decided`(use_tool) →
  `acted` → tool result → `received_feedback`。
- [ ] Phase 00 留下的 xfail 测试中，"上一轮提到的事下一轮能回忆"类的用例**仍
  然 xfail**——这要等 Phase 03 BeliefProjection 真实化才能转绿。
- [ ] `AlphaAgent.respond()` 对外签名与 Phase 00 末态完全一致。
- [ ] Event log 中每条事件的 `causal_parents` 至少有一项（除 `perceived` 之
  外）。

## 6. 风险与备注

- **Attend 与 Interpret 是否合并**。理论上 Attend 是注意焦点抽取，Interpret
  是与已有信念比对——两件事。但第一版规则化实现里 Attend 几乎是 Interpret
  的前置过滤。如果实施时发现合并更清晰，可以合，但要写进 dev_record，让后续
  阶段知道是有意为之。
- **Stage 之间不要共享可变状态**。每步只接收前步 output，每步只 emit 自己的
  事件。共享只能走 controller 注入的依赖（event log、projection、clock）。
- **Effector 的 LLM call 是这一阶段最大的复杂度**。它要处理：tool loop（沿
  用现有 ToolExecutor 抽象）、错误重试、超时。允许把这部分直接借用现有
  `runtime/tools.py` 的实现，但调用应通过 `Effector.execute(decision)` 收口。
- **Subject 是单例**。Phase 02 永远使用 `SUBJECT_SELF`（Phase 01 §3.1 定义
  的常量）——不再有 per-user subject 这一回事。"这条消息来自哪个用户"通过
  CounterpartRef 表达。
- **CounterpartRouter 的稳定 hash**。从 (platform, user_id) 推 CounterpartId
  时必须用稳定 hash（例如 `sha1(platform + ":" + user_id)[:16]`），不能跨进
  程变。否则同一用户重启后变成新 Counterpart，事件日志关联断。
- **Reactive 不阻塞，busy 响应必须显式处理**。`respond()` 拿不到锁时立即返
  回 busy 文本——CLI / Gateway 必须把它当成系统级响应显示给用户，**不要**
  把它写进 conversation_messages 或重试 N 次。Gateway 实现者要注意：用户
  看到 busy 后再发同一句话，对主体来说是全新的 stimulus（事件层面上
  与"上一句被拒"无任何因果连接）。
- **不发 loop_acquired/released 事件作为热路径**。Phase 01 §3.5 的
  loop_* 事件只在 `acquired.waiting > 1s` 或显式 yield 时 emit。Reactive
  几乎从不写 loop_acquired，避免日志膨胀。
- **tick_id 的作用**。本阶段就引入，避免后续 reflection / consolidation 想要
  跨事件聚合时找不到锚。

## 7. 后续衔接

- Phase 03 替换 `BeliefProjection` stub 为真实实现，`Interpreter.interpret`
  能拿到真正的 recalled beliefs。
- Phase 04 替换 `ContextWindowProjection` stub，foreground/background/recalled
  正式可用。
- Phase 05 让 `ReflectorL1.audit` 真出 Reflection。
- Phase 06 让 `ProcedureProjection.match` 能命中并影响 Decider。
- Phase 09 让 Effector 把"拼 prompt"那块剥离到 Renderer。
- Phase 10 让 Stimulus 多一类来源（self_signal / clock_tick）。
