# 从零设计认知系统 Cognition From Scratch

## 0. 这一文档的视角

前两份文档（`cognition.md` 与 `memory_design.md`）是从两个不同的方向来的：

- `cognition.md`：理论框架，主语是"认知"，七层 + 十五字段。
- `memory_design.md`：工程实现，主语是"记忆"，六层 M0–M5 + Controller。

这一份文档以 cognition.md 为起点。**主语是认知，存储是从认知派生的副产物**。
如果完全不考虑现有 memory 实现、从 cognition.md 的需求一步步推导，我会得到
一个**以信念为中心、以事件日志为基底、以多并发 loop 为运行机制、以元认知为
内建机制**的系统。下文把这个系统写完整。

工程目标不是模拟人脑，而是：

```text
让一个软件主体能在时间上：
  形成、组织、使用、修正、反思自己对世界与自我的理解。
  并且这一切是可审计、可回放、可暂停、可替换的。
```

## 1. 设计的七个起点决断

在动手画结构图之前，先把方向性决断写清楚——这些决断会贯穿后续所有章节。

1. **中心抽象不是 memory，而是 Belief。** Memory 是 Belief 的存储侧投影。
   "我记得 X"是 belief 的副作用，不是 belief 本身。
2. **基底是 append-only 的 cognitive event log。** 当前 belief 状态是日志的
   projection，不是真实状态本身。这一点决定了审计、回放、分叉、并发都是可行
   的——也决定了任何"原地修改 belief"的接口都是错的。
3. **不只一条 loop。** 反应式 agent loop 只是其中之一。同等位的还有反思 loop /
   巩固 loop / 驱动 loop，它们以不同节奏跑、读同一份日志、写不同的 projection。
4. **认知主体 first-class。** `Subject` 不是 user_id 或 scope_key，而是带角
   色、能力、需求、价值排序、已知偏见的完整对象。所有 cognition 事件都挂在某
   个 Subject 名下。
5. **元认知是第二阶认知系统，不是注释。** 它分三级：L1 监控、L2 控制、
   L3 自我模型。L1 读 L0 trace，L2 改下一轮 L0 策略，L3 是 L0 在自我维度的
   慢投影。
6. **价值层 first-class。** 不是标量 importance。每条 Belief 带
   `ValueProfile`（这条信念对生存/社会/道德/功利/审美/终极各维的关切度），每
   个 Subject 带 `ValueLens`（价值优先级排序）。冲突由 ValueLens 解。
7. **渲染与认知解耦。** Belief graph 是源；prompt / tool block / debug stream
   / belief diff 都是 view。渲染器是一个独立模块，不能让"如何拼 prompt"这种
   I/O 细节倒灌到 belief 设计里。

后面所有设计都遵守这七条。

## 2. 三个核心抽象

### 2.1 Subject 主体

```python
@dataclass(frozen=True)
class Subject:
    id: SubjectId                       # 稳定标识
    role: Role                          # operator / guest / system / agent_self
    capabilities: list[Capability]      # 当下能做什么（语言、工具、领域知识）
    declared_needs: list[Need]          # 当前需求快照
    value_lens: ValueLens               # 价值排序
    self_model: SelfModel               # L3 元认知慢投影
    membership: list[GroupRef]          # 所属群体、组织、角色集合
    known_biases: list[BiasMarker]      # 已识别的偏向
    held_at: Instant                    # 这一刻的 Subject 快照时间
```

`Subject` 不是数据库行，是一个**时刻投影**。每个 cognitive event 都附带当时
有效的 Subject 引用（`SubjectId@version`）；过去某一刻的 Subject 可以从 event
log 重建。

`SelfModel` 是 L3 元认知层维护的稳态摘要：

```python
@dataclass(frozen=True)
class SelfModel:
    capabilities_self_assessed: dict[Capability, ConfidenceCurve]
    typical_failure_modes: list[FailurePattern]
    preferred_strategies: list[StrategyRef]
    stable_preferences: list[BeliefRef]
    typical_value_tradeoffs: list[ValueTradeoff]
```

注意：`stable_preferences` 是**对 belief 的引用**而非内容拷贝。一旦底层 belief
变了，self_model 在下一轮 consolidation 里自动同步。

### 2.2 Belief 信念

cognition.md 的认知单元字段全部 first-class，没有任何一项藏在 metadata 里：

```python
@dataclass(frozen=True)
class Belief:
    # 标识
    id: BeliefId
    subject: SubjectRef                 # 这条信念属于哪个主体

    # 内容
    object: str                         # 这条信念关于"什么"
    content: NLStatement                # 自然语言表述
    cognitive_type: CognitiveType       # 见 2.4
    structure: StructuredClaim | None   # 可选的弱结构 (subj, pred, obj, relations)

    # 证据
    sources: list[EvidenceRef]          # 指向 CognitiveEvent 或外部 Perception
    derivation: DerivationTrace | None  # 如果是从其它信念推出来的，记下推链

    # 元属性
    confidence: float                   # [0, 1]
    applicability: Applicability        # 时间窗、范围、前提条件
    value_profile: ValueProfile         # 见 2.5

    # 关联结构
    relations: list[BeliefRelation]     # 支持 / 反对 / 限定 / 抽象 / 实例

    # 主体与情境
    formed_in: SituationRef             # 这条信念形成时的情境
    holder_role: Role                   # 形成时主体扮演的角色

    # 行动相关
    action_orientation: list[ActionHint]
    feedback_history: list[FeedbackEntry]

    # 生命周期
    update_policy: UpdatePolicy         # 什么条件下会修改这条信念
    status: Lifecycle                   # candidate / active / superseded / archived / retracted
    held_since: Instant
    held_until: Instant | None
    superseded_by: BeliefRef | None
    supersedes: BeliefRef | None

    # 元认知
    self_audit: list[ReflectionRef]     # 这条信念被自审过的记录
```

四个关键设计：

1. **Belief 是 immutable**。要改就发新事件、产生新 belief、把旧 belief
   `superseded`。任何代码路径不允许 `mutate(belief)`。
2. **每个字段都可以追溯到一个 CognitiveEvent**。`sources` / `derivation` /
   `self_audit` 都是事件引用。不允许"凭空出现"的 belief。
3. **`status` 不是简单 enum，是状态机**。状态转移本身是 CognitiveEvent。
4. **`applicability` 不是字符串备注**，是 typed：时间窗 + scope + 前提条件
   列表。这是 cognition.md "适用范围"字段唯一不被丢失的方式。

### 2.3 CognitiveEvent 认知事件

所有"认知动作"都是事件。事件是这个系统唯一的写入单元。

```python
@dataclass(frozen=True)
class CognitiveEvent:
    id: EventId
    kind: CognitiveEventKind
    subject: SubjectRef                 # 谁产生的
    situation: SituationRef             # 在什么情境下
    inputs: list[Reference]             # 引用了哪些 belief / perception / decision
    outputs: list[Reference]            # 产生了哪些 belief / decision / reflection
    rationale: NLStatement              # 这一步的自述（为什么这样推）
    timestamp: Instant
    actor: ActorRef                     # 哪个 loop / 哪个模块产生的
    causal_parents: list[EventId]       # 因果父事件
```

事件类型（首版列表）：

```python
CognitiveEventKind = Literal[
    # Reactive loop 内部
    "perceived",
    "attended",
    "interpreted",
    "judged",
    "decided",
    "acted",
    "received_feedback",
    "reflected",
    "revised",

    # 信念生命周期
    "belief_formed",
    "belief_strengthened",
    "belief_weakened",
    "belief_superseded",
    "belief_retracted",

    # 元认知
    "bias_detected",
    "strategy_changed",
    "self_model_updated",

    # 学习
    "procedure_learned",
    "value_lens_shifted",

    # Drive / 外部
    "goal_set",
    "goal_satisfied",
    "external_signal_received",
]
```

事件日志是这个系统的**唯一权威**。其它所有数据——当前 belief 集、Subject 当
前状态、procedure 库——都是日志的 projection。

### 2.4 CognitiveType（内容形态）

cognition.md 内容层的八种，一次都写清楚：

```python
CognitiveType = Literal[
    "fact",        # 单点断言：北京是中国首都
    "concept",     # 抽象归类：什么是"风险"
    "relation",    # 关系：收入与教育的关联
    "causal",      # 因果：通胀导致购买力下降
    "rule",        # 条件规则：在 X 条件下要 Y
    "procedure",   # 程序：如何完成某类任务的步骤
    "value",       # 价值：什么重要、什么值得
    "self",        # 自我：我擅长什么、我害怕什么
]
```

这是**认知形态**，不是**记忆层级**。一条信念是 fact 还是 causal，跟它存在哪
里、属于哪种 memory 没关系——它属于 cognition.md 第一层"内容层"。在这个从零
设计里，记忆层级这个概念**根本不出现**——因为没有"记忆"这个独立子系统，只有
事件日志和它的投影。

### 2.5 ValueProfile 与 ValueLens

每条 Belief 自带价值剖面：

```python
ValueKind = Literal[
    "existence",   # 安全、资源、健康
    "social",      # 尊重、归属、地位
    "moral",       # 公平、正义、责任
    "utility",     # 效率、成本、收益
    "aesthetic",   # 优雅、秩序、风格
    "ultimate",    # 意义、自由、真理
]

@dataclass(frozen=True)
class ValueProfile:
    weights: dict[ValueKind, float]    # 这条信念对每一维的关切度
```

主体自带价值透镜：

```python
@dataclass(frozen=True)
class ValueLens:
    priority: list[ValueKind]                       # 主体的价值排序
    sensitivity: dict[ValueKind, float]             # 主体对每一维违反的敏感度
    tradeoff_preferences: list[ValueTradeoff]       # 已学到的具体取舍偏好
```

**冲突解决**：两条 belief 矛盾时，谁的 ValueProfile 在 Subject.ValueLens 的
priority 排序里更靠前、且 sensitivity 加权后分更高，谁胜出。失败方进入
`superseded`，事件类型 `belief_superseded`，rationale 写明用的是哪条 lens。

这把"什么重要"从开发者口味里解放出来。同一系统给不同主体用，可以接 plug-in
的 ValueLens，自然得到不同的行为。

## 3. 七个一级类型完整 schema

除了 Subject / Belief / CognitiveEvent，还有四个一级类型支撑闭环：

### 3.1 Situation

```python
@dataclass(frozen=True)
class Situation:
    id: SituationId
    physical: PhysicalContext        # 时间、地理、设备
    social: SocialContext            # 在场他人、权威结构、关系
    institutional: InstitutionalContext  # 规则、合规、平台政策
    informational: InformationalContext  # 信息质量、过载程度、来源可信
    cultural: CulturalContext        # 语言、习俗、风格规范
    historical: HistoricalContext    # 当前任务态摘要；具体的"最近发生过什么"由 ContextWindow（§5）承担
```

每一维都是**类型化结构**，不是 dict。`HistoricalContext` 包含当前 goal、open
questions、pending tasks、last action 等运行时态——这是 Situation 与
Reactive Loop 的接口面。

### 3.2 Perception

```python
@dataclass(frozen=True)
class Perception:
    id: PerceptionId
    source: PerceptionSource                # user_message / clock_tick / webhook / agent_signal
    raw: Any                                # 原始载荷
    surface_intent: list[IntentMarker]      # 浅层意图标注
    raised_entities: list[EntityRef]        # 提到的实体
    subject: SubjectRef                     # 在哪个主体的视角下被感知
    situation: SituationRef                 # 当时情境
    received_at: Instant
```

Perception **不假设来源是用户消息**。`source` 可以是 clock、webhook、
其它 agent。这把"agent loop"降级为"Perception 的一种来源类型"，不是核心。

### 3.3 Judgment

```python
@dataclass(frozen=True)
class Judgment:
    id: JudgmentId
    claim: NLStatement                      # 这一轮成立的命题
    supports: list[BeliefRef]               # 哪些 belief 支撑它
    undermined_by: list[BeliefRef]          # 哪些 belief 削弱它
    applicable_under: Applicability         # 仅在这些条件下成立
    confidence: float
    value_weights: dict[ValueKind, float]   # 这个判断对价值各维的权重
    formed_in: SituationRef
    expires_at: Instant | None              # 默认本轮结束就失效
```

Judgment 是**短命**的：默认本轮结束就 expire。只有当 consolidation loop 看到
同样的 judgment 在 N 轮内反复成立，才会被 promote 成 Belief。

### 3.4 Decision

```python
@dataclass(frozen=True)
class Decision:
    id: DecisionId
    action: Action                          # respond / ask / use_tool / refuse / defer / no_op
    payload: Any                            # action 的具体内容
    justified_by: list[JudgmentRef]
    expected_feedback: ExpectedFeedback     # 期待的反馈形态（用于第6步对账）
    fallback: Decision | None
    decided_at: Instant
```

`expected_feedback` 是关键：**Decider 必须声明这次决策期望看到什么反馈**。
否则后面的 Feedback 步无法对账，feedback loop 就断了。

### 3.5 Reflection

```python
@dataclass(frozen=True)
class Reflection:
    id: ReflectionId
    level: Literal["L1", "L2", "L3"]
    kind: ReflectionKind
    severity: Severity
    target: ReflectionTarget                # 指向 belief / judgment / decision / loop_run
    finding: NLStatement                    # 自审说的是什么
    suggested_remedy: RemedyHint
    created_at: Instant
```

`ReflectionKind` 例如：

```text
contradiction-accepted
low-confidence-high-stakes
situation-mismatch
unsupported-tool-call
premature-auto-approval
value-tradeoff-not-justified
overfitting-to-recent
self-model-update-needed
```

### 3.6 Procedure

```python
@dataclass(frozen=True)
class Procedure:
    id: ProcedureId
    trigger: TriggerPattern                 # 什么 Perception+Judgment 配合时触发
    steps: list[Step]                       # 有序步骤
    expected_outcome: NLStatement
    learned_from: list[EventId]             # 从哪些成功经验抽出来的
    success_count: int
    failure_count: int
    confidence: float
```

Procedure 是 cognition.md "程序认知"的兑现。它不是普通信念，而是"如何做"的固
化模式。学习路径在 §8 详述。

### 3.7 完整类型清单

```text
Subject       一级，主体
Situation     一级，情境
Perception    一级，感知事件（带 Subject + Situation）
Belief        一级，主语
Judgment      一级，短命的成立命题
Decision      一级，行动选择
Reflection    一级，元认知记录
Procedure     一级，固化的策略

CognitiveEvent  唯一的写入单元，因果地连接以上类型
```

## 4. 反应式闭环 The Reactive Loop

一次 Reactive Loop 的完整步序：

```text
┌─────────────────────────────────────────────────────────────────┐
│ 1. Perceive                                                       │
│    inputs: raw signal + current Subject + current Situation      │
│    output: Perception                                             │
│    event: "perceived"                                             │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Attend                                                         │
│    pick the salient subset of the Perception                     │
│    output: AttentionFocus { entities, claims, value_signals }    │
│    event: "attended"                                              │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Interpret                                                      │
│    recall relevant Beliefs (projection query)                    │
│    compute stance: consistent / contradicting / novel / ambiguous │
│    output: Interpretation                                         │
│    event: "interpreted"                                           │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Judge                                                          │
│    combine Interpretation with Subject.ValueLens                 │
│    form one or more Judgments                                    │
│    event: "judged"                                                │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Decide                                                         │
│    match Judgments against Procedures library                    │
│    produce Decision (with expected_feedback)                     │
│    event: "decided"                                               │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Act                                                            │
│    execute Decision in the environment                            │
│    event: "acted"                                                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Feedback                                                       │
│    compare actual outcome with Decision.expected_feedback        │
│    annotate involved Beliefs with FeedbackEntry                  │
│    event: "received_feedback"                                     │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. Reflect (L1)                                                  │
│    run rule-based audit over steps 1–7                            │
│    emit Reflections                                               │
│    event: "reflected"                                             │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. Revise                                                         │
│    apply Reflection-driven changes:                              │
│      - form new Beliefs                                           │
│      - supersede old Beliefs                                      │
│      - emit candidate Procedures                                  │
│    event: "revised" + downstream events                          │
└─────────────────────────────────────────────────────────────────┘
```

伪代码：

```python
def reactive_tick(
    stimulus: Stimulus,
    subject_id: SubjectId,
    thread_id: ThreadId,        # 决定从哪个 ContextWindow 取上下文
) -> LoopResult:
    subject = SubjectProjection.current(subject_id)
    situation = SituationBuilder.observe(stimulus, subject)

    perception = Perceiver.perceive(stimulus, subject, situation)
    emit("perceived", perception)

    focus = Attender.focus(perception, subject)
    emit("attended", focus)

    # 装配本轮可见的上下文窗口（详见 §5）
    window = ContextWindowProjection.get(thread_id, subject, at=now())

    recalled = BeliefProjection.recall(focus, subject)
    interpretation = Interpreter.interpret(focus, window, recalled, subject)
    emit("interpreted", interpretation)

    judgments = Judger.judge(interpretation, subject.value_lens)
    emit_all("judged", judgments)

    procedures = ProcedureProjection.match(judgments, subject)
    decision = Decider.decide(judgments, procedures, subject, window=window)
    emit("decided", decision)

    outcome = Effector.execute(decision, window=window)
    emit("acted", outcome)

    feedback = FeedbackReader.compare(decision, outcome)
    emit("received_feedback", feedback)

    reflections = ReflectorL1.audit(
        perception, focus, interpretation, judgments,
        decision, outcome, feedback,
    )
    emit_all("reflected", reflections)

    revisions = Reviser.derive(
        perception, judgments, decision, feedback, reflections,
    )
    for r in revisions:
        emit(r.event_kind, r.payload)

    return LoopResult(decision=decision, reflections=reflections)
```

注意所有数据传递都是值传递；任何一步都不修改前一步的 output。所有 mutation
都通过 `emit(...)` 经 event log，再被 projection 看到。这样一个 tick 是可
回放、可单步、可分叉的。

## 5. 上下文窗口 ContextWindow

为什么单独列一章：Reactive Loop 是主体对外的主通道，每一轮 tick 都要决定
"这一刻能看到哪些东西"。如果只让 `Situation.historical` 一句"当前任务态摘要"
草草带过，会出问题——要么把所有 raw 都塞 prompt 把预算炸了，要么过度压缩把
可引述的原始信息丢了。`§4` 的 Reactive Loop 必须能从一个**显式的、有结构的、
可调度的**上下文容器里取材，这个容器就是 ContextWindow。

### 5.1 定位：是 projection，不是新存储

ContextWindow 与 BeliefProjection、SubjectProjection 同等位——都是事件日志
的 projection。它自己不持有数据，所有内容都从事件日志 derive：

```python
@dataclass(frozen=True)
class ContextWindow:
    thread_id: ThreadId
    foreground: list[Perception]           # 最近 K 个原始感知，逐字
    background: CompressedSummary | None   # 早于 foreground 的部分，已压缩
    recalled: list[BeliefRef]              # 本轮 Interpret 召回的相关信念
    recent_judgments: list[JudgmentRef]    # 本轮形成或仍活跃的判断
    matched_procedures: list[ProcedureRef] # 本轮匹配到的程序
    subject_at: SubjectRef                 # 当时主体快照（含 version）
    situation_at: SituationRef             # 当时情境快照
    assembled_at: Instant
```

四条不能违反的原则：

1. **Raw 永远在事件日志里，永不丢。** `foreground` 只是"这些 raw 当前可见"的
   索引，不是 raw 的副本。需要原文随时回事件日志取。
2. **Background 是有损投影。** 由 Consolidation Loop 维护。它的产出走单独
   的 `context_compressed` 事件 + 一行 projection 行，带 `derived_from:
   [event_ids]`，可随时重算。
3. **压缩是 projection-time，不是 source-of-truth-time。** 事件日志里**永远
   不会**用"压缩过的内容"替代"原始内容"。换 LLM、换 budget、换压缩策略都
   不动事件日志。
4. **ContextWindow 自身不参与认知。** 它没有信念、没有判断、没有反思、没有
   决策，是 view/plumbing，地位与 Renderer 同。**但生成 background 的压缩
   动作是认知**（它在判断"哪些值得保留"），属于 Consolidation Loop，要产生
   cognitive event。这两件事不能混：

```text
ContextWindow（数据结构）   非认知，view
压缩动作（行为）            认知，Consolidation Loop 的事
```

### 5.2 在 Reactive Loop 各步骤里的使用

```text
1 Perceive    ── 只看当前 raw stimulus，不需要 window
3 Interpret   ── 需要 foreground (raw) 做立场比对
                 + background (compressed) 做主题连续性
                 + recalled Beliefs 做长期对照
                 三者一起喂规则或 LLM
5 Decide      ── 默认只用 Judgment + Procedure
                 若 Action 需要"引述上文"才回 foreground 取 raw
6 Act         ── Renderer 把 ContextWindow 渲染成 LLM/工具输入
                 此时由 Renderer 的预算决定 foreground 多少条不压缩、
                 background 用多深的摘要
9 Revise      ── 写 belief 时，sources 引用具体 PerceptionId (raw)，
                 不是摘要——证据链必须能追溯到原文
```

整轮 tick 只装配一次 window：在 Attend 之后、Interpret 之前调
`ContextWindowProjection.get(thread_id, subject, at=now)`，后续所有步骤共享
这一个引用，不重复 query projection。

### 5.3 Consolidation 怎么压缩

压缩在 Consolidation Loop 跑，触发条件可以是多策略叠加：

```text
- 条数：foreground 超过 K 条 → 把最老的 J 条挤进 background
- token 预算：估算 foreground token 数超过阈值
- 主题切换：识别到话题转换，旧主题整段压
- 时间窗：超过 T 时长未刷新就强制压一次
- 显式信号：Reactive 第 9 步 Revise 标记某段已被吸收进 belief
```

压缩动作本身是 cognitive event：

```python
emit("context_compressed", {
    "thread_id": ...,
    "absorbed_event_ids": [...],
    "produced_summary": ...,
    "compression_policy": "...",
    "actor": "ConsolidationLoop",
    "preserved_anchors": [...],   # 必须保留的实体/引文 id
})
```

这让压缩本身可审计、可回放；将来想换压缩策略，replay 这条事件链就能在
projection 里重新产出 background，不动事件日志。

### 5.4 两类 Thread：会话上下文与认知上下文

主体不止有"和用户聊天"这一种 context。至少要分两类：

```python
class ThreadKind(str, Enum):
    CONVERSATION = "conversation"   # 与某个外部对象（用户/其它 agent）的对话线索
    COGNITION    = "cognition"      # 主体内部连贯的思考链（drive loop 自激发）
```

两种 thread 各有自己的 ContextWindowProjection，按 `thread_id` 分。一个
Reactive tick 可以**同时打开多个 window**——例如"我在跟用户讨论 A，但内部
还挂着一个未结的关于 B 的思考"。Interpret/Judge/Decide 在同一 tick 内可以
跨 thread 引用对方的 foreground 与 background。

短期 agent 只会有 conversation thread；Drive Loop 一启动，cognition thread
就自然出现。这是长期运行的主体必备的能力——没有它，主体的"内心独白"无处寄
存，每次 tick 都要从零重建思路。

### 5.5 与 cognition.md 的对应

```text
foreground (raw)          ≈ 过程层"感知 / 注意"的输入材料
background (compressed)   ≈ 情境层 historical + 结构层时间维度
recalled Beliefs          ≈ 结构层网络结构的检索结果
matched Procedures        ≈ 内容层"程序认知"的本轮激活
subject_at + situation_at ≈ 主体层 + 情境层的当下投影
```

ContextWindow 在 Interpret 步合流——这正是 cognition.md "解释建构"那一步
的工程化兑现：把 raw 输入、近期主题、长期信念、当下主体与情境一次性铺开，
留给 Judger 做下一步。

## 6. 多 Loop 并发

Reactive Loop 不是全部。完整认知系统有四条 loop：

```text
┌──────────────────────────────────────────────────────────────────┐
│ Reactive Loop  (per stimulus)                                     │
│   节奏：与外界刺激同步                                              │
│   职责：把一次感知走完七阶段，产 Decision                            │
│   写：perceived/attended/.../revised 全套事件                       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Reflective Loop  (continuous, L2/L3)                              │
│   节奏：流式跟随 Reactive 的事件，必要时阻塞它                       │
│   职责：L2 控制——看到 Reactive 走偏，重写下一轮策略                  │
│         L3 自我模型——把累积 Reflection 投影成 SelfModel             │
│   写：strategy_changed / self_model_updated                       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Consolidation Loop  (idle / scheduled)                            │
│   节奏：每 N 分钟、或主体空闲时                                      │
│   职责：把短命 Judgment 投影成稳定 Belief                            │
│         合并重复 Belief；归档失去 applicability 的 Belief             │
│         从重复成功 Decision 抽取 Procedure                           │
│         维护每条 thread 的 ContextWindow.background（压缩，见 §5.3）│
│   写：belief_formed / belief_superseded / procedure_learned /      │
│       context_compressed                                            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Drive Loop  (intent-paced, optional)                              │
│   节奏：由 open goal 触发，可秒级也可日级                            │
│   职责：从未满足的 goal 中生成主动 Perception                        │
│         驱动 Reactive Loop 在无外部刺激时也跑                        │
│   写：goal_set / external_signal_received（self-issued）            │
└──────────────────────────────────────────────────────────────────┘
```

四条 loop 的数据共享方式是**唯一的**：读 event log + 写 event log。没有共享
可变状态。这避免了并发条件。每条 loop 都可以独立暂停、重启、替换实现。

并发模型推荐：

```text
Reactive 是主线程（或主 task），其它三条是后台 task。
Reflective 可以中断 Reactive：在 Reactive 进入下一步前，检查 reflective
  是否发出了 "strategy_changed"，有则套用新策略。
Consolidation 永远不阻塞 Reactive。
Drive 通过给 Perceiver 投递自生 Stimulus 来触发 Reactive；从外部看，drive
  loop 与外部输入是同构的。
```

## 7. 元认知三级

cognition.md 把元认知列在第七层。在这套设计里，元认知不是层，而是**第二阶认知
系统**——它自己也有 Perceive/Interpret/Judge/Decide，只不过它的"外部世界"就是
L0 的 event log。

### L1 监控 Monitor

读 L0 trace，按规则发 Reflection。不改 L0 行为。

例：

```python
def audit_low_confidence_high_stakes(judgments, beliefs):
    for j in judgments:
        if j.confidence < 0.4 and j.value_weights.get("existence", 0) > 0.7:
            yield Reflection(
                level="L1",
                kind="low-confidence-high-stakes",
                severity="warning",
                target=j.id,
                finding="acting on a low-confidence judgment with existence stakes",
                suggested_remedy="ask user to confirm before acting",
            )
```

L1 是**纯只读**——不写 belief、不改决策。它只贡献 Reflection 事件。

### L2 控制 Control

读 L0 trace + L1 Reflection，**可以阻断或重写下一轮 L0 策略**。

例：当 L1 在最近 K 个 tick 内产生 ≥M 条同 kind 的 warning，L2 触发：

```python
def control_when_recurring_bias_detected(reflections):
    if count_recent(reflections, kind="contradiction-accepted") >= 3:
        return StrategyChange(
            target_stage="Interpret",
            new_policy="require explicit user confirmation before accepting "
                       "any new claim that contradicts active belief",
            valid_for_turns=5,
        )
```

L2 写 `strategy_changed` 事件。下一个 Reactive tick 在每个 stage 入口先查最近
的活跃 strategy override，有则套用。

### L3 自我模型 Self-Model

读全量 L1/L2 历史 + Reactive 历史，慢节奏地投影 `Subject.SelfModel`。

例：

```python
def update_self_model(events: list[CognitiveEvent]) -> SelfModel:
    failure_modes = extract_recurring_failures(events)
    capability_curves = derive_capability_estimates(events)
    return SelfModel(
        capabilities_self_assessed=capability_curves,
        typical_failure_modes=failure_modes,
        ...
    )
```

L3 每天 / 每数百轮跑一次。它的产出回写 Subject，下一轮 Reactive 使用更新后的
Subject。这就是"主体随时间自我修正"。

### 三级的关系

```text
L0 (Reactive)  ──产 events──▶  L1 (Monitor)        只读，发 Reflection
L0 + L1                  ──▶  L2 (Control)        改下一轮 L0 策略
L0 + L1 + L2             ──▶  L3 (Self-Model)     更新 Subject

任意一级都不能改更高一级。L3 是认知系统的"不动点"——但它由低层不断推动迁移。
```

## 8. 学习的五条路径

cognition.md 强调"认知是动态系统"。这套设计里"学习"被拆成五条**互不混淆**的
路径，每条都有触发条件和更新机制：

| 学习类型      | 触发                                | 写入                              | 谁负责         |
| --------- | --------------------------------- | ------------------------------- | ----------- |
| 1. 信念形成   | Reactive 第9步看到 novel claim 被采用    | `belief_formed`                 | Revise      |
| 2. 信念修正   | Interpret 检测到 contradiction       | `belief_superseded`             | Revise      |
| 3. 策略学习   | 同 trigger 下 Decision N 次成功        | `procedure_learned`             | Consolidation |
| 4. 价值学习   | 同 Subject 的 tradeoff 决策模式持续偏向某方向  | `value_lens_shifted`            | L3          |
| 5. 自我模型学习 | Reflection kind 在长窗口里反复出现         | `self_model_updated`            | L3          |

五条路径都通过事件日志写入，所以都是**可审计、可回滚**的。回滚一条 belief 修
正只是把那个事件之后的 projection 重算；不需要 "undo" 接口。

## 9. 数据基底：Event Log + Projections

这部分是从零设计的核心好处之一——**一切都是日志，一切都是从日志投影**。

### 9.1 写路径

```text
任何 CognitionService:
  - 不直接修改 Belief / Subject / Procedure
  - 只调 EventLog.append(event)
  - Projection 异步消费 event log，更新读视图

EventLog 保证：
  - append-only
  - causally ordered within a subject
  - durable
```

### 9.2 读路径

不同的 projection 服务不同的查询：

```text
BeliefProjection
  ── 关系视图：按 cognitive_type / subject / entity / status 查询
  ── 全文/向量视图：按内容相似度查询
  ── 关联视图：belief relation graph

SubjectProjection
  ── 主体当前快照，含 SelfModel

ProcedureProjection
  ── 按 trigger pattern 查询匹配的 procedures

ReflectionProjection
  ── 按 kind / severity / 时间窗 查询

SituationProjection
  ── 历史情境的可检索摘要

ContextWindowProjection
  ── 每个 thread_id 一个，foreground (raw 索引) + background (lossy summary)
  ── Reactive Loop 每轮装配一次；Consolidation Loop 维护 background
```

每个 projection 都是**幂等可重建**的：把对应 event 流重放一遍就能得到当前
视图。这意味着可以随时增加新 projection 而不动写路径。

### 9.3 存储独立性

整个设计**不绑定具体存储**。EventLog 的实现可以是：

```text
SQLite       单机、嵌入式
Postgres     多用户、需要事务
Kafka        高吞吐、跨服务
S3 + DDB     分布式、便宜
本地文件      离线/测试
```

Projection 的实现可以是：

```text
内存索引       开发期
SQLite        默认
向量库        语义检索
图数据库       关联结构查询
```

存储选型是 deploy 时的事，不是设计时的事。这个设计能容纳任意组合。

### 9.4 时间维度

每个 belief 既有 `held_since` / `held_until`（信念有效期），也可以通过日志
查到 `formed_at`（信念形成时刻）、`last_referenced_at`（最近一次被用到）。这
让 cognition.md 的"过去经验 → 当前判断 → 未来预期"三段时间结构是 native 的：

```text
formed_at        — 何时形成
held_since       — 主体从何时开始相信它
last_referenced  — 何时最近一次影响 Reactive
expected_to_hold — applicability 的未来段
held_until       — 主体何时不再相信它
```

每个时间点都是真实的事件锚。

## 10. 表达载体：从信念图到渲染器

LLM 是一个**消费者**，不是核心。如果未来换成多模态模型、规划器、规则引擎，认
知系统不应该跟着重构。

源是 belief graph：subject + 当前 active beliefs + 关系边 + 价值剖面 +
Reflection。**这个图是认知系统的事实**。

渲染器是独立模块，对源做 view：

```text
TextChatRenderer       → 拼成 OpenAI chat completions 格式
ToolUseRenderer        → 拼成 Anthropic tool-use block
StreamThoughtRenderer  → 渲染"内心独白"用于 debug
DiffRenderer           → 渲染两次 tick 之间的 belief 差
GraphSnapshotRenderer  → 渲染 belief 关系图
EvidenceRenderer       → 渲染一条 belief 的全部证据链
```

每个渲染器实现同一接口：

```python
class CognitionRenderer(Protocol):
    def render(self, view: CognitionView, budget: RenderBudget) -> Any: ...
```

`CognitionView` 是从 projection 抽出来的纯数据切片，渲染器拿到的是这个切片。
渲染器之间互不知道。

这样一来：

- **更换 LLM 厂商**：只换 renderer，不动 cognition core。
- **加多模态**：加一个 renderer，老的还能用。
- **调试**：用 GraphSnapshot/Diff renderer 看认知发生了什么。

## 11. 与 Agent Loop 的关系

Agent loop 在这个设计里是 **Reactive Loop 的一个入口**：用户 stimulus 进
Perceiver → 走完 Reactive → Decision 是 LLM 调用 / tool call / 回复 → Act
执行 → Feedback 来自 tool result 或用户下一条消息。

但同等位置还有：

```text
Clock Stimulus       — 周期性触发 Reactive，做 housekeeping 或自检
Drive Stimulus       — Drive loop 把未满足 goal 转成 self-stimulus
Webhook Stimulus     — 外部事件（监控告警、消息到达）触发
Inter-agent Stimulus — 其它认知主体的消息
```

所有 Stimulus 都被 Perceiver 统一处理。Perceiver 不假设 stimulus 是
"user_message"。

工具调用也是 Decision 的一种：

```python
Action = Literal[
    "respond_text",      # 用文本回应
    "ask_clarification", # 反问
    "use_tool",          # 调工具
    "refuse",            # 拒绝
    "defer",             # 延迟
    "no_op",             # 不动作（drive/clock 唤醒后判断不需要动）
    "self_signal",       # 给自己发一条 perception
]
```

`self_signal` 让 Reactive 能驱动自己：例如 Decider 发现需要分两步思考，它可
以决定"先发自我信号 X，第二个 Reactive tick 再答用户"。这就比"一次 LLM 调用走
完"更接近真实认知。

## 12. 实施分阶段

完整系统比较大，但每阶段都能独立验证。

### Phase 0 — 类型与事件日志

- 定义所有 frozen 类型：Subject / Belief / Situation / Perception /
  Judgment / Decision / Reflection / Procedure / CognitiveEvent。
- 实现 EventLog 接口（append + stream + replay）。
- 实现一个内存版 + 一个落盘版。
- 验收：能 append 一串事件，重启后从日志重建出等价 projection。

### Phase 1 — Reactive 单 tick 跑通

- 实现 Perceiver / Attender / Interpreter / Judger / Decider /
  Effector / FeedbackReader / Reviser。
- 每个 stage 是纯函数（除了 emit）。
- 第一版用规则实现 Interpreter / Judger / Decider；LLM 只在 Effector
  里出现，作为一种执行手段。
- 同步实现最小可用的 `ContextWindowProjection`：foreground = 该 thread 最近 K
  个 Perception，没有 background（压缩留到 Phase 4），recalled 暂时为空。
  Reactive 第 3 步开始消费 window。
- 验收：单条用户 stimulus 走完 9 步，事件日志里能看到完整 trace；连续多轮后
  foreground 能正确滚动。

### Phase 2 — BeliefProjection v1

- 实现按 cognitive_type / subject / entity 查询的索引。
- 实现 supersede / merge 的 projection 逻辑。
- 验收：同一断言两次出现不重复成 belief；矛盾两次出现自动 supersede。

### Phase 3 — Reflector L1

- 实现 5–8 条规则化反思。
- Reflection 写日志。
- 验收：能在用户连续矛盾输入时自动报"contradiction-accepted"。

### Phase 4 — Consolidation Loop

- 周期性 task：scan Judgment → promote 稳定者为 Belief；扫 Decision
  历史 → 抽 Procedure。
- **加入 ContextWindow.background 压缩**：按条数 / token / 主题切换 / 时间窗
  触发，emit `context_compressed` 事件，更新 ContextWindowProjection；保留
  关键实体/引文 anchor。
- 验收：第三次出现同模式任务时，Reactive 第 5 步能匹配到 Procedure；超过 K
  条 foreground 后，最老条目正确进入 background 且 raw 仍可在事件日志取回。

### Phase 5 — ValueLens 与冲突解决

- Subject 带 ValueLens。
- Interpreter 检测矛盾后，由 ValueLens 决胜并发 `belief_superseded`。
- 验收：同一主体配置不同 ValueLens 时，对同样输入有不同信念演化。

### Phase 6 — Reflector L2

- 监测重复 reflection → 发 `strategy_changed`。
- Reactive 每个 stage 入口检查活跃 strategy override。
- 验收：在 reflexive 反复矛盾后，下一轮 Reactive 自动加严某 stage 的策略。

### Phase 7 — Renderer 解耦

- 抽出 CognitionView + Renderer 接口。
- 第一个 renderer 就是 TextChat（兼容现行 LLM）。
- 加 GraphSnapshot 与 Diff renderer 供 debug。
- 验收：调试时能可视化"信念图随时间变化"。

### Phase 8 — Drive Loop

- 实现 GoalRegistry 与 self-stimulus generator。
- 验收：未关闭的 goal 在主体空闲时能触发 Reactive 跑一轮思考。

### Phase 9 — Reflector L3 与 SelfModel

- 长窗口分析 Reflection → 更新 Subject.SelfModel。
- SelfModel 反喂下一轮 Reactive 的 Subject。
- 验收：长期使用后 Subject.SelfModel 能描述出主体的典型 failure mode。

## 13. 取舍与边界

### 13.1 为什么是 Belief 中心，不是 Memory 中心

| 维度       | Memory 中心            | Belief 中心            |
| -------- | ------------------- | -------------------- |
| 一级类型     | 记忆条目                | 信念                   |
| 修改方式     | upsert/delete       | append-only event    |
| 主体性      | 边界（scope）           | first-class Subject  |
| 价值层      | 标量 importance       | typed ValueProfile   |
| 元认知      | 外挂                  | 内建第二阶系统              |
| 接 LLM    | 直接拼 prompt          | 经 renderer 解耦         |
| 适合场景     | 简单助理                | 长期主体、可审计、可分叉          |

代价：写入永远是 append，projection 需要异步更新；不能"原地改一条记忆"。这
是一致性换可审计性、可分叉性的取舍——对长期运行的主体值得。

### 13.2 为什么 Event-sourced

+ 完整审计
+ 任意时刻回放
+ 多 projection 并存
+ Cheap branching（A/B 测试两个 Reactive 策略）
+ 并发 loop 无锁共享

- 写放大
- Projection 落后
- Schema 演化需要 versioned event

第一版的负担可以接受，长期回报大。

### 13.3 为什么元认知是第二阶系统而非注释

+ L1/L2/L3 各自有清晰职责
+ 元认知逻辑能复用 Reactive 的设施（也是事件、也走 projection）
+ 主体可以选关掉某一级
+ 自我模型成为 belief 的一部分，自我感是涌现而非假装

- 引入更多事件类型，存储与渲染更复杂

不引入元认知，系统终究无法解释"自己为什么这样想"，无法真正修正自己的偏见。

### 13.4 为什么 Per-Subject 而非 Per-System

cognition.md 强调认知一定属于某个主体。不同主体的 ValueLens / SelfModel /
KnownBiases 都不同。Per-System 的认知会变成"系统的认知"——但系统没有立场。

代价：所有事件都要带 Subject；多主体场景下要考虑跨主体信念共享如何处理（共
享区 + 视角化覆盖）。

## 14. 显式不做

为了不让设计漂移，列出**不打算覆盖**的：

1. **不做完整 Bayesian belief network。** Belief 有 confidence 标量与
   supports/undermined_by 关系；不做严格概率推断。LLM 替我们做模糊推理。
2. **不做形式化 ontology。** concept 类 belief 是自然语言，不是 OWL/RDF。
3. **不做规划器。** Decider 用 Procedure + LLM；不做 STRIPS 式规划。
4. **不做自修改代码。** Reflector L2 改的是 strategy 数据，不改自己的源码。
5. **默认不开 Drive Loop。** 自主目标生成要显式开启；默认主体是被动的。
6. **不做多主体共识协议。** 单主体认知系统；多主体协作另开设计。
7. **不做情绪建模。** ValueProfile 是认知层抽象；具体的情绪状态不是这一系统
   的职责（虽然 cognition.md 提到过情绪——我把它视为 ValueLens.sensitivity 的
   一个特例，不单建一层）。
8. **不做实时性能优化。** Projection 是 eventual；要 hard real-time 的场景
   要在外面套缓存或预计算。

## 15. 一句话概括

> **认知是一个 Subject 在时间中，对 Beliefs 做出修改的过程；这些修改作为
> CognitiveEvents 被记录、被反思、被巩固，并按 ValueLens 解冲突。Reactive
> Loop 只是这条时间线的一种触发方式；Memory 只是这条时间线的一种 projection；
> Prompt 只是这条时间线的一种渲染。**

存储、prompt、agent loop、LLM 厂商——全部是可替换组件。
Belief、CognitiveEvent、Subject、ValueLens——是不可替换的核心。

如果未来要把这套系统从 alpha-agent 抽出去做成独立框架，只要带走第 2、3、4、
8 节描述的核心抽象与事件日志，加上任意一组 projection 实现，就能在新宿主上重
建认知主体。这是这个从零设计相对于"在 memory 之上加薄层"最关键的好处。
