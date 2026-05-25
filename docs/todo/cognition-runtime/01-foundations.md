# Phase 01 — 类型与事件日志

**Status:** pending
**Depends on:** Phase 00
**Scope:** L
**Design ref:** `cognition_from_scratch.md` §1 §2 §3 §9；README 三条架构不变量

## 0. 目标

建立认知运行时的**类型骨架**、**唯一写入通道**与**单主体串行调度**——也即
README 三条不变量的首发实现。这一阶段完成后，代码库里有：

- 一个 `src/alpha_agent/cognition/` 新模块，包含所有 first-class 类型的
  frozen 数据类。Subject 是单例（Agent 自己），Counterpart 是一对多（被服务
  方），两者类型都齐全。
- 一张新表 `cognitive_events`，append-only——整个系统唯一的写入通道。
- 一张新表 `counterpart_view`，CounterpartProjection 的物化视图（事件流仍是
  源头）。
- 一个 `EventLog` 接口与两份实现：内存版（测试用）、SQLite 版（生产用）。
- 一个 `Projection` 基类与一个 trivial demo projection，证明 replay 闭环跑得
  通。
- 一个 `LoopCoordinator`，单主体串行锁的实现，承载"非抢占"调度约束。
- 不需要任何认知行为——这一阶段不调 LLM，不接 Reactive Loop。它只交付**地
  基**。

## 1. 范围

### 1.1 In scope

- 全部一级类型定义：
  - **Subject**（单例，Agent 自己）
  - **Counterpart**（一对多，被服务方）
  - Belief / CognitiveEvent / Situation / Perception / Judgment / Decision /
    Reflection / Procedure / ContextWindow / ValueProfile / ValueLens /
    CognitiveType / ValueKind / Stimulus / ThreadId
- Belief 含 `about: list[Reference]`——指向被该信念描述的对象（Counterpart、
  实体、Subject 自己）。
- Perception 含 `from_counterpart: CounterpartRef | None`——感知来源。
- Stimulus.kind 枚举含 `user_message / tool_result / clock_tick / self_signal
  / webhook / inter_agent`，每种来源决定 `from_counterpart` 怎么填。
- `EventLog` 协议 + 内存实现 + SQLite 实现。
- `Projection` 基类与 replay 协议。
- **CounterpartProjection** 与 `counterpart_view` 表（首次 observe / identity
  升级 / relationship 改变 / trust 更新等事件的物化视图）。
- **LoopCoordinator**：单主体串行锁，所有 loop 走它申请。
- 事件 kind 枚举（`CognitiveEventKind`），含 Counterpart 相关与 lock 相关事
  件（见 §3.5）。
- 一个 demo projection：`EventCountByKind`——纯计数，只为验证 replay。
- 单元测试：append、replay、idempotent rebuild、Counterpart projection 重建、
  LoopCoordinator 串行/排队公平性。

### 1.2 Out of scope

- Reactive Loop 的任何 stage 实现（Phase 02）。
- BeliefProjection（Phase 03）——本阶段只定义 Belief 类型与 about 字段，
  不实现查询。
- ContextWindowProjection（Phase 04）——本阶段只定义 ContextWindow 类型。
- Reflector / Consolidation / Drive / Renderer（后续 Phase）。
- 把 `AlphaAgent.respond()` 接到新事件日志——这一阶段它仍走 Phase 00 留下
  的最简版。

## 2. 任务清单

### 2.1 数据模型 / 类型

- [ ] `cognition/models/_ids.py`：新类型别名集合
  - 标识：`SubjectId`、`CounterpartId`、`BeliefId`、`EventId`、`SituationId`、
    `PerceptionId`、`JudgmentId`、`DecisionId`、`ReflectionId`、`ProcedureId`、
    `ThreadId`。
  - 引用：`Reference`（联合）、`SubjectRef`、`CounterpartRef`、`BeliefRef`、
    `JudgmentRef`、`ProcedureRef`、`ReflectionRef`、`EvidenceRef`、
    `SituationRef`、`EntityRef`、`ActorRef`、`StrategyRef`。
  - 主体相关：`Capability`、`Need`、`Role`、`GroupRef`、`BiasMarker`、
    `ConfidenceCurve`、`FailurePattern`、`ValueTradeoff`、`InteractionPattern`、
    `SelfModel`。
  - Counterpart 相关：`CounterpartRole`、`Relationship`、`ServiceCommitment`、
    `StyleHint`。
  - 内容相关：`BeliefRelation`、`ActionHint`、`FeedbackEntry`、`UpdatePolicy`、
    `Lifecycle`、`IntentMarker`、`ExpectedFeedback`、`Action`、`Severity`、
    `ReflectionKind`、`ReflectionTarget`、`RemedyHint`、`TriggerPattern`、
    `Step`、`NLStatement`、`StructuredClaim`、`DerivationTrace`、
    `Applicability`、`CompressedSummary`、`MetaEval`。
  - 情境相关：`PhysicalContext`、`SocialContext`（**含 Counterpart 引用**）、
    `InstitutionalContext`、`InformationalContext`、`CulturalContext`、
    `HistoricalContext`。
  - 时间：`Instant`。
- [ ] `cognition/models/enums.py`：
  - `CognitiveType`、`ValueKind`、`ThreadKind`、`CognitiveEventKind`、
    `CounterpartRole`、`StimulusKind`。
- [ ] `cognition/models/subject.py`：`Subject`——单例，含
  `served_counterparts: list[CounterpartRef]`。同模块导出常量
  `SUBJECT_SELF: SubjectId = SubjectId("agent:self")`——所有引用主体 id
  的代码（emitter / coordinator / event log / projection）都从这里 import，
  不允许出现字面量 `"agent:self"`。
- [ ] `cognition/models/counterpart.py`：`Counterpart` + `ServiceCommitment` +
  `Relationship` + `StyleHint`。
- [ ] `cognition/models/belief.py`：`Belief`，含 `about: list[Reference]`
  表达"这条信念关于谁"。
- [ ] `cognition/models/situation.py`：`Situation`；其 SocialContext 引用
  当前在场的 Counterpart 列表。
- [ ] `cognition/models/perception.py`：`Perception`，含
  `from_counterpart: CounterpartRef | None`；`Stimulus`，含
  `source: CounterpartRef | None`。
- [ ] `cognition/models/judgment.py`：`Judgment`。
- [ ] `cognition/models/decision.py`：`Decision`。
- [ ] `cognition/models/reflection.py`：`Reflection`。
- [ ] `cognition/models/procedure.py`：`Procedure`。
- [ ] `cognition/models/context_window.py`：`ContextWindow`，含
  `counterpart: CounterpartRef | None`——会话 thread 关联到一个 Counterpart；
  cognition thread 为 None。一并定义 `CompressedSummary`。
- [ ] `cognition/models/thread.py`：`ThreadId` 数据类与 `ThreadKind` 枚举
  实现（CONVERSATION / COGNITION）；提供 `from_session(session_id,
  source_metadata)` 与 `cognition(subject_id, topic)` 两个构造器。Phase 02
  及之后所有 stimulus 路由都引用此模块。
- [ ] `cognition/models/value.py`：`ValueProfile`、`ValueLens`。
- [ ] `cognition/models/event.py`：`CognitiveEvent`。
- [ ] `cognition/models/__init__.py`：聚合 export。

所有 dataclass 一律 `frozen=True`。所有列表/字典字段都用 `field(default_factory=...)`。

### 2.2 EventLog 与 EventEmitter

- [ ] `cognition/event_log/base.py`：`EventLog` Protocol（见 §3.2）。
- [ ] `cognition/event_log/memory.py`：`InMemoryEventLog`。
- [ ] `cognition/event_log/sqlite.py`：`SQLiteEventLog`。
- [ ] `cognition/emitter.py`：`EventEmitter`——`EventLog.append` 的轻量
  wrapper，把 caller 不关心的字段（id 生成、timestamp、actor、
  `subject_version`、causal_parents）自动填好。所有 Reactive stage /
  worker / aggregator 写事件都走这里，不直接调 `EventLog.append`。
- [ ] `state/schema.sql`：在 Phase 00 留下的 schema.sql 末尾追加
  `cognitive_events` 表（见 §3.3）。注意 schema 文件归属——事件日志虽然属于
  cognition，但表与现有 `conversation_messages` 共库，所以 DDL 仍放在
  `state/schema.sql`。
- [ ] `state/store.py`：扩展 `StateStore` 以提供事件日志相关连接（不暴露事件
  逻辑，只提供 sqlite connection）。
- [ ] `cognition/event_log/sqlite.py` 通过 `StateStore` 取连接，不自己 open。

### 2.3 Projection

- [ ] `cognition/projections/base.py`：
  - `Projection` 抽象基类，方法：`apply(event) -> None`、`reset() -> None`、
    `view() -> ProjectionView`。
  - `ProjectionView` 标记接口。
- [ ] `cognition/projections/registry.py`：`ProjectionRegistry`——按名字
  / 类型查找已注册的 projection 实例。后续阶段（02 起）所有需要读多张
  projection 的代码（CognitiveController / ReflectorL1 / Renderer /
  Consolidation worker）都通过 registry 取，不直接持有具体 projection 引用。
- [ ] `cognition/projections/event_count.py`：demo `EventCountByKind`
  projection。
- [ ] `cognition/projections/counterpart.py`：`CounterpartProjection`——
  处理 counterpart_first_observed / counterpart_identified /
  counterpart_relationship_changed / service_committed / service_fulfilled /
  service_failed / trust_updated 等事件，物化到 `counterpart_view` 表。提供
  `get(id) / list_active() / by_role(role)` 三个查询。
- [ ] `cognition/projection_runner.py`：通用 replay runner——给定一个
  EventLog 与一组 Projection，按事件流顺序 dispatch。

### 2.4 调度：LoopCoordinator

- [ ] `cognition/coordinator.py`：`LoopCoordinator`（见 §3.6）。
  - 内部维护 `current_holder: str | None` + FIFO waiter queue（带优先级）。
  - 提供 `acquire(req)` 阻塞 context manager——给调度型 loop（L2 / Drive
    / Consolidation / L3）用。
  - 提供 `try_acquire(req)` 非阻塞 context manager——给 Reactive 专用，
    锁被占时不等待，直接抛 `LockBusy(holder, since)`。
  - 提供 `current_holder()`、`waiting()`、`yield_to_higher_priority()`。
  - 锁粒度：每个 SubjectId 一把锁。因为整个系统只有一个 Subject，本阶段其
    实是单全局锁；保留 `subject_id` 参数为将来可能扩展留口。
- [ ] `cognition/coordinator.py` 同文件：`LoopPriority` 枚举（reactive=0、
  l2=1、drive=2、consolidation=3、l3=4；数字小者优先；reactive 只用于
  日志标记，不参与排队优先级）。
- [ ] 关键 invariant 测试：
  - try_acquire 在锁被持有时立即抛 LockBusy，不阻塞。
  - acquire 在锁被持有时阻塞直到 holder 释放（或自觉 yield 让位）。
  - 持锁中的 holder 不会被强制中断——`yield_to_higher_priority()` 是
    holder 自己调，coordinator 永远不发 cancel。

### 2.5 测试

- [ ] `tests/cognition/test_event_log_memory.py`：append/iter/length。
- [ ] `tests/cognition/test_event_log_sqlite.py`：append、replay 不丢序、跨
  进程持久化。
- [ ] `tests/cognition/test_projection_runner.py`：用 EventCountByKind 验证
  - 同一事件流跑两次得到等价 view；
  - 中途 reset 后能从头重建；
  - 新增 projection 后能从历史回放出当前状态。
- [ ] `tests/cognition/test_types_frozen.py`：所有模型类型都是 frozen 且可
  hash（如果需要）。
- [ ] `tests/cognition/test_counterpart_projection.py`：
  - first_observed / identified / relationship_changed / trust_updated 事件
    序列 → view 状态正确；drop view → replay → 等价。
- [ ] `tests/cognition/test_belief_about_field.py`：构造 Belief 含 `about=
  [counterpart_ref, entity_ref]`，序列化/反序列化往返一致。
- [ ] `tests/cognition/test_loop_coordinator_serial.py`：
  - 两个调度型 acquire 串行执行；高优先级 acquire 在低优先级 holder 释放
    前阻塞。
- [ ] `tests/cognition/test_loop_coordinator_try_acquire_busy.py`：
  - 低优先级 holder 持锁中 → Reactive try_acquire 立即抛 LockBusy，
    LockBusy.holder 与 since 字段正确填充。
- [ ] `tests/cognition/test_loop_coordinator_try_acquire_free.py`：
  - 无 holder → Reactive try_acquire 立即拿到锁，正常退出。
- [ ] `tests/cognition/test_loop_coordinator_yield.py`：
  - 低优先级 holder 在 `max_chunk_duration` 后调 yield → 若有更高调度型
    优先级等待则让位，否则 holder 续锁。
- [ ] `tests/cognition/test_loop_coordinator_no_preemption.py`：
  - 即使 Reactive 反复 try_acquire 失败，coordinator 不会强制让低优先级
    holder 释放——只能由 holder 主动 yield。

### 2.6 文档

- [ ] 更新 `AGENTS.md` 项目导航：`src/alpha_agent/cognition/` 一行 + 简介。
- [ ] 在仓库根 `README.md` 加一行 "Cognition runtime under construction; see
  `docs/todo/cognition-runtime/`"（如果 Phase 00 还没加）。

## 3. 接口契约（草案）

### 3.1 类型字段总览

完整字段表见 `cognition_from_scratch.md` §2、§3、§5.1。这里只列**必须 Phase
01 落齐**的字段（不能等到后续阶段）：

```python
@dataclass(frozen=True)
class CognitiveEvent:
    id: EventId
    kind: CognitiveEventKind
    subject: SubjectRef
    subject_version: int
    situation: SituationRef
    inputs: list[Reference] = field(default_factory=list)
    outputs: list[Reference] = field(default_factory=list)
    rationale: NLStatement = ""
    timestamp: Instant = ...
    actor: ActorRef = ...
    causal_parents: list[EventId] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)   # 阶段性补字段时用
```

`payload` 是逃生通道：阶段性补字段时先塞 payload，过两阶段稳定了再 promote 为
正式 field（伴随事件 schema 版本号 bump）。

#### Subject（单例）

```python
SUBJECT_SELF: SubjectId = SubjectId("agent:self")

@dataclass(frozen=True)
class Subject:
    id: SubjectId                                # 固定为 SUBJECT_SELF
    role: Role                                   # 通常 "agent"
    capabilities: list[Capability]               # 当前能力快照
    declared_needs: list[Need]                   # 当前需求（罕用，主要靠 goal）
    value_lens: ValueLens                        # 价值排序，Phase 07 演化
    self_model: SelfModel                        # L3 投影，Phase 11 演化
    served_counterparts: list[CounterpartRef]    # 当前在服务哪些 Counterpart
    known_biases: list[BiasMarker]               # 已识别偏向
    held_at: Instant
```

#### SelfModel

```python
@dataclass(frozen=True)
class SelfModel:
    capabilities_self_assessed: dict[Capability, ConfidenceCurve] = field(default_factory=dict)
    typical_failure_modes: list[FailurePattern] = field(default_factory=list)
    preferred_strategies: list[StrategyRef] = field(default_factory=list)
    stable_preferences: list[BeliefRef] = field(default_factory=list)
    typical_value_tradeoffs: list[ValueTradeoff] = field(default_factory=list)
    interaction_patterns_by_counterpart_role: dict[CounterpartRole, InteractionPattern] = field(default_factory=dict)
```

SelfModel 全字段一次写齐——Phase 01 不需要等到 Phase 11 才声明
`interaction_patterns_by_counterpart_role`。Phase 11 只是第一个把这些字段
写出非空值的阶段；schema 在 Phase 01 就稳定，避免后续 schema 演化。

#### Counterpart（一对多）

```python
@dataclass(frozen=True)
class Counterpart:
    id: CounterpartId
    role: CounterpartRole                        # user / operator / peer_agent / system / anonymous
    identity: dict[str, Any]                     # 平台 / 显示名 / 联系方式
    relationship: Relationship                   # served_by_agent / instructed_agent / consulted / observed
    service_contract: list[ServiceCommitment]
    trust_level: float                           # [0, 1]，Agent 对其输入的信任度
    communication_style: list[StyleHint]         # 风格倾向；Renderer 用
    first_seen_at: Instant
    last_interaction_at: Instant
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### Belief（含 `about`）

```python
@dataclass(frozen=True)
class Belief:
    id: BeliefId
    subject: SubjectRef                          # 永远是 Agent 自己
    about: list[Reference]                       # 这条信念关于谁：CounterpartRef / EntityRef / SubjectRef
    object: str                                  # 信念指向的对象描述
    content: NLStatement
    cognitive_type: CognitiveType
    structure: StructuredClaim | None
    sources: list[EvidenceRef]
    confidence: float
    applicability: Applicability
    value_profile: ValueProfile
    relations: list[BeliefRelation]
    formed_in: SituationRef
    holder_role: Role
    action_orientation: list[ActionHint]
    feedback_history: list[FeedbackEntry] = field(default_factory=list)
    update_policy: UpdatePolicy
    status: Lifecycle
    held_since: Instant
    held_until: Instant | None = None
    superseded_by: BeliefRef | None = None
    supersedes: BeliefRef | None = None
    self_audit: list[ReflectionRef] = field(default_factory=list)
```

注意：`subject` 字段虽然是 first-class，但本系统只有一个 Subject，所以这一字
段实际上是常量 `agent:self`。保留它是为了将来如果设计扩展到多主体框架时不破
坏 schema。**信念真正的"关于谁"信息在 `about` 里**。

#### Perception（含 `from_counterpart`）

```python
@dataclass(frozen=True)
class Stimulus:
    kind: StimulusKind                # user_message / tool_result / clock_tick /
                                       # self_signal / webhook / inter_agent
    source: CounterpartRef | None     # 用户消息 / 同伴 agent 时填；clock/self_signal 填 None
    payload: Any
    thread_id: ThreadId
    received_at: Instant

@dataclass(frozen=True)
class Perception:
    id: PerceptionId
    source_kind: StimulusKind
    from_counterpart: CounterpartRef | None    # 镜像 Stimulus.source；为 None
                                                # 表示来自 Subject 自己或非 Counterpart 环境
    raw: Any
    surface_intent: list[IntentMarker]
    raised_entities: list[EntityRef]
    subject: SubjectRef
    situation: SituationRef
    received_at: Instant
```

#### Situation（SocialContext 引用 Counterparts）

```python
@dataclass(frozen=True)
class SocialContext:
    present_counterparts: list[CounterpartRef]   # 当前情境下"在场"的对方
    authority_hints: dict[CounterpartRef, str]   # e.g. counterpart:user_a → "operator"
    group_dynamics: list[str]                    # 私聊 / 群聊 / 公开广播 等
```

#### ContextWindow（关联 Counterpart）

```python
@dataclass(frozen=True)
class ContextWindow:
    thread_id: ThreadId
    counterpart: CounterpartRef | None           # conversation thread 有；cognition thread 为 None
    foreground: list[Perception]
    background: CompressedSummary | None
    recalled: list[BeliefRef]
    recent_judgments: list[JudgmentRef]
    matched_procedures: list[ProcedureRef]
    subject_at: SubjectRef
    situation_at: SituationRef
    assembled_at: Instant
```

### 3.2 EventLog 协议

```python
class EventLog(Protocol):
    def append(self, event: CognitiveEvent) -> EventId: ...
    def get(self, event_id: EventId) -> CognitiveEvent: ...
    def iter(
        self,
        *,
        subject: SubjectRef | None = None,
        kinds: Iterable[CognitiveEventKind] | None = None,
        since: Instant | None = None,
        until: Instant | None = None,
    ) -> Iterator[CognitiveEvent]: ...
    def length(self, *, subject: SubjectRef | None = None) -> int: ...
```

关键约定：

- `append` 是**唯一**的写入入口。
- 同一 subject 下的事件按 monotonic ordinal 序，跨 subject 不保证全局序。
- 删除不存在——`append("belief_retracted", ...)` 是事件，不是物理删事件。

### 3.3 `cognitive_events` 表

```sql
CREATE TABLE IF NOT EXISTS cognitive_events (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    subject_version INTEGER NOT NULL,
    situation_id TEXT,
    actor TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    inputs TEXT NOT NULL DEFAULT '[]',
    outputs TEXT NOT NULL DEFAULT '[]',
    causal_parents TEXT NOT NULL DEFAULT '[]',
    payload TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    UNIQUE(subject_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_cognitive_events_subject_time
    ON cognitive_events(subject_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_cognitive_events_kind_time
    ON cognitive_events(kind, timestamp);
```

`schema_version` 为后续事件 schema 演化预留——v1 永远向前兼容地能 replay。

### 3.4 Projection 协议

```python
class Projection(ABC):
    name: ClassVar[str]
    handles: ClassVar[frozenset[CognitiveEventKind]]

    @abstractmethod
    def apply(self, event: CognitiveEvent) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def view(self) -> Any: ...

class ProjectionRegistry:
    """所有 projection 的查询入口；按 name 或 type 取出实例。"""
    def register(self, projection: Projection) -> None: ...
    def get(self, name: str) -> Projection: ...
    def get_typed(self, cls: type[P]) -> P: ...
    def all(self) -> Iterable[Projection]: ...

class ProjectionRunner:
    def __init__(self, log: EventLog, registry: ProjectionRegistry): ...
    def replay_all(self) -> None: ...
    def apply_one(self, event: CognitiveEvent) -> None: ...
```

`handles` 是该 projection 关心的事件 kind 白名单——避免每个 projection 把
全量事件都过一遍。`ProjectionRegistry` 是后续阶段的统一注入点：
CognitiveController、Reflector、Renderer、worker 等都接收 registry，按
需取出具体 projection；测试时只需注册 stub 版即可替换。

### 3.5 事件 kind 首版列表

```python
CognitiveEventKind = Literal[
    # Reactive loop 内部（Phase 02）
    "perceived", "attended", "interpreted", "judged",
    "decided", "acted", "received_feedback", "reflected", "revised",

    # 信念生命周期（Phase 03）
    "belief_formed", "belief_strengthened", "belief_weakened",
    "belief_superseded", "belief_retracted",

    # 元认知（Phase 05/08/11）
    "bias_detected", "strategy_changed", "strategy_expired",
    "self_model_updated",

    # 学习与价值（Phase 06/07）
    "procedure_learned", "procedure_strengthened", "procedure_weakened",
    "procedure_matched",
    "value_lens_shifted", "context_compressed",
    "consolidation_conflict_queued", "conflict_kept_for_human_review",
    "belief_archived", "belief_form_pending_confirmation",
    "context_anchor_set", "context_anchor_cleared",

    # Counterpart 生命周期
    "counterpart_first_observed",
    "counterpart_identified",
    "counterpart_relationship_changed",
    "service_committed",
    "service_fulfilled",
    "service_failed",
    "trust_updated",

    # LoopCoordinator（用于调度审计，不强制 emit 每次 acquire/release）
    "loop_acquired",
    "loop_released",
    "loop_yielded",

    # Drive / 外部（Phase 10）
    "goal_set", "goal_satisfied", "goal_abandoned", "goal_progressed",
    "external_signal_received",
]
```

`loop_*` 事件**只在 acquire 等待 > 1s 或主动 yield 时**写日志；高频 acquire
不写，避免日志膨胀。

### 3.6 LoopCoordinator API

```python
class LoopPriority(IntEnum):
    REACTIVE      = 0    # 标记用；Reactive 走 try_acquire，不进入排队
    L2            = 1
    DRIVE         = 2
    CONSOLIDATION = 3
    L3            = 4

@dataclass(frozen=True)
class LoopAcquireRequest:
    loop_name: str
    priority: LoopPriority
    max_chunk_duration: timedelta   # 自觉分块上限；不是强制硬中断时间

class LockBusy(Exception):
    """try_acquire 在锁被占时抛出。"""
    def __init__(self, holder: str, since: Instant):
        self.holder = holder
        self.since = since

class LoopCoordinator:
    """
    单 Subject 内的串行调度器。
    - 同一时刻至多一个 holder。
    - 调度型 loop 用 acquire（阻塞 + 优先级排队）。
    - Reactive 用 try_acquire（非阻塞；锁忙时抛 LockBusy，由 caller 决定如
      何回复用户）。
    - 没有抢占——锁拥有者自己决定何时让出。
    """

    def __init__(self, subject_id: SubjectId): ...

    @contextmanager
    def acquire(self, req: LoopAcquireRequest) -> Iterator[None]:
        """阻塞直到拿到锁。给 L2 / Drive / Consolidation / L3 用。"""

    @contextmanager
    def try_acquire(self, req: LoopAcquireRequest) -> Iterator[None]:
        """非阻塞。锁忙时抛 LockBusy。给 Reactive 专用。"""

    def yield_to_higher_priority(self) -> bool:
        """holder 自觉调用；若有更高 priority 等待则返回 True 并放锁。
        即使没有调度型等待者，coordinator 也会瞬时释放锁、给可能正在
        探测的 Reactive 一个窗口、然后让 holder 续锁。"""

    def current_holder(self) -> str | None: ...
    def waiting(self) -> list[tuple[str, LoopPriority]]: ...
```

约定：

- Reactive 拿不到锁时**不阻塞**——它捕获 LockBusy 后立即返回 "agent is
  busy" 系统提示，且这一次 stimulus **不写任何 cognitive event 也不写
  conversation_messages**。从主体视角，这次请求未曾发生。具体 contract
  在 Phase 02 §2.4 落地。
- 低优先级 holder 在每个 `max_chunk_duration` 边界 **MUST** 调
  `yield_to_higher_priority()`；这是 contract，违反就算 bug。
- coordinator 不杀线程、不发信号；纯协作式。
- 单 Subject 时 coordinator 是单 mutex；保留 subject_id 参数是为将来扩展。

### 3.7 `counterpart_view` 表

```sql
CREATE TABLE IF NOT EXISTS counterpart_view (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    identity TEXT NOT NULL DEFAULT '{}',           -- JSON dict
    relationship TEXT NOT NULL DEFAULT 'observed',
    service_contract TEXT NOT NULL DEFAULT '[]',
    trust_level REAL NOT NULL DEFAULT 0.5,
    communication_style TEXT NOT NULL DEFAULT '[]',
    first_seen_at TEXT NOT NULL,
    last_interaction_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    last_event_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_counterpart_role
    ON counterpart_view(role, last_interaction_at DESC);
```

`gateway/session.py`（Phase 00 之后保留的部分）应该在新会话到达时调用
`CounterpartProjection.upsert_from_source_metadata(...)` 触发
`counterpart_first_observed` 事件——这件具体接线属于 Phase 02 的工作，本阶段
只提供 projection 接口与表。

## 4. 文件清单

### 4.1 新增

```text
src/alpha_agent/cognition/__init__.py
src/alpha_agent/cognition/models/__init__.py
src/alpha_agent/cognition/models/_ids.py
src/alpha_agent/cognition/models/enums.py
src/alpha_agent/cognition/models/subject.py
src/alpha_agent/cognition/models/counterpart.py
src/alpha_agent/cognition/models/belief.py
src/alpha_agent/cognition/models/situation.py
src/alpha_agent/cognition/models/perception.py
src/alpha_agent/cognition/models/judgment.py
src/alpha_agent/cognition/models/decision.py
src/alpha_agent/cognition/models/reflection.py
src/alpha_agent/cognition/models/procedure.py
src/alpha_agent/cognition/models/context_window.py
src/alpha_agent/cognition/models/value.py
src/alpha_agent/cognition/models/event.py
src/alpha_agent/cognition/models/thread.py
src/alpha_agent/cognition/event_log/__init__.py
src/alpha_agent/cognition/event_log/base.py
src/alpha_agent/cognition/event_log/memory.py
src/alpha_agent/cognition/event_log/sqlite.py
src/alpha_agent/cognition/emitter.py
src/alpha_agent/cognition/projections/__init__.py
src/alpha_agent/cognition/projections/base.py
src/alpha_agent/cognition/projections/registry.py
src/alpha_agent/cognition/projections/event_count.py
src/alpha_agent/cognition/projections/counterpart.py
src/alpha_agent/cognition/projection_runner.py
src/alpha_agent/cognition/coordinator.py
tests/cognition/__init__.py
tests/cognition/test_event_log_memory.py
tests/cognition/test_event_log_sqlite.py
tests/cognition/test_projection_runner.py
tests/cognition/test_types_frozen.py
tests/cognition/test_counterpart_projection.py
tests/cognition/test_belief_about_field.py
tests/cognition/test_loop_coordinator_serial.py
tests/cognition/test_loop_coordinator_try_acquire_busy.py
tests/cognition/test_loop_coordinator_try_acquire_free.py
tests/cognition/test_loop_coordinator_yield.py
tests/cognition/test_loop_coordinator_no_preemption.py
```

### 4.2 修改

```text
src/alpha_agent/state/schema.sql       追加 cognitive_events 与 counterpart_view 两张表
src/alpha_agent/state/store.py         暴露给 cognition 的 sqlite connection helper
AGENTS.md                              项目导航补 cognition/ 一行（含 coordinator/）
```

### 4.3 删除

无。

## 5. 验收标准

- [ ] `uv run pytest tests/cognition/ -q` 全绿。
- [ ] 模块 import 检查：`python -c "from alpha_agent.cognition import models,
  event_log, projections, coordinator"` 不报错。
- [ ] SQLite event log：append 1000 条 → 关进程 → 重开 → iter 出来 1000 条且
  顺序一致。
- [ ] EventCountByKind projection：清空状态后 `replay_all`，view 与逐事件
  apply 等价。
- [ ] CounterpartProjection：drop `counterpart_view` → replay → 等价。
- [ ] 所有模型类型 `dataclasses.is_dataclass(x)` 与 `getattr(x, "__hash__",
  None) is not None`（frozen=True 给定的 hash）通过。
- [ ] `Belief.about` 字段在序列化/反序列化中保持 list 顺序与引用类型。
- [ ] LoopCoordinator 单元测试覆盖：
  - 串行 acquire / release；
  - 高优先级阻塞等待，holder 主动 yield 时让位；
  - 无 yield 时不被强制中断（即 acquire 调用时不会触发任何信号 / 异步取消）。
- [ ] `AlphaAgent.respond()` 行为与 Phase 00 末态**完全一致**——这一阶段
  不接触它。

## 6. 风险与备注

- **类型一次写齐 vs 渐进**。我选一次写齐——所有 frozen 类型不引入运行时成本，
  但避免后续每阶段都得 hack 字段。代价是 Phase 01 文件多，但每个文件都小。
- **schema 演化策略**。事件 schema 一旦上线，破坏性改是禁止的；新增字段优
  先（向前兼容），并 bump `schema_version`。如果将来发现某个字段必须重命名，
  做法是"新增字段 + 双写一段时间 + 旧字段 deprecate"——但这不应该在 Phase
  01 操心。
- **payload 逃生通道**。某些字段还没想清楚时（例如 `derivation` 怎么序列化），
  先放 payload。下一阶段再 promote。
- **measure twice, cut once**。在动手前把所有类型字段过一遍 cognition.md 十
  五字段对照——Belief 是否全字段都到位？Subject 是否包含 SelfModel？Belief
  是否有 `about`？Perception 是否有 `from_counterpart`？这一阶段改字段成本最
  低，下阶段就要碰事件 schema 了。
- **Subject vs Counterpart 不要混**。Belief 上的 `subject` 永远是
  `SUBJECT_SELF`；"这条 belief 是关于谁"通过 `about` 列表表达。code review
  时盯紧这条不变量，否则后续 audit / renderer 全乱。
- **Counterpart 创建时机**。Counterpart 不预先存在。Phase 02 之后，gateway
  收到新平台用户的第一条消息 → emit `counterpart_first_observed` → projection
  物化 → Reactive 拿到 CounterpartRef。本阶段只准备 projection 与表，不接
  gateway。
- **LoopCoordinator 是协作而非抢占**。低优先级 loop 不调 yield 就一直占着，
  Reactive 反复 try_acquire 收 busy、用户体验下降。worker 实现者必须自觉调
  yield。Phase 06/08/10/11 各自的 worker 文档都会把这条复述一遍。
- **测试避免依赖具体 timestamp / id 生成**。注入 clock / id generator，事件
  比较时归一化。

## 7. 后续衔接

Phase 02 在这一阶段交付的类型 / event_log / coordinator 之上：

- 实现 7 个 stage 模块 + 1 个 controller。
- 用 `EventLog.append(...)` 发 perceived / attended / interpreted / judged /
  decided / acted / received_feedback / reflected / revised 事件。
- 用 `InMemoryEventLog` 跑测试；用 `SQLiteEventLog` 接 `AlphaAgent.respond()`。
- 在 `respond()` 入口 `LoopCoordinator.try_acquire(LoopPriority.REACTIVE,
  ...)` ——锁忙时立即抛 LockBusy，由 `respond()` 转成 busy 文本返回；锁
  空闲则正常走 9 步链。这是 Reactive 与其它 loop 并发安全的唯一保证。
- 把 `source_metadata` → CounterpartRef 的映射在 `respond()` 里做：调
  `CounterpartProjection.upsert_from_source_metadata(...)`，可能 emit
  `counterpart_first_observed` 事件，再把得到的 CounterpartRef 放进 Stimulus
  与 Perception。

Phase 03 / 04 各自继承 `Projection` 实现自己的视图：
- Phase 03 BeliefProjection 必须支持按 `about` 字段查询（"关于 user_a 的信
  念有哪些"）。
- Phase 04 ContextWindowProjection 必须区分 conversation thread（关联
  Counterpart）与 cognition thread（无 Counterpart）。

Phase 06 / 08 / 10 / 11 各自的 loop 都通过本阶段提供的 LoopCoordinator 申请
锁，并自觉分块 yield。
