# Phase 11 — Reflector L3 / SelfModel

**Status:** pending
**Depends on:** Phase 05, Phase 06, Phase 08
**Scope:** M
**Design ref:** `cognition_from_scratch.md` §2.1（Subject.SelfModel）、§7（L3）、
§8（学习路径 5）；README 不变量 1/2

## 0. 目标

补完元认知三级的最后一级——L3 SelfModel：跨长窗口聚合 L1/L2 反思 + Reactive
历史 + Consolidation 输出 → 投影出 `Subject.SelfModel` 的稳态字段
（能力自评 / 典型失误 / 偏好策略 / 稳定偏好引用 / 价值取舍模式 / 按对方角色
聚合的交互模式）。

这一阶段是认知系统的"长期自我感"——主体在长时间运行后，能给出"我倾向于…
…"、"我经常在…失败"、"我对 operator 比对 user 更精简"这类自述。

L3 是最慢的 loop，priority 最低，仍走 LoopCoordinator，但 cadence 是日级，
几乎从不与 Reactive 冲突；冲突时无条件让位。SelfModel 描述的是 **Agent 自
己**——不是用户。L3 不写关于具体 Counterpart 的 belief（那是 Phase 06
summarize_counterpart worker 的事），它只写 Subject.SelfModel；其中可以
包含按 CounterpartRole 聚合的交互模式（"我与 user 角色平均成功率 X、与
operator 角色平均成功率 Y"），这仍是关于 Agent 的描述。

## 1. 范围

### 1.1 In scope

- `ReflectorL3`：低频（默认每日一次）跑全量聚合。
- 5 个 SelfModel 字段的投影：
  - `capabilities_self_assessed`
  - `typical_failure_modes`
  - `preferred_strategies`
  - `stable_preferences`
  - `typical_value_tradeoffs`
- emit `self_model_updated` 事件。
- SubjectProjection 接收 self_model_updated → 反映在 `Subject.SelfModel`。
- Reactive 下一轮装配 Subject 时拿到的就是新 SelfModel。
- CLI：`alpha cognition self-model [--subject]`。
- 测试：构造长事件流 → 跑 L3 → 验证 SelfModel 字段。

### 1.2 Out of scope

- LLM-based 自我叙述。
- 多主体对比。
- SelfModel 主动驱动 Reactive 决策——SelfModel 只通过更新 Subject 间接影响。

## 2. 任务清单

### 2.1 L3 主模块

- [ ] `cognition/reflectors/l3.py`：`ReflectorL3`。
  - `run_once()`：聚合 → 计算字段 → emit。
  - 入口先 `coordinator.acquire(LoopAcquireRequest(loop_name="reflector_l3",
    priority=LoopPriority.L3, max_chunk_duration=timedelta(seconds=60)))`。
  - 每个 aggregator 跑完调 yield；若有调度型等待者就停下下次再续
    （aggregator 需 resumable，与 Phase 06 worker 同要求）。
  - 通过 Phase 06 通用 scheduler 调度，触发条件：

    ```python
    ScheduleTrigger(
        min_interval=timedelta(hours=24),
        max_interval=None,                    # 没新东西就永远不跑
        watches=frozenset({
            "reflected", "bias_detected",
            "strategy_changed", "belief_superseded",
            "procedure_learned", "value_lens_shifted",
        }),
        min_new_events=50,                    # 量变到一定程度才更新 SelfModel
    )
    ```

    用户长期不用时 L3 不跑——SelfModel 也不会无意义地反复 emit 同样内容。
    `min_new_events=50` 与 §2.4 的 12h emit throttle 互相加强，确保
    SelfModel 不抖动。
- [ ] `cognition/reflectors/l3_aggregators/`：每个 SelfModel 字段一个聚合器
  文件。

### 2.2 聚合器

- [ ] `capabilities_aggregator.py`：扫 procedure_view 的 success/failure 比
  + 工具 / cognitive_type 的成功率 → 输出 `dict[Capability, ConfidenceCurve]`。
- [ ] `failure_modes_aggregator.py`：扫 reflection_view 的 kind 分布（窗口
  默认 30 天）→ 输出 `list[FailurePattern]`。
- [ ] `preferred_strategies_aggregator.py`：扫 strategy_view 的活跃时长 +
  set_by="reflector_l2" → 输出 `list[StrategyRef]`。
- [ ] `stable_preferences_aggregator.py`：从 belief_view 取 cognitive_type=
  "value" 且 confidence ≥ 0.8 且 about=[]（即 Agent 自己的稳定偏好，非关于
  某 Counterpart 的）→ `list[BeliefRef]`。
- [ ] `tradeoff_aggregator.py`：扫 belief_superseded 事件的
  `decisive_value_kinds` 分布 → `list[ValueTradeoff]`。
- [ ] `interaction_patterns_aggregator.py`（新）：按 CounterpartRole 聚合：
  - 与 user 角色对方 tick 数 / 成功率 / 平均 reflection severity
  - 与 operator 角色对方 tick 数 / ...
  - 输出 `dict[CounterpartRole, InteractionPattern]`
  - SelfModel 新增字段 `interaction_patterns_by_counterpart_role`。
  - 这是关于 **Agent 自己**的描述（"我对 operator 通常更精简"），不是关于
    具体 Counterpart 的描述。

### 2.3 SubjectProjection 接收

- [ ] `cognition/projections/subject.py`：补 handle `self_model_updated`，
  更新 `subject_view` 表（如果尚未建，本阶段建；见 §3.1）。
- [ ] Reactive `SubjectProjection.current()` 从 subject_view 取最新
  SelfModel（系统单 Subject，无 id 参数）。

### 2.4 限速 & 防漂

- [ ] `cognition/reflectors/l3.py` 强制：同主体每 12h 最多 emit 一次
  `self_model_updated`（即使 run_once 跑多次）。
- [ ] payload 内含 diff，便于 audit。

### 2.5 CLI

- [ ] `alpha cognition self-model [--subject]`：打印 SelfModel 字段。
- [ ] `alpha cognition self-model history [--subject] [--last N]`：列最近若
  干 `self_model_updated` 事件。
- [ ] `alpha cognition reflect-l3 --once [--subject]` 手动触发。

### 2.6 测试

- [ ] `tests/cognition/test_capabilities_aggregator.py`。
- [ ] `tests/cognition/test_failure_modes_aggregator.py`。
- [ ] `tests/cognition/test_preferred_strategies_aggregator.py`。
- [ ] `tests/cognition/test_stable_preferences_aggregator.py`。
- [ ] `tests/cognition/test_tradeoff_aggregator.py`。
- [ ] `tests/cognition/test_l3_emit_throttling.py`。
- [ ] `tests/cognition/test_self_model_propagates_to_subject.py`：emit →
  下一次 SubjectProjection.current 含新字段。
- [ ] `tests/cognition/test_cli_self_model.py`。

### 2.7 文档

- [ ] AGENTS.md。
- [ ] `cognition/reflectors/README.md` 写完整三级元认知映射表（L1 / L2 /
  L3 各读什么 / 写什么）。

## 3. 接口契约

### 3.1 `subject_view`

```sql
CREATE TABLE IF NOT EXISTS subject_view (
    id TEXT PRIMARY KEY,
    role TEXT,
    capabilities TEXT NOT NULL DEFAULT '[]',
    declared_needs TEXT NOT NULL DEFAULT '[]',
    value_lens_id TEXT,                         -- 指 subject_value_lens
    self_model TEXT NOT NULL DEFAULT '{}',      -- JSON of SelfModel
    served_counterparts TEXT NOT NULL DEFAULT '[]',
    known_biases TEXT NOT NULL DEFAULT '[]',
    held_at TEXT NOT NULL,
    last_event_id TEXT NOT NULL
);
```

字段与 Phase 01 `Subject` 一一对应。`subject_view` 是 SubjectProjection 的
物化结果——`role` / `capabilities` 等由 Reactive 初次 perceive 时填；
`served_counterparts` 由 Reactive 在 Counterpart 首次出现时追加；
`self_model` 列由本阶段写。`interaction_patterns_by_counterpart_role` 是
SelfModel 内的字段，存在 `self_model` JSON 内，不单独建列。

### 3.2 聚合器协议

```python
class SelfModelAggregator(Protocol):
    field_name: ClassVar[str]   # e.g. "capabilities_self_assessed"

    def compute(
        self,
        subject: SubjectRef,
        log: EventLog,
        projections: ProjectionRegistry,
        window: AggregationWindow,
    ) -> Any: ...

@dataclass(frozen=True)
class AggregationWindow:
    since: Instant
    until: Instant
```

### 3.3 事件

```python
"self_model_updated"
{
    "before": { ... SelfModel fields ... },
    "after":  { ... SelfModel fields ... },
    "window": {"since": ..., "until": ...},
    "aggregators_run": ["capabilities_self_assessed", "..."],
}
```

事件的 subject 由 `CognitiveEvent.subject` 字段表达，不在 payload 里重复。

## 4. 文件清单

### 4.1 新增

```text
src/alpha_agent/cognition/reflectors/l3.py
src/alpha_agent/cognition/reflectors/l3_aggregators/__init__.py
src/alpha_agent/cognition/reflectors/l3_aggregators/capabilities_aggregator.py
src/alpha_agent/cognition/reflectors/l3_aggregators/failure_modes_aggregator.py
src/alpha_agent/cognition/reflectors/l3_aggregators/preferred_strategies_aggregator.py
src/alpha_agent/cognition/reflectors/l3_aggregators/stable_preferences_aggregator.py
src/alpha_agent/cognition/reflectors/l3_aggregators/tradeoff_aggregator.py
src/alpha_agent/cognition/reflectors/l3_aggregators/interaction_patterns_aggregator.py
tests/cognition/test_capabilities_aggregator.py
tests/cognition/test_failure_modes_aggregator.py
tests/cognition/test_preferred_strategies_aggregator.py
tests/cognition/test_stable_preferences_aggregator.py
tests/cognition/test_tradeoff_aggregator.py
tests/cognition/test_interaction_patterns_aggregator.py
tests/cognition/test_l3_emit_throttling.py
tests/cognition/test_self_model_propagates_to_subject.py
tests/cognition/test_cli_self_model.py
```

### 4.2 修改

```text
src/alpha_agent/state/schema.sql                追加 subject_view（若尚未建）
src/alpha_agent/cognition/projections/subject.py 替换为完整 projection 实现
src/alpha_agent/cognition/loops/consolidation.py 加 L3 调度（24h cadence）
src/alpha_agent/cognition/models/event.py       新事件 kind 已在 Phase 01 含 self_model_updated
src/alpha_agent/cognition/reflectors/README.md  L1/L2/L3 完整映射表
src/alpha_agent/cli.py                          alpha cognition self-model / reflect-l3
AGENTS.md
```

### 4.3 删除

无。

## 5. 验收标准

- [ ] `uv run pytest tests/cognition/test_*_aggregator.py
  tests/cognition/test_l3_*.py tests/cognition/test_self_model_*.py -q`
  全绿。
- [ ] 构造一个长事件流 fixture（≥200 事件、覆盖 procedure 成功失败、reflection
  不同 kind、strategy 启用、belief_superseded 含 decisive_value_kinds）→ 跑
  `reflect-l3 --once` → SelfModel 5 个字段都非空且合理。
- [ ] 同主体连续 `reflect-l3 --once` 两次 → 第二次因 throttling 不 emit。
- [ ] emit 后下一次 SubjectProjection.current 含更新后的 SelfModel。
- [ ] `alpha cognition self-model` 能打印；`history` 能列变更。
- [ ] L3 不直接写 Belief / Strategy / Lens——只写 self_model_updated 与
  subject_view。

## 6. 风险与备注

- **聚合窗口选择**。默认 30 天对 failure_modes 合适，对 stable_preferences
  应该更长（90 天）。每个 aggregator 自己定窗口。
- **SelfModel 更新过快**。throttling 12h 是为了防止 SelfModel 抖动。生产数
  据看下来可能要更慢（如 7 天一次主要更新 + diff 阈值）。
- **diff 显著性**。before/after 完全一样时不 emit（避免噪音事件）。
- **SubjectProjection 历史**。SelfModel 演化要可回放：每次 `self_model_
  updated` 都完整记录 before/after，而非只记 after。
- **L3 不应改 Subject.role / membership** 这类身份字段——那些由 perceive
  阶段或外部信号设定。L3 只动 self_model 子树。

## 7. 后续衔接

完整完成后，整套认知运行时具备：

- 感知 → 解释 → 判断 → 决策 → 行动 → 反馈 → 反思 → 修正（Phase 02-09）
- 三级元认知（L1 监控 Phase 05，L2 控制 Phase 08，L3 自我模型 Phase 11）
- 价值层 first-class（Phase 07）
- 多 loop 并发（Reactive Phase 02、Consolidation Phase 06、Drive Phase 10）
- 渲染解耦（Phase 09）
- 长期信念与短期上下文分离（Phase 03/04）

未列入本计划但可作下一轮的方向：

- LLM-assisted Interpreter / Reflector / Aggregator（每个都按已有接口加一
  个 plugin）。
- 多主体协作协议。
- 主动 goal 提议（Drive Loop 的扩展）。
- 多模态 Renderer。
- 远端 / 分布式 event log。

这些都是新 phase，不在本计划范围。
