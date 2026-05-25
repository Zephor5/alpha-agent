# Phase 07 — ValueLens 与冲突解决

**Status:** pending
**Depends on:** Phase 03, Phase 05, Phase 06
**Scope:** M
**Design ref:** `cognition_from_scratch.md` §2.5, §7（学习路径 4）；README
不变量 1/2

## 0. 目标

让 `Subject.ValueLens` 真正影响认知行为：

- Interpreter 在 stance=contradicting 时不再把两条 belief 都标 supports，而
  是按 ValueLens 决出胜者。
- Phase 06 留下的 `consolidation_conflict_queued` 队列被本阶段消费。
- 同一组事件流在不同 ValueLens 配置下能产生不同最终信念集合。

ValueLens 属于 Subject（Agent 自己），不属于 Counterpart——Counterpart 的
`trust_level` 是 Agent 对其的信任度，是 belief 层面的属性，不进 ValueLens。
但当两条矛盾 belief 来自不同 trust_level 的 Counterpart（可在 `belief.about
/ belief.sources` 看到），resolver 把 trust_level 作为一个加权项参与决策
（v1 简化：trust 高的来源胜出 +ε margin）。

调度归属：

- **同步解决路径**（Interpreter / Reviser 在 Reactive tick 内调 resolver）
  跟随 Reactive 持锁，不自己 acquire。
- **`resolve_queued_conflicts` 和 `learn_value_lens` 两个 worker** 跑在
  ConsolidationLoop 里，沿用 Phase 06 的 acquire(CONSOLIDATION) + 分块
  yield 协议——这两个文件物理上放在 `cognition/loops/workers/`，逻辑上属于
  ValueLens 范畴。

## 1. 范围

### 1.1 In scope

- `ValueProfile` 自动派生（从 Belief 内容启发式推权重；规则在本阶段文档内
  定义，设计动机见 `docs/cognition/cognition_from_scratch.md`）。
- `ValueLens` 持久化与读取（默认 lens + 主体覆写）。
- Conflict resolution 算法：给定两条矛盾 belief + lens → 返回胜者。
- Interpreter / Reviser 接入：检测冲突 → 调 resolver → emit
  `belief_superseded` 并带 lens 解释。
- Consolidation 的 `consolidation_conflict_queued` 消费者：扫该队列 → 调
  resolver → 决出 supersede 或转人审。
- ValueLens 学习的**最小闭环**：连续 N 次 tradeoff 偏向某方向 → emit
  `value_lens_shifted`（具体更新策略 v1 简单，Phase 11 L3 完善）。

### 1.2 Out of scope

- 完全的 ValueLens 自适应（Phase 11 做）。
- 用户级 lens 配置 UI（CLI 一条命令够用）。
- 价值层多人协商。

## 2. 任务清单

### 2.1 ValueProfile 派生

- [ ] `cognition/value/profile_derivation.py`：`derive_value_profile(content,
  structure, cognitive_type, entities) -> ValueProfile`。
  - 关键词触发 + cognitive_type 默认权重 + entity 类型权重。
  - 规则集放在同模块，方便测试覆盖。

### 2.2 ValueLens 存取

- [ ] `cognition/value/lens.py`：
  - `default_value_lens()` 返回内置默认 priority。
  - `load_lens(subject_id) -> ValueLens`：从 `subject_value_lens` 表读，缺失
    fallback 到 default。
  - `save_lens(subject_id, lens, emitter)`：emit `value_lens_shifted` 事件 +
    更新表。
- [ ] `state/schema.sql` 追加 `subject_value_lens` 表（见 §3.1）。

### 2.3 Resolver

- [ ] `cognition/value/resolver.py`：
  - `resolve_conflict(left: Belief, right: Belief, lens: ValueLens)
    -> ConflictResolution`。
  - 算法：对每条 belief，计算 `score = Σ lens.sensitivity[v] *
    lens.priority_rank(v) * belief.value_profile.weights[v]`，分高者胜。
  - 平手时返回 `tie`，由调用方决定（默认转人审 / 默认保留更新者）。

### 2.4 Reactive 接入

- [ ] `cognition/stages/interpret.py`：检测 stance=contradicting →
  在 Interpretation 上多一个字段 `proposed_resolution: ConflictResolution`。
- [ ] `cognition/stages/revise.py`：根据 Interpretation.proposed_resolution
  发 `belief_superseded`，事件 payload 记录用了哪条 lens 的哪个 priority。

### 2.5 Consolidation 消费冲突队列

- [ ] `cognition/loops/workers/resolve_queued_conflicts.py`：新 worker。
  - 扫 `consolidation_conflict_queued` 事件（最近未消化的）。
  - 调 resolver → emit `belief_superseded` 或 `conflict_kept_for_human_review`。
  - ScheduleTrigger：

    ```python
    ScheduleTrigger(
        min_interval=timedelta(minutes=5),
        max_interval=timedelta(hours=6),
        watches=frozenset({"consolidation_conflict_queued"}),
        min_new_events=1,
    )
    ```
- [ ] 注册到 `ConsolidationLoop`。

### 2.6 ValueLens 学习 v1

- [ ] `cognition/loops/workers/learn_value_lens.py`：
  - 扫最近 K 个 `belief_superseded`（带 lens 解释）。
  - 若同一 ValueKind 反复胜出 → 该维度 sensitivity +=δ，emit
    `value_lens_shifted`。
  - 限速：每 24h 最多 1 次 shift（防抖）。
  - ScheduleTrigger：

    ```python
    ScheduleTrigger(
        min_interval=timedelta(hours=1),
        max_interval=timedelta(days=1),
        watches=frozenset({"belief_superseded"}),
        min_new_events=5,    # 需要足够样本才考虑 shift
    )
    ```

### 2.7 CLI

- [ ] `alpha cognition lens show [subject]`。
- [ ] `alpha cognition lens set [subject] --priority existence,utility,moral,...`。

### 2.8 测试

- [ ] `tests/cognition/test_value_profile_derivation.py`：关键词/类型/实体 →
  权重。
- [ ] `tests/cognition/test_resolver_winner.py`：明显胜者。
- [ ] `tests/cognition/test_resolver_tie.py`：平手处理。
- [ ] `tests/cognition/test_lens_shapes_supersede.py`：同样矛盾，不同 lens
  → 不同 supersede 方向。
- [ ] `tests/cognition/test_consolidation_resolves_queued_conflicts.py`。
- [ ] `tests/cognition/test_value_lens_learning_v1.py`：连续 N 次同维度胜出
  → shift。
- [ ] `tests/cognition/test_cli_lens.py`。

### 2.9 文档

- [ ] AGENTS.md。

## 3. 接口契约

### 3.1 `subject_value_lens` 表

```sql
CREATE TABLE IF NOT EXISTS subject_value_lens (
    subject_id TEXT PRIMARY KEY,
    priority TEXT NOT NULL,                  -- JSON list[ValueKind]
    sensitivity TEXT NOT NULL DEFAULT '{}',  -- JSON dict[ValueKind, float]
    tradeoff_preferences TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL,
    last_event_id TEXT NOT NULL              -- 最近 value_lens_shifted 事件 id
);
```

### 3.2 Resolver

```python
@dataclass(frozen=True)
class ConflictResolution:
    winner_id: BeliefId
    loser_id: BeliefId
    tie: bool
    rationale: str
    by_lens_priority: list[ValueKind]
    margin: float

def resolve_conflict(
    left: Belief,
    right: Belief,
    lens: ValueLens,
) -> ConflictResolution: ...
```

### 3.3 事件

```python
"value_lens_shifted"
{
    "before": { "priority": [...], "sensitivity": {...} },
    "after":  { "priority": [...], "sensitivity": {...} },
    "trigger": "...",   # e.g. "learn_value_lens worker observed 5 utility wins"
}

"belief_superseded" 现在 payload 多一项：
{
    ...原有字段...,
    "decisive_value_kinds": ["utility", "existence"],
}

"conflict_kept_for_human_review"
{"belief_ids": [...], "reason": "tie under current lens"}
```

事件的 subject 由 `CognitiveEvent.subject` 字段表达，不在 payload 里重复。

## 4. 文件清单

### 4.1 新增

```text
src/alpha_agent/cognition/value/__init__.py
src/alpha_agent/cognition/value/profile_derivation.py
src/alpha_agent/cognition/value/lens.py
src/alpha_agent/cognition/value/resolver.py
src/alpha_agent/cognition/loops/workers/resolve_queued_conflicts.py
src/alpha_agent/cognition/loops/workers/learn_value_lens.py
tests/cognition/test_value_profile_derivation.py
tests/cognition/test_resolver_winner.py
tests/cognition/test_resolver_tie.py
tests/cognition/test_lens_shapes_supersede.py
tests/cognition/test_consolidation_resolves_queued_conflicts.py
tests/cognition/test_value_lens_learning_v1.py
tests/cognition/test_cli_lens.py
```

### 4.2 修改

```text
src/alpha_agent/state/schema.sql                追加 subject_value_lens
src/alpha_agent/cognition/stages/interpret.py   多 proposed_resolution
src/alpha_agent/cognition/stages/revise.py      用 lens 决 supersede
src/alpha_agent/cognition/projections/subject.py 读 subject_value_lens
src/alpha_agent/cognition/loops/consolidation.py 注册新 worker
src/alpha_agent/cognition/models/event.py       新事件 kind
src/alpha_agent/cli.py                          alpha cognition lens
AGENTS.md
```

### 4.3 删除

无。

## 5. 验收标准

- [ ] `uv run pytest tests/cognition/test_value_*.py
  tests/cognition/test_resolver_*.py tests/cognition/test_lens_*.py -q` 全绿。
- [ ] 构造同样的两条矛盾 belief，分别用两组 ValueLens 参数调用 resolver →
  supersede 方向不同；持久化路径仍只验证单 Subject 的当前 lens。
- [ ] `alpha cognition lens show` 能打印当前 lens；`set` 修改后 emit
  `value_lens_shifted` 并被 SubjectProjection 反映。
- [ ] Phase 06 留下的 conflict_queued 事件，跑一次 consolidate 后被消化。
- [ ] `value_lens_shifted` 对 `SUBJECT_SELF` 在 24h 内最多 1 次。

## 6. 风险与备注

- **派生权重的稳定性**。关键词列表会不全。本阶段尽量覆盖明显类别，剩下
  fallback 到默认 profile；记录"未匹配关键词"日志用于后续扩词典。
- **平手情况**。平手的 supersede 默认不发——记 `conflict_kept_for_human
  _review`。Phase 08 L2 可决定要不要自动选个方向，本阶段不做。
- **lens 学习要克制**。v1 对 `SUBJECT_SELF` 每 24h 最多 1 次 shift，且只调
  sensitivity，不动 priority。priority 改变是大事，等 Phase 11 L3 设计完整
  流程。
- **解释链审计**。每条 `belief_superseded` 现在带 `decisive_value_kinds` —
  这是后续审计"为什么这条 belief 输了"的关键。

## 7. 后续衔接

- Phase 08 L2 可设规则："如果同一主体 24h 内 `value_lens_shifted` 发生 ≥3
  次 → 触发 strategy_changed 暂停自动 shift"。
- Phase 11 L3 用完整 lens 演化历史更新 SelfModel.typical_value_tradeoffs。
