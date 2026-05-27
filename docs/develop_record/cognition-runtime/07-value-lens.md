# Phase 07 — ValueLens 与冲突解决

**Status:** implemented deterministic v1
**Depends on:** Phase 03, Phase 05, Phase 06
**Scope:** M
**Design ref:** `cognition_from_scratch.md` §2.5, §7（学习路径 4）；README
不变量 1/2

## 0. 目标

让 `Subject.ValueLens` 真正影响认知行为：

- Interpreter 在检测到已召回 belief 之间的矛盾时会附带
  `proposed_resolution`，queued conflict 的 durable supersede 由
  Consolidation worker 执行。
- Phase 06 留下的 `consolidation_conflict_queued` 队列被本阶段消费。
- 同一组事件流在不同 ValueLens 配置下能产生不同最终信念集合。

ValueLens 属于 Subject（Agent 自己），不属于 Counterpart——Counterpart 的
`trust_level` 是 Agent 对其的信任度，是 belief 层面的属性，不进 ValueLens。
v1 resolver 只使用项目已有的 `ValueKind` 语义：
`safety/honesty/helpfulness/autonomy/efficiency/learning`。Counterpart
`trust_level` 仍是后续增强，不参与本阶段评分。

调度归属：

- **同步解释路径**（Interpreter 在 Reactive tick 内调 resolver）
  跟随 Reactive 持锁，不自己 acquire；Reviser 自动 durable supersede 留给后续阶段。
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
- Interpreter 接入：检测召回 belief 之间的冲突 → 调 resolver → 在
  `interpreted` payload 上记录 `proposed_resolution`。
- Durable supersede 接入：`resolve_queued_conflicts` worker 消费队列并 emit
  `belief_superseded` 或 `conflict_kept_for_human_review`。
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

- [x] `cognition/value/profile_derivation.py`：`derive_value_profile(content,
  structure, cognitive_type, entities) -> ValueProfile`。
  - 关键词触发 + cognitive_type 默认权重 + entity 类型权重。
  - 规则集放在同模块，方便测试覆盖。
- [x] 普通 belief materialization 会在 incoming `value_profile.weights` 为空时
  派生 profile；Consolidation worker 新建 belief 的公共 helper 也会派生 profile。

### 2.2 ValueLens 存取

- [x] `cognition/value/lens.py`：
  - `default_value_lens()` 返回内置默认 priority。
  - `load_lens(subject_id) -> ValueLens`：从 `subject_value_lens` 表读，缺失
    fallback 到 default。
  - `save_lens(subject_id, lens, emitter)`：emit `value_lens_shifted` 事件 +
    更新表。
- [x] `state/schema.sql` 追加 `subject_value_lens` 表（见 §3.1）。

### 2.3 Resolver

- [x] `cognition/value/resolver.py`：
  - `resolve_conflict(left: Belief, right: Belief, lens: ValueLens)
    -> ConflictResolution`。
  - 算法：对每条 belief，计算 `score = Σ lens.sensitivity[v] *
    lens.priority_rank(v) * belief.value_profile.weights[v]`，分高者胜。
  - 平手时返回 `tie`，由调用方决定（默认转人审 / 默认保留更新者）。

### 2.4 Reactive 接入

- [x] `cognition/stages/interpret.py`：检测召回 belief 冲突 →
  在 Interpretation 上多一个字段 `proposed_resolution: ConflictResolution`。
- [ ] `cognition/stages/revise.py` 自动发 durable `belief_superseded` 仍
  deferred；v1 durable 路径收口在 queued conflict worker，避免 Reactive tick
  里引入未成熟的直接长程写入策略。

### 2.5 Consolidation 消费冲突队列

- [x] `cognition/loops/workers/resolve_queued_conflicts.py`：新 worker。
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
- [x] 注册到 `ConsolidationLoop`。

### 2.6 ValueLens 学习 v1

- [x] `cognition/loops/workers/learn_value_lens.py`：
  - 扫最近 K 个 `belief_superseded`（带 lens 解释）。
  - 若同一 ValueKind 反复胜出 → 该维度 sensitivity +=δ，emit
    `value_lens_shifted`。
  - 限速：每 24h 最多 1 次 shift（防抖）。
  - rate-limit 的当前时间按 event-log 顺序的最后事件判断，不依赖事件 id
    字典序。
  - 成功/no-op run 只处理 `last_processed_event_id` 之后的事件；yielded
    resume 在这个 post-checkpoint 窗口内从 metadata cursor 之后开始并 wrap。
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

- [x] `alpha cognition lens show [subject]`。
- [x] `alpha cognition lens set [subject] --priority safety,honesty,...`。v1
  使用当前项目 `ValueKind`，不是旧草案里的 `existence/utility/moral`。

### 2.8 测试

- [x] `tests/cognition/test_value_lens_phase07.py` 覆盖关键词/类型/实体派生、
  resolver winner/tie、同样矛盾不同 lens、queued conflict supersede/tie、
  lens learning 限速与 sensitivity shift、CLI show/set、SubjectProjection replay、
  empty-profile queued conflict 派生、非时间顺序 event id 的学习限速回归，以及
  成功 checkpoint 后不重算旧 supersede 事件的回归。

### 2.9 文档

- [x] AGENTS.md。

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
    "trigger": "...",   # e.g. "learn_value_lens worker observed 5 efficiency wins"
}

"belief_superseded" 现在 payload 多这些项：
{
    ...原有字段...,
    "decisive_value_kinds": ["safety", "honesty"],
    "value_lens_explanation": "...",
    "resolution_margin": 1.0
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
tests/cognition/test_value_lens_phase07.py
docs/develop_record/phase-07-value-lens.md
```

### 4.2 修改

```text
src/alpha_agent/state/schema.sql                追加 subject_value_lens
src/alpha_agent/cognition/stages/types.py       Interpretation.proposed_resolution
src/alpha_agent/cognition/stages/interpret.py   多 proposed_resolution
src/alpha_agent/cognition/stages/revise.py      未改，durable supersede 由 worker 负责
src/alpha_agent/cognition/projections/subject.py 读 subject_value_lens
src/alpha_agent/cognition/loops/consolidation.py 学习 worker 配置默认值
src/alpha_agent/cognition/loops/workers/__init__.py 注册新 worker
src/alpha_agent/cognition/models/event.py       Phase 06 已含所需事件 kind
src/alpha_agent/cli.py                          alpha cognition lens
tests/test_cli_agent_loop.py                    初始化表清单包含 subject_value_lens
README.md
AGENTS.md
```

### 4.3 删除

无。

## 5. 验收标准

- [x] `uv run pytest tests/cognition/test_value_lens_phase07.py -q` 全绿。
- [x] 构造同样的两条矛盾 belief，分别用两组 ValueLens 参数调用 resolver →
  supersede 方向不同；持久化路径仍只验证单 Subject 的当前 lens。
- [x] `alpha cognition lens show` 能打印当前 lens；`set` 修改后 emit
  `value_lens_shifted` 并被 SubjectProjection 反映。
- [x] Phase 06 留下的 conflict_queued 事件，跑一次 consolidate 后被消化。
- [x] `value_lens_shifted` 对 `SUBJECT_SELF` 在 24h 内最多 1 次。

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
