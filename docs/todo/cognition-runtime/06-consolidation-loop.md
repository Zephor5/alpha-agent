# Phase 06 — Consolidation Loop + 背景压缩

**Status:** pending
**Depends on:** Phase 03, Phase 04, Phase 05
**Scope:** L
**Design ref:** `cognition_from_scratch.md` §5.3, §6, §7（学习路径 1-3）；
README 不变量 1/2

## 0. 目标

引入第二条 loop——**Consolidation Loop**，与 Reactive 并发跑，节奏低频
（idle / scheduled）。它的职责是把短命 Judgment 投影成稳定 Belief、合并重
复 Belief、归档过期 Belief、从重复成功 Decision 抽取 Procedure、维护每条
thread 的 ContextWindow.background（压缩）、以及为每个 active Counterpart
维护一条 digest belief。

完成后：

- 长期使用主体的 ContextWindow.foreground 不再无限增长，老的 perception 自
  动进 background 摘要。
- Procedure 库开始非空——同模式任务第三次出现时 Decider 能命中。
- 重复 belief 被合并；过期 belief 进入 archived 状态。
- 每个 Counterpart 有一条随时间更新的 digest belief，给 Renderer 在开场时
  快速取用。

这是首个走 LoopCoordinator 的非 Reactive loop，所有 worker 必须：

- 启动时 `coordinator.acquire(LoopAcquireRequest(loop_name="consolidation",
  priority=LoopPriority.CONSOLIDATION, max_chunk_duration=timedelta(seconds=30)))`。
- 在每个 ≤ 30s 的工作单元后**主动调** `coordinator.yield_to_higher_priority()`。
  若有更高优先级的调度型 loop（如 L2）等待则让位、本次 run_once 中断；
  下次 scheduler 触发时续跑。即使没有调度型等待者，每次 yield 也会瞬时释放
  锁，让恰在此刻试探的 Reactive 有机会 try_acquire 成功——这是把 busy
  响应率压下去的唯一办法。
- 因此 worker 实现必须 **resumable**（带 checkpoint），不能假设一次 run_once
  走完整个 backlog。
- 注意：Reactive 在 worker 持锁期间会直接收到 LockBusy 并返回 busy 提示
  （不阻塞、不写事件）。worker 持锁越久，busy 响应率越高——所以 30s chunk
  既是公平阀，也是用户体验阀。

## 1. 范围

### 1.1 In scope

- `ConsolidationLoop` 调度器（in-process scheduler 或外部 cron，二选一）。
- 六件工作（每件一个 worker，独立 acquire+yield）：
  1. **Judgment → Belief 提升**：同 claim 在 N tick 内重复 ≥M 次 →
     `belief_formed`。
  2. **Belief 合并**：同 about/object/cognitive_type/normalized_content
     的多条 active belief → 选最新一条保留，其余 `belief_superseded`。
  3. **Belief 归档**：applicability.valid_until 已过 → `belief_retracted`
     或 status=archived（用 `belief_archived` 事件）。
  4. **Procedure 学习**：同 trigger pattern 下 Decision 成功 ≥K 次 →
     `procedure_learned`。
  5. **ContextWindow.background 压缩**：见 §3.3。
  6. **Counterpart 摘要**：对每个 active Counterpart，从
     `BeliefProjection.recall_about(counterpart_ref)` 取关于该 Counterpart
     的活跃 belief → 生成一条 cognitive_type="self"（描述对方）的
     digest belief，about=[counterpart_ref]，supersede 旧 digest（如有）。
     这条 digest 给 Renderer 在打开该 Counterpart 的会话时快速取用。
- 配置：
  - 是否启用（`config.consolidation.enabled`）。
  - 周期（默认 5 min）。
  - 各阈值（N、M、K 等）。
- CLI：`alpha cognition consolidate --now` 手动触发；`--dry-run` 预览。

### 1.2 Out of scope

- ValueLens 解冲突（Phase 07）——本阶段合并 belief 时只处理完全等价的，遇到
  矛盾就发 `consolidation_conflict_queued` 事件留给后续解决。
- Drive Loop（Phase 10）——Consolidation 不主动生成 stimulus。
- 跨主体合并——主体内单独跑。

## 2. 任务清单

### 2.1 调度

- [ ] `cognition/loops/__init__.py`。
- [ ] `cognition/loops/consolidation.py`：`ConsolidationLoop`。
  - 内部 6 个 worker（一件工作一个），每个 worker 是个 callable，**带
    checkpoint 状态**便于 resumable。
  - 提供 `run_once()` 串行跑全部 worker（CLI / 测试用）。在 worker 之间与
    worker 内分块边界 **必须** 调 `coordinator.yield_to_higher_priority()`。
  - 通过 `register_all_workers()` 把 6 个 worker 注册到共享 Scheduler；
    生产环境由 Scheduler 持续 tick，不需要在 ConsolidationLoop 上单独
    启停（其它 loop 同理）。
  - 整个 loop 入口先 `coordinator.acquire(consolidation, max_chunk=30s)`。
- [ ] `cognition/loops/scheduler.py`：通用 in-process scheduler（也给 Phase
  08 L2、Phase 10 Drive、Phase 11 L3 复用）。
  - 调度策略：**时间 + 关注内容**（见 §3.6）。每个 worker 声明
    `ScheduleTrigger`，scheduler 在 wake-up 时调 `should_run(worker, now)`
    决定是否真正 acquire 锁——空 backlog 期间不 acquire、不占锁、不触发
    Reactive busy。
  - 每个 worker 的 last_run_at / last_processed_event_id 存在
    `cognition_worker_checkpoint` 表（见 §3.7）；scheduler 启动时从表恢复。

### 2.2 Workers

- [ ] `cognition/loops/workers/promote_judgment.py`：
  - 扫最近 N tick 的 `judged` 事件。
  - 按 claim normalized 聚合 → 若 ≥M 次 → emit `belief_formed`。
- [ ] `cognition/loops/workers/merge_beliefs.py`：
  - 扫 BeliefProjection.list_active 找 normalized 等价的多条。
  - 选 confidence 最高那条留 active，其余 emit `belief_superseded`
    （指向保留的）。
  - 遇到 normalized 不等价但 subject/predicate 同 → emit
    `consolidation_conflict_queued`，留 Phase 07 处理。
- [ ] `cognition/loops/workers/archive_expired.py`：
  - applicability.valid_until < now → emit `belief_archived`。
- [ ] `cognition/loops/workers/learn_procedure.py`：
  - 扫最近若干 decided + received_feedback 事件。
  - 按 trigger pattern hash 聚合，成功 ≥K 次 → emit `procedure_learned`。
- [ ] `cognition/loops/workers/compress_context.py`：
  - 扫每个 thread_id 的 ContextWindow（系统只有一个 Subject，无须二维 key）。
  - 若 foreground 长度 > K 或 token 估算超阈值 → 取最老 J 条 → 生成摘要 →
    emit `context_compressed`。
  - 摘要生成 v1：deterministic concatenation + heuristic salience-based
    truncation。Phase 09 Renderer 接入后可选 LLM 摘要。
  - 必须 preserve anchors（context_anchor_set 标过的不能进 background）。
- [ ] `cognition/loops/workers/summarize_counterpart.py`：
  - 扫 CounterpartProjection.list_active() 取每个 Counterpart。
  - 对每个 Counterpart 调 `BeliefProjection.recall_about(counterpart_ref)`。
  - 若关于该 Counterpart 的活跃 belief 数 ≥ N（默认 5）或自上次 digest 起
    新增 ≥ M（默认 3）→ 生成一条 digest belief：
    - cognitive_type="concept"（关于对方的归纳；不是 "self"——"self" 在
      cognition.md §2.4 专指 Agent 自己）
    - about=[counterpart_ref]
    - content = 摘要文字（v1 deterministic：按 cognitive_type 分组、按
      confidence 排序 top-N）
    - sources = 所有被摘要的 belief ids
    - supersedes = 旧 digest（若有）
  - emit `belief_formed`（新 digest）+ `belief_superseded`（旧 digest）。
  - **不**修改被摘要的原始 belief——它们继续作为 evidence 存在。
  - chunk 边界：每处理完一个 Counterpart 调一次 yield。

### 2.3 ContextWindow projection 接收新事件

- [ ] `cognition/projections/context_window.py` 已经在 Phase 04 留好对
  `context_compressed` 事件的处理；本阶段补完：把对应 perception_ids 从
  foreground 移到 background_summary_id 引用。
- [ ] 新增 `context_window_background` 表存摘要本身（见 §3.3）。

### 2.4 Procedure projection 接收新事件

- [ ] `cognition/projections/procedure.py` 替换 stub：
  - handle `procedure_learned`、`procedure_strengthened`、
    `procedure_weakened`。
  - `match(judgments, subject)` 实现：按 trigger pattern 模糊匹配。
- [ ] `procedure_view` 表（见 §3.4）。

### 2.5 测试

- [ ] `tests/cognition/test_consolidation_promote_judgment.py`。
- [ ] `tests/cognition/test_consolidation_merge_beliefs.py`。
- [ ] `tests/cognition/test_consolidation_archive_expired.py`。
- [ ] `tests/cognition/test_consolidation_learn_procedure.py`。
- [ ] `tests/cognition/test_consolidation_compress_context.py`。
- [ ] `tests/cognition/test_consolidation_summarize_counterpart.py`：
  - 给关于 user_a 的 6 条 active belief → run_once → 出现一条 digest
    belief，about=[user_a]，supersedes 列表为空。
  - 再加 4 条 → 再 run_once → 新 digest，supersedes 指向上一条 digest。
  - drop belief_view 重 replay → 等价。
- [ ] `tests/cognition/test_consolidation_conflict_queued.py`。
- [ ] `tests/cognition/test_consolidation_idempotent.py`：跑两次 run_once
  结果不变。
- [ ] `tests/cognition/test_consolidation_yield_opens_window_for_reactive.py`：
  - consolidation 进行中、Reactive 持续 try_acquire 收 busy → worker 在下
    一个分块边界 yield → coordinator 瞬时让出锁 → Reactive try_acquire
    成功一次、跑完 tick → consolidation 从 checkpoint 续跑。
- [ ] `tests/cognition/test_consolidation_no_preemption.py`：
  - Reactive 反复 try_acquire 收 busy 期间 consolidation 进行中，coordinator
    不发任何强制中断；consolidation 必须**自己**调 yield 才让出窗口。
- [ ] `tests/cognition/test_cli_consolidate.py`。

### 2.6 文档

- [ ] AGENTS.md。
- [ ] 在 `cognition/loops/README.md` 列五个 worker 与各自阈值。

## 3. 接口契约

### 3.1 Worker 协议

```python
class ConsolidationWorker(Protocol):
    name: ClassVar[str]
    trigger: ClassVar[ScheduleTrigger]                 # 见 §3.6
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]]

    def run(
        self,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        coordinator: LoopCoordinator,
        config: ConsolidationConfig,
        checkpoint: WorkerCheckpoint,                  # 上次进度
    ) -> WorkerReport: ...

@dataclass(frozen=True)
class WorkerReport:
    worker: str
    inspected: int
    emitted: int
    notes: list[str]
    yielded_to_higher_priority: bool                   # True 则提前返回
    new_checkpoint: WorkerCheckpoint                   # 写回 checkpoint 表
```

- Worker 收 `coordinator` 是为了在 chunk 边界调
  `coordinator.yield_to_higher_priority()`。
- Worker 收 `checkpoint` 是为了 resumable——从上次中断处继续。
- 系统单 Subject，`subject` 参数省略。
- 注意 worker.run **不负责判断"要不要跑"**——那是 Scheduler 的责任（§3.6
  `should_run`）。worker.run 一旦被调用，就说明有 backlog 值得处理。

### 3.2 调度

```python
class ConsolidationLoop:
    """ConsolidationLoop 不自己 schedule。它把所有 worker 注册到共享的
    Scheduler（§3.6）；scheduler 按各 worker 的 ScheduleTrigger 决定何时调
    run。CLI 触发的 --now 走 run_once 跳过 trigger。"""
    def __init__(
        self,
        scheduler: Scheduler,
        coordinator: LoopCoordinator,
        ...,
    ): ...
    def register_all_workers(self) -> None: ...
    def run_once(self) -> list[WorkerReport]:
        """绕过 trigger，强制跑一遍所有 worker（CLI / 测试用）。"""
```

### 3.3 `context_window_background` 表

```sql
CREATE TABLE IF NOT EXISTS context_window_background (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    derived_from_event_ids TEXT NOT NULL DEFAULT '[]',
    preserved_anchors TEXT NOT NULL DEFAULT '[]',
    compression_policy TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ctx_bg_thread_time
    ON context_window_background(thread_id, created_at DESC);
```

`context_window_view.background_summary_id` 引用此表的 id。

### 3.4 `procedure_view` 表

```sql
CREATE TABLE IF NOT EXISTS procedure_view (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    trigger_pattern TEXT NOT NULL,
    steps TEXT NOT NULL DEFAULT '[]',
    expected_outcome TEXT NOT NULL DEFAULT '',
    learned_from_event_ids TEXT NOT NULL DEFAULT '[]',
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_procedure_trigger
    ON procedure_view(trigger_pattern);
```

### 3.5 新事件

```python
"belief_archived"
{"belief_id": ..., "reason": ...}

"procedure_learned"
{"procedure": { ... }, "learned_from_event_ids": [...]}

"procedure_strengthened" / "procedure_weakened"
{"procedure_id": ..., "delta": +0.1}

"context_compressed"
{
    "thread_id": ...,
    "absorbed_event_ids": [...],
    "produced_summary_id": "...",
    "compression_policy": "deterministic_v1",
    "preserved_anchors": [...],
}

"consolidation_conflict_queued"
{"conflict_kind": "...", "belief_ids": [...]}
```

### 3.6 通用 Scheduler 协议（也被 Phase 08/10/11 复用）

```python
@dataclass(frozen=True)
class ScheduleTrigger:
    min_interval: timedelta                    # 最短间隔
    max_interval: timedelta | None             # 最长间隔；None 表示无兜底
    watches: frozenset[CognitiveEventKind]     # 在意的事件
    min_new_events: int = 1                    # backlog 阈值

@dataclass(frozen=True)
class WorkerCheckpoint:
    worker_name: str
    last_run_at: Instant | None
    last_processed_event_id: EventId | None
    last_status: Literal["ok", "yielded", "skipped_no_backlog", "error"]

class Scheduler:
    """所有调度型 worker 共享一个 scheduler。"""

    def register(self, worker: Worker, trigger: ScheduleTrigger) -> None: ...

    def should_run(self, worker: Worker, now: Instant) -> bool:
        """时间 + backlog 双闸门。Scheduler 在每个 wake-up 调一次。
        - now - last_run_at < min_interval → False
        - now - last_run_at >= max_interval (若设) → True
        - 否则：count(events in watches after last_processed_event_id)
                >= min_new_events → True

        三种 trigger 模式（约定，不是代码分支）：
        - 纯 backlog 驱动：max_interval=None，事件来才跑。例：DriveLoop。
        - backlog + 兜底：max_interval > min_interval，常见模式。例：L2。
        - 纯时钟驱动：min_interval == max_interval，每 N 跑一次不看事件。
          例：archive_expired / expire_strategies。
        """

    def tick(self, now: Instant) -> None:
        """对每个注册的 worker：should_run 为 True 才 acquire 锁、run。
        run 完更新 checkpoint。should_run 为 False 不 acquire——
        从而不触发 Reactive 的 busy 响应。"""
```

worker 协议（§3.1）相应加一个 `trigger` 类字段：

```python
class ConsolidationWorker(Protocol):
    name: ClassVar[str]
    trigger: ClassVar[ScheduleTrigger]
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]]
    def run(self, ...): ...
```

通常 `trigger.watches` 与 `handles_event_kinds` 大幅重叠但不必相同——例如
`compress_context` worker 实际写 `context_compressed`，但它 watches 的是
`perceived`（因为 foreground 长度增长由 perceived 驱动）。

### 3.7 `cognition_worker_checkpoint` 表

```sql
CREATE TABLE IF NOT EXISTS cognition_worker_checkpoint (
    worker_name TEXT PRIMARY KEY,
    last_run_at TEXT,
    last_processed_event_id TEXT,
    last_status TEXT NOT NULL DEFAULT 'ok',
    metadata TEXT NOT NULL DEFAULT '{}'        -- worker 自定义状态，
                                                -- 例如 summarize_counterpart
                                                -- 上次扫到的 counterpart_id
);
```

resumable worker 在每个 chunk 边界把进度写进 `metadata`；下次 run 从该位置
继续。

### 3.8 各 worker 的默认 trigger 表

| worker                  | min_interval | max_interval | watches                                           | min_new_events |
| ---------------------- | ------------ | ------------ | ------------------------------------------------- | -------------- |
| promote_judgment       | 5 min        | 6 h          | `judged`                                          | 10             |
| merge_beliefs          | 30 min       | 24 h         | `belief_formed`                                   | 5              |
| archive_expired        | 6 h          | 6 h          | ∅ (纯时钟)                                         | 0              |
| learn_procedure        | 30 min       | 24 h         | `decided` ∪ `received_feedback`                   | 10             |
| compress_context       | 10 min       | 6 h          | `perceived`                                       | 12             |
| summarize_counterpart  | 30 min       | 7 d          | `belief_formed` ∪ `belief_superseded`             | 3              |

`archive_expired` 是纯时钟驱动（min==max）——belief 过期由 `valid_until`
决定，不由新事件触发；每 6h 扫一遍 active belief 就够了。其它 worker 都
是 backlog + 兜底模式。

## 4. 文件清单

### 4.1 新增

```text
src/alpha_agent/cognition/loops/__init__.py
src/alpha_agent/cognition/loops/consolidation.py
src/alpha_agent/cognition/loops/scheduler.py
src/alpha_agent/cognition/loops/workers/__init__.py
src/alpha_agent/cognition/loops/workers/promote_judgment.py
src/alpha_agent/cognition/loops/workers/merge_beliefs.py
src/alpha_agent/cognition/loops/workers/archive_expired.py
src/alpha_agent/cognition/loops/workers/learn_procedure.py
src/alpha_agent/cognition/loops/workers/compress_context.py
src/alpha_agent/cognition/loops/workers/summarize_counterpart.py
src/alpha_agent/cognition/loops/README.md
tests/cognition/test_scheduler_should_run_backlog_floor.py
tests/cognition/test_scheduler_should_run_time_ceiling.py
tests/cognition/test_scheduler_skips_when_empty_backlog.py
tests/cognition/test_scheduler_checkpoint_persistence.py
tests/cognition/test_consolidation_promote_judgment.py
tests/cognition/test_consolidation_merge_beliefs.py
tests/cognition/test_consolidation_archive_expired.py
tests/cognition/test_consolidation_learn_procedure.py
tests/cognition/test_consolidation_compress_context.py
tests/cognition/test_consolidation_summarize_counterpart.py
tests/cognition/test_consolidation_conflict_queued.py
tests/cognition/test_consolidation_idempotent.py
tests/cognition/test_consolidation_yield_opens_window_for_reactive.py
tests/cognition/test_consolidation_no_preemption.py
tests/cognition/test_cli_consolidate.py
```

### 4.2 修改

```text
src/alpha_agent/state/schema.sql                              追加 context_window_background / procedure_view / cognition_worker_checkpoint 三张表
src/alpha_agent/cognition/projections/context_window.py       补完 context_compressed
src/alpha_agent/cognition/projections/procedure.py            替换 stub
src/alpha_agent/cognition/models/event.py                     新增事件 kind
src/alpha_agent/config.py                                     新增 consolidation 配置段
src/alpha_agent/cli.py                                        alpha cognition consolidate
AGENTS.md
```

### 4.3 删除

无。

## 5. 验收标准

- [ ] `uv run pytest tests/cognition/test_consolidation_*.py -q` 全绿。
- [ ] `alpha cognition consolidate --dry-run` 列出预期变化但不写库。
- [ ] 跑一个长会话脚本：连续 50 轮对话 → consolidate → foreground 长度 ≤
  K，background_summary_id 非空。
- [ ] 重复成功 3 次"用 X 工具完成 Y"模式 → procedure_view 出现对应 procedure。
- [ ] 重启进程后 ConsolidationLoop 状态从 event log 重建一致。
- [ ] run_once 跑两次 → 第二次所有 worker 的 emitted=0（幂等）。

## 6. 风险与备注

- **scheduler 选型**。in-process scheduler 简单但单 worker 卡住会拖后续。外
  部 cron / systemd timer 更稳但需要 ops 介入。v1 先 in-process，文档里写明
  "未来可换"。
- **Resumable worker 是硬要求，不是 nice-to-have**。每个 worker 必须在
  yield 边界 checkpoint 自己的进度（最近处理到哪条 belief / Counterpart /
  thread）；下次 run_once 从 checkpoint 继续。否则一旦 Reactive 让位次数多，
  worker 会反复从头扫描永远跑不完。checkpoint 存储建议放
  `state/consolidation_checkpoint` 表（简单 key-value）。
- **30 秒 chunk 不是绝对值**。复杂 worker（如 LLM 摘要 if 启用）单步可能 >
  30s。worker 实现者应在每个 belief / Counterpart / thread 处理完毕后立即调
  yield；30s 是"最长应该容忍的非 yield 窗口"。
- **压缩质量**。v1 deterministic 摘要会丢细节但稳定可重放。LLM 摘要更好但不
  确定性高。本阶段先 deterministic；Phase 09 Renderer 完成后可加 LLM
  摘要作为可选 policy（仍走 `context_compressed` 事件，policy 字段记录）。
- **学习 procedure 的误报**。规则化 trigger pattern 容易学到"看似规律的偶然
  连续"。设置较高阈值（K ≥ 3）+ trigger normalize 限制（必须含相同主要工具
  / 同类动作）。
- **conflict_queued 事件不消费会堆积**。Phase 07 来消费。本阶段在 CLI
  metrics 里把这个队列长度暴露出来。
- **subject-by-subject 还是全局调度**。v1 每次 run_once 取所有 active
  subject 跑一遍。subject 多时改为优先级调度（最近活跃优先）。

## 7. 后续衔接

- Phase 07 ValueLens 消费 `consolidation_conflict_queued`，决出冲突。
- Phase 09 Renderer 能可选启用 LLM-based 压缩 policy。
- Phase 11 L3 SelfModel 读 procedure_view 的 success/failure 比例，更新主体
  的 capabilities_self_assessed。
