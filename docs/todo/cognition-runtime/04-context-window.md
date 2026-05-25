# Phase 04 — ContextWindowProjection（前景版）

**Status:** pending
**Depends on:** Phase 01, Phase 02
**Scope:** S
**Design ref:** `cognition_from_scratch.md` §5；README 不变量 1

## 0. 目标

把 Phase 02 的 `ContextWindowProjection` stub 升级成正式实现，**前景与召回**
两部分到位。背景压缩留给 Phase 06 Consolidation Loop。

ContextWindow 在 Phase 01 已含 `counterpart: CounterpartRef | None`（会话
thread 关联到具体 Counterpart，cognition thread 为 None）。本阶段的物化表
`context_window_view` 把 counterpart 落到列上，Reactive 装配 window 时一并
带进 `BeliefRecallParams`，让 recall 默认按对方过滤。

完成后：

- 按 `thread_id` 取出 ContextWindow，foreground 是最近 K 个 perception 的
  raw 引用，recalled 来自 `BeliefProjection.recall(...)` 的结果。
- 支持两类 thread：`conversation` 与 `cognition`。
- 一个 Reactive tick 可同时打开多个 window（跨 thread 引用）。

## 1. 范围

### 1.1 In scope

- `ContextWindowProjection.get(thread_id, subject, at)` 正式实现。
- foreground 取 last K perception（K 可配置，默认 12，由 perception event 直接
  列出）。
- recalled 字段调 `BeliefProjection.recall(BeliefRecallParams(focus=...,
  counterpart=window.counterpart))` 取活跃信念。
- 两类 `ThreadKind` 支持：`CONVERSATION`、`COGNITION`。Drive Loop 的
  self-stimulus 写到 cognition thread；用户输入写到 conversation thread。
- `recent_judgments` / `matched_procedures` 字段从最近若干 tick 的 events
  里聚合。
- thread 隔离：不同 thread_id 的 window 互不渗透。

### 1.2 Out of scope

- `background` 压缩——Phase 06。这一阶段 `background = None`。
- 多 thread 之间的"思维迁移"协议——超出本计划范围，将来另文档。
- Renderer 怎么用 window 拼 prompt——Phase 09。

## 2. 任务清单

### 2.1 Projection 实现

- [ ] `cognition/projections/context_window.py` 替换 stub：
  - 维护 `thread_id → ContextWindow` 物化表 `context_window_view`（系统单
    Subject，thread_id 全局唯一）。
  - `apply(event)`：
    - `perceived` 事件：把 PerceptionId 追加到对应 thread 的 foreground 队列；
      超过 K 时把最老的挤出（不删，只移出 foreground——Phase 06 才进 background）。
    - `judged`：记录到 `recent_judgments`，最多保留 M 个。
    - `procedure_matched`（Phase 06 才出现的事件，本阶段提前埋好处理逻辑）：
      记录到 `matched_procedures`。
- [ ] `get(thread_id, at)` 返回当前快照——按 thread_id 查找。
- [ ] 提供 `mark_anchor(thread_id, perception_id)` 接口，允许 Reactive 显式
  标记"这条 perception 不能被挤出 foreground"（例如开场介绍）。

### 2.2 Thread 路由

`ThreadId` 类型与构造器在 Phase 01 `cognition/models/thread.py` 已就位
（`from_session`、`cognition`）。本阶段做 **stimulus → thread_id 的策略
集中**：

- [ ] `cognition/threads.py`：`StimulusRouter`，把 Stimulus 路由到 thread_id
  的规则集中在这里。
  - `user_message` / `tool_result` / `webhook` / `inter_agent` → 走
    `ThreadId.from_session(session_id, source_metadata)`。
  - `self_signal` / `clock_tick` → 走 `ThreadId.cognition(subject_id, topic)`。
- [ ] Phase 02 `respond()` 与 Phase 10 Drive Loop 都通过 StimulusRouter 取
  thread_id，避免每个 caller 自己写 if/else。

### 2.3 Reactive 接入

- [ ] `cognition/controller.py`：
  - 一轮 tick 开始前装配 window。
  - Decider 与 Effector 接收 window 作为参数（Phase 02 已经签名留好）。
  - Effector 内部拼 prompt 时显式用 `window.foreground` 的 perception 内容
    （Phase 09 抽到 Renderer）。

### 2.4 测试

- [ ] `tests/cognition/test_context_window_foreground_roll.py`：
  - 连发 K+5 个 perceived 事件 → foreground 长度恰为 K，最老 5 个被移出。
- [ ] `tests/cognition/test_context_window_recall_join.py`：
  - 已有 belief → tick 产生 perceived → recalled 字段包含相关 belief。
- [ ] `tests/cognition/test_context_window_thread_isolation.py`：
  - thread A 的 perception 不出现在 thread B 的 foreground。
- [ ] `tests/cognition/test_context_window_counterpart_link.py`：
  - 同一 Counterpart 在不同 session 开新 thread，`list_threads_by_counterpart`
    能取出全部对应 thread_id。
  - conversation thread 的 counterpart 字段非空；cognition thread 为 None。
- [ ] `tests/cognition/test_context_window_anchor.py`：
  - mark_anchor 后，即使发 K+10 个 perceived，被锚定的不被挤出。
- [ ] `tests/cognition/test_context_window_rebuild.py`：
  - drop `context_window_view` → replay → 等价。

### 2.5 文档

- [ ] 在 `cognition/projections/context_window.py` 模块 docstring 写清楚
  "raw 永远在事件日志，foreground 只是引用"这条原则。
- [ ] 更新 AGENTS.md 项目导航。

## 3. 接口契约（草案）

### 3.1 `context_window_view` 表

```sql
CREATE TABLE IF NOT EXISTS context_window_view (
    thread_id TEXT PRIMARY KEY,
    thread_kind TEXT NOT NULL,                     -- conversation / cognition
    counterpart_id TEXT,                           -- conversation thread 必填；cognition 为 NULL
    foreground_ids TEXT NOT NULL DEFAULT '[]',     -- list[PerceptionId]
    anchored_ids TEXT NOT NULL DEFAULT '[]',       -- list[PerceptionId]
    recent_judgment_ids TEXT NOT NULL DEFAULT '[]',
    matched_procedure_ids TEXT NOT NULL DEFAULT '[]',
    background_summary_id TEXT,                    -- 由 Phase 06 写
    last_event_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ctx_window_counterpart
    ON context_window_view(counterpart_id, thread_kind);
CREATE INDEX IF NOT EXISTS idx_ctx_window_kind
    ON context_window_view(thread_kind);
```

### 3.2 Projection 主接口

```python
class ContextWindowProjection(Projection):
    name = "context_window"
    handles = frozenset({
        "perceived", "judged",
        # 以下事件会在后续 phase 出现，但本阶段提前接收
        "procedure_matched", "context_compressed",
        "context_anchor_set", "context_anchor_cleared",
    })

    def get(
        self,
        thread_id: ThreadId,
        at: Instant | None = None,
    ) -> ContextWindow: ...

    def list_threads_by_counterpart(
        self,
        counterpart: CounterpartRef,
    ) -> list[ThreadId]: ...   # renderer / consolidation 用

    def mark_anchor(
        self,
        thread_id: ThreadId,
        perception_id: PerceptionId,
        emitter: EventEmitter,
    ) -> None: ...   # emit "context_anchor_set"
```

`at` 参数允许查询历史时刻的 window，对 audit / debug 很有用。v1 实现：若 `at`
非空，从 `cognitive_events` replay 到 `at` 时刻重建 window；若为空走物化表。

### 3.3 StimulusRouter

`ThreadId` 的 dataclass 与构造器在 Phase 01 `cognition/models/thread.py`
定义；这里只列本阶段新增的路由器：

```python
class StimulusRouter:
    @staticmethod
    def route(
        stimulus: Stimulus,
        session_id: str | None = None,
    ) -> ThreadId: ...
```

具体路由规则（由 `stimulus.kind` 分派）：

- `user_message` / `tool_result` / `webhook` / `inter_agent` →
  `ThreadId.from_session(session_id, stimulus.payload.get("source_metadata"))`
- `self_signal` →
  `ThreadId.cognition(SUBJECT_SELF, topic=stimulus.payload["goal_id"])`
- `clock_tick` →
  `ThreadId.cognition(SUBJECT_SELF, topic="clock")`

注意 `route` 是 stimulus 工厂阶段的辅助——Phase 02 的 `respond()` 因为只
有 user_message 一类，直接 `ThreadId.from_session(...)` 即可；Phase 10
Drive Loop 走 `self_signal` 时用 StimulusRouter 集中策略。

## 4. 文件清单

### 4.1 新增

```text
src/alpha_agent/cognition/threads.py
tests/cognition/test_context_window_foreground_roll.py
tests/cognition/test_context_window_recall_join.py
tests/cognition/test_context_window_thread_isolation.py
tests/cognition/test_context_window_anchor.py
tests/cognition/test_context_window_rebuild.py
```

### 4.2 修改

```text
src/alpha_agent/state/schema.sql                          追加 context_window_view
src/alpha_agent/cognition/projections/context_window.py   替换 stub
src/alpha_agent/cognition/controller.py                   装配 window
src/alpha_agent/cognition/stages/effector.py              内部用 window.foreground
src/alpha_agent/cognition/models/event.py                 新增 context_anchor_* kind
```

### 4.3 删除

无。

## 5. 验收标准

- [ ] `uv run pytest tests/cognition/test_context_window_*.py -q` 全绿。
- [ ] 连续 20 轮对话，foreground 长度始终 ≤ K（默认 12）。
- [ ] `alpha debug prompt --show-window` 能打印当前 window 的 foreground /
  recalled / recent_judgments / matched_procedures 各字段。
- [ ] 双线程演示：构造一个 cognition thread（手工 emit perceived 到该 thread）
  → 该 thread window 与 conversation thread window 独立。
- [ ] drop view → 重启 → 等价。

## 6. 风险与备注

- **K 与 M 的初始默认值**。foreground K=12、recent_judgments M=8 是直觉值。
  Phase 06 Consolidation 接入后看实际表现再调。
- **anchor 滥用风险**。如果调用方乱标 anchor，foreground 会无限扩张。要在
  `mark_anchor` 接口内强制总 anchor 数 ≤ K/2，超额拒绝。
- **跨 thread 引用怎么写在事件里**。本阶段先不实现跨 thread 引用，只保证两类
  thread 都能独立运行；跨 thread 在 Phase 10 Drive Loop 真正用到时再设计。
- **at 参数的历史查询**性能差（需要 replay）。inspection 场景用即可，热路径
  不要走。
- **背景为空时 Renderer 怎么用**。Phase 09 Renderer 看到 background=None 时
  只渲染 foreground+recalled。这是合理 fallback。

## 7. 后续衔接

- Phase 06 Consolidation Loop 写 `context_compressed` 事件 → 本 projection
  接收并把对应 perception_ids 从 foreground 挤出，存到 background_summary_id。
- Phase 09 Renderer 把 ContextWindow 渲染成 prompt。
- Phase 10 Drive Loop 用 `ThreadId.cognition(...)` 写入 cognition thread 的
  perceived 事件。
