# Phase 10 — Drive Loop

**Status:** pending
**Depends on:** Phase 02, Phase 06
**Scope:** M
**Design ref:** `cognition_from_scratch.md` §6 (Drive Loop), §11；README
不变量 1/2

## 0. 目标

引入**第三条 loop**——Drive Loop。它把"未满足的 goal"转成 self-stimulus，
驱动 Reactive Loop 在无外部输入时也跑——即"主体内心独白"。

完成后：

- `Goal` 是 first-class 类型，有 set / satisfied / abandoned 三态。Goal 可
  以是 Counterpart-scoped：例如"回答 user_a 的未结问题"，通过
  `for_counterpart: CounterpartRef | None` 字段表达。
- Drive Loop 按节奏扫 active goal → 选一个 → 生成 self-stimulus → 注入
  `CognitiveController.reactive_tick(...)`（kind="self_signal"）。
- self-stimulus 走 cognition thread，不污染对外 conversation thread；但当
  goal 关联到某 Counterpart 时，Reactive tick 处理时把该 counterpart 写进
  `Situation.social.present_counterparts`，让 `BeliefProjection.recall`
  自然拉到对应 belief。
- 默认 disabled——主体保持被动。配置开启后才主动。

Drive Loop 的运行分两步：

1. **选 goal + 生成 self_signal**：低优先级工作，要 acquire
   `LoopPriority.DRIVE`。
2. **触发 Reactive tick 处理 self_signal**：enqueue stimulus 到 reactive 入
   口，由 Reactive 自己 try_acquire（REACTIVE）——Drive 不会"绕过"协调。
   若 Reactive 锁正忙（罕见，因 Drive 自己刚释放 DRIVE 锁不久），这次
   self_signal 被丢弃，下次 scheduler tick 再试。

## 1. 范围

### 1.1 In scope

- `Goal` 类型与 `goal_view` projection。
- `GoalRegistry`：set_goal / satisfy / abandon。
- `DriveLoop` 调度器（复用 Phase 06 的 Scheduler 与 checkpoint 表）。
- 简单 goal selection 策略（优先级 + 上次活动时间）。
- self-stimulus 生成器：把 goal 状态包成 `Stimulus(kind="self_signal", ...)`。
- 配置开关 + CLI 控制。
- 测试覆盖闭环。

### 1.2 Out of scope

- LLM-based goal 生成（自主目标），v1 只接受用户 / 外部系统 set goal。
- 复杂的 goal 依赖图。
- 多 goal 并行 self-stimulus（v1 一时刻只一个）。

## 2. 任务清单

### 2.1 类型

- [ ] `cognition/models/goal.py`：`Goal` frozen dataclass（id / description /
  target_outcome / priority / status / created_at / updated_at / source /
  linked_belief_ids / **for_counterpart: CounterpartRef | None**）。无
  subject_id——系统只有一个 Subject。
- [ ] 把 `Goal` 加到 `cognition/models/__init__.py` export。
- [ ] `CognitiveEventKind` 增加 `goal_set` / `goal_satisfied` /
  `goal_abandoned` / `goal_progressed`。

### 2.2 GoalRegistry

- [ ] `cognition/goals/registry.py`：
  - `set_goal(...)` → emit `goal_set`。
  - `satisfy(goal_id, evidence)` → emit `goal_satisfied`。
  - `abandon(goal_id, reason)` → emit `goal_abandoned`。
  - `progress(goal_id, note)` → emit `goal_progressed`。
- [ ] `cognition/projections/goal.py`：`GoalProjection`。
- [ ] `state/schema.sql` 追加 `goal_view` 表（见 §3.1）。

### 2.3 DriveLoop

- [ ] `cognition/loops/drive.py`：`DriveLoop`。
  - `run_once()`：扫 active goals → 选一个 → 决定要不要生成 self-stimulus。
  - 入口先 `coordinator.acquire(LoopAcquireRequest(loop_name="drive",
    priority=LoopPriority.DRIVE, max_chunk_duration=timedelta(seconds=10)))`。
  - 选 goal 与生成 stimulus 是轻量级工作，单次 run_once 即完成；释放锁后
    enqueue stimulus 到 reactive 入口（Reactive 自己 try_acquire REACTIVE
    锁——若锁忙就丢弃这次 self_signal，下次 scheduler tick 再试）。
  - 通过 `CognitiveController.reactive_tick(stimulus, thread_id=
    ThreadId.cognition(SUBJECT_SELF, topic=goal_id))` 触发 Reactive。
  - Stimulus.source = goal.for_counterpart（若有），让 Reactive 知道这次思考
    "是关于 user_a 的"。
  - 通过 Phase 06 通用 scheduler 调度，触发条件：

    ```python
    ScheduleTrigger(
        min_interval=timedelta(minutes=5),    # 全局节流
        max_interval=None,                    # 无 goal 就永远不跑，没兜底
        watches=frozenset({"goal_set", "goal_progressed",
                            "received_feedback"}),
        min_new_events=1,
    )
    ```

    没人 set 过 goal、没人推过 goal 进度、也没有任何反馈，DriveLoop 根本
    不 acquire——主体没活儿干就别叫醒它。
- [ ] cooldown 配置：每 goal 至少间隔 T 秒才能再生成 self-stimulus（goal 维
  度节流，独立于 scheduler 全局节流）。

### 2.4 self-stimulus 路由

- [ ] `cognition/stages/perceive.py`：识别 `Stimulus.kind="self_signal"` →
  Perception.source = `self_signal`，situation.historical 与 conversation
  thread 隔离。
- [ ] ThreadId.cognition 已经在 Phase 04 准备好；本阶段保证 Drive Loop 用对。

### 2.5 配置 & CLI

- [ ] `config.cognition.drive.enabled = false`（默认关），与 Phase 06 的
  `config.cognition.consolidation.*` 使用同一 `cognition` 配置命名空间。
- [ ] `alpha cognition goals list [--active] [--subject]`。
- [ ] `alpha cognition goals set --description ... [--priority N]`。
- [ ] `alpha cognition goals satisfy <id> --evidence ...`。
- [ ] `alpha cognition goals abandon <id> --reason ...`。
- [ ] `alpha cognition drive --once` 手动跑一次（即使 enabled=false）。

### 2.6 测试

- [ ] `tests/cognition/test_goal_registry_emits.py`。
- [ ] `tests/cognition/test_goal_projection_rebuild.py`。
- [ ] `tests/cognition/test_drive_loop_triggers_reactive.py`：set goal →
  drive run_once → 看到 perceived (self_signal) 与后续完整 reactive 链。
- [ ] `tests/cognition/test_drive_loop_cooldown.py`：cooldown 内不重复触发。
- [ ] `tests/cognition/test_drive_loop_disabled_by_default.py`。
- [ ] `tests/cognition/test_cli_goals.py`。

### 2.7 文档

- [ ] AGENTS.md。
- [ ] `cognition/loops/README.md` 加 Drive Loop 说明。

## 3. 接口契约

### 3.1 `goal_view`

```sql
CREATE TABLE IF NOT EXISTS goal_view (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    target_outcome TEXT NOT NULL DEFAULT '',
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',   -- active / satisfied / abandoned
    source TEXT NOT NULL DEFAULT 'user',     -- user / reflector_l2 / external
    for_counterpart TEXT,                    -- CounterpartId | NULL
    linked_belief_ids TEXT NOT NULL DEFAULT '[]',
    last_drive_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_goal_status_priority
    ON goal_view(status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_goal_for_counterpart
    ON goal_view(for_counterpart, status);
```

系统只有一个 Subject，goal_view 没有 subject_id 列。

### 3.2 Goal & Stimulus

```python
@dataclass(frozen=True)
class Goal:
    id: GoalId
    description: str
    target_outcome: str
    priority: int
    status: Literal["active", "satisfied", "abandoned"]
    source: Literal["user", "reflector_l2", "external"]
    for_counterpart: CounterpartRef | None
    linked_belief_ids: list[BeliefId]
    created_at: Instant
    updated_at: Instant
    last_drive_at: Instant | None

# Phase 01 留下的 Stimulus 已经有 "self_signal" kind 与 source 字段。
# DriveLoop 用：
Stimulus(
    kind="self_signal",
    source=goal.for_counterpart,    # None 或目标关联的 Counterpart
    payload={"goal_id": ..., "drive_reason": "active goal needs progress"},
    thread_id=ThreadId.cognition(subject_id=SUBJECT_SELF, topic=goal_id),
    received_at=utc_now(),
)
```

### 3.3 事件

```python
"goal_set"
{"goal": { ... fields ... }, "source": "user"}

"goal_satisfied"
{"goal_id": ..., "evidence": "..."}

"goal_abandoned"
{"goal_id": ..., "reason": "..."}

"goal_progressed"
{"goal_id": ..., "note": "...", "linked_event_ids": [...]}
```

## 4. 文件清单

### 4.1 新增

```text
src/alpha_agent/cognition/models/goal.py
src/alpha_agent/cognition/goals/__init__.py
src/alpha_agent/cognition/goals/registry.py
src/alpha_agent/cognition/projections/goal.py
src/alpha_agent/cognition/loops/drive.py
tests/cognition/test_goal_registry_emits.py
tests/cognition/test_goal_projection_rebuild.py
tests/cognition/test_drive_loop_triggers_reactive.py
tests/cognition/test_drive_loop_cooldown.py
tests/cognition/test_drive_loop_disabled_by_default.py
tests/cognition/test_cli_goals.py
```

### 4.2 修改

```text
src/alpha_agent/state/schema.sql                追加 goal_view
src/alpha_agent/cognition/models/event.py       新增 goal_* kind
src/alpha_agent/cognition/stages/perceive.py    支持 self_signal
src/alpha_agent/cognition/loops/README.md
src/alpha_agent/config.py                       cognition.drive 配置段
src/alpha_agent/cli.py                          alpha cognition goals / drive
AGENTS.md
```

### 4.3 删除

无。

## 5. 验收标准

- [ ] `uv run pytest tests/cognition/test_goal_*.py
  tests/cognition/test_drive_loop_*.py -q` 全绿。
- [ ] 设一个 goal "answer pending question X" → `alpha cognition drive --once`
  → event log 出现完整 self_signal Reactive 链。
- [ ] cooldown 内重复 run_once 不再触发同 goal。
- [ ] 默认配置下 enabled=false，Drive Loop 不自动跑（仅手动 `--once` 可用）。
- [ ] satisfy/abandon 后该 goal 不再被选。
- [ ] self_signal 走 cognition thread，conversation thread 的 ContextWindow
  不受污染。

## 6. 风险与备注

- **自主行为的安全边界**。Drive Loop 一旦上线会让主体"自己说话"。默认关闭、
  cooldown、显式 enable 是三道闸门。生产环境开启前必须有 strategy override
  能立刻停。
- **goal_satisfied 谁来判**。v1 手动 satisfy（用户 CLI）。后续可加规则：
  当目标 linked_belief 出现 belief_formed → 自动 satisfy。但 v1 不做。
- **cognition thread 名字冲突**。`ThreadId.cognition(subject_id, goal_id)`
  是 stable key——同一 goal 多次 drive 共用一个 thread，ContextWindow 累积
  得到主体在该 goal 上的"思路"。
- **goal 太多会刷屏**。`SUBJECT_SELF` 的 active goal 上限默认 8，超过拒绝
  set；用户要 abandon 老的才能加新的。

## 7. 后续衔接

- Phase 11 L3 可读 GoalProjection 看主体的 goal 完成率、平均时长，更新
  SelfModel。
- 远期：加 LLM-based 自主 goal proposal——单独 phase，不在本计划。
