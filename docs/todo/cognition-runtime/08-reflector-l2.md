# Phase 08 — Reflector L2（控制）

**Status:** pending
**Depends on:** Phase 05, Phase 07
**Scope:** M
**Design ref:** `cognition_from_scratch.md` §7（L2 部分）；README 不变量 2

## 0. 目标

把元认知从"只读监控"升级为"可改下一轮策略"。L2 监测 L1 反思历史 +
Reactive 事件流的累积模式 → 当满足触发条件时发 `strategy_changed` 事件 →
Reactive 各 stage 在入口检查活跃 strategy override 并套用。

L2 不动 belief，不动 ValueLens，只改"接下来怎么走"的元规则——所以它不会
与 Reactive 改的数据竞争，只会与 Reactive 争 coordinator 锁。

L2 是调度型 loop，通过 LoopCoordinator 申请锁，priority=L2（仅次于
Reactive）。它的工作单元短（几秒级），但仍要遵守"自觉 yield"协议——单次
run_once 跑完所有 rule 评估即释放，不要长期占锁。

StrategyOverride 可选带 `for_counterpart: CounterpartRef | None` 字段——
例如 `require_explicit_confirm_on_contradiction` 只对 trust_level < 0.5
的 Counterpart 触发。

**匹配规则（固定）**：Reactive stage 在套用 strategy 前比对当前
`perception.from_counterpart` 与 `strategy.for_counterpart`：

- `strategy.for_counterpart is None` → 全局生效，所有 perception 都套用。
- `strategy.for_counterpart == perception.from_counterpart` → 套用。
- 其余 → 跳过。

注意是匹配 perception 的来源，**不是** matching Goal.for_counterpart 或
ContextWindow.counterpart——因为同一 tick 内可能跨 thread 引用，但触发当
前规则评估的"对方"以 perception 来源为准。Drive Loop 产生的 self_signal
perception 的 from_counterpart = goal.for_counterpart（见 Phase 10 §3.2），
两者自然一致。

v1 默认所有 override 是 global（for_counterpart=None），Counterpart-scoped
作为可选扩展。

## 1. 范围

### 1.1 In scope

- `ReflectorL2`：
  - 周期性扫 `reflection_view` + 部分 cognitive events。
  - 一组聚合规则：监测重复 reflection、连续 contradiction、反复 lens shift
    等。
  - 发 `strategy_changed` 事件，附 strategy 描述 + 有效期。
- `StrategyProjection`：维护当前活跃的 strategy override 列表。
- Reactive stages 在入口查询活跃 strategy → 套用对应行为（例如要求显式确认、
  关闭 auto-approve、加严 stance 判定）。
- 一个 strategy override 的内置实现（DSL 太重，v1 用 enum + payload）。
- 限制：每个 strategy 必须有过期时间（事件级 valid_until），到期自动失效。

### 1.2 Out of scope

- 自由形态的策略 DSL。
- L2 自己修改 L2（自指太深，留远期）。
- 跨主体策略共享。

## 2. 任务清单

### 2.1 L2 主模块

- [ ] `cognition/reflectors/l2.py`：`ReflectorL2`。
  - `run_once()`：扫近期 reflection + events → emit 若干 `strategy_changed`。
  - 入口先 `coordinator.acquire(LoopAcquireRequest(loop_name="reflector_l2",
    priority=LoopPriority.L2, max_chunk_duration=timedelta(seconds=15)))`。
  - 每条规则评估完后调 yield。
  - 通过 Phase 06 通用 scheduler 调度，触发条件：

    ```python
    ScheduleTrigger(
        min_interval=timedelta(minutes=1),
        max_interval=timedelta(hours=6),
        watches=frozenset({"reflected", "bias_detected"}),
        min_new_events=1,            # 哪怕 1 条新反思也值得评估
    )
    ```

    完全没新反思的周期内 scheduler 不 acquire 锁；最长 6h 兜底
    （lens_shift_flap 等需要时间窗判断的规则需要定期跑一次）。
- [ ] `cognition/reflectors/l2_rules/`：每条 L2 规则一个文件。

### 2.2 v1 L2 规则集

| name                              | 触发                                         | 产出 strategy                          |
| --------------------------------- | -------------------------------------------- | ------------------------------------ |
| recurring-contradiction-accepted | 同 kind L1 reflection 30 min 内 ≥3 次         | `require_explicit_confirm_on_contradiction` |
| feedback-surprise-streak          | 同 trigger 下 feedback 不匹配预期连续 ≥5 次  | `disable_auto_procedure_match_for_trigger` |
| lens-shift-flap                    | 同 lens shift 方向反复 ≥3 次（24h 内）       | `freeze_lens_learning_for_24h`       |
| premature-novel-auto-form-burst   | 1 小时内 novel 自动 belief_formed ≥5 次       | `require_confirm_before_novel_form`  |

### 2.3 StrategyProjection

- [ ] `cognition/projections/strategy.py`：`StrategyProjection`。
  - handles `strategy_changed`、`strategy_expired`。
  - `active(subject) -> list[StrategyOverride]`：当前所有有效 override。
  - `is_active(subject, name) -> bool`。
- [ ] `strategy_view` 表（见 §3.2）。

### 2.4 Reactive 套用 strategy

- [ ] `cognition/controller.py`：每个 stage 入口都从 StrategyProjection 取
  当前 override，传给 stage。
- [ ] 改 stage：
  - `interpret.py`：若 `require_confirm_before_novel_form` 活跃 → 标记
    Interpretation 含 `requires_confirmation=True`。
  - `decide.py`：若 `disable_auto_procedure_match_for_trigger` 活跃 → 跳过
    Procedure 匹配。
  - `revise.py`：若 `require_explicit_confirm_on_contradiction` 活跃 →
    matchedcontradicting 时 emit `belief_form_pending_confirmation` 而非
    `belief_formed`。
  - `loops/workers/learn_value_lens.py`：若 `freeze_lens_learning_for_24h`
    活跃 → 跳过。

### 2.5 过期机制

- [ ] `cognition/loops/workers/expire_strategies.py`：扫 active strategy 检查
  `valid_until` < now → emit `strategy_expired`。
  - ScheduleTrigger（纯时钟驱动型——只看时间不看 backlog）：

    ```python
    ScheduleTrigger(
        min_interval=timedelta(hours=1),
        max_interval=timedelta(hours=1),   # min == max → 每小时强制跑
        watches=frozenset(),
        min_new_events=0,
    )
    ```

    `min_interval == max_interval` 是"纯时钟扫描"型 worker 的标准模式：
    既不需要事件触发、也不需要兜底窗口，每 N 分/小时定时跑一次。

### 2.6 CLI

- [ ] `alpha cognition strategies [--active] [--all]`。
- [ ] `alpha cognition strategy-expire <id>` 手动撤销。

### 2.7 测试

- [ ] 每条规则一个 test：模拟 reflection 流 → 验证 emit。
- [ ] `tests/cognition/test_strategy_applied_in_interpret.py`。
- [ ] `tests/cognition/test_strategy_applied_in_decide.py`。
- [ ] `tests/cognition/test_strategy_applied_in_revise.py`。
- [ ] `tests/cognition/test_strategy_for_counterpart_matching.py`：
  - global override（for_counterpart=None）→ 对来自任何 Counterpart 的
    perception 都套用。
  - override.for_counterpart=user_a → 仅在 perception.from_counterpart
    == user_a 时套用；user_b 来源不套用。
  - Drive Loop self_signal（其 perception.from_counterpart = goal.
    for_counterpart）→ 套用结果与对应 user 直接交互时一致。
- [ ] `tests/cognition/test_strategy_expires_on_clock.py`。
- [ ] `tests/cognition/test_cli_strategies.py`。

### 2.8 文档

- [ ] AGENTS.md。
- [ ] `cognition/reflectors/l2_rules/README.md` 列规则与对应 strategy。

## 3. 接口契约

### 3.1 StrategyOverride

```python
@dataclass(frozen=True)
class StrategyOverride:
    id: StrategyId
    name: str                              # 例如 "require_explicit_confirm_on_contradiction"
    payload: dict[str, Any]                # 规则特定参数
    target_stages: list[str]               # interpret / decide / revise / lens_learning
    for_counterpart: CounterpartRef | None # Counterpart-scoped 策略；None 表示全局
    set_by: str                            # "reflector_l2" / "user"
    set_at: Instant
    valid_until: Instant
```

### 3.2 `strategy_view` 表

```sql
CREATE TABLE IF NOT EXISTS strategy_view (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    target_stages TEXT NOT NULL DEFAULT '[]',
    for_counterpart TEXT,                  -- CounterpartId | NULL
    set_by TEXT NOT NULL,
    set_at TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'  -- active / expired / revoked
);

CREATE INDEX IF NOT EXISTS idx_strategy_status_validity
    ON strategy_view(status, valid_until);
CREATE INDEX IF NOT EXISTS idx_strategy_for_counterpart
    ON strategy_view(for_counterpart, status);
```

### 3.3 事件

```python
"strategy_changed"
{
    "strategy": { ...StrategyOverride fields... },
    "triggered_by_reflection_ids": [...],
    "triggered_by_rule": "recurring-contradiction-accepted",
}

"strategy_expired"
{"strategy_id": "...", "reason": "valid_until reached"}

"belief_form_pending_confirmation"
{"belief": { ... }, "reason": "strategy:require_explicit_confirm_on_contradiction"}
```

## 4. 文件清单

### 4.1 新增

```text
src/alpha_agent/cognition/reflectors/l2.py
src/alpha_agent/cognition/reflectors/l2_rules/__init__.py
src/alpha_agent/cognition/reflectors/l2_rules/recurring_contradiction_accepted.py
src/alpha_agent/cognition/reflectors/l2_rules/feedback_surprise_streak.py
src/alpha_agent/cognition/reflectors/l2_rules/lens_shift_flap.py
src/alpha_agent/cognition/reflectors/l2_rules/premature_novel_auto_form_burst.py
src/alpha_agent/cognition/reflectors/l2_rules/README.md
src/alpha_agent/cognition/projections/strategy.py
src/alpha_agent/cognition/loops/workers/expire_strategies.py
tests/cognition/test_l2_rule_*.py                 (4 files)
tests/cognition/test_strategy_applied_in_interpret.py
tests/cognition/test_strategy_applied_in_decide.py
tests/cognition/test_strategy_applied_in_revise.py
tests/cognition/test_strategy_for_counterpart_matching.py
tests/cognition/test_strategy_expires_on_clock.py
tests/cognition/test_cli_strategies.py
```

### 4.2 修改

```text
src/alpha_agent/state/schema.sql                            追加 strategy_view
src/alpha_agent/cognition/controller.py                     stages 入口注入 active strategies
src/alpha_agent/cognition/stages/interpret.py
src/alpha_agent/cognition/stages/decide.py
src/alpha_agent/cognition/stages/revise.py
src/alpha_agent/cognition/loops/workers/learn_value_lens.py 查 strategy 决定是否跳过
src/alpha_agent/cognition/loops/consolidation.py            注册 expire_strategies
src/alpha_agent/cognition/models/event.py                   新事件 kind
src/alpha_agent/cli.py                                      alpha cognition strategies
AGENTS.md
```

### 4.3 删除

无。

## 5. 验收标准

- [ ] `uv run pytest tests/cognition/test_l2_*.py tests/cognition/test_strategy_*.py -q`
  全绿。
- [ ] 演示：连续 3 次矛盾输入 → L2 emit `strategy_changed` →
  第 4 次矛盾输入时 Reactive 走"需要显式确认"分支。
- [ ] strategy valid_until 到 → 自动 emit `strategy_expired` → 下一轮恢复
  正常行为。
- [ ] `alpha cognition strategies --active` 列出当前活跃 strategies。
- [ ] L2 不会 emit 任何 `belief_*` 事件（只发 strategy_*）。

## 6. 风险与备注

- **strategy 名字稳定性**。每个 name 是 stage 代码识别的 enum；命名一旦发布
  不能改，否则历史事件 replay 行为异变。如要改，加新 name 并 deprecate 旧的。
- **过度策略化风险**。L2 emit 太多 strategy 会让 Reactive 行为反复变。在
  `strategy_view` 设硬上限：同主体同时 active strategies ≤ 5。
- **strategy 来源透明**。`set_by` 字段必须可信——`reflector_l2` / `user` /
  `gateway_admin`。
- **测试时间控制**。strategy 过期要测，必须注入 clock。
- **stage 检查代价**。每个 stage 入口都查 active strategies，要做缓存（per
  tick 缓存一次，不要每次都查 DB）。

## 7. 后续衔接

- Phase 11 L3 SelfModel 用 strategy_changed 历史推断"这个主体倾向于何时被
  L2 干预"。
- 将来可加 UI 让用户手动 set strategy（同接口，`set_by="user"`）。
