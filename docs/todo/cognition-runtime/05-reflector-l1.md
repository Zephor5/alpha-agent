# Phase 05 — Reflector L1（监控）

**Status:** completed
**Depends on:** Phase 02, Phase 03
**Scope:** S
**Design ref:** `cognition_from_scratch.md` §7（L1 部分）；README 不变量 1/2

## 0. 目标

把 Phase 02 留下的 `ReflectorL1.audit(...)` 占位（恒返回空）实现成真正的规则
化反思器。它**只读**地审视刚走完的 9 步链，按一组规则发出 Reflection 事件；
不修改任何 Belief、不改变下一轮策略。L2/L3 留给 Phase 08/11。

L1 在 Reactive tick 的 reflect 步骤里同步执行，完全在 Reactive 持锁期间，
不自己 acquire 协调锁。规则评估时可读 `ctx.perception.from_counterpart`
与对应 Counterpart 的 trust_level——例如把 `trust_level < 0.3` 作为
`low-confidence-high-stakes` 等规则的额外触发条件。

## 1. 范围

### 1.1 In scope

- 5–8 条 v1 反思规则（见 §3.2）。
- Reflection 事件写日志：`reflected` + 每条 Reflection 单独的
  `bias_detected`（如适用）。
- `ReflectionProjection`：按 kind / severity / 时间窗 / 目标 belief id 查询。
- CLI：`alpha cognition reflections [--severity warning] [--last N]`。
- 测试覆盖每条规则的正反例。

### 1.2 Out of scope

- 跨轮聚合反思（Phase 08 L2 做）。
- 用 Reflection 改下一轮策略（Phase 08 L2 做）。
- 用 Reflection 更新 Subject.SelfModel（Phase 11 L3 做）。
- LLM-assisted reflection（远期，留给独立 phase）。

## 2. 任务清单

### 2.1 Reflector 模块

- [x] `cognition/reflectors/__init__.py`。
- [x] `cognition/reflectors/l1.py`：规则引擎 `ReflectorL1`。
  - `AuditContext` 接收 ProjectionRegistry（规则按需读 projection）。
  - `audit(ctx: AuditContext) -> list[Reflection]`。
  - 内部把规则注册成 plug-in callable，便于增删。
- [x] `cognition/stages/reflect.py`：stage wrapper 装配完整 tick context，
  接收 `audit(perception, focus, interpretation, judgments, decision, outcome,
  feedback, subject, window, ...)`。
- [x] `cognition/reflectors/rules/`：每条规则一个文件，统一签名（见 §3.1）。

### 2.2 v1 规则集

按 §3.2 实现 6 条规则。

- [x] `low-confidence-high-stakes`
- [x] `contradiction-accepted`
- [x] `situation-mismatch`
- [x] `unsupported-tool-call`
- [x] `premature-novel-auto-form`
- [x] `feedback-surprise`

### 2.3 Projection 与查询

- [x] `cognition/projections/reflection.py`：`ReflectionProjection`。
- [x] `state/schema.sql` 追加 `reflection_view` 表。

### 2.4 Reactive 接入

- [x] `cognition/controller.py`：把 Phase 02 留下的空 reflector 调用替换。
- [x] `cognition/stages/reflect.py`：调 `ReflectorL1.audit(...)`，emit
  `reflected` 事件（含 reflection_count）+ 每条 Reflection 一个独立事件。

### 2.5 CLI

- [x] `cli.py`：新增 `alpha cognition reflections` 子命令。

### 2.6 测试

- [x] 每条规则两个测试：触发 / 不触发。
- [x] `tests/cognition/test_reflector_no_op_when_nothing_to_audit.py`：
  健康 tick → 空列表。
- [x] `tests/cognition/test_reflection_projection_rebuild.py`。
- [x] `tests/cognition/test_cli_reflections.py`。

### 2.7 文档

- [x] 更新 AGENTS.md。
- [x] 在 cognition/reflectors/ README 简列 v1 规则及触发条件。

## 3. 接口契约

### 3.1 规则签名

```python
class ReflectionRule(Protocol):
    name: ClassVar[str]

    def evaluate(self, ctx: AuditContext) -> Iterator[Reflection]: ...

@dataclass(frozen=True)
class AuditContext:
    tick_id: str
    perception: Perception
    focus: AttentionFocus
    interpretation: Interpretation
    judgments: list[Judgment]
    decision: Decision
    outcome: Outcome
    feedback: Feedback
    subject: Subject
    counterpart: Counterpart | None         # 当前 thread 的对方；从 perception 路由
    projections: ProjectionRegistry          # 规则按需取 BeliefProjection /
                                              # CounterpartProjection 等
```

把整个 `ProjectionRegistry` 传进来——单一规则可能要读 belief、counterpart、
strategy 多张表，不要在 AuditContext 上一个字段一个字段地暴露。

### 3.2 规则列表

| name                          | 触发                                          | severity |
| ----------------------------- | -------------------------------------------- | -------- |
| low-confidence-high-stakes    | judgment.confidence<0.4 且 existence/safety 权重>0.7 | warning  |
| contradiction-accepted        | 同时 supports & contradicting 含同一 belief    | blocker  |
| situation-mismatch            | judgment.applicable_under 与当前 situation 不符 | info     |
| unsupported-tool-call         | decision.action=use_tool 但无任何 judgment 要求 | warning  |
| premature-novel-auto-form     | stance=novel 且 confidence<0.5 且反馈明确记录 formed_belief_ids | warning  |
| feedback-surprise             | feedback.matched_expected=False 且 surprises 非空 | info     |

### 3.3 `reflection_view` 表

```sql
CREATE TABLE IF NOT EXISTS reflection_view (
    id TEXT PRIMARY KEY,
    tick_id TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'L1',
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    target_kind TEXT NOT NULL,         -- belief / judgment / decision / loop_run
    target_id TEXT NOT NULL,
    finding TEXT NOT NULL,
    suggested_remedy TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reflection_severity
    ON reflection_view(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reflection_kind
    ON reflection_view(kind, created_at DESC);
```

### 3.4 事件

```python
# reflected (Phase 02 已埋)
{
    "tick_id": ...,
    "reflection_count": N,
    "reflection_ids": [...],
    "reflections": [{...完整 Reflection record...}],
}

`ReflectionProjection` 以 `reflected.payload["reflections"]` 作为完整重建
来源；`bias_detected` 是每条 finding 的索引/审计事件，不承载完整 finding
record。

# bias_detected (新)
{
    "tick_id": ...,
    "reflection_id": "...",
    "kind": "...",
    "severity": "...",
    "target": {"kind": "...", "id": "..."},
}
```

## 4. 文件清单

### 4.1 新增

```text
src/alpha_agent/cognition/reflectors/__init__.py
src/alpha_agent/cognition/reflectors/l1.py
src/alpha_agent/cognition/reflectors/rules/__init__.py
src/alpha_agent/cognition/reflectors/rules/low_confidence_high_stakes.py
src/alpha_agent/cognition/reflectors/rules/contradiction_accepted.py
src/alpha_agent/cognition/reflectors/rules/situation_mismatch.py
src/alpha_agent/cognition/reflectors/rules/unsupported_tool_call.py
src/alpha_agent/cognition/reflectors/rules/premature_novel_auto_form.py
src/alpha_agent/cognition/reflectors/rules/feedback_surprise.py
src/alpha_agent/cognition/projections/reflection.py
tests/cognition/test_rule_low_confidence_high_stakes.py
tests/cognition/test_rule_contradiction_accepted.py
tests/cognition/test_rule_situation_mismatch.py
tests/cognition/test_rule_unsupported_tool_call.py
tests/cognition/test_rule_premature_novel_auto_form.py
tests/cognition/test_rule_feedback_surprise.py
tests/cognition/test_reflector_no_op_when_nothing_to_audit.py
tests/cognition/test_reflection_projection_rebuild.py
tests/cognition/test_cli_reflections.py
```

### 4.2 修改

```text
src/alpha_agent/state/schema.sql            追加 reflection_view
src/alpha_agent/cognition/stages/reflect.py 接 ReflectorL1
src/alpha_agent/cognition/controller.py     注入 ReflectorL1
src/alpha_agent/cli.py                      alpha cognition reflections 子命令
AGENTS.md                                   项目导航补 reflectors/
```

### 4.3 删除

无。

## 5. 验收标准

- [x] `uv run pytest tests/cognition/test_rule_*.py -q` 全绿。
- [x] `alpha cognition reflections --last 10` 能列出最近反思。
- [x] `contradiction-accepted` 的触发语义有规则级正反例覆盖；完整“连续输入
  两条偏好后由第二条触发”依赖 Phase 06+ 的 belief 形成/巩固路径。
- [x] 任何 tick 跑完，event log 里都有 `reflected` 事件（即使 reflection_count=0）。
- [x] drop reflection_view → replay → 等价。

## 6. 风险与备注

- **规则数量克制**。v1 6 条够用。规则集要小步加，每条新规则要有正反例。
- **Reflection 误报代价**。Phase 05 的 reflection **不**触发任何行动；只在
  日志里显形。即使误报多也不破坏 Reactive Loop。Phase 08 L2 开始用它们改策
  略时才需要严格调参。
- **规则之间的耦合**。每条规则只看 AuditContext，规则之间无序号依赖——后续
  可任意增删、并行评估。
- **subject_id 隐式**。系统只有一个 Subject，reflection_view 不冗余
  subject_id 列；事件本身带 subject_id 已够审计。

## 7. 后续衔接

- Phase 08 L2 读 ReflectionProjection 的最近窗口聚合，决定要不要发
  `strategy_changed`。
- Phase 11 L3 读全量 ReflectionProjection 提取 typical_failure_modes 写到
  SelfModel。
