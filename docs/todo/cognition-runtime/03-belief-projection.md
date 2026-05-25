# Phase 03 — BeliefProjection v1

**Status:** pending
**Depends on:** Phase 01, Phase 02
**Scope:** M
**Design ref:** `cognition_from_scratch.md` §9.2；README 不变量 1

## 0. 目标

把 Phase 02 的 `BeliefProjection` stub 换成真正可查询的信念视图。这一阶段
完成后：

- 事件日志里的 `belief_formed` / `belief_strengthened` / `belief_weakened` /
  `belief_superseded` / `belief_retracted` 能正确投影成 active belief 集合。
- `Interpreter.interpret(...)` 能拿到 recalled = "与当前焦点相关的活跃信念
  列表"；默认 scope 包含 "关于当前 thread Counterpart 的信念" + "全局信念
  （about 为空）"。
- 重启进程后从事件日志 replay 能重建等价 projection。
- "用户上轮提到偏好 X，下轮能回忆"类用例转绿。

`Belief.about: list[Reference]` 在 Phase 01 已经是类型层 first-class。本阶
段第一次把"按 about 查询"做到底——既支持显式按 Counterpart 拉、也支持 recall
路径自动 join。

## 1. 范围

### 1.1 In scope

- 真正的 `BeliefProjection`：按 `cognitive_type` / `about` / `entity` /
  `status` 查询。
- `recall(params: BeliefRecallParams) -> list[Belief]` 实现：基于焦点
  entities + about 过滤（含当前 thread 关联的 Counterpart + 无 about 的全
  局信念）。
- `Belief` 的 supersede / merge / retract projection 逻辑。
- 一份持久化存储：`belief_view` 表（projection 物化表，可从事件日志重建），
  内含 `about_index` 倒排支持 about 查询。
- 从历史事件流冷启动重建 view 的能力。
- Phase 00 中 xfail 的"长期记忆"类测试转绿。

### 1.2 Out of scope

- 向量/全文检索（暂用 entity 倒排 + 子串匹配）。
- 冲突解决——本阶段 supersede 只接受**显式**事件（即上游已经决定要
  supersede 谁），不做自动判定。Phase 07 ValueLens 才做自动冲突解决。
- recall 排序——本阶段返回全部相关 active belief，排序留给 Phase 09 Renderer
  阶段或后续 retrieval scoring。

## 2. 任务清单

### 2.1 Projection 物化表

- [ ] `state/schema.sql` 追加 `belief_view` 表（见 §3.1）。这是 BeliefProjection
  物化结果，**事件日志是源头，这张表可任意 drop & rebuild**。

### 2.2 Projection 实现

- [ ] `cognition/projections/belief.py` 替换 stub：
  - `apply(event)` 处理：`belief_formed` / `belief_strengthened` /
    `belief_weakened` / `belief_superseded` / `belief_retracted` /
    `belief_archived`。
  - `recall(params: BeliefRecallParams) -> list[Belief]`：scope = entity
    overlap + about 过滤（含当前 Counterpart + 无 about 的全局信念）+ status
    = active。
  - `recall_about(counterpart_ref) -> list[Belief]`：直接拉关于某 Counterpart
    的所有活跃信念（renderer / consolidation 都会用）。
  - `get_by_id(belief_id) -> Belief`。
  - `list_active() -> Iterator[Belief]`（不需要 subject 参数——系统只有一个
    Subject）。
- [ ] `cognition/projections/belief.py` 内附 `BeliefRecallParams` 数据类，
  封装 focus / counterpart / 限制条件，便于将来扩展不破坏签名。

### 2.3 Interpreter 接入

- [ ] `cognition/stages/interpret.py`：把 stub `recalled=[]` 改成调用
  `BeliefProjection.recall(BeliefRecallParams(focus=focus,
  counterpart=window.counterpart))`。
- [ ] Interpreter stance 判定（v1）：规则化
  - 焦点 claim 与 recalled belief 的 normalized content 完全等价 → consistent
  - 焦点 claim 与 recalled belief 同 subject/predicate 但 object 不同 → contradicting
  - 焦点 claim 引入了 active belief 集合中未出现的新实体 + 新断言 → novel
  - 焦点 claim 与 recalled 部分重叠但不明确 → ambiguous

### 2.4 Reviser 接入

- [ ] `cognition/stages/revise.py`：从 Interpretation + Decision 推导出 Belief
  形成事件：
  - stance=novel + Decision 不是 refuse → `belief_formed`
  - stance=contradicting + Decision 表态采纳新主张 → 先 emit `belief_formed`
    （新主张），再 emit `belief_superseded`（指向旧主张）
  - 用户显式说"忘记 X" → `belief_retracted`
- 每条 belief_formed 事件的 payload 包含：完整 Belief 字段。Projection 据此
  物化 `belief_view` 行。

### 2.5 测试

- [ ] `tests/cognition/test_belief_projection_apply.py`：
  - apply belief_formed → view 出现
  - apply belief_superseded → 旧 row 状态变 superseded，新 row active
  - apply belief_retracted → 状态变 retracted
- [ ] `tests/cognition/test_belief_projection_rebuild.py`：
  - 跑一串事件流 → drop view → replay → view 等价
- [ ] `tests/cognition/test_recall_by_counterpart.py`：
  - 给 thread 关联 counterpart:user_a → recall 只返回 about=[user_a] 或
    about=[] 的 belief，about=[user_b] 的不返回。
- [ ] `tests/cognition/test_recall_entity_overlap.py`：
  - focus.entities = ["python"] → 只 recall 涉及 "python" 实体的 belief
- [ ] `tests/cognition/test_recall_about_explicit.py`：
  - `recall_about(counterpart:user_a)` 返回所有关于 user_a 的活跃 belief，
    不受 entity 过滤限制。
- [ ] `tests/cognition/test_phase_00_xfail_now_pass.py`：
  - 把 Phase 00 留下的"长期记忆"xfail 用例搬过来，去掉 xfail 标记

### 2.6 文档

- [ ] AGENTS.md 项目导航更新：列出 `belief_view` 表与 `BeliefProjection`。
- [ ] 在仓库 README 加一行"Beliefs are now recallable across sessions"。

## 3. 接口契约（草案）

### 3.1 `belief_view` 表

```sql
CREATE TABLE IF NOT EXISTS belief_view (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    cognitive_type TEXT NOT NULL,
    structure TEXT NOT NULL DEFAULT '{}',
    sources TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.5,
    applicability TEXT NOT NULL DEFAULT '{}',
    value_profile TEXT NOT NULL DEFAULT '{}',
    relations TEXT NOT NULL DEFAULT '[]',
    formed_in_situation TEXT,
    holder_role TEXT,
    action_orientation TEXT NOT NULL DEFAULT '[]',
    update_policy TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'active',
    held_since TEXT NOT NULL,
    held_until TEXT,
    supersedes TEXT,
    superseded_by TEXT,
    last_event_id TEXT NOT NULL                -- 最近一条影响本行的事件
);

CREATE INDEX IF NOT EXISTS idx_belief_view_status
    ON belief_view(status);
CREATE INDEX IF NOT EXISTS idx_belief_view_type
    ON belief_view(cognitive_type, status);

-- entity 倒排表：每条 belief 涉及的每个 entity 占一行
CREATE TABLE IF NOT EXISTS belief_entity_index (
    belief_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY(belief_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_belief_entity_lookup
    ON belief_entity_index(entity_id, belief_id);

-- about 倒排表：每条 belief 的每个 about 引用占一行
CREATE TABLE IF NOT EXISTS belief_about_index (
    belief_id TEXT NOT NULL,
    about_kind TEXT NOT NULL,       -- counterpart / entity / subject
    about_id TEXT NOT NULL,
    PRIMARY KEY(belief_id, about_kind, about_id)
);

CREATE INDEX IF NOT EXISTS idx_belief_about_lookup
    ON belief_about_index(about_kind, about_id, belief_id);
```

Subject 列移除——系统只有一个 Subject，列冗余无意义。entity 与 about 都走
独立倒排表（同一形式），避免一个用 JSON LIKE、一个用倒排的不对称。
projection 在 apply 时同步写主表 + 两张倒排；rebuild 时三张表一起 drop &
重建。

### 3.2 Projection 主接口

```python
@dataclass(frozen=True)
class BeliefRecallParams:
    focus: AttentionFocus
    counterpart: CounterpartRef | None = None   # 当前 thread 关联的对方
    include_global: bool = True                  # 是否同时拉 about=[] 的全局信念
    types: frozenset[CognitiveType] | None = None
    limit: int = 32

class BeliefProjection(Projection):
    name = "belief"
    handles = frozenset({
        "belief_formed", "belief_strengthened", "belief_weakened",
        "belief_superseded", "belief_retracted", "belief_archived",
    })

    def recall(self, params: BeliefRecallParams) -> list[Belief]: ...
    def recall_about(self, ref: Reference) -> list[Belief]: ...
    def get_by_id(self, belief_id: BeliefId) -> Belief | None: ...
    def list_active(self) -> Iterator[Belief]: ...
```

### 3.3 事件 payload

```python
# belief_formed
{
    "tick_id": ...,
    "belief": { ... full Belief fields ... },
}

# belief_superseded
{
    "tick_id": ...,
    "old_belief_id": "...",
    "new_belief_id": "...",
    "reason": "...",
}

# belief_retracted
{
    "tick_id": ...,
    "belief_id": "...",
    "reason": "...",
}
```

## 4. 文件清单

### 4.1 新增

```text
tests/cognition/test_belief_projection_apply.py
tests/cognition/test_belief_projection_rebuild.py
tests/cognition/test_recall_by_counterpart.py
tests/cognition/test_recall_entity_overlap.py
tests/cognition/test_recall_about_explicit.py
tests/cognition/test_phase_00_xfail_now_pass.py
```

### 4.2 修改

```text
src/alpha_agent/state/schema.sql                  追加 belief_view
src/alpha_agent/cognition/projections/belief.py   替换 stub
src/alpha_agent/cognition/stages/interpret.py     recalled 接真实 projection
src/alpha_agent/cognition/stages/revise.py        发 belief_* 事件
tests/test_agent_loop.py                          移除"长期记忆"用例的 xfail 标
tests/test_cli_agent_loop.py                      同上
```

### 4.3 删除

无。

## 5. 验收标准

- [ ] `uv run pytest tests/cognition/ -q` 全绿。
- [ ] 删除 `belief_view` 表 → 重启 → `BeliefProjection` 从 cognitive_events
  replay → 等价于删除前。
- [ ] 跨 session 测试：session A 形成 belief "user prefers Python" → 关掉 →
  session B 同主体下 ask "what language do I prefer?" → 答案能引用该 belief。
- [ ] Phase 00 中 xfail 的"长期记忆"类用例转绿（至少 3 条）。
- [ ] `alpha debug prompt` 新增 `--show-recall` 选项，能打印本轮 recall 出的
  belief 列表。

## 6. 风险与备注

- **物化表 vs 完全 replay**。冷启动一个长期主体（百万事件）跑全 replay 会慢。
  v1 物化到 `belief_view`，重启时只 replay 自 last_event_id 起的增量。详细
  保留点见 Projection 基类的 `last_processed_event_id` 字段。
- **entity / about 均走倒排表**（`belief_entity_index` /
  `belief_about_index`）。SQLite 没有原生 array 类型；早期版本曾考虑
  "entity 走 JSON LIKE、about 走倒排"的混合方案，后来统一改成两边都倒排，
  好处是查询语义对称、可索引、写入路径简单（apply belief_formed 时三张表
  一次性 upsert）。
- **belief id 的稳定性**。同一断言两次出现要不要复用 id？v1 不复用——每次
  `belief_formed` 都新 id；去重靠 Phase 06 Consolidation。这样事件日志保
  持简单。
- **stance 判定是规则化的**。第一版不接 LLM；后续阶段可以加 LLM-assisted
  Interpreter 作为可插拔替代。
- **scope 隔离**。SubjectId 之间默认完全隔离。共享区在后续 phase 才考虑
  （cognition.md 七层之外，跨主体设计另开文档）。

## 7. 后续衔接

- Phase 04 ContextWindowProjection 会消费 `BeliefProjection.recall(...)` 的
  结果作为 ContextWindow.recalled 字段。
- Phase 05 ReflectorL1 的"contradiction-accepted"规则需要查 BeliefProjection
  当前 active 状态。
- Phase 06 Consolidation 会扫 BeliefProjection 找重复 belief 合并；并通过
  promote Judgment 形成新 belief。
- Phase 07 ValueLens 在解冲突时读 BeliefProjection。
