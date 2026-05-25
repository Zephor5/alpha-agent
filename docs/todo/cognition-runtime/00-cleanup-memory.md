# Phase 00 — 清理现有 memory 机制

**Status:** pending
**Depends on:** —（起点）
**Scope:** L
**Design ref:** `docs/cognition/cognition_from_scratch.md` §13.1 / §13.2 / AGENTS.md
"DO NOT FOLLOW PARTIAL COMPATIBILITY FIRST" 规则。

## 0. 目标

把现有 `src/alpha_agent/memory/` 子系统按目标架构所需的状态清理干净，让后续
Phase 01–11 能在没有遗留 memory 概念污染的地基上长。这一阶段交付**一个能跑、
但不带任何长期记忆**的 Alpha Agent：会话内可对话（用 conversation_messages 维
持一轮上下文），但不再有 episodic/semantic/procedural/candidate/scene/persona
等"记忆条目"概念，也没有 MemoryController / extractor / consolidation 调用。

为什么 cleanup 自己一阶段：

- 现有 memory 子系统跨 14 个 Python 文件、13 个 SQLite 表、若干 CLI 子命令、
  众多测试与 README 段落。半清不清的状态会让 Phase 01 同时背负"建新事件日
  志"和"和旧 memory 共存"两件事，违反 AGENTS.md 的反兼容层规则。
- 这一阶段不引入任何新设计，只做删除与收缩。失败成本低，可独立 merge。
- 留下的"干净 baseline"是后续每阶段都能 fork 对照的基线。

## 1. 范围

### 1.1 In scope

- 删除 `src/alpha_agent/memory/` 中由本计划新架构取代的所有模块与表。
- 重写 `src/alpha_agent/memory/schema.sql`（更名为 `state/schema.sql` 或保留
  路径，但语义重置——只保留与 cognition 无关的运行时表）。
- 收缩 `src/alpha_agent/runtime/agent.py`：去掉 MemoryController / Retriever /
  Extractor / Consolidation 的所有调用；`AlphaAgent.respond()` 只剩
  "append conversation_message → 拼一个最简 prompt → call LLM → append assistant
  message → return"。
- 删除 / 重命名 `alpha memory ...` CLI 子命令族；保留 `alpha debug prompt`
  作为后续 cognition inspection 的占位。
- 删除依赖被删模块的测试；保留行为型测试（"用户说 X，agent 回 Y"）但允许它
  们暂时失败，作为 Phase 02 完成后要恢复的红线。
- 更新 `README.md`、`AGENTS.md` 项目导航、`docs/TODO.md` 中关于 memory 的段
  落，标明"memory 子系统已废止，新认知运行时见 `docs/todo/cognition-runtime/`"。
- 把 `docs/doing/memory-system-optimization-phases.md` 移到
  `docs/develop_record/`，明确归档。

### 1.2 Out of scope

- 不引入 `src/alpha_agent/cognition/` 任何目录或类型（Phase 01 做）。
- 不引入新事件日志表（Phase 01 做）。
- 不动 gateway / LLM provider / tool registry / skills 模块。
- 不动 `conversation_messages`、`runtime_traces`、`gateway_session_mappings`、
  `gateway_dedup` 这 4 张与 memory 无关的表。

## 2. 任务清单

### 2.1 决策表（先写下来）

- [ ] 在本文档底部 §6 风险与备注里补一张表，逐项记录每个 memory/* 模块、每
  张表、每个 CLI 子命令、每个测试文件的处置（delete / keep / move）。表是这一
  阶段实际工作的"清单"。

### 2.2 数据库 schema 重写

- [ ] 删除以下表（drop schema + 删初始化 SQL）：
  - `session_context_states`
  - `episodic_memories`
  - `semantic_memories`
  - `procedural_memories`
  - `memory_candidates`
  - `memory_decisions`
  - `memory_access_log`
  - `entity_nodes`
  - `relation_edges`
- [ ] 保留以下表，schema 不动：
  - `conversation_messages`
  - `runtime_traces`
  - `gateway_session_mappings`
  - `gateway_dedup`
- [ ] 删除全部相关索引。
- [ ] 把 `src/alpha_agent/memory/schema.sql` 移到
  `src/alpha_agent/state/schema.sql`，模块也跟着改名（见 2.3）。

### 2.3 Python 模块删除

- [ ] 删除以下文件：
  - `src/alpha_agent/memory/consolidation.py`
  - `src/alpha_agent/memory/controller.py`
  - `src/alpha_agent/memory/episodic.py`
  - `src/alpha_agent/memory/extractor.py`
  - `src/alpha_agent/memory/persistence.py`
  - `src/alpha_agent/memory/procedural.py`
  - `src/alpha_agent/memory/retrieval.py`
  - `src/alpha_agent/memory/review.py`
  - `src/alpha_agent/memory/salience.py`
  - `src/alpha_agent/memory/semantic.py`
- [ ] 把 `memory/models.py` 与 `memory/store.py` 收缩为只含 `conversation_messages`
  与 `runtime_traces` 的访问。把整个目录改名为
  `src/alpha_agent/state/`，因为这两张表是"会话状态"，与"长期记忆"无关。
- [ ] 在 `state/__init__.py` 中只 export `ConversationMessage`、`RuntimeTrace`、
  `StateStore`。
- [ ] 检查仓库 grep：`grep -r "alpha_agent.memory" src/ tests/` 全部清零。

### 2.4 Runtime 收缩

- [ ] `src/alpha_agent/runtime/agent.py`：
  - 删除对 MemoryController / Retriever / Extractor / Consolidation / Semantic /
    Episodic / Procedural / Review 的所有 import 与字段。
  - `AlphaAgent.respond()` 删到只剩：
    1. append user `ConversationMessage`
    2. 拼一个最简单的 system + 最近 N 条 conversation_messages 的 prompt
    3. LLM call（保留 tool loop 与重试逻辑）
    4. append assistant `ConversationMessage`
    5. 返回响应
  - 文件目标行数 < 400（当前 1500+）。
- [ ] `src/alpha_agent/runtime/prompt_builder.py`：
  - 删除 persona / scene / semantic / episodic / procedural 渲染段落。
  - 只保留"system prompt + uncompressed conversation messages + user query"。
- [ ] `src/alpha_agent/runtime/context_compression.py`、
  `src/alpha_agent/runtime/session_context.py`：
  - 删除 StructuredSessionState、SessionContextManager 中跟 memory 相关的逻
    辑。SessionContextManager 收缩为只提供"取这条 session 最近 N 条
    conversation_messages"。
  - context_compression.py 整文件先删除（后续压缩在 Phase 06 重写）。

### 2.5 CLI 收缩

- [ ] `src/alpha_agent/cli.py`：
  - 删除整个 `alpha memory ...` 子命令族（list / inspect / approve / reject /
    edit / forget / consolidate / metrics / diagnostics 等全部）。
  - 保留 `alpha debug prompt`，使其只输出当前最简 prompt。
  - 保留 `alpha chat`、`alpha ask`、`alpha config`、`alpha skills`、
    `alpha gateway`。

### 2.6 测试清理

- [ ] 删除以下测试文件：
  - `tests/test_memory_store.py`
  - `tests/test_memory_extraction_eval.py`
  - `tests/test_memory_review.py`
  - `tests/test_retrieval.py`
  - `tests/test_consolidation.py`
  - `tests/memory_eval.py`
- [ ] `tests/test_prompt_builder.py`：保留，但删除所有断言已删除渲染段的用
  例；只留对最简 prompt 的断言。
- [ ] `tests/test_agent_loop.py`、`tests/test_cli_agent_loop.py`：保留行为
  断言，但更新对话期望——因为不再有 long-term memory，所以"上一次说过 X，下次
  能回忆"的用例此阶段会失败。在测试里加 `pytest.mark.xfail` 并标注 `reason=
  "depends on cognition Phase 02+"`。
- [ ] 跑一遍 `uv run pytest`，确认剩余测试全绿（除上面 xfail 的）。

### 2.7 文档更新

- [ ] `README.md`：删除整段"Memory inspection / review / forget" 介绍；加一
  段醒目的"Long-term cognition is being rebuilt; see
  `docs/todo/cognition-runtime/`"。
- [ ] `AGENTS.md` 的 "Project Content Navigation"：
  - 把 `memory/` 改成 `state/`，描述也改成 "session-level state tables only;
    long-term cognition lives in `cognition/` (under construction)"。
  - 在 docs 段下加 `docs/todo/cognition-runtime/` 一行。
- [ ] `docs/TODO.md`：把"keep memory native to Alpha Agent: working,
  episodic, semantic, procedural, salience..."这段反向写：
  "Memory-as-records is being replaced by an event-sourced cognition runtime;
  see `docs/todo/cognition-runtime/`."
- [ ] `git mv docs/doing/memory-system-optimization-phases.md
  docs/develop_record/memory-system-optimization-phases-completed.md`。
- [ ] `docs/doing/` 删空，或加一个一行的占位说明。

## 3. 接口契约（草案）

清理阶段不引入新接口，但要保证收缩后 `AlphaAgent.respond` 的签名与返回类型
**保持不变**——这样 gateway / CLI 调用方完全感知不到 cleanup：

```python
class AlphaAgent:
    def respond(
        self,
        user_message: str,
        session_id: str,
        source_metadata: dict[str, Any] | None = None,
    ) -> AgentTurnResult: ...
```

`AgentTurnResult.debug` 暂时只放：

```python
{
    "session_id": "...",
    "llm_round_count": ...,
    "tool_call_count": ...,
    "note": "long-term cognition disabled; see docs/todo/cognition-runtime/",
}
```

## 4. 文件清单

### 4.1 删除

```text
src/alpha_agent/memory/consolidation.py
src/alpha_agent/memory/controller.py
src/alpha_agent/memory/episodic.py
src/alpha_agent/memory/extractor.py
src/alpha_agent/memory/persistence.py
src/alpha_agent/memory/procedural.py
src/alpha_agent/memory/retrieval.py
src/alpha_agent/memory/review.py
src/alpha_agent/memory/salience.py
src/alpha_agent/memory/semantic.py
src/alpha_agent/runtime/context_compression.py
tests/test_memory_store.py
tests/test_memory_extraction_eval.py
tests/test_memory_review.py
tests/test_retrieval.py
tests/test_consolidation.py
tests/memory_eval.py
```

### 4.2 移动 / 重命名

```text
src/alpha_agent/memory/        →  src/alpha_agent/state/
src/alpha_agent/memory/schema.sql →  src/alpha_agent/state/schema.sql
docs/doing/memory-system-optimization-phases.md
  →  docs/develop_record/memory-system-optimization-phases-completed.md
```

### 4.3 收缩修改

```text
src/alpha_agent/state/models.py        （只保留 ConversationMessage、RuntimeTrace）
src/alpha_agent/state/store.py         （只保留对应两表的 CRUD）
src/alpha_agent/state/schema.sql       （只保留 4 张表）
src/alpha_agent/runtime/agent.py       （目标 < 400 行）
src/alpha_agent/runtime/prompt_builder.py  （目标 < 150 行）
src/alpha_agent/runtime/session_context.py （目标 < 100 行）
src/alpha_agent/cli.py                  （删除 memory 子命令）
tests/test_prompt_builder.py
tests/test_agent_loop.py
tests/test_cli_agent_loop.py
README.md
AGENTS.md
docs/TODO.md
```

## 5. 验收标准

- [ ] `grep -rn "alpha_agent.memory" src/ tests/` 输出为空。
- [ ] `grep -rn "MemoryController\|MemoryRetriever\|MemoryExtractor\|
  ConsolidationService\|SemanticMemory\|EpisodicMemory\|ProceduralMemory\|
  MemoryCandidate" src/` 输出为空。
- [ ] 删除后的 schema.sql 只有 4 张表（`conversation_messages`、
  `runtime_traces`、`gateway_session_mappings`、`gateway_dedup`）。
- [ ] `uv run pytest -q` 全绿（带 xfail 的不算红）。
- [ ] `alpha chat` 能跑一轮对话，但同一 session 重启后**没有任何长期记忆**。
- [ ] `alpha memory ...` 子命令全部不存在；`alpha --help` 输出里没有
  `memory` 字样。
- [ ] `AGENTS.md` 项目导航里 `memory/` 改成 `state/`，README 中 memory 段被
  替换为"under construction"提示。
- [ ] `docs/doing/` 不再含 memory-system-optimization-phases.md。

## 6. 风险与备注

- **决策表必须先写**。在本文档下方追加一张表（模块名 / 表名 / 处置 / 备注），
  审过再开 PR。事后看决策依据，比看 diff 高效得多。
- **xfail 测试不要无限期保留**。Phase 02 完成时这些 xfail 应该全部转回绿色或
  被显式删除（重写为新的认知行为测试）。
- **不要复用旧文件名继续承载新含义**。例如 `consolidation.py`、
  `retrieval.py` 这种名字别在新 `cognition/` 里复用，避免 git blame
  / search 工具把新旧两份代码混在一起看。
- **删除 develop_record 之前要备份**。`memory-system-optimization-phases.md`
  里的设计取舍仍有参考价值，归档要保住。
- **数据迁移**：项目规则 `AGENTS.md` 说"DO NOT CONSIDER COMPATIBILITY WHEN
  CODING, INCLUDING EXISTED DATA IN DATABASE"——所以现有用户 SQLite 文件里的
  memory 数据**直接弃用**。文档应明确告知用户清库或重建。

## 7. 后续衔接

Phase 01 在这一阶段交付的"干净地基"上：

- 新建 `src/alpha_agent/cognition/` 目录。
- 加 `cognitive_events` 表与 EventLog 实现。
- 引入核心类型（Subject / Belief / CognitiveEvent / Situation / Perception /
  Judgment / Decision / Reflection / Procedure / ContextWindow / ValueProfile /
  ValueLens）。

注意：Phase 01 不会再回到 `state/` 里加东西，但会在 schema.sql 里追加事件日
志相关表。`state/` 与 `cognition/` 是两个独立子系统：state 负责会话流水，
cognition 负责长期主体与事件日志。
