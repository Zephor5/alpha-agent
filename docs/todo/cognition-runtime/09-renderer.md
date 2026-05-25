# Phase 09 — Renderer 解耦

**Status:** completed
**Depends on:** Phase 02, Phase 03, Phase 04, Phase 05
**Scope:** M
**Design ref:** `cognition_from_scratch.md` §10；README 不变量 1

## 0. 目标

把“如何把认知状态变成 LLM 输入”从 Effector 内部剥离为 renderer 接口。
Renderer 是 view -> payload 的纯渲染边界：Reactive tick 仍然由 Controller
和 Effector 负责，renderer 不拥有 loop、tool execution、projection mutation。

完成后：

- Effector 不再直接拼 chat messages；默认使用 `TextChatRenderer`。
- Runtime tool loop 行为保留：renderer 只生成初始 messages，assistant tool-call
  和 tool-result messages 仍由 runtime/effector 的工具循环追加。
- Phase 02 的 `runtime/prompt_builder.py` 已删除，`src/` 中不再 import。
- CLI debug/inspection 走 renderer：
  - `alpha debug prompt --renderer text_chat`
  - `alpha cognition graph --format mermaid|dot`
  - `alpha cognition diff <tick_id_a> <tick_id_b>`
  - `alpha cognition evidence <belief_id>`

## 1. 实际完成范围

### 1.1 In scope completed

- `src/alpha_agent/cognition/render/`
  - `CognitionView`
  - `Renderer` Protocol
  - `RenderBudget`
  - `RenderResult`
  - `build_view`
  - `TextChatRenderer`
  - `GraphSnapshotRenderer`
  - `DiffRenderer`
  - `EvidenceRenderer`
- `TextChatRenderer` 输出 OpenAI chat-completions messages。
- Counterpart-aware text rendering：
  - role 选择 system prompt 模板；
  - communication style 进入 system prompt；
  - low trust counterpart 使 recalled beliefs 标为 user-reported/unverified；
  - no counterpart 使用默认模板。
- `GraphSnapshotRenderer` 输出 deterministic Mermaid/DOT belief graph。
- `DiffRenderer` 基于当前 event log 做 tick-to-tick event-kind diff，覆盖已存在
  的 belief/value-lens/strategy event kinds。
- `EvidenceRenderer` 基于 event log 回溯 belief lifecycle 事件，并输出 inputs /
  outputs，能在事件携带 perception inputs 时回溯到 perception id。
- `alpha debug prompt --renderer text_chat` 通过 renderer 输出 prompt preview。
- `alpha cognition graph/diff/evidence` 提供最小可用 deterministic text 输出。
- 旧 prompt_builder 测试迁移到 renderer 和 session-context 测试。

### 1.2 Explicitly not completed in Phase 09

以下能力依赖后续 Phase 的模型或投影，不在 Phase 09 标完成：

- LLM-based compression renderer。
- Anthropic/multimodal tool-use block renderer。
- End-to-end streaming prompt rendering。
- 语义级 strategy/lens diff。Phase 09 只做 event-kind diff；Phase 07/08/11 完成
  对应投影后再升级。
- Counterpart digest。当前没有 Phase 06 digest 投影，`counterpart_digest` 保留在
  `CognitionView` 中但默认为空。

## 2. 接口契约

### 2.1 CognitionView

`CognitionView` 是 renderer 消费的不可变数据切片。当前字段覆盖：

- `subject`
- `counterpart`
- `situation`
- `window`
- `recalled_beliefs`
- `counterpart_digest`
- `active_judgments`
- `matched_procedures`
- `active_strategies`
- `recent_reflections`
- `assembled_at`
- `current_query`
- `chat_history`
- `metadata`

`chat_history` 只用于 debug/runtime transcript preview；Reactive Effector 主要使用
`window.foreground` 和 `current_query`。

### 2.2 Renderer

```python
@dataclass(frozen=True)
class RenderBudget:
    max_tokens: int = 2048
    per_section_tokens: dict[str, int] = field(default_factory=dict)
    style_hints: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class RenderResult:
    payload: Any
    used_tokens: int
    dropped_sections: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

class Renderer(Protocol):
    name: ClassVar[str]
    def render(self, view: CognitionView, budget: RenderBudget) -> RenderResult: ...
```

## 3. 文件清单

### 3.1 新增

```text
src/alpha_agent/cognition/render/__init__.py
src/alpha_agent/cognition/render/base.py
src/alpha_agent/cognition/render/build_view.py
src/alpha_agent/cognition/render/diff.py
src/alpha_agent/cognition/render/evidence.py
src/alpha_agent/cognition/render/graph_snapshot.py
src/alpha_agent/cognition/render/text_chat.py
src/alpha_agent/cognition/render/view.py
src/alpha_agent/cognition/render/README.md
tests/cognition/render_helpers.py
tests/cognition/test_cli_render.py
tests/cognition/test_diff_renderer.py
tests/cognition/test_evidence_renderer.py
tests/cognition/test_graph_snapshot.py
tests/cognition/test_text_chat_renderer.py
tests/cognition/test_text_chat_renderer_counterpart.py
tests/cognition/test_view_builder.py
tests/test_session_context.py
```

### 3.2 修改

```text
src/alpha_agent/cli.py
src/alpha_agent/cognition/controller.py
src/alpha_agent/cognition/stages/__init__.py
src/alpha_agent/cognition/stages/effector.py
src/alpha_agent/runtime/agent.py
tests/cognition/test_context_window_projection.py
```

### 3.3 删除

```text
src/alpha_agent/runtime/prompt_builder.py
tests/test_prompt_builder.py
```

## 4. 验收标准

- [x] `uv run ruff check .`
- [x] targeted renderer/runtime tests pass during implementation.
- [x] `rg "prompt_builder" src/alpha_agent` returns no matches.
- [x] `alpha debug prompt --renderer text_chat` is covered by CLI test.
- [x] `alpha cognition graph --format mermaid` is covered by CLI test.
- [x] `alpha cognition diff <tick_a> <tick_b>` is covered by CLI test.
- [x] `alpha cognition evidence <belief_id>` is covered by CLI test.

Final full-suite verification is recorded in `docs/develop_record/phase-09-renderer.md`.

## 5. 后续衔接

- Phase 06 digest/consolidation can populate `counterpart_digest`.
- Phase 07/08/11 can replace event-kind diff with semantic lens/strategy/self-model diff.
- A future provider phase can add Anthropic/multimodal renderers without changing Effector.
