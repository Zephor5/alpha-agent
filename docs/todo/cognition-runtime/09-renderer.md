# Phase 09 — Renderer 解耦

**Status:** pending
**Depends on:** Phase 02, Phase 03, Phase 04
**Scope:** M
**Design ref:** `cognition_from_scratch.md` §10；README 不变量 1

## 0. 目标

把"如何把认知状态变成 LLM 输入"这件事从 Effector 内部剥离出来，变成可插拔
的 Renderer 接口。同时引入两个 debug-friendly 渲染器（graph snapshot、diff）
便于审计。

完成后：

- Effector 不直接拼 chat messages；它接收一个 `CognitionView` 并交给注入的
  Renderer。
- LLM 厂商切换 / 多模态扩展 / 调试可视化都不动 cognition core。
- Phase 02 标记 deprecated 的 `runtime/prompt_builder.py` 真正删除。

Renderer 是 view→payload 的纯函数，不在 LoopCoordinator 管辖范围——它在
Reactive 的 Effector 内部跑，跟随 Reactive 持锁。但它必须消费 Counterpart
信息：

1. `ContextWindow.counterpart`（来自 Phase 04）告诉 Renderer 当前会话对方
   是谁。
2. `CounterpartProjection.get(counterpart_id)` 取出对方的
   `communication_style` / `role` / `trust_level`，作为
   `RenderBudget.style_hints`。
3. TextChatRenderer 据此调整：
   - **style**：communication_style hints 影响 system prompt 的风格段。
   - **trust**：trust_level 低时，把对方近期 belief 标"unverified"提示
     LLM。
   - **role**：operator 角色获得更精简、协议化的 prompt；user 获得更自然
     的交互式 prompt。
4. EvidenceRenderer 在回溯 belief 时，优先标出 `about=[counterpart]` 的关
   联，方便 audit"为什么 agent 这样回应 user_a"。

## 1. 范围

### 1.1 In scope

- `CognitionView`：纯数据切片，渲染器消费的统一入参。
- `Renderer` Protocol + `RenderBudget`。
- 4 个内置 renderer：
  - `TextChatRenderer`：OpenAI chat completions 格式。
  - `GraphSnapshotRenderer`：当前 active beliefs 关系图（用于 inspection）。
  - `DiffRenderer`：两次 tick 之间 belief / strategy / lens 的变化。
  - `EvidenceRenderer`：一条 belief 的完整证据链（从 event log 回溯）。
- Effector 切换到 Renderer 调用。
- 删除 `runtime/prompt_builder.py`。
- CLI：`alpha debug prompt` 接 TextChatRenderer 输出；`alpha cognition graph
  / diff / evidence <id>` 走对应 renderer。

### 1.2 Out of scope

- 真正接 Anthropic tool-use blocks（要新 renderer，留独立 phase）。
- LLM-based 压缩 renderer（Phase 06 备注里提过，远期实现）。
- 端到端流式 prompt（要 Effector 改造，远期）。

## 2. 任务清单

### 2.1 View 与 Renderer 基础

- [ ] `cognition/render/__init__.py`。
- [ ] `cognition/render/view.py`：`CognitionView` 数据类。
- [ ] `cognition/render/base.py`：`Renderer` Protocol、`RenderBudget`、
  `RenderResult`。
- [ ] `cognition/render/build_view.py`：从 Subject / Situation /
  ContextWindow / Beliefs / Judgments / Procedures / Strategies 组装一个
  view。Effector 装配时调。

### 2.2 内置 renderer

- [ ] `cognition/render/text_chat.py`：`TextChatRenderer`。
  - 输入 view + budget → chat messages list。
  - 渲染顺序：system prompt（含对方风格段）→ strategy reminders
    (system-reminder 包裹) → counterpart_digest（若有）→ recalled beliefs →
    ContextWindow.background → foreground → user query。
  - 每段都 budget-aware（沿用 Phase 00 阶段简版的 per-section budget 思路）。
  - 当 view.counterpart 非空：
    - system prompt 末尾追加 communication_style 派生的风格指令；
    - trust_level < 0.5 时给 recalled belief 段加"User-reported, not verified
      by agent"前缀；
    - role 不同时切不同 system prompt 模板（user / operator / peer_agent / 
      system / anonymous 五份）。
  - 当 view.counterpart 为 None（cognition thread / drive loop self-signal）：
    走 default system prompt，无对方风格段。
- [ ] `cognition/render/graph_snapshot.py`：输出 dot / mermaid 文本。
- [ ] `cognition/render/diff.py`：输入 two `tick_id` → 输出变更列表。
- [ ] `cognition/render/evidence.py`：输入 belief_id → 从 event log 回溯
  formed / strengthened / superseded 链 + 引用 perception ids → 输出文本。

### 2.3 Effector 接入

- [ ] `cognition/stages/effector.py`：
  - 不再自己拼 messages，调 `Renderer.render(view, budget)`。
  - 接收注入的 renderer（默认 TextChatRenderer）。
  - 把 render 结果中的 messages 喂给 LLM provider。

### 2.4 CLI

- [ ] `alpha debug prompt`：默认 TextChatRenderer，加 `--renderer
  <name>` 切换。
- [ ] `alpha cognition graph [--subject] [--format dot|mermaid]`。
- [ ] `alpha cognition diff <tick_id_a> <tick_id_b>`。
- [ ] `alpha cognition evidence <belief_id>`。

### 2.5 删除 prompt_builder

- [ ] `git rm src/alpha_agent/runtime/prompt_builder.py`。
- [ ] 移除所有 import。
- [ ] `tests/test_prompt_builder.py`：迁移有意义的 assertion 到
  `tests/cognition/test_text_chat_renderer.py`，文件删除。

### 2.6 测试

- [ ] `tests/cognition/test_view_builder.py`：从 mock projections → view 字段
  正确。
- [ ] `tests/cognition/test_text_chat_renderer.py`：budget / 章节顺序 / 空字段
  处理。
- [ ] `tests/cognition/test_text_chat_renderer_counterpart.py`：
  - 同一 view 配不同 counterpart.role → system prompt 模板不同。
  - communication_style hint → 风格段出现对应文本。
  - trust_level=0.2 → recalled belief 段加 "User-reported" 前缀。
  - counterpart=None → 走 default 模板，无风格段。
- [ ] `tests/cognition/test_graph_snapshot.py`：dot 与 mermaid 输出可解析。
- [ ] `tests/cognition/test_diff_renderer.py`：构造两次 tick → diff 输出正确。
- [ ] `tests/cognition/test_evidence_renderer.py`：belief 形成 → superseded
  链能完整渲染。
- [ ] `tests/cognition/test_cli_render.py`。

### 2.7 文档

- [ ] AGENTS.md。
- [ ] `cognition/render/README.md` 列 renderer 与适用场景。

## 3. 接口契约

### 3.1 CognitionView

```python
@dataclass(frozen=True)
class CognitionView:
    subject: Subject                       # 永远是 agent:self
    counterpart: Counterpart | None        # 当前 thread 的对方；cognition thread 为 None
    situation: Situation
    window: ContextWindow
    recalled_beliefs: list[Belief]
    counterpart_digest: Belief | None      # Phase 06 生成的关于 counterpart 的 digest
    active_judgments: list[Judgment]
    matched_procedures: list[Procedure]
    active_strategies: list[StrategyOverride]
    recent_reflections: list[Reflection]
    assembled_at: Instant
```

### 3.2 Renderer

```python
@dataclass(frozen=True)
class RenderBudget:
    max_tokens: int
    per_section_tokens: dict[str, int] = field(default_factory=dict)
    style_hints: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class RenderResult:
    payload: Any                # messages / dot / diff text / evidence text
    used_tokens: int
    dropped_sections: list[str]
    notes: list[str]

class Renderer(Protocol):
    name: ClassVar[str]
    def render(self, view: CognitionView, budget: RenderBudget) -> RenderResult: ...
```

### 3.3 build_view 简化签名

```python
def build_view(
    *,
    thread_id: ThreadId,
    situation: Situation,
    projections: ProjectionRegistry,
    clock: Callable[[], Instant] = utc_now,
) -> CognitionView: ...
```

`subject` 由 `projections.subject.current()` 内部取出，`counterpart` 由
window.counterpart + `projections.counterpart.get(...)` 取出——caller 不
重复传。

## 4. 文件清单

### 4.1 新增

```text
src/alpha_agent/cognition/render/__init__.py
src/alpha_agent/cognition/render/view.py
src/alpha_agent/cognition/render/base.py
src/alpha_agent/cognition/render/build_view.py
src/alpha_agent/cognition/render/text_chat.py
src/alpha_agent/cognition/render/graph_snapshot.py
src/alpha_agent/cognition/render/diff.py
src/alpha_agent/cognition/render/evidence.py
src/alpha_agent/cognition/render/README.md
tests/cognition/test_view_builder.py
tests/cognition/test_text_chat_renderer.py
tests/cognition/test_graph_snapshot.py
tests/cognition/test_diff_renderer.py
tests/cognition/test_evidence_renderer.py
tests/cognition/test_cli_render.py
```

### 4.2 修改

```text
src/alpha_agent/cognition/stages/effector.py     通过 Renderer 拼 prompt
src/alpha_agent/cognition/controller.py          注入 Renderer
src/alpha_agent/cli.py                           debug prompt 支持 --renderer + 新子命令
AGENTS.md
```

### 4.3 删除

```text
src/alpha_agent/runtime/prompt_builder.py
tests/test_prompt_builder.py
```

## 5. 验收标准

- [ ] `uv run pytest tests/cognition/test_*_renderer.py
  tests/cognition/test_view_builder.py -q` 全绿。
- [ ] `grep -rn "prompt_builder" src/` 输出为空。
- [ ] `alpha debug prompt --renderer text_chat` 与现行 prompt 形态等价。
- [ ] `alpha cognition graph --format mermaid` 输出 mermaid，能在 typical
  渲染器（mermaid live editor）画出。
- [ ] `alpha cognition diff <tick_a> <tick_b>` 至少包含 belief / lens /
  strategy 三类变化。
- [ ] `alpha cognition evidence <belief_id>` 能回溯到具体 perception id 与
  原始消息。

## 6. 风险与备注

- **budget 的尺度**。chat completions 是 token-budget；mermaid 是 node count
  budget。基类不强制一种语义——`RenderBudget.max_tokens` 是建议值，每个
  renderer 自己决定怎么用。
- **build_view 性能**。每 tick 都跑会 query 多个 projection。最简策略：
  Reactive 装配 window 之后保存 view 引用到 controller，stage 间复用。
- **多 renderer 同时使用**。同一 view 用 text_chat 和 graph_snapshot 各跑一
  次完全合法——view 不可变，renderer 无状态。inspection 工具会用这点。
- **删除 prompt_builder 时小心 import**。仓库里可能有 CLI / gateway 代码引
  prompt_builder——Phase 02 已经 deprecate 了大部分，但 Phase 09 前要全局
  grep 一次。

## 7. 后续衔接

- Phase 10 Drive Loop 不直接用 renderer——它通过 emit perceived 进 reactive
  流。
- Phase 11 L3 可用 EvidenceRenderer 作为 SelfModel 解释证据。
- 远期：加 AnthropicToolUseRenderer / StreamThoughtRenderer / 多模态 renderer
  都按本阶段接口加一个文件即可。
