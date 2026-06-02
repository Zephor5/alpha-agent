# memory_recall 检索改造计划

## 目标

让 `memory_recall` 能稳定回答普通自然语言问题，同时保持记忆检索是显式工具调用，不恢复每轮自动注入动态记忆的 prompt 形态。

本任务实现这些内容：

- 扩展 `memory_recall` 的结构化检索意图 schema。
- 为中英文、代码标识符、版本号、路径、API 名称等混合文本建立确定性搜索分词。
- 在 `BeliefProjection` 中维护 SQLite FTS5 双索引。
- 通过多路候选召回合并 belief。
- 用可解释、确定性的打分排序候选。

结果仍然只查询 belief。是否调用 `memory_recall` 由模型决定，runtime 不增加动态记忆注入规则。

## 当前问题

当前 `memory_recall` 把整个 `query` 当成一个 entity-like probe，然后由 `BeliefProjection.recall(...)` 通过 `belief_entity_index` 或 `normalized_content LIKE "%normalized(query)%"` 匹配。

这对短关键词稳定，但对普通问题很脆弱。例如：

```text
query: what examples do I prefer?
belief: User prefers Python examples.
```

这两段文本没有完整子串关系，当前策略可能召回失败。

中文和技术文本还有额外问题：默认词法分词不一定能保留 `memory_recall`、`v3.0.1`、`GPT-5.4-mini`、`C++`、`OpenAI API`、源码路径等领域 token。

## 设计边界

- `memory_recall` 是显式工具，runtime 不判断何时需要 recall。
- stable counterpart profile 继续作为 session 稳定上下文，和动态 recall 分离。
- tool 的模型可见输出只包含 `content`、`type`、`scope`。
- 候选生成可以多路召回，但检索能力必须由 projection/tool API 承载，不在 runtime 写临时 SQL。
- 排序必须确定、可测试、可解释。
- 分词不能依赖“列举所有可能技术 token 格式”的正则。

## 工具契约

保留现有字段：

```json
{
  "query": "string",
  "scope": "counterpart | global | both",
  "types": ["factual", "preference"],
  "max_results": 4
}
```

增加结构化检索意图：

```json
{
  "keywords": ["examples", "Python"],
  "entities": ["Python", "coding examples"],
  "intent": "preference lookup"
}
```

字段规则：

- `query`：必填，检索文本，最长 300 字符。
- `keywords`：可选，词法检索词，最多 12 项，每项最长 80 字符。
- `entities`：可选，entity/object probe，最多 8 项，每项最长 120 字符。
- `intent`：可选，短检索目的描述，只用于 trace/debug 上下文，最长 120 字符。
- `scope`：默认 `both`。
- `types`：可选，直接映射 `CognitiveType`，最多 8 项。
- `max_results`：默认 4，范围 1-8。

只传 `query` 的旧调用必须继续有效。非法字段、过长字符串、错误数组类型、未知 enum 值，都通过现有 recoverable tool failure 路径失败。

## 依赖和配置

本任务引入 `jieba` 作为中文 run 的分词工具。

实现要求：

- 更新 `pyproject.toml` 和 `uv.lock`。
- 所有 `jieba` 调用集中在项目自有 wrapper 内，业务代码不直接 import `jieba`。
- wrapper 支持可选用户词典；用户词典不存在时正常运行。
- 用户词典路径使用项目相对路径或配置路径，不能写入本机绝对路径。

建议默认用户词典位置：

```text
docs/todo/memory_recall_userdict.example.txt
```

如果实际实现需要运行时词典，应使用配置项或项目相对路径；todo 文档只要求 wrapper 具备加载能力。

## 搜索分词

新增一个确定性 mixed tokenizer。目标不是完美中文分词，而是让 belief search text 对中英文混合、代码标识符、版本号、路径、API 名称有稳定候选召回。

建议模块：

```text
src/alpha_agent/cognition/search_tokenizer.py
```

### script-run 切分

先把文本切成三类 run：

- CJK run：连续 CJK ideograph 文本。
- technical run：连续非 CJK 有意义文本，包括 ASCII 字母、数字、下划线，以及版本号/路径/代码常用结构符。
- separator run：无检索意义的空白或标点。

technical run 可以包含内部空格，但只在空格两侧都是 ASCII 字母、数字或技术结构字符时保留。CJK 和 technical 之间的空格、中文标点、普通分隔标点作为 separator。

示例：

```text
用户希望在 v3.0.1 里支持 GPT-5.4-mini、memory_recall 和 src/alpha_agent/runtime/agent.py
```

切分为：

```text
[CJK] 用户希望在
[TECH] v3.0.1
[CJK] 里支持
[TECH] GPT-5.4-mini
[TECH] memory_recall
[CJK] 和
[TECH] src/alpha_agent/runtime/agent.py
```

### CJK run

CJK run 交给 `jieba` wrapper。

要求：

- 分词结果去空白。
- ASCII 部分 casefold。
- 过滤空 token 和纯标点 token。
- 输出顺序稳定。

### technical run

technical run 默认受保护，不用正则枚举所有格式。每个 technical run 同时输出原始 token 和派生 token。

处理规则：

1. 原始 token：trim 后 casefold；只要包含有意义的字母或数字就保留。
2. 派生 token：按 `_`、`-`、`.`、`/`、`:`、`@`、`#`、`+` 拆分。
3. 对常见 exact lookup 形式保留结构符，例如 `memory_recall`、`v3.0.1`、`c++`、`c#`、`agent.py`。
4. 派生 token 可以继续拆下划线组合，例如 `alpha_agent` 额外产生 `alpha`、`agent`。
5. 去重但保留首次出现顺序。

示例：

| 输入 | 原始 token | 派生 token |
| --- | --- | --- |
| `v3.0.1` | `v3.0.1` | `v3`, `3`, `0`, `1` |
| `GPT-5.4-mini` | `gpt-5.4-mini` | `gpt`, `5`, `4`, `mini` |
| `memory_recall` | `memory_recall` | `memory`, `recall` |
| `OpenAI API` | `openai api` | `openai`, `api` |
| `src/alpha_agent/runtime/agent.py` | `src/alpha_agent/runtime/agent.py` | `src`, `alpha_agent`, `alpha`, `agent`, `runtime`, `agent.py`, `py` |
| `C++` | `c++` | `c` |
| `C#` | `c#` | `c` |

原始 token 不能被派生 token 替代；两类都要进入搜索文本。

### search_terms

为每条 belief 构造 `search_terms`：

- mixed-tokenized `content`
- mixed-tokenized `object`
- `normalized_content`
- `belief_entity_index` 中已有 entity id

`search_terms` 用于 term FTS。原始 `content` 仍进入 trigram FTS，作为 tokenizer 未覆盖格式的兜底。

## FTS 索引

FTS 表由 `BeliefProjection` 管理。当前 `BeliefProjection` 有自己的 `_SCHEMA`，同时 `state/schema.sql` 是数据库 baseline；实现时必须两边一致更新：

- `src/alpha_agent/cognition/projections/belief.py` 的 `_SCHEMA`
- `src/alpha_agent/state/schema.sql`

### term FTS

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS belief_search_terms_fts
USING fts5(
    belief_id UNINDEXED,
    search_terms,
    object,
    tokenize = "unicode61 remove_diacritics 1 tokenchars '_-#./:+'"
);
```

用途：

- 英文词。
- `jieba` 产出的中文词。
- technical 原始 token。
- technical 派生 token。
- object/entity probe。

### trigram FTS

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS belief_search_trigram_fts
USING fts5(
    belief_id UNINDEXED,
    content,
    object,
    normalized_content,
    tokenize = "trigram"
);
```

用途：

- 中文子串兜底。
- mixed tokenizer 没覆盖的技术格式。
- 版本、路径、代码标识符的子串召回。

### 生命周期

FTS5 virtual table 不提供 `belief_id` 主键约束。更新时必须先删后插，避免同一 belief 出现多行。

要求：

- `_upsert_belief(...)` 成功写入 `belief_view` 后，同一个 transaction 内先 `DELETE WHERE belief_id = ?`，再插入两张 FTS 表。
- `_supersede(...)`、`_mark_status(...)` 将 belief 变为非 active 时，同一个 transaction 内删除旧 belief 的 FTS 行。
- 如果 supersede 事件携带新 belief，新 belief 的 active FTS 行由 `_upsert_belief(...)` 写入。
- `reset()` 必须清空两张 FTS 表。
- rebuild 时通过 replay 事件重建 FTS，不做独立旁路。
- 查询 FTS 后必须 join 回 active `belief_view`，不能只信 FTS 行。
- 如果运行环境没有 FTS5，相关测试必须明确失败，不做静默 fallback。

## FTS 查询构造

所有来自用户或 LLM 的 `query`、`keywords`、`entities` 都必须经过 FTS query builder，不能直接拼到 `MATCH` 字符串里。

建议新增内部函数：

```text
build_term_fts_query(tokens: Sequence[str]) -> str
build_trigram_fts_query(probes: Sequence[str]) -> str
```

要求：

- 使用 SQL 参数绑定传入 MATCH 表达式。
- 对 FTS5 特殊字符做转义或双引号 phrase 包裹。
- 空 token 不生成 MATCH 条件。
- term FTS 查询使用 mixed tokenizer 输出的 token。
- trigram FTS 只使用长度至少 3 的 probe；短于 3 的 probe 走 entity/object/substring 路径。
- `C++`、`C#`、`v3.0.1`、路径、引号、冒号不能导致 MATCH 语法错误。

## 候选生成

先把所有输入规整为 probes：

- `query_tokens`：mixed-tokenized `query`
- `keyword_tokens`：mixed-tokenized `keywords`
- `entity_tokens`：mixed-tokenized `entities`
- `raw_probes`：原始 `query`、`keywords`、`entities` 的 normalized 形式

候选来源：

1. `belief_entity_index` exact match：使用 `entity_tokens` 和 `raw_probes`。
2. object exact/partial match：使用 `entity_tokens`、`keyword_tokens`、`raw_probes`。
3. term FTS：查询 `search_terms` 和 `object`。
4. trigram FTS：查询 raw content、object、normalized content。
5. normalized-content substring fallback：只对非空且有意义的 probes 使用。

Scope/type 是所有候选路径的过滤条件，不是独立候选来源。不要因为 belief 在当前 scope 内就直接进入结果；它必须至少命中一个 query、keyword、entity、object、FTS 或 substring 信号。

候选合并：

- 按 belief id 合并重复候选。
- 合并时保留所有命中原因。
- 排除 `object` 以 counterpart digest/profile prefix 开头的 belief。
- `scope=counterpart` 且当前 session 没有 counterpart 时返回空结果。

Scope 语义：

- `counterpart`：只查当前 session counterpart 相关 belief。
- `global`：只查没有 `about` refs 的 belief。
- `both`：当前 session counterpart 相关 belief + global belief。

## 打分

每个候选生成一个内部对象：

```text
ScoredBeliefCandidate(
    belief: Belief,
    scope: "counterpart" | "global",
    score: float,
    reasons: tuple[str, ...],
)
```

建议模块边界：

- projection 负责返回候选及 FTS rank/命中来源。
- tool/scorer 负责把候选变成 `ScoredBeliefCandidate` 并裁剪模型输出。

打分组件：

```text
score =
  scope_score
  + type_score
  + entity_score
  + object_score
  + term_fts_score
  + trigram_fts_score
  + substring_score
  + confidence_score
  + recency_tiebreak
```

初始权重：

- `scope_score`：counterpart match +4，global in `both` +1。
- `type_score`：请求了 `types` 且命中 +2。
- `entity_score`：entity exact match +4。
- `object_score`：object exact match +3，partial match +1。
- `term_fts_score`：由 term FTS rank 转成 0-4。
- `trigram_fts_score`：由 trigram FTS rank 转成 0-2。
- `substring_score`：normalized content/object 包含 query/keyword/entity fragment +1。
- `confidence_score`：belief confidence 映射到 0-1。
- `recency_tiebreak`：0-0.25，只作弱 tie-break。

排序：

1. 分数高优先。
2. counterpart-scoped 优先于 global。
3. confidence 高优先。
4. `held_since` 新优先。
5. belief id 字典序稳定排序。

## 可解释性

普通 tool 输出保持紧凑：

```json
{
  "results": [
    {
      "content": "User prefers Python examples.",
      "type": "preference",
      "scope": "counterpart"
    }
  ]
}
```

测试和 debug helper 可以检查 `ScoredBeliefCandidate`：

```json
{
  "belief_id": "belief:...",
  "score": 8.7,
  "reasons": ["scope:counterpart", "entity:python", "type:preference"]
}
```

解释信息不进入普通模型可见 tool output。

## 实施任务

### Task 1：更新依赖和工具契约测试

验收标准：

- `pyproject.toml` 和 `uv.lock` 包含 `jieba`。
- `memory_recall` schema 包含 `keywords`、`entities`、`intent`。
- 只传 `query` 的调用仍然通过。
- 无效数组、过长字符串、未知字段和非法 enum 走现有 recoverable tool failure。

相关文件：

- `pyproject.toml`
- `uv.lock`
- `src/alpha_agent/tools/memory_recall.py`
- `tests/cognition/test_memory_recall_tool.py`

### Task 2：实现 mixed tokenizer

验收标准：

- CJK run 通过项目自有 `jieba` wrapper 分词。
- technical run 保留原始 token 并生成派生 token。
- tokenization 确定、去重、保留首次出现顺序。
- 测试覆盖：
  - `v3.0.1`
  - `GPT-5.4-mini`
  - `memory_recall`
  - `OpenAI API`
  - `src/alpha_agent/runtime/agent.py`
  - `C++` 和 `C#`
  - `用户喜欢Python3.12示例`
  - `小六是assistant名字`

相关文件：

- `src/alpha_agent/cognition/search_tokenizer.py`
- `tests/cognition/test_search_tokenizer.py`

### Task 3：实现 FTS schema 和生命周期

验收标准：

- `BeliefProjection._SCHEMA` 和 `state/schema.sql` 都包含两张 FTS 表。
- `_upsert_belief(...)` 同步 active FTS 行。
- `_supersede(...)`、`_mark_status(...)` 删除非 active belief 的 FTS 行。
- `reset()` 清空 FTS 表。
- rebuild 后 FTS 结果和 active `belief_view` 一致。
- 测试覆盖 create、supersede、retract/archive、reset、rebuild、无 stale row。

相关文件：

- `src/alpha_agent/cognition/projections/belief.py`
- `src/alpha_agent/state/schema.sql`
- `tests/cognition/test_belief_projection_apply.py`

### Task 4：实现 FTS query builder 和候选召回 API

验收标准：

- `MATCH` 表达式不直接拼接用户/LLM 输入。
- `C++`、`C#`、`v3.0.1`、路径、引号、冒号不会触发 FTS 语法错误。
- 短于 3 字符的 trigram probe 不走 trigram MATCH。
- projection 暴露候选召回 API，tool 不直接查 state tables。
- 候选必须有实际命中信号，不能只是 scope 内 active belief。

相关文件：

- `src/alpha_agent/cognition/projections/belief.py`
- `tests/cognition/test_recall_entity_overlap.py`
- `tests/cognition/test_memory_recall_tool.py`

### Task 5：实现多路候选合并和可解释打分

验收标准：

- entity、object、term FTS、trigram FTS、substring 候选能合并。
- 重复 belief 按 id 合并，保留所有 reason。
- scope/type 过滤对所有路径一致。
- digest/profile belief 被排除。
- counterpart result 在同等相关性下优先 global。
- exact entity/object match 优先 loose FTS。
- 测试断言候选顺序和 reason，不只断言数量。

相关文件：

- `src/alpha_agent/tools/memory_recall.py`
- `tests/cognition/test_memory_recall_tool.py`

### Task 6：runtime 和文档回归

验收标准：

- runtime 仍通过 `ToolExecutionContext.extensions` 传 recall context。
- 不增加自动动态记忆注入。
- system prompt guidance 保持简短，不写规则化 recall gate。
- tool traces 和持久化 tool messages 形状不变。
- README 说明 stable profile、dynamic recall、memory write proposal 的区别。

相关文件：

- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cognition/render/text_chat.py`
- `README.md`
- `tests/test_agent_loop.py`

## 验证

先跑定向测试：

```bash
uv run pytest tests/cognition/test_search_tokenizer.py -q
uv run pytest tests/cognition/test_memory_recall_tool.py -q
uv run pytest tests/cognition/test_belief_projection_apply.py -q
uv run pytest tests/cognition/test_recall_entity_overlap.py -q
uv run pytest tests/test_agent_loop.py -q
```

最终验证：

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

## 风险

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| `jieba` 没切中领域词 | 中 | 保留 technical 原始 token、entity/object 路径和 trigram FTS |
| technical tokenizer 派生出噪声 token | 中 | 保持 `max_results` 小，并用可解释打分测试排序 |
| FTS term rank 放大宽泛词 | 中 | FTS 分数设上限，让 scope/entity/object 信号占主导 |
| 中文 query 和 belief 表述差异大 | 中 | 同时使用 jieba term 和 trigram 候选 |
| 召回无关记忆 | 高 | 所有候选必须有命中信号，并强制 scope/type 过滤 |

## 不做

- Procedure search。
- runtime recall gate。
- 自动 prompt 注入动态 belief。
