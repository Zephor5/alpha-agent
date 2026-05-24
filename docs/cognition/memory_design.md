下面给一个**从零开始、面向 LLM 的类人记忆系统设计**。目标不是神经科学上完全模拟人脑，而是吸收人脑记忆的几个关键特性：

```text
分层
联想
遗忘
巩固
情景化
语义化
可回溯
可修正
```

我会把它设计成一个工程系统，而不是抽象概念。

> Implementation note: Alpha's current runtime no longer uses a separate
> priority-pruned `working_memory` table for recent conversation context.
> Short-term session context is now derived from append-only
> `conversation_messages` plus optional `session_context_states` rows whose
> `metadata.projection` stores structured state. The projection records current
> goal, decisions, open questions, pending tasks, user constraints, relevant
> files/entities, and last action, and later compression reads that metadata
> instead of reconstructing state from clipped Markdown.
> Long-term retrieval remains query-dependent across episodic, semantic, and
> procedural memory, with scene/persona projections stored as source-backed
> semantic memories.

---

# 1. 先定义目标：LLM 记忆到底要解决什么

一个 LLM 记忆系统不应该只是：

```text
历史对话 -> embedding -> vector search
```

那太粗糙，长期运行后会出现：

```text
重复记忆
过时记忆
错误记忆
召回噪音
缺乏上下文
不知道哪些记忆可信
不知道哪些记忆重要
```

更合理的目标是：

```text
让 LLM 在需要时，能找回：
1. 用户是谁
2. 用户长期偏好是什么
3. 过去发生过什么
4. 当前任务处于什么阶段
5. 某个结论来自哪些原始证据
6. 哪些记忆已经过时或被覆盖
7. 哪些记忆应该主动影响回答
```

所以我们要设计的不是一个“知识库”，而是一个：

```text
LLM-oriented Memory Operating System
```

---

# 2. 总体结构：多层记忆，而不是单层向量库

我建议采用 6 层：

```text
Memory System
├── M0 Working Memory        工作记忆
├── M1 Episodic Memory       情景记忆 / 事件日志
├── M2 Atomic Semantic Memory 原子语义记忆
├── M3 Scene Memory          场景/主题记忆
├── M4 Persona / Profile     用户画像 / 长期偏好
├── M5 Procedural Memory     程序记忆 / 行为策略
└── Memory Controller        记忆控制器
```

可以类比人脑：

| 系统层     | 类人脑部分                | 工程含义            |
| ------- | -------------------- | --------------- |
| M0 工作记忆 | 当前注意力                | 当前上下文、当前任务状态    |
| M1 情景记忆 | episodic memory      | 原始对话、事件、工具调用    |
| M2 语义记忆 | semantic memory      | 抽取出的事实、偏好、关系    |
| M3 场景记忆 | schema / event model | 某个项目/主题的阶段性总结   |
| M4 用户画像 | 长期自我/他人模型            | 用户偏好、习惯、稳定约束    |
| M5 程序记忆 | procedural memory    | 固定工作流、回答策略、操作习惯 |

核心原则是：

```text
M1 保真
M2 抽象
M3 组织
M4 稳定
M5 行为化
```

---

# 3. 核心思想：event log + semantic memory + scene memory + profile

最小可行架构可以是：

```text
原始事件日志
    ↓
原子记忆抽取
    ↓
场景聚合
    ↓
长期画像
    ↓
召回注入 / 工具查询
```

也就是：

```text
M1 Event Log
  -> M2 Atomic Memory
    -> M3 Scene Memory
      -> M4 Persona
        -> Prompt Injection / Memory Tools
```

这和单纯知识图谱的区别是：

```text
知识图谱更适合表达“事实关系”
类人记忆系统还需要表达“经历、上下文、重要性、时间、情绪权重、习惯、遗忘和修正”
```

---

# 4. M0：工作记忆 Working Memory

工作记忆是当前对话/当前任务中最活跃的内容。

它不一定持久化，或者只短期持久化。

数据结构可以是：

```ts
interface WorkingMemory {
  sessionId: string;
  currentGoal?: string;
  currentTaskState?: string;
  activeEntities: string[];
  activeTopics: string[];
  recentMessages: Message[];
  scratchpadSummary?: string;
  openQuestions: string[];
  constraints: string[];
}
```

例如当前这轮讨论里，工作记忆可能是：

```json
{
  "currentGoal": "设计一个面向 LLM 的类人记忆系统",
  "activeTopics": ["知识图谱", "人脑记忆", "Agent Memory", "分层记忆"],
  "constraints": [
    "不是为了展示图谱",
    "面向 LLM 召回和长期记忆",
    "需要落到代码实现层面"
  ]
}
```

M0 的作用：

```text
1. 帮助当前对话保持连续性
2. 决定本轮应该检索哪些长期记忆
3. 避免每次都从长期记忆里全量召回
4. 给 memory controller 提供当前注意力焦点
```

M0 可以放在 Redis、内存、session store 里。

Alpha 当前实现里，M0 不是单独表，而是：

```text
conversation_messages
  + session_context_states.structured_projection
  + uncompressed transcript tail
```

`session_context_states` 不再保存普通“压缩聊天摘要”，而是保存可检查的
session state：current goal、decisions、open questions、pending tasks、
user constraints、relevant files/entities、last action。当前用户消息仍然是
prompt 中最后一条真实用户消息，session state 只能作为低优先级背景。

---

# 5. M1：情景记忆 Episodic Memory

M1 是最底层、最重要的一层。它保存“发生过什么”。

不要一开始就只保存总结。总结会丢信息，也可能引入幻觉。

M1 应该 append-only：

```text
用户说了什么
助手答了什么
调用了什么工具
工具返回了什么
用户是否纠正了助手
某个任务是否成功
某个结论是在哪次对话里产生的
```

数据结构：

```ts
interface MemoryEvent {
  id: string;
  tenantId?: string;
  userId: string;
  agentId?: string;
  sessionId: string;

  type:
    | "user_message"
    | "assistant_message"
    | "tool_call"
    | "tool_result"
    | "system_event"
    | "correction"
    | "task_result";

  content: string;

  metadata: {
    toolName?: string;
    taskId?: string;
    source?: string;
    language?: string;
    client?: string;
  };

  timestamp: string;

  embeddingStatus: "pending" | "ready" | "failed";

  salience?: number;
}
```

数据库表：

```sql
CREATE TABLE memory_events (
  id TEXT PRIMARY KEY,
  tenant_id TEXT,
  user_id TEXT NOT NULL,
  agent_id TEXT,
  session_id TEXT NOT NULL,
  type TEXT NOT NULL,
  content TEXT NOT NULL,
  metadata JSON,
  timestamp DATETIME NOT NULL,
  embedding_status TEXT DEFAULT 'pending',
  salience REAL DEFAULT 0
);
```

配套索引：

```sql
CREATE INDEX idx_memory_events_user_time
ON memory_events(user_id, timestamp DESC);

CREATE INDEX idx_memory_events_session
ON memory_events(session_id, timestamp ASC);
```

如果用 PostgreSQL，可以加：

```sql
CREATE INDEX idx_memory_events_fts
ON memory_events
USING gin(to_tsvector('simple', content));
```

M1 的关键要求：

```text
1. 原始性：尽量保存原始内容
2. 可回放：能重建某次对话或任务过程
3. 可溯源：上层记忆必须能指回 M1
4. 不轻易删除：除非用户要求删除或合规需要
```

---

# 6. M2：原子语义记忆 Atomic Semantic Memory

M2 是从 M1 中抽取出来的结构化记忆。

它不是完整知识图谱，而是更适合 LLM 使用的“记忆卡片”。

例如：

```text
用户经常咨询 Java / Spring Boot / Redis / MyBatis Plus 相关问题。
用户喜欢代码实现层面的细节，而不是泛泛而谈。
用户正在思考如何为 LLM 设计长期记忆系统。
用户认为如果查询不是为了展示，而是为了 LLM 记忆，知识图谱可以更松散。
```

数据结构：

```ts
interface AtomicMemory {
  id: string;
  userId: string;
  scope: "user" | "project" | "agent" | "global";

  type:
    | "fact"
    | "preference"
    | "instruction"
    | "goal"
    | "project"
    | "skill"
    | "correction"
    | "relationship";

  content: string;

  subject?: string;
  predicate?: string;
  object?: string;

  entities: string[];

  confidence: number;  // 0 - 1
  importance: number;  // 0 - 100
  stability: number;   // 0 - 1，越高越长期稳定
  recency: number;     // 动态计算
  accessCount: number;

  validFrom?: string;
  validUntil?: string;

  status: "active" | "superseded" | "deprecated" | "deleted";

  sourceEventIds: string[];

  createdAt: string;
  updatedAt: string;
}
```

注意这里我同时保留了：

```text
自然语言 content
可选 subject/predicate/object
entities
sourceEventIds
confidence / importance / stability
```

原因是：

```text
content 给 LLM 读
subject/predicate/object 给机器做弱结构查询
entities 做倒排索引
sourceEventIds 做证据链
confidence/importance/stability 做召回排序
```

表结构：

```sql
CREATE TABLE atomic_memories (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  type TEXT NOT NULL,

  content TEXT NOT NULL,

  subject TEXT,
  predicate TEXT,
  object TEXT,

  entities JSON,

  confidence REAL DEFAULT 0.8,
  importance REAL DEFAULT 50,
  stability REAL DEFAULT 0.5,
  access_count INTEGER DEFAULT 0,

  valid_from DATETIME,
  valid_until DATETIME,

  status TEXT DEFAULT 'active',

  source_event_ids JSON NOT NULL,

  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL
);
```

向量表：

```sql
CREATE TABLE memory_embeddings (
  memory_id TEXT PRIMARY KEY,
  embedding VECTOR,
  model TEXT,
  created_at DATETIME
);
```

实体索引表：

```sql
CREATE TABLE memory_entities (
  entity TEXT NOT NULL,
  memory_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  weight REAL DEFAULT 1.0,
  PRIMARY KEY(entity, memory_id)
);
```

这样就形成了轻量图谱能力，但不强迫你把所有东西都拆成三元组。

---

# 7. M2 为什么不要直接做完整知识图谱

完整知识图谱的问题是：

```text
1. 抽取成本高
2. schema 难设计
3. 关系类型容易爆炸
4. LLM 消费 triples 不如自然语言舒服
5. 用户偏好、习惯、风格很难完全三元组化
```

所以我建议 M2 是：

```text
自然语言原子记忆为主
轻量实体/关系字段为辅
```

也就是：

```json
{
  "content": "用户偏好回答包含代码实现层面的细节。",
  "type": "preference",
  "subject": "user",
  "predicate": "prefers_answer_style",
  "object": "code-level implementation details",
  "entities": ["user", "code implementation", "answer style"],
  "confidence": 0.9,
  "importance": 85,
  "sourceEventIds": ["evt_123", "evt_456"]
}
```

这比纯三元组更适合 LLM。

---

# 8. M3：场景记忆 Scene Memory

M3 是把多个 M2 聚合成一个“场景”或“主题”。

例如：

```text
场景：LLM 记忆系统设计
包含：
- 用户前面讨论过知识图谱是否适合 LLM 记忆
- 用户关注 TencentDB Agent Memory 的实现
- 用户想从零设计类人脑记忆系统
- 用户倾向工程实现和代码层面的设计
```

数据结构：

```ts
interface SceneMemory {
  id: string;
  userId: string;

  title: string;
  summary: string;

  topicTags: string[];
  entities: string[];

  status: "active" | "archived" | "deprecated";

  memoryIds: string[];
  sourceEventIds: string[];

  timeline: {
    startAt: string;
    lastUpdatedAt: string;
  };

  markdown?: string;
}
```

表结构：

```sql
CREATE TABLE scene_memories (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  topic_tags JSON,
  entities JSON,
  status TEXT DEFAULT 'active',
  memory_ids JSON,
  source_event_ids JSON,
  start_at DATETIME,
  last_updated_at DATETIME,
  markdown TEXT
);
```

M3 可以保存成 Markdown 文件，也可以进数据库。

我倾向于两份都保留：

```text
数据库：用于索引、权限、状态管理
Markdown：用于 LLM 阅读、人类审计、调试
```

例如：

```markdown
# LLM Memory System Design

## Summary

用户正在设计一个面向 LLM 的长期记忆系统，倾向于分层设计，而不是单纯知识图谱或向量库。

## Stable Preferences

- 偏好代码实现层面的解释
- 喜欢先讲架构，再落到表结构和流程
- 对知识图谱、Agent Memory、LLM 记忆有持续兴趣

## Related Atomic Memories

- mem_001: 用户认为知识图谱如果只是给 LLM 记忆用，可以更松散
- mem_002: 用户关注 TencentDB Agent Memory 的 L0/L1/L2/L3 分层
- mem_003: 用户想设计接近人脑记忆模型的 LLM 记忆系统

## Evidence

- evt_101
- evt_102
- evt_103
```

M3 的作用：

```text
1. 把碎片化 M2 组织成长期主题
2. 减少召回噪音
3. 给 LLM 提供“上下文地图”
4. 支持 drill-down 到 M2/M1
```

Alpha 当前实现将 scene 保存为 `semantic_memories.memory_type = "scene"`，
而不是新增独立表。scene 只从 reviewed、active atomic semantic memories
生成，并且这些 atomic memories 必须还能解析到 `conversation_messages`
源消息。reviewed 当前定义为 approved 或 auto_approved candidate lineage
产生的 semantic memory。scene 的 active `source_memory_ids` 只指向仍然 active
的 reviewed M2 原子记忆，metadata 中的 `source_message_ids` 指向当前 active
证据对应的 M1 transcript。source memory 被 deleted、superseded 或
conflict_review 后，新的 active scene 会去掉该 source；旧 scene 只作为
superseded audit chain 保留。`drill_down_semantic_memory()` 可以从 active scene
回溯到 active atomic memories 和原始消息。

---

# 9. M4：用户画像 Persona / Profile Memory

M4 是最高层的稳定记忆。

它不应该包含所有细节，而应该包含对用户长期有用的信息：

```text
用户长期偏好
交流风格
技术背景
工作方式
常见任务类型
回答偏好
约束偏好
```

数据结构：

```ts
interface UserPersona {
  userId: string;

  summary: string;

  preferences: PersonaItem[];
  skillsAndInterests: PersonaItem[];
  communicationStyle: PersonaItem[];
  recurringProjects: PersonaItem[];
  constraints: PersonaItem[];

  updatedAt: string;
}

interface PersonaItem {
  id: string;
  content: string;
  confidence: number;
  stability: number;
  sourceSceneIds: string[];
  sourceMemoryIds: string[];
}
```

例如：

```json
{
  "summary": "用户偏好深入、工程化、代码实现层面的技术解释，经常讨论 Java/Spring Boot、Redis、OpenAI SDK、Agent memory 和知识图谱。",
  "preferences": [
    {
      "content": "用户喜欢 step-by-step 的技术分析，而不是泛泛总结。",
      "confidence": 0.95,
      "stability": 0.9
    },
    {
      "content": "用户倾向中文讨论复杂系统设计，但也会练习英文表达。",
      "confidence": 0.85,
      "stability": 0.8
    }
  ]
}
```

表结构：

```sql
CREATE TABLE persona_items (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  category TEXT NOT NULL,
  content TEXT NOT NULL,
  confidence REAL DEFAULT 0.8,
  stability REAL DEFAULT 0.8,
  source_scene_ids JSON,
  source_memory_ids JSON,
  status TEXT DEFAULT 'active',
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL
);
```

也可以额外生成一个给 LLM 注入的 `persona.md`：

```markdown
# User Persona

## Communication Preferences

- Prefer detailed, implementation-level explanations.
- Often asks for step-by-step reasoning.
- Likes practical code/config examples.

## Technical Interests

- Java / Spring Boot
- Redis / MyBatis / MySQL
- OpenAI SDK and agent runtime
- Knowledge graph and LLM memory systems

## Response Guidelines

- Start with architecture, then drill down into implementation.
- Avoid shallow conceptual answers.
- Include trade-offs and edge cases.
```

M4 不能频繁更新。它应该由 M3 聚合后定期更新。

Alpha 当前实现将 persona 保存为
`semantic_memories.memory_type = "persona"`。persona 只使用 active、
high-confidence、high-stability 的 reviewed semantic memories 作为直接
source evidence。scene summary 可以作为生成 persona 文案的上下文，但不能作为
persona 的 `source_memory_ids` 暴露；persona drill-down 必须回到 atomic
semantic memories 和 transcript messages。pending/rejected candidates 不参与
persona 生成。persona 和 scene 在 prompt 中都属于 reference-only context，
不能覆盖当前用户消息中的显式要求。

---

# 10. M5：程序记忆 Procedural Memory

这一层经常被忽略，但对 Agent 很重要。

程序记忆不是“用户说过什么”，而是：

```text
以后遇到某类任务时，应该怎么做。
```

例如：

```text
如果用户要求“看一下某开源项目”，应该：
1. 先读 README
2. 再看目录结构
3. 再定位核心入口
4. 再看数据模型
5. 再分析 pipeline
6. 最后总结优缺点和可借鉴点
```

这不是普通事实，而是工作流。

数据结构：

```ts
interface ProceduralMemory {
  id: string;
  userId?: string;
  scope: "user" | "agent" | "global";

  trigger: string;
  procedure: string[];

  examples: string[];

  confidence: number;
  successCount: number;
  failureCount: number;

  createdAt: string;
  updatedAt: string;
}
```

表结构：

```sql
CREATE TABLE procedural_memories (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  scope TEXT NOT NULL,
  trigger TEXT NOT NULL,
  procedure JSON NOT NULL,
  examples JSON,
  confidence REAL DEFAULT 0.8,
  success_count INTEGER DEFAULT 0,
  failure_count INTEGER DEFAULT 0,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL
);
```

这层对 Agent 的长期效率很重要。因为很多“记忆”其实不是内容，而是：

```text
如何和这个用户协作
如何完成某类任务
这个用户通常期待什么输出结构
```

---

# 11. Memory Controller：记忆控制器

这是整个系统的大脑。

它负责：

```text
1. 判断什么该写入记忆
2. 判断写入哪一层
3. 判断什么时候巩固
4. 判断什么时候遗忘
5. 判断本轮召回什么
6. 判断记忆之间是否冲突
7. 判断是否需要更新 persona
```

可以拆成几个模块：

```text
MemoryController
├── CaptureManager       捕获
├── Extractor            抽取
├── Deduplicator         去重
├── Consolidator         巩固
├── Retriever            召回
├── Ranker               排序
├── ForgettingManager    遗忘
├── ConflictResolver     冲突解决
└── PromptComposer       注入 prompt
```

---

# 12. 写入流程：从对话到长期记忆

完整写入流程：

```text
用户/助手对话
    ↓
写入 M1 event log
    ↓
判断是否值得抽取
    ↓
LLM 抽取 M2 atomic memories
    ↓
去重 / 冲突检测
    ↓
写入 M2
    ↓
异步聚合 M3 scene
    ↓
周期性更新 M4 persona
    ↓
必要时更新 M5 procedure
```

伪代码：

```ts
async function onTurnCommitted(turn: ConversationTurn) {
  const events = await eventStore.appendTurn(turn);

  await embeddingQueue.enqueue(events);

  const shouldExtract = await capturePolicy.shouldExtract(turn);

  if (!shouldExtract) {
    return;
  }

  const atomicCandidates = await extractor.extractAtomicMemories({
    events,
    recentContext: await eventStore.getRecent(turn.sessionId, 20),
    existingPersona: await personaStore.getCompact(turn.userId),
  });

  const decisions = await deduplicator.resolve({
    candidates: atomicCandidates,
    existing: await retriever.findSimilarMemories(atomicCandidates),
  });

  const writtenMemories = await memoryStore.applyDecisions(decisions);

  await sceneQueue.enqueue({
    userId: turn.userId,
    memoryIds: writtenMemories.map(m => m.id),
  });

  await personaQueue.maybeEnqueue(turn.userId);
}
```

---

# 13. M2 抽取 prompt 应该怎么设计

抽取器不要把所有内容都记下来。它应该只抽取长期有价值的信息。

抽取规则：

```text
应该记：
- 稳定偏好
- 明确指令
- 长期项目
- 重要事实
- 用户纠正
- 反复出现的主题
- 已完成或未完成任务状态

不应该记：
- 临时闲聊
- 一次性状态
- 敏感信息，除非用户明确要求
- 没有把握的推测
- 低价值细节
```

LLM 输出 JSON：

```json
{
  "memories": [
    {
      "type": "preference",
      "content": "用户偏好技术解释深入到代码实现层面。",
      "entities": ["user", "technical explanation", "code implementation"],
      "importance": 85,
      "confidence": 0.9,
      "stability": 0.8,
      "source_event_ids": ["evt_1", "evt_2"]
    }
  ]
}
```

抽取 prompt 可以大概这样：

```text
You are a memory extraction module.

Extract only durable, useful memories for future LLM interactions.

Do not extract temporary details, trivial facts, or unsupported guesses.

For each memory, output:
- type
- content
- entities
- importance
- confidence
- stability
- source_event_ids

Memory types:
- fact
- preference
- instruction
- goal
- project
- correction
- skill
- relationship
```

---

# 14. 去重与冲突处理

这是长期记忆系统的难点。

新记忆进入前，要找相似旧记忆：

```text
candidate memory
    ↓
vector search
BM25 search
entity search
same type filter
    ↓
existing memory candidates
    ↓
LLM / rule-based dedup decision
```

决策类型：

```ts
type DedupDecision =
  | { action: "store"; newMemory: AtomicMemory }
  | { action: "skip"; reason: string }
  | { action: "update"; targetId: string; newContent: string }
  | { action: "merge"; targetIds: string[]; mergedContent: string }
  | { action: "supersede"; oldId: string; newMemory: AtomicMemory };
```

例子：

```text
旧记忆：
用户喜欢详细解释。

新记忆：
用户喜欢技术回答深入到代码实现层面。

决策：
update / merge

合并后：
用户偏好详细、工程化、深入到代码实现层面的技术解释。
```

冲突例子：

```text
旧记忆：
用户偏好用 Java。

新记忆：
用户现在这个项目偏好用 TypeScript。

不能直接覆盖。
应该变成：
- 用户长期常用 Java
- 在当前项目中用户偏好 TypeScript
```

所以记忆必须有 scope：

```text
global user preference
project-specific preference
session-specific constraint
```

---

# 15. 召回流程：不是 topK vector search 就完了

召回应该分多路：

```text
当前 query
    ↓
query understanding
    ↓
多路召回
      - M4 persona
      - M3 active scenes
      - M2 atomic memories
      - M1 raw events if needed
      - M5 procedural memories
    ↓
rerank
    ↓
prompt composition
```

召回器设计：

```ts
interface RecallRequest {
  userId: string;
  sessionId: string;
  query: string;
  activeTask?: string;
  activeEntities?: string[];
  maxTokens: number;
}

interface RecallResult {
  personaItems: PersonaItem[];
  sceneSummaries: SceneMemory[];
  atomicMemories: AtomicMemory[];
  rawEvents?: MemoryEvent[];
  procedures?: ProceduralMemory[];
}
```

多路召回：

```ts
async function recall(req: RecallRequest): Promise<RecallResult> {
  const queryEmbedding = await embed(req.query);

  const [
    persona,
    scenes,
    vectorHits,
    keywordHits,
    entityHits,
    procedures,
  ] = await Promise.all([
    personaStore.getCompact(req.userId),
    sceneStore.findActiveScenes(req.userId, req.query),
    memoryStore.vectorSearch(req.userId, queryEmbedding),
    memoryStore.keywordSearch(req.userId, req.query),
    memoryStore.entitySearch(req.userId, req.activeEntities ?? []),
    procedureStore.match(req.userId, req.query),
  ]);

  const atomicMemories = rerankAndMerge({
    vectorHits,
    keywordHits,
    entityHits,
    query: req.query,
  });

  return {
    personaItems: persona,
    sceneSummaries: scenes,
    atomicMemories,
    procedures,
  };
}
```

排序分数可以是：

```text
score =
  semantic_similarity * 0.35
+ keyword_score       * 0.20
+ entity_overlap      * 0.15
+ importance          * 0.15
+ recency             * 0.05
+ confidence          * 0.05
+ access_frequency    * 0.05
```

伪代码：

```ts
function rankMemory(memory, queryFeatures) {
  return (
    0.35 * memory.semanticScore +
    0.20 * memory.keywordScore +
    0.15 * memory.entityScore +
    0.15 * normalize(memory.importance) +
    0.05 * recencyScore(memory.updatedAt) +
    0.05 * memory.confidence +
    0.05 * accessScore(memory.accessCount)
  );
}
```

---

# 16. Prompt 注入策略

不要把所有记忆都塞进 prompt。

我建议分三类注入：

```text
System Prompt:
  - 稳定 persona
  - 程序记忆
  - 长期回答偏好

Developer / Context:
  - 当前场景 summary
  - 当前任务状态

User Context 前缀:
  - 本轮 query 相关的 atomic memories
```

示例：

```text
[Long-term user profile]
- User prefers implementation-level technical explanations.
- User often asks for step-by-step reasoning and practical code examples.

[Relevant scenes]
- The user is designing a long-term memory architecture for LLM agents.
- Recent discussion compared knowledge graphs, human memory, and TencentDB Agent Memory.

[Relevant memories]
- User thinks a knowledge graph for LLM memory can be looser than one used for visualization.
- User is interested in L0/L1/L2/L3 layered memory design.
```

注意：

```text
稳定内容放 system，利于 prompt cache。
动态召回内容放 user context 前缀，并用 `<system-reminder>...</system-reminder>` 包裹，避免破坏 system prompt cache。
```

---

# 17. 遗忘机制：类人记忆必须会遗忘

没有遗忘的记忆系统会越来越差。

遗忘不是简单删除，而是分几种：

```text
1. 时间衰减 decay
2. 被覆盖 superseded
3. 低置信度 deprecated
4. 用户要求删除 deleted
5. 归档 archived
```

记忆状态：

```text
active
superseded
deprecated
archived
deleted
```

时间衰减公式可以很简单：

```ts
function decay(memory: AtomicMemory, now: Date) {
  const ageDays = daysBetween(memory.updatedAt, now);

  const decayRate =
    memory.stability > 0.8 ? 0.001 :
    memory.stability > 0.5 ? 0.005 :
    0.02;

  return Math.exp(-decayRate * ageDays);
}
```

长期稳定偏好衰减慢：

```text
用户喜欢代码级解释
用户常用 Java
用户偏好中文技术讨论
```

临时记忆衰减快：

```text
用户今天在调某个 bug
用户刚才问某个命令
用户当前任务临时约束
```

召回时不要只看创建时间，而应该看：

```text
importance
stability
updatedAt
accessCount
confidence
scope
```

---

# 18. 巩固机制：从碎片到长期画像

人脑有 consolidation，LLM 记忆也需要。

可以设计成异步 pipeline：

```text
实时：
  写 M1
  必要时抽 M2

分钟级：
  M2 去重、合并、冲突处理

小时级：
  M2 -> M3 场景聚合

天级：
  M3 -> M4 用户画像更新

长期：
  低价值记忆归档 / 压缩 / 删除
```

调度策略：

```text
新用户前几轮：频繁抽取，快速建立画像
稳定用户：降低抽取频率
高重要事件：立即抽取
空闲时：做场景聚合和 persona 更新
```

伪代码：

```ts
class ConsolidationScheduler {
  async onNewAtomicMemories(userId: string, memoryIds: string[]) {
    await this.sceneQueue.enqueue({
      userId,
      memoryIds,
      delaySeconds: 60,
    });

    const shouldUpdatePersona = await this.shouldUpdatePersona(userId);

    if (shouldUpdatePersona) {
      await this.personaQueue.enqueue({
        userId,
        delayMinutes: 10,
      });
    }
  }

  async shouldUpdatePersona(userId: string) {
    const changedScenes = await sceneStore.countChangedSinceLastPersona(userId);
    const elapsed = await personaStore.elapsedSinceLastUpdate(userId);

    return changedScenes >= 3 || elapsed.days >= 1;
  }
}
```

---

# 19. 轻量知识图谱层：可选，但很有价值

虽然我不建议一开始做完整知识图谱，但建议加一层轻量实体关系索引。

实体表：

```sql
CREATE TABLE entities (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  type TEXT,
  aliases JSON,
  created_at DATETIME,
  updated_at DATETIME
);
```

关系表：

```sql
CREATE TABLE entity_relations (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  subject_entity_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  object_entity_id TEXT,
  object_value TEXT,
  confidence REAL DEFAULT 0.8,
  source_memory_ids JSON,
  status TEXT DEFAULT 'active',
  created_at DATETIME,
  updated_at DATETIME
);
```

但要控制关系类型，不要爆炸。

可以先限定 10～20 种：

```text
likes
dislikes
uses
works_on
asked_about
prefers
belongs_to_project
has_constraint
corrected
related_to
```

对于 LLM 记忆系统，图谱层主要用来：

```text
1. 实体归一化
2. 多跳召回
3. 消歧
4. 关系约束
5. 辅助排序
```

而不是取代 M2/M3/M4。

---

# 20. 推荐的最终架构

整体架构可以是：

```text
                         ┌─────────────────────┐
                         │     LLM / Agent      │
                         └──────────┬──────────┘
                                    │
                          before prompt build
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Memory Retriever   │
                         └──────────┬──────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Persona M4      │      │   Scene M3        │      │ Atomic Memory M2 │
└──────────────────┘      └──────────────────┘      └──────────────────┘
                                                             │
                                                             ▼
                                                    ┌──────────────────┐
                                                    │ Event Log M1      │
                                                    └──────────────────┘

after turn committed
          │
          ▼
┌──────────────────┐
│ Capture Manager   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Event Store M1    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Extractor M2      │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Dedup / Conflict  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Consolidation     │
│ M2 -> M3 -> M4    │
└──────────────────┘
```

---

# 21. 最小可实现版本 MVP

如果从零做，不要一开始做太大。可以分阶段。

## Phase 1：基础长期记忆

实现：

```text
M1 event_log
M2 atomic_memory
embedding search
BM25 search
source_event_ids
basic prompt injection
```

先不要做 M3/M4。

表：

```text
memory_events
atomic_memories
memory_embeddings
```

接口：

```ts
interface MemoryService {
  recordTurn(turn: ConversationTurn): Promise<void>;
  extractMemories(sessionId: string): Promise<AtomicMemory[]>;
  recall(userId: string, query: string): Promise<AtomicMemory[]>;
}
```

---

## Phase 2：去重和冲突

增加：

```text
similar memory search
LLM dedup
update / merge / supersede
confidence / importance / stability
```

接口：

```ts
interface DedupService {
  resolve(candidates: AtomicMemory[]): Promise<DedupDecision[]>;
}
```

---

## Phase 3：场景记忆

增加：

```text
scene_memories
scene markdown
M2 -> M3 聚合
scene navigation 注入
```

接口：

```ts
interface SceneService {
  updateScenes(userId: string, memoryIds: string[]): Promise<void>;
  recallScenes(userId: string, query: string): Promise<SceneMemory[]>;
}
```

---

## Phase 4：用户画像

增加：

```text
persona_items
persona.md
M3 -> M4 更新
稳定 persona 注入 system prompt
```

接口：

```ts
interface PersonaService {
  updatePersona(userId: string): Promise<void>;
  getPromptPersona(userId: string): Promise<string>;
}
```

---

## Phase 5：程序记忆和策略化 Agent

增加：

```text
procedural_memories
workflow recall
task-specific procedure injection
success/failure feedback
```

---

# 22. 一个完整的接口设计

```ts
interface MemorySystem {
  // 写入原始事件
  appendEvent(event: MemoryEvent): Promise<void>;

  // 记录一轮对话
  recordTurn(turn: ConversationTurn): Promise<void>;

  // 从原始事件抽取原子记忆
  extractAtomicMemories(input: ExtractInput): Promise<AtomicMemory[]>;

  // 去重、合并、冲突处理
  consolidateAtomicMemories(memories: AtomicMemory[]): Promise<void>;

  // 更新场景记忆
  updateScenes(userId: string, memoryIds: string[]): Promise<void>;

  // 更新用户画像
  updatePersona(userId: string): Promise<void>;

  // 召回
  recall(request: RecallRequest): Promise<RecallResult>;

  // 生成 prompt 上下文
  buildPromptMemoryContext(request: RecallRequest): Promise<string>;

  // 用户删除/修改记忆
  deleteMemory(userId: string, memoryId: string): Promise<void>;
  updateMemory(userId: string, memoryId: string, patch: Partial<AtomicMemory>): Promise<void>;
}
```

---

# 23. Prompt Memory Context 的格式

最终给 LLM 的上下文应该简洁、分层、带来源。

例如：

```text
# Long-Term Memory

## User Profile

- The user prefers detailed, implementation-level technical explanations.
- The user often asks for architecture first, then code/data model details.

## Active Scene

The user is designing a human-like memory system for LLM agents. Recent discussion covered:
- whether memory should be modeled as a knowledge graph
- TencentDB Agent Memory's L0/L1/L2/L3 design
- event logs, atomic memories, scene memories, and persona profiles

## Relevant Atomic Memories

1. User thinks knowledge graphs for LLM memory can be looser than display-oriented KGs.
   Source: mem_123

2. User wants designs to go down to code implementation level.
   Source: mem_456

## Available Memory Tools

- memory_search(query): search atomic memories
- conversation_search(query): search raw past conversations
- scene_read(scene_id): read full scene memory
```

---

# 24. 最重要的设计取舍

## 取舍一：自然语言记忆 vs 知识图谱

我的建议：

```text
主存储：自然语言 atomic memory
辅助索引：entities + optional triples
```

不要反过来。

因为 LLM 最擅长消费自然语言，过度结构化会让系统复杂度暴涨。

---

## 取舍二：自动注入 vs 工具查询

两者都要有。

```text
自动注入：
  少量、高置信、高相关的记忆

工具查询：
  让 Agent 在需要时主动查更多
```

不要自动注入太多，否则会污染上下文。

---

## 取舍三：总结 vs 原文

不要只保存总结。

正确做法：

```text
M1 保存原文
M2 保存抽取事实
M3 保存场景总结
M4 保存长期画像
```

上层总结必须能回溯到底层原文。

---

## 取舍四：记忆准确性 vs 召回覆盖率

宁可少记一点，也不要乱记。

尤其是用户画像：

```text
一次出现的不一定是长期偏好
多次出现、被用户明确表达、或者对未来明显有帮助，才适合进入 M4
```

---

# 25. 最终一句话架构

我会这样定义这个系统：

```text
一个面向 LLM 的类人记忆系统，不应该是单一知识图谱，也不应该是单一向量库，而应该是：

append-only event log
+ atomic semantic memory
+ lightweight entity graph
+ scene-level consolidation
+ persona/profile abstraction
+ procedural memory
+ salience/decay/conflict control
+ hybrid retrieval
+ traceable prompt injection
```

更短一点：

```text
M1 记录经历
M2 抽取事实
M3 组织场景
M4 形成画像
M5 固化行为
Controller 负责记忆、遗忘、召回和修正
```

如果你要做一个真的能长期运行的 LLM 记忆系统，这个结构会比“知识图谱 + 向量库”更稳，也更接近人脑记忆的工程近似。
