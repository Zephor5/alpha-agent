# Cognition Runtime — 落地执行计划

把 `docs/cognition/cognition_from_scratch.md` 拆成一组可独立落地、可独立验收的
阶段。每个阶段一个文档，文档结构统一，可直接当成执行清单。

## 与其它文档的关系

```text
docs/cognition/cognition.md
  ── 理论框架（七层 + 十五字段）

docs/cognition/cognition_from_scratch.md
  ── 架构设计（事件日志 + 多 loop + ContextWindow + 元认知三级）
  ── 本计划的"为什么这样设计"权威，本计划只回答"怎么做、做到哪算完"

docs/doing/memory-system-optimization-phases.md
  ── 已完成的 memory 子系统优化记录
  ── 本计划 Phase 00 要清理的对象

docs/todo/cognition-runtime/  （本目录）
  ── 12 个阶段文档 + README
```

## 三条架构不变量（所有阶段共享）

这套设计有三条贯穿始终的不变量。它们不属于任何一阶段，而是任何一阶段都必须
遵守的基础约定。

### 不变量 1：主体是单数，对方是复数

```text
Subject       = Agent 自己。整个认知运行时**只有一个** Subject。
                它持有 ValueLens、SelfModel、KnownBiases、served_counterparts。

Counterpart   = Agent 服务 / 对话 / 观察的"对方"。**一对多**。
                典型角色：user / operator / peer_agent / system / anonymous。
                每个 Counterpart 有 identity、role、relationship、
                service_contract、trust_level、communication_style。

Belief        = Subject 持有的信念。可以是关于 Counterpart、关于实体、
                关于 Subject 自己。`about: list[Reference]` 字段表达"关于谁"。
                "user_a 偏好 Python" 是一条 Belief，about=[counterpart:user_a]。

Perception    = 一次感知。`from_counterpart: CounterpartRef | None` 表达
                "来自谁"。user_message 来自某个 Counterpart；clock_tick /
                self_signal 来源不是 Counterpart。

ValueLens / SelfModel / KnownBiases  = 永远属于 Subject，不属于 Counterpart。
                Agent 不"代用户思考"——它对 user 的理解都是自己的 Belief。
```

这条不变量保证认知主体唯一、信念归属清晰：用户偏好不是"用户自己的 memory"，
而是"agent 对该用户的 belief"——可以被修正、可以被撤回。Counterpart 自己说
什么是 Perception，会进 event log。

### 不变量 2：单主体串行、非抢占、Reactive 非阻塞

整个系统只有一个 Subject，所有 loop 都在它身上跑。两条互补规则：

```text
单 Subject 内：任意时刻只有一个 loop 在运行。
              所有调度型 loop（Consolidation / L2 / Drive / L3）通过
              LoopCoordinator 排队，FIFO + 优先级。

抢占策略：    无。低优先级 loop 在跑时，更高优先级的也不强行打断。
              类比："你不能打断主体的睡觉"——consolidation 这类"睡眠期更新"
              一旦开始就要让它跑完（或自觉 yield 后退出）。

Reactive 例外：
              Reactive 用 try_acquire（非阻塞）。如果锁被低优先级 loop 占
              着，**不等待**，立刻向用户返回一条系统提示，例如：
                  "Agent is currently consolidating memory; please retry in
                   ~30s."
              且这一次交互**不写任何 cognitive event**——既不 emit
              perceived，也不写 conversation_messages。从主体视角看，这次
              请求"未曾发生"。

排队优先级（仅对调度型 loop 间生效）：
  1. Reflective L2 (out-of-band, 分钟级)
  2. Drive Loop (主动 stimulus 入队)
  3. Consolidation Loop (小时级)
  4. Reflective L3 (日级)
  Reactive 不入此队——它要么瞬时拿到锁、要么瞬时回 busy。

公平性 & 让路：
  低优先级 loop 跑太长会让连续多次 Reactive 全部被回 busy。对策："自觉分块"
  仍然保留——但目的从"让 Reactive 不再等"变成"让下一次 Reactive 不再被
  回 busy"。每个 worker 在 max_chunk_duration 边界主动 yield，让其它调度
  型 loop 排队上来；下次 Reactive 试 acquire 时若恰在 yield 缝隙，就能拿
  到。
    - Consolidation worker：每跑 ≤ max_chunk_duration（默认 30s）yield 一次。
    - L3 aggregator：≤ 60s 一片。
    - yield 不强制 holder 退出——但若有更高优先级（调度型）等待，coordinator
      让出锁；若只有 Reactive 在外面试探，coordinator 也短暂释放锁、给
      Reactive 一个窗口、然后让 holder 续锁。
```

设计选择的代价：消费者要处理 "busy" 响应（CLI / Gateway 把它显示出来，
不当成正常对话）。换来的是：主体的认知不会在请求洪峰下排队失控，"睡眠期"
工作不会被无限挤后。

LoopCoordinator 的接口与事件 kind 见 Phase 01 §3.6；调度型 loop 的 worker
协议里都强制带 `max_chunk_duration` 字段。

调度策略：时间 + 关注内容，不是纯时间 cadence。每个调度型 worker 声明：

```text
min_interval        最短间隔（防过频）
max_interval        最长间隔（兜底，避免某些"看长窗口"的规则永远不跑）
watches             一组 CognitiveEventKind——本 worker 在意的事件
min_new_events      在 watches 窗口内至少要有这么多新事件才触发
```

Scheduler 在每个 wake-up 检查 `should_run(worker, now)`：

- `now - last_run_at < min_interval` → 跳过（频率下限）
- `now - last_run_at >= max_interval` → 跑（兜底）
- 否则查 worker.last_processed_event_id 之后 watches 类事件数 →
  `>= min_new_events` 才跑，否则跳过

效果：一个完整 cadence 周期内若没有任何相关交互，worker 根本不 acquire 锁，
也就不会触发 Reactive 的 busy 响应。L3 这种日级 worker 也不会在"用户两周没
用"的情况下还每天空跑一次。具体协议见 Phase 06 §3.x（通用 scheduler 与
checkpoint 表）。

### 不变量 3：事件日志是宪法

`cognitive_events` 表是 append-only。projection 是从日志投影的物化视图，可随
时 drop & rebuild。任何阶段的"写"都必须 emit 事件、不允许直接改 projection
表。`schema_version` 字段为后续事件 schema 演化预留——新增字段优先，破坏性
改动禁止。

---

## 总体原则

承袭仓库 `AGENTS.md` 的硬规则：

- **不做兼容层**。每阶段都向目标架构直推，不留"既能跑旧路径又能跑新路径"的开
  关。Phase 00 把现有 memory 系统按计划清理掉，后续阶段在干净地基上长。
- **全局视角**。每阶段都要考虑对 CLI、gateway、tests、README、AGENTS.md 导航
  的影响，落地清单里都要列。
- **重复 3 次即抽公共**。各阶段 Stage / Projection / Reflector 之间会有共享
  辅助函数，每阶段文档里都标了"可共享候选"。
- **不写本机绝对路径**。所有路径都是项目根的相对路径。
- **CounterpartRef 一路传递**。任何产生 Perception / Belief / Decision 的代
  码都必须能从输入中识别或继承一个 `CounterpartRef`（可能是 None，表示来自
  Subject 自己或环境）。
- **调度型 loop 走 LoopCoordinator**。任何不在 Reactive tick 内同步执行的
  loop 都必须通过 LoopCoordinator 申请锁，并在 `max_chunk_duration` 边界
  自觉 yield。

## 阶段总览

| Phase | Title                          | Depends on   | Scope  | Status  |
| ----- | ------------------------------ | ------------ | ------ | ------- |
| 00    | 清理现有 memory 机制           | —            | L (大) | completed |
| 01    | 类型与事件日志                 | 00           | L (大) | completed |
| 02    | Reactive 单 tick 跑通          | 01           | L (大) | completed |
| 03    | BeliefProjection v1            | 01, 02       | M (中) | completed |
| 04    | ContextWindowProjection（前景）| 01, 02       | S (小) | completed |
| 05    | Reflector L1                   | 02, 03       | S (小) | completed |
| 06    | Consolidation Loop + 背景压缩  | 03, 04, 05   | L (大) | completed |
| 07    | ValueLens 与冲突解决           | 03, 05, 06   | M (中) | completed |
| 08    | Reflector L2                   | 05, 07       | M (中) | completed |
| 09    | Renderer 解耦                  | 02, 03, 04   | M (中) | completed |
| 10    | Drive Loop                     | 02, 06       | M (中) | completed |
| 11    | Reflector L3 / SelfModel       | 05, 06, 08   | M (中) | completed |

Phase 01 是 L 而非 M，因为它一次落齐所有 first-class 类型 + 事件日志 +
LoopCoordinator + Counterpart projection——这套地基所有后续阶段都依
赖，不能渐进。

依赖图（→ 表示"必须先于"）：

```text
00 → 01 → 02 ─┬─→ 03 ─┬─→ 05 ─┬─→ 06 ─┬─→ 07 ─→ 08 ─┐
              │       │       │       │              │
              ├─→ 04 ─┤       │       ├─→ 10         ├─→ 11
              │       │       │       │              │
              └──────────────→ 09      └──────────────┘
```

具体依赖（来自各 Phase 文档头部）：

- 03 ← 01, 02
- 04 ← 01, 02
- 05 ← 02, 03
- 06 ← 03, 04, 05
- 07 ← 03, 05, 06
- 08 ← 05, 07
- 09 ← 02, 03, 04
- 10 ← 02, 06
- 11 ← 05, 06, 08

也就是说：

- 00 → 01 → 02 是一条强直线，必须先打通。
- 03 与 04 都依赖 02，可并行（不同的 projection）。
- 05 在 03 之后就能开；06 要等 03/04/05 都到位。
- 07 / 08 / 11 是元认知与价值链的延伸；07 消费 06 留下的冲突队列并注册
  consolidation worker，所以不能早于 06。
- 09 渲染器可在 02/03/04 之后并行插入；10 Drive Loop 需要 Phase 06 的通用
  Scheduler，所以在 06 之后插入。

## 推荐落地顺序

最小可用主体（MVP）：00 → 01 → 02 → 04 → 03 → 05 → 09

到这里主体已能：感知、解释、判断、决策、行动、反思、用结构化方式渲染给 LLM。

完整主体：补 06 → 07 → 08 → 11，并按需要在 06 之后加 10。

## 每阶段文档结构

所有阶段文档结构一致：

```text
0. 目标
1. 范围（In scope / Out of scope）
2. 任务清单（数据模型 / 模块 / 测试 / 文档）
3. 接口契约（关键类型与函数签名草案）
4. 文件清单（新增 / 修改 / 删除）
5. 验收标准
6. 风险与备注
7. 后续衔接（下一阶段会从本阶段消费什么）
```

每条任务都是 markdown checkbox。一阶段完成时，文档顶部 Status 改为
`completed`，并把该阶段的"实际选择记录"写进同目录的 `docs/develop_record/`。

## 给执行者的几条提醒

1. **顺序不可乱**。依赖图不是建议，是强依赖。跳过 Phase 00 直接开 01 会污染
   数据库。
2. **每阶段交付时同步更新 AGENTS.md 与 README.md 的项目导航**。这是 AGENTS.md
   要求的"全局视角"。
3. **跨阶段共享代码**统一放 `src/alpha_agent/cognition/_shared/` 或合适的
   现有 util 目录，不要在多个 stage 里重复。
4. **事件 schema 一旦写出来就避免破坏性改**。事件日志是 append-only 的"宪
   法"。新增字段优先；旧事件 replay 要永远能跑过。
5. **测试不写实现细节**。测行为：given 一串事件，then projection 是某个形状。
   不要测"调用了哪个内部函数"。
