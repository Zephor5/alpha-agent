# Alpha Agent TODO

This document turns the Hermes implementation review into Alpha Agent's own
near-term roadmap. The goal is practical usability parity where it matters:
messaging access, reliable agent turns, observability, and operations. It is
not a plan to copy Hermes internals. Alpha Agent's core direction remains its
own explicit cognition-inspired memory runtime.

## Guiding Decisions

- Keep memory native to Alpha Agent: working, episodic, semantic, procedural,
  salience, retrieval, and consolidation stay as first-class internal services.
- Add messaging through a thin gateway layer, not by merging platform logic into
  the core agent runtime.
- Normalize all platforms into one small internal message model before invoking
  the agent.
- Prefer simple, inspectable sync/async boundaries. Platform adapters may need
  async I/O, but the memory and agent core should remain understandable.
- Build for one human operator first. Avoid a broad plugin marketplace, massive
  slash-command surface, or multi-agent orchestration until the single-user path
  is useful.

## Hermes Review Notes

Useful Hermes reference points:

- Hermes `gateway/platforms/base.py`
- Hermes `gateway/platforms/weixin.py`
- Hermes `gateway/platforms/feishu.py`
- Hermes `gateway/session.py`
- Hermes `gateway/run.py`
- Hermes `agent/conversation_loop.py`
- Hermes `agent/tool_executor.py`
- Hermes `gateway/config.py`
- Hermes `gateway/status.py`

What to borrow conceptually:

- Platform adapters convert raw platform events into a common message event.
- A gateway owns auth, deduplication, session routing, command handling, typing
  indicators, queueing, and outbound delivery.
- Session identity must be platform-aware: DM, group, thread, and per-user
  isolation are different decisions.
- Long-running turns need visible progress and recoverable state.
- Platform integrations need tests around auth, dedup, message normalization,
  routing, and delivery failures.

What not to copy:

- Hermes' broad memory-provider/plugin system.
- Hermes' very large gateway runner shape.
- Hermes' large slash-command surface as an initial target.
- Hermes' specific Weixin iLink assumption as a generic "WeChat bot" answer.
- Hermes' context compression and tool loop details unless Alpha Agent reaches
  the same scale pressures.

## P0: Usability Foundation

These are prerequisites for making Alpha Agent usable outside `alpha chat`.

- [x] Add `gateway/` package with clear layers:
  - `gateway/models.py`
  - `gateway/session.py`
  - `gateway/adapters/base.py`
  - `gateway/runner.py`
  - `gateway/config.py`
  - `gateway/status.py`
- [x] Define platform-neutral models:
  - `ConversationSource`: platform, chat_id, chat_type, user_id, user_name,
    thread_id, message_id, metadata.
  - `InboundMessage`: source, text, message_type, attachments, received_at,
    platform_message_id, raw metadata.
  - `OutboundMessage`: text, attachments, reply_to, thread metadata, visibility.
  - `DeliveryResult`: success, message_id, error, retryable.
- [x] Define a small `PlatformAdapter` interface:
  - `connect(handler)`
  - `disconnect()`
  - `send(source, outbound)`
  - `send_typing(source)`
  - optional `on_processing_start(source)` / `on_processing_complete(source)`.
- [x] Implement platform-aware session key generation:
  - DM session.
  - Group shared session.
  - Group per-user session.
  - Thread session.
  - Thread per-user session.
- [x] Store gateway session mappings in SQLite instead of ad hoc files.
- [x] Decide how external platform identities map to Alpha memory scope:
  - global user memory.
  - platform-specific user memory.
  - chat/thread-local working memory.
- [x] Add inbound message deduplication:
  - platform update/message id.
  - fallback text fingerprint with short TTL.
  - persisted dedup state for webhook platforms.
- [x] Add per-session active-turn guard:
  - one running turn per session by default.
  - busy-turn queue admission signal; durable queue storage/draining is
    deferred to the real adapter runner.
  - `/stop`, `/reset`, `/status` bypass normal busy guard.
- [x] Add gateway runtime logs:
  - `~/.alpha-agent/logs/agent.log`
  - `~/.alpha-agent/logs/gateway.log`
  - `~/.alpha-agent/logs/errors.log`
  - include session_id, platform, chat_id hash, user_id hash.
- [x] Add `alpha gateway` CLI group:
  - `alpha gateway run`
  - `alpha gateway status`
  - `alpha gateway doctor`
  - `alpha gateway stop` remains intentionally deferred until a real
    long-running adapter runner and PID/lock stop path exist.

P0 implementation notes:

- Gateway session mappings and dedup state are persisted in SQLite.
- Active-turn guarding is thread-safe inside one process; multi-process
  distributed locking is not part of P0.
- Runtime logging uses JSONL helpers that include `session_id`, `platform`, and
  hashed external chat/user identifiers when message context is available.
- `alpha gateway run` is currently an honest operational smoke stub. It
  initializes the database, status file, and log files, then exits cleanly when
  no platform adapters are configured.

## P1: Agent Loop Improvements

Current `AlphaAgent.respond()` is a clean MVP. The next step is to keep it
explicit while making it useful for real channels and longer tasks.

- [x] Split the turn pipeline into named services without hiding flow:
  - event write.
  - working memory update.
  - retrieval.
  - prompt build.
  - model call.
  - assistant event write.
  - extraction.
  - consolidation trigger decision.
  - delivery event emission.
- [x] Add structured turn events:
  - `turn.started`
  - `memory.retrieved`
  - `llm.started`
  - `llm.completed`
  - `memory.extracted`
  - `turn.completed`
  - `turn.failed`
- [x] Add tool execution as an explicit subsystem:
  - keep tool registry small.
  - no hidden agent framework.
  - include `tool.started`, `tool.completed`, `tool.failed`.
  - require deterministic result serialization into events.
- [x] Add interrupt/cancel support:
  - cancellation flag by session_id.
  - gateway `/stop` command.
  - safe cleanup of in-flight provider/tool call where possible.
- [x] Add bounded retry policy:
  - provider HTTP retry for transient errors.
  - no infinite agent loops.
  - record retry count in turn debug metadata.
- [x] Add prompt/debug inspection for channel turns:
  - `alpha debug prompt --session ...`
  - include gateway source context.
  - include retrieved memory ids and ranking scores.
- [x] Add memory review mode:
  - show extracted candidates before storing.
  - approve/reject/edit candidates.
  - start with CLI.

P1 Agent Loop implementation notes:

- The turn pipeline is split into explicit runtime methods while keeping
  `AlphaAgent.respond()` as the visible orchestration path.
- Structured turn, memory, LLM, tool, completion, and failure events are stored
  in the normal event log metadata.
- Tool execution remains bounded and explicit. Caller-supplied tool calls are
  local one-shot executions. Provider-returned OpenAI-compatible tool calls now
  run through a bounded multi-step loop controlled by `max_tool_iterations` and
  `max_llm_rounds`: each assistant `tool_calls` message is followed immediately
  by matching `role=tool` results before the next model call. When the bound is
  reached, the runtime makes one no-tools `finalize` request for a best-effort
  answer; if that still requests tools, the turn fails observably.
- DeepSeek and OpenAI-compatible providers share the same tool-call wire model:
  `tools`, `tool_choice`, assistant `tool_calls`, and `role=tool` messages with
  `tool_call_id`. Missing provider tool ids and `finish_reason=tool_calls`
  without normalized calls fail before execution. Recoverable provider tool
  execution failures are recorded as `tool.failed` events and returned to the
  model as deterministic JSON tool results so the next LLM round can correct.
- Cancellation is synchronous and cooperative. The runtime checks session
  cancellation flags at safe boundaries before/after user event, retrieval, LLM,
  and tool stages. It cannot preempt a blocking provider or tool call that does
  not return control.
- Retry is bounded around transient provider HTTP failures only; retry counts
  are recorded in turn debug metadata and runtime events.
- `alpha debug prompt MESSAGE` remains supported. `alpha debug prompt
  --session ...` can include gateway source fields and prints the built prompt,
  retrieved memory ids, and retrieval scores from `memory_access_log`.
- Prompt construction separates stable system identity, optional reference-only
  retrieved context, and the current user message as a real chat message. The
  current turn is not written into working memory before retrieval, and
  procedural memories require explicit textual relevance before their procedure
  bodies can enter the prompt.
- `alpha memory review MESSAGE --session ...` previews extracted candidates
  without writing semantic/episodic/procedural memory. CLI flags support
  approve-all, reject-all, per-candidate approve/reject, and selected candidate
  edit-and-approve. Platform review commands for Feishu/WeChat remain future
  adapter work tracked in the integration sections.

## P1: Feishu Integration

Feishu should likely be the first serious platform integration because Hermes'
implementation shows a mature and official-ish integration path.

- [ ] Decide first transport:
  - Webhook is easier to deploy behind a public callback.
  - WebSocket is easier for local/private operation if app permissions allow it.
- [ ] Add dependencies behind an optional extra:
  - `alpha-agent[feishu]`
  - likely `lark-oapi`, plus `aiohttp` or equivalent if webhook mode is chosen.
- [ ] Add config:
  - `ALPHA_FEISHU_ENABLED`
  - `ALPHA_FEISHU_CONNECTION_MODE`
  - `ALPHA_FEISHU_APP_ID`
  - `ALPHA_FEISHU_APP_SECRET`
  - `ALPHA_FEISHU_VERIFICATION_TOKEN`
  - `ALPHA_FEISHU_ENCRYPT_KEY`
  - `ALPHA_FEISHU_ALLOWED_USERS`
  - `ALPHA_FEISHU_REQUIRE_MENTION`
- [ ] Implement text MVP:
  - receive DM text.
  - receive group text only when bot is mentioned.
  - strip self mention before passing text to Alpha.
  - send plain text replies.
  - apply allowlist before invoking agent.
- [ ] Normalize identity carefully:
  - preserve `open_id`, `user_id`, and `union_id` in source metadata.
  - prefer stable `union_id` for memory scope when available.
  - do not leak raw IDs into prompt unless needed.
- [ ] Add webhook security if webhook mode is implemented:
  - content-type check.
  - max body size.
  - verification token.
  - signature validation with timing-safe compare.
  - basic per-IP/app rate limit.
- [ ] Add per-chat serial processing:
  - one active turn per Feishu chat/thread.
  - queue follow-up bursts.
  - debounce rapid text bursts only after the simple path works.
- [ ] Add processing state:
  - typing indicator or reaction while processing.
  - failure reaction/message on exception.
- [ ] Add second-stage Feishu features:
  - reply/thread context.
  - image/file receive.
  - file/image send.
  - interactive card buttons for approve/deny memory candidates.
  - expose memory review decisions through Feishu commands or cards.
  - reaction events as command inputs only if genuinely useful.
- [ ] Add tests:
  - webhook token/signature validation.
  - group mention gating.
  - allowlist.
  - identity normalization.
  - dedup.
  - outbound send payload.

## P1: WeChat / Weixin Integration

Do not treat "WeChat bot" as one implementation. Choose the target channel
before writing code.

- [ ] Decide target:
  - personal WeChat via Tencent iLink Bot API.
  - WeCom / 企业微信.
  - official account.
  - third-party bridge.
- [ ] If choosing iLink, document constraints first:
  - availability and account requirements.
  - whether ordinary group chat is supported for this bot identity.
  - QR login lifecycle.
  - token refresh/expiration behavior.
  - compliance and operational risk.
- [ ] Add config for chosen transport only after target is decided.
- [ ] For iLink-style implementation, treat these as core requirements:
  - long-poll receive loop.
  - persisted account token.
  - per-peer `context_token` cache.
  - send with context token, then retry without it on stale session.
  - message id and fingerprint dedup.
  - text chunking for long replies.
  - basic typing status if supported.
  - memory review command/buttons after the CLI review flow proves useful.
  - conservative media support later.
- [ ] Keep Alpha's internal source model platform-neutral:
  - platform=`weixin`.
  - chat_id from peer/group id.
  - user_id from sender id.
  - context_token stays adapter metadata, not agent memory.
- [ ] Add WeChat tests:
  - update normalization.
  - context token cache behavior.
  - stale context token fallback.
  - dedup.
  - text chunking.
  - auth/allowlist.

## P1: Memory-First Product Usability

These items make Alpha Agent feel different from a generic chat bot.

- [ ] Add memory review commands:
  - `alpha memory candidates`
  - `alpha memory approve <id>`
  - `alpha memory reject <id>`
  - `alpha memory edit <id>`
- [ ] Store extracted candidates separately before promotion when review mode is on.
- [ ] Add confidence/source display to channel replies when memory is used.
- [ ] Add "what do you remember about me?" command.
- [ ] Add "forget this" / "forget memory id" support.
- [ ] Add per-channel memory policy:
  - DM can write semantic memory by default.
  - group chats may require explicit "remember".
  - platform/system messages should never become semantic facts.
- [ ] Add consolidation modes:
  - manual.
  - after N turns.
  - scheduled only after gateway scheduler exists.
- [ ] Add duplicate/contradiction handling:
  - detect conflicting semantic facts by subject/predicate.
  - mark older fact superseded instead of deleting immediately.
  - expose conflict review in CLI.

## P2: Engineering And Operations

- [ ] Add config validation:
  - `alpha doctor`
  - `alpha gateway doctor`
  - check DB path, provider config, platform credentials, optional deps.
- [ ] Add structured logging with secret redaction:
  - API keys.
  - platform tokens.
  - webhook signatures.
  - raw platform user IDs when privacy mode is enabled.
- [ ] Add runtime status:
  - PID file.
  - gateway lock.
  - status JSON with started_at, connected platforms, last error, active sessions.
- [ ] Add clean shutdown:
  - disconnect adapters.
  - flush logs.
  - mark active turns interrupted.
- [ ] Add service templates:
  - Docker Compose for local/private deployment.
  - systemd user service for Linux.
  - launchd plist later if macOS background operation matters.
- [ ] Add hermetic test script:
  - fixed `TZ=UTC`.
  - fixed `PYTHONHASHSEED`.
  - credentials cleared unless a test explicitly sets them.
  - runs ruff, mypy, pytest.
- [ ] Add gateway-specific tests:
  - session key rules.
  - adapter contract tests.
  - dedup store.
  - busy session queue.
  - command bypass behavior.
  - status file behavior.
- [ ] Add release checklist:
  - migrations compatible.
  - `.env.example` updated.
  - README command examples checked.
  - mock provider path still works without API key.

## P2: Channel Commands

Keep channel commands small and operational.

- [ ] `/status`: current session id, provider, memory counts, active turn state.
- [ ] `/reset`: reset working memory/session context for this channel.
- [ ] `/stop`: cancel active turn.
- [ ] `/remember <text>`: explicit memory write candidate.
- [ ] `/forget <id>`: mark memory inactive/superseded.
- [ ] `/consolidate`: run manual consolidation if authorized.
- [ ] `/debug prompt`: admin-only prompt inspection.

Avoid adding broad model switching, plugin management, update commands, kanban
commands, or multi-agent controls until Alpha Agent has stable messaging and
memory review.

## Suggested Build Order

1. Add gateway models, session key logic, adapter interface, and tests.
2. Add gateway runner for local in-process adapter tests.
3. Add the minimum reliable turn lifecycle:
   - structured turn events.
   - active-turn guard.
   - `/stop` cancellation path.
   - bounded provider retry.
   - prompt/retrieval debug metadata for channel turns.
4. Wire CLI `alpha gateway run/status/doctor`.
5. Add Feishu text MVP with allowlist, mention gating, and tests.
6. Add channel commands `/status`, `/reset`, `/stop`, `/remember`.
7. Add memory candidate review flow.
8. Decide WeChat target after confirming the real account/channel constraints.
9. Add chosen WeChat adapter.
10. Add service/runtime status and deployment templates.
