# Alpha Agent TODO

This document turns the Hermes implementation review into Alpha Agent's own
near-term roadmap. The goal is practical usability parity where it matters:
messaging access, reliable agent turns, observability, and operations. It is
not a plan to copy Hermes internals. Alpha Agent's core direction remains its
own cognition runtime, rebuilt after the Phase 00 state baseline.

## Guiding Decisions

- Memory-as-records has been removed by Phase 00. Long-term cognition is being
  rebuilt as an event-sourced cognition runtime; see
  `docs/todo/cognition-runtime/`.
- Add messaging through a thin gateway layer, not by merging platform logic into
  the core agent runtime.
- Normalize all platforms into one small internal message model before invoking
  the agent.
- Prefer simple, inspectable sync/async boundaries. Platform adapters may need
  async I/O, but the state baseline and agent core should remain understandable.
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
    platform_thread_id, message_id, metadata.
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
- [x] Preserve external platform identity metadata for the current state
  baseline:
  - platform user, chat, and thread fields remain source metadata.
  - long-term cognition routing is deferred to `docs/todo/cognition-runtime/`.
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
  - `alpha daemon start` owns local and gateway runtime turns in the background.
  - `alpha daemon run` runs the same owner in the foreground.
  - `alpha daemon status`
  - `alpha daemon restart`
  - `alpha daemon stop`
  - `alpha gateway status`
  - `alpha gateway doctor`

P0 implementation notes:

- Gateway session mappings and dedup state are persisted in SQLite.
- Active-turn guarding is daemon-owned and shared by local CLI turns and gateway
  turns for the same session id.
- Runtime logging uses JSONL helpers that include `session_id`, `platform`, and
  hashed external chat/user identifiers when message context is available.
- `alpha gateway run` no longer starts an independent runtime; it points users
  to `alpha daemon start`.

## P1: Agent Loop Improvements

Current `AlphaAgent.respond()` is a clean MVP. The next step is to keep it
explicit while making it useful for real channels and longer tasks.

- [x] Split the turn pipeline into named services without hiding flow:
  - append session source message.
  - session context projection/compression.
  - recent session state loading.
  - prompt build.
  - model call.
  - assistant message write.
  - runtime diagnostic traces.
- [x] Add structured runtime traces:
  - `llm.started`
  - `llm.completed`
  - `turn.failed`
- [x] Add tool execution as an explicit subsystem:
  - keep tool registry small.
  - no hidden agent framework.
  - include `tool.started`, `tool.completed`, `tool.failed`.
  - return only tool raw output in transcript tool messages; keep tool name and
    diagnostic metadata in trace/session metadata.
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
  - include the recent conversation state used to build the prompt.

P1 Agent Loop implementation notes:

- The turn pipeline is split into explicit runtime methods while keeping
  `AlphaAgent.respond()` as the visible orchestration path.
- User, assistant, and tool transcript content is stored in
  `session_messages`; operational diagnostics are stored as
  `runtime_traces`.
- Tool execution remains bounded and explicit. Caller-supplied tool calls are
  local one-shot executions. Provider-returned OpenAI-compatible tool calls run
  through one bounded agent loop controlled by `max_tool_iterations` and
  `max_llm_rounds`: the initial model call, tool-result follow-up calls, and
  finalization call share one loop state. Each assistant `tool_calls` message is
  followed immediately by matching `role=tool` results before the next model
  call. When the bound is reached, the runtime makes one `finalize` request for
  a best-effort answer with the same tool schema and `tool_choice="none"` so
  provider prefix caches are not invalidated by dropping tool definitions; if
  that still requests tools, the turn fails observably.
- DeepSeek and OpenAI-compatible providers share the same tool-call wire model:
  `tools`, `tool_choice`, assistant `tool_calls`, and `role=tool` messages with
  `tool_call_id`. Missing provider tool ids and `finish_reason=tool_calls`
  without normalized calls fail before execution. Recoverable provider tool
  execution failures are recorded as `tool.failed` traces and returned to the
  model as plain tool-output text so the next LLM round can correct.
- Cancellation is synchronous and cooperative. The runtime checks session
  cancellation flags at safe boundaries before/after user message persistence,
  state loading, LLM, and tool stages. It cannot preempt a blocking provider or
  tool call that does not return control.
- Retry is bounded around transient provider HTTP failures only; retry counts
  are recorded in turn debug metadata and runtime traces.
- `alpha debug prompt MESSAGE` remains supported. `alpha debug prompt
  --session ...` can include gateway source fields and prints the built prompt
  from the current state baseline without writing runtime access rows.
- Prompt construction keeps only the stable identity as a `system` message,
  followed by source stream messages after the latest compressed handover and
  the current user message. There is no long-term recall, extraction, candidate
  lifecycle, retrieval ranking, or consolidation in the Phase 00 baseline.

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
  - preserve stable identity fields for future cognition counterpart routing.
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
  - cognition review controls after the cognition runtime defines review
    objects and decisions.
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
  - cognition review command/buttons after the cognition review flow exists.
  - conservative media support later.
- [ ] Keep Alpha's internal source model platform-neutral:
  - platform=`weixin`.
  - chat_id from peer/group id.
  - user_id from sender id.
  - context_token stays adapter metadata, not cognition state.
- [ ] Add WeChat tests:
  - update normalization.
  - context token cache behavior.
  - stale context token fallback.
  - dedup.
  - text chunking.
  - auth/allowlist.

## P1: Cognition Product Usability

These items make Alpha Agent feel different from a generic chat bot after the
cognition runtime phases define the underlying objects.

- [ ] Add cognition review commands once Phase 02+ defines durable review
  records and decisions.
- [ ] Add confidence/source display to cognition inspection, prompt debug, and
  diagnostics after belief projection exists.
- [ ] Add "what do you know about me?" inspection on top of projected beliefs.
- [ ] Add correction/forget semantics on top of cognition events and belief
  projection.
- [ ] Add per-channel cognition policy:
  - DM can create trusted observations under explicit rules.
  - group chats require clear routing and write policy.
  - platform/system messages should never become durable user facts.
- [ ] Add consolidation/reporting only after the cognition event log and
  reflection phases are in place.

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

- [ ] `/status`: current session id, provider, cognition status, active turn
  state.
- [ ] `/reset`: reset session context for this channel.
- [ ] `/stop`: cancel active turn.
- [ ] `/remember <text>`: explicit cognition observation/review request after
  the cognition review model exists.
- [ ] `/forget <id>`: apply correction/forget semantics after belief projection
  supports them.
- [ ] `/consolidate`: run manual cognition consolidation if authorized.
- [ ] `/debug prompt`: admin-only prompt inspection.

Avoid adding broad model switching, plugin management, update commands, kanban
commands, or multi-agent controls until Alpha Agent has stable messaging and
cognition review.

## Suggested Build Order

1. Add gateway models, session key logic, adapter interface, and tests.
2. Add gateway runner for local in-process adapter tests.
3. Add the minimum reliable turn lifecycle:
   - structured runtime traces.
   - active-turn guard.
   - `/stop` cancellation path.
   - bounded provider retry.
   - prompt/state debug metadata for channel turns.
4. Wire CLI `alpha daemon start/status/stop` and `alpha gateway status/doctor`.
5. Add Feishu text MVP with allowlist, mention gating, and tests.
6. Add channel commands `/status`, `/reset`, `/stop`, `/remember`.
7. Add cognition review flow after the cognition runtime phases define it.
8. Decide WeChat target after confirming the real account/channel constraints.
9. Add chosen WeChat adapter.
10. Add service/runtime status and deployment templates.
