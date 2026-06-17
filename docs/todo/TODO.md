# Alpha Agent TODO

This is Alpha Agent's near-term roadmap, originally derived from a Hermes
implementation review. The gateway shell, the agent turn loop, and the
event-sourced cognition runtime have all landed (see README "Status & roadmap"
and the archived build plans under `docs/develop_record/`). What remains is
real platform reach, a user-facing cognition surface, channel control commands,
and operations/deployment. The goal is still practical single-operator
usability, not copying Hermes internals.

Already in place (do not re-plan; kept here only as the baseline these items
build on):

- Gateway foundation: platform-neutral models, a `PlatformAdapter` interface,
  platform-aware session keys with SQLite mappings, inbound dedup, a
  daemon-owned active-turn guard, JSONL runtime logs, and `alpha daemon
  start/run/restart/status/stop` + `alpha gateway status/doctor`.
- Agent turn loop: explicit pipeline, structured runtime traces, bounded tool
  subsystem with `ToolSpec` governance, cooperative cancellation, bounded
  provider retry, and `alpha debug prompt`.
- Cognition runtime: event log + projections (belief, counterpart, goal,
  subject), background extraction / consolidation / conflict review / summary /
  archival workers, counterpart routing, the synchronous Drive Loop with goals,
  and the `memory_recall` / `memory_propose` tools.

## Guiding Decisions

- Long-term cognition is an event-sourced runtime, not memory-as-records. The
  original phase-by-phase build plan is archived at
  `docs/develop_record/cognition-runtime/`; treat it as history, not as current
  spec. Active design plans still live in `docs/todo/` (e.g.
  `governed_tool_run_contract_plan.md`).
- Add messaging through a thin gateway adapter layer, not by merging platform
  logic into the core agent runtime.
- Normalize every platform into the one internal message model
  (`gateway/models.py`) before invoking the agent.
- Prefer simple, inspectable sync/async boundaries. Platform adapters may need
  async I/O, but the daemon-owned runtime and state baseline stay synchronous
  and understandable.
- Build for one human operator first. Avoid a plugin marketplace, a broad
  slash-command surface, or multi-agent orchestration until the single-user
  path is genuinely useful.

## Platform Adapter Guidance (from Hermes review)

Because the gateway foundation already exists, new platform work is an
adapter + config + wiring + tests on top of it, not new core plumbing. The
daemon owns the runtime turn and the single-active-turn guard per session id.

Still-relevant Hermes reference points when implementing an adapter:
`gateway/platforms/base.py`, `weixin.py`, `feishu.py`, `gateway/session.py`,
`gateway/run.py`, `gateway/config.py`, `gateway/status.py`.

Guardrails — do not copy from Hermes:

- its broad memory-provider/plugin system,
- its very large gateway runner shape,
- its large slash-command surface as an initial target,
- its Weixin iLink assumption as a generic "WeChat bot" answer,
- its context-compression and tool-loop details beyond Alpha's current scale.

Every adapter still needs tests around auth, dedup, message normalization,
routing, and delivery failures.

## P1: Feishu Integration

Feishu is the likely first serious platform integration. The gateway base,
session keys, dedup, active-turn guard, and logging are already shared, so the
work is the adapter, its config, and its tests.

- [ ] Decide first transport:
  - Webhook is easier to deploy behind a public callback.
  - WebSocket is easier for local/private operation if app permissions allow it.
- [ ] Add dependencies behind an optional extra:
  - `alpha-agent[feishu]`
  - likely `lark-oapi`, plus `aiohttp` or equivalent if webhook mode is chosen.
- [ ] Add config (none of these exist yet; `gateway/config.py` currently ships
  no real adapters):
  - `ALPHA_FEISHU_ENABLED`, `ALPHA_FEISHU_CONNECTION_MODE`
  - `ALPHA_FEISHU_APP_ID`, `ALPHA_FEISHU_APP_SECRET`
  - `ALPHA_FEISHU_VERIFICATION_TOKEN`, `ALPHA_FEISHU_ENCRYPT_KEY`
  - `ALPHA_FEISHU_ALLOWED_USERS`, `ALPHA_FEISHU_REQUIRE_MENTION`
- [ ] Implement text MVP:
  - receive DM text.
  - receive group text only when bot is mentioned.
  - strip self mention before passing text to Alpha.
  - send plain text replies.
  - apply allowlist before invoking agent.
- [ ] Normalize identity carefully:
  - preserve `open_id`, `user_id`, and `union_id` in source metadata.
  - route stable identity into cognition counterpart identity; the existing
    `counterpart_router` maps source metadata to a `CounterpartRef`.
  - do not leak raw IDs into the prompt unless needed.
- [ ] Add webhook security if webhook mode is implemented:
  - content-type check, max body size.
  - verification token, signature validation with timing-safe compare.
  - basic per-IP/app rate limit.
- [ ] Add per-chat serial processing:
  - the daemon active-turn guard already enforces one active turn per session;
    the remaining adapter work is queueing/draining follow-up bursts.
  - debounce rapid text bursts only after the simple path works.
- [ ] Add processing state:
  - typing indicator or reaction while processing.
  - failure reaction/message on exception.
- [ ] Add second-stage Feishu features:
  - reply/thread context.
  - image/file receive, file/image send.
  - reaction events as command inputs only if genuinely useful.
- [ ] Add tests: webhook token/signature validation, group mention gating,
  allowlist, identity normalization, dedup, outbound send payload.

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
  - QR login lifecycle; token refresh/expiration behavior.
  - compliance and operational risk.
- [ ] Add config for the chosen transport only after the target is decided.
- [ ] For an iLink-style implementation, treat these as core requirements:
  - long-poll receive loop; persisted account token.
  - per-peer `context_token` cache.
  - send with context token, then retry without it on stale session.
  - message id and fingerprint dedup; text chunking for long replies.
  - basic typing status if supported; conservative media support later.
- [ ] Keep Alpha's internal source model platform-neutral:
  - `platform=weixin`, `chat_id` from peer/group id, `user_id` from sender id.
  - `context_token` stays adapter metadata, not cognition state.
- [ ] Add tests: update normalization, context token cache behavior, stale
  context token fallback, dedup, text chunking, auth/allowlist.

## P1: Cognition Product Usability

The cognition runtime now provides the objects these features need — belief and
summary projections (with FTS recall ranking), counterpart projection,
background consolidation/conflict review, the Drive Loop/goals, and the
`memory_recall`/`memory_propose` tools. The work below is the missing
user-facing surface on top of that runtime.

- [ ] Add a "what do you know about me?" inspection built on projected beliefs
  and the existing recall ranking (CLI today exposes goals and import status
  but no belief inspection).
- [ ] Surface provenance in cognition inspection and `alpha debug prompt`:
  belief `authority`, `validity` window, `sources`/evidence, and lifecycle
  (there is no scalar "confidence" field — present authority + validity +
  sources instead).
- [ ] Add user-facing correction / forget semantics. Consolidation already
  supports supersede/retract/archive decisions internally; expose an explicit
  user path that emits the corresponding cognitive events and reflects in the
  projection.
- [ ] Add cognition review commands once the review surface (above) settles —
  approve/reject/edit pending or low-authority beliefs.
- [ ] Add per-channel cognition write policy:
  - DM can create trusted observations under explicit rules.
  - group chats require clear routing and a write policy.
  - platform/system messages must never become durable user facts.
- [ ] Add user-facing consolidation reporting/digests (the background
  consolidation itself already runs).

## P2: Engineering And Operations

- [ ] Add a top-level `alpha doctor` for provider/DB/config validation.
  `alpha gateway doctor` already checks DB path, log dir, LLM provider, gateway
  tables, and configured adapters; extend the credential/optional-dep checks
  once real adapters exist.
- [ ] Extend log redaction to a privacy mode. Config secret masking
  (`config show/get`) and bash-output secret redaction already exist, and
  gateway JSONL logs already hash external chat/user ids; still missing is a
  privacy-mode toggle plus redaction of platform tokens and webhook signatures
  once adapters land.
- [ ] Add clean shutdown for adapters: disconnect connected platforms and mark
  in-flight turns interrupted. Daemon socket/lock teardown already exists; this
  becomes real work once an adapter holds a live connection.
- [ ] Add service templates:
  - Docker Compose for local/private deployment.
  - systemd user service for Linux.
  - launchd plist later if macOS background operation matters.
- [ ] Add a hermetic test script (CI currently runs the three commands directly):
  - fixed `TZ=UTC` and `PYTHONHASHSEED`.
  - credentials cleared unless a test explicitly sets them.
  - runs ruff, mypy, pytest.
- [ ] Add a release checklist:
  - migrations/schema rebuild verified.
  - `.env.example` and `config.example.toml` updated.
  - README command examples checked.
  - mock provider path still works without an API key.

Note: runtime status (PID file, lock, status JSON with `started_at`, adapters,
background state/last-error) and gateway-specific tests (session keys, dedup,
busy guard, command bypass, status file) are already implemented and are no
longer tracked here.

## P2: Channel Commands

The active-turn guard already lets `/stop`, `/reset`, `/status` bypass the busy
check, but no handler dispatches them — they currently fall through to the model
as plain text. Runtime cancellation exists (`AgentManager.cancel` /
`_check_canceled`); these commands need to be wired to it.

- [ ] `/status`: current session id, provider, cognition status, active-turn state.
- [ ] `/reset`: reset session context for this channel.
- [ ] `/stop`: cancel the active turn (wire to existing runtime cancellation).
- [ ] `/remember <text>`: explicit cognition observation/review request, built on
  `memory_propose`.
- [ ] `/forget <id>`: apply correction/forget semantics once the cognition
  surface above supports them.
- [ ] `/debug prompt`: admin-only prompt inspection (the `alpha debug prompt`
  CLI already exists; this exposes it as a channel command).

Avoid broad model switching, plugin management, update commands, kanban
commands, or multi-agent controls until messaging and the cognition review
surface are stable.

## Suggested Build Order

1. Add the Feishu text MVP adapter (allowlist, mention gating, identity
   normalization, tests) on top of the existing gateway foundation.
2. Wire channel command handlers (`/status`, `/reset`, `/stop`) into gateway
   dispatch, reusing the existing cancellation path.
3. Build the cognition product surface: "what do you know about me?",
   provenance display, and correction/forget, on top of belief projection and
   the memory tools.
4. Add `/remember` and `/forget` once the cognition surface supports them.
5. Add operations polish: top-level `alpha doctor`, hermetic test script, and a
   privacy-mode log redaction toggle.
6. Decide the WeChat target after confirming real account/channel constraints,
   then add the chosen adapter.
7. Add deployment templates (Docker Compose, systemd) and the release checklist.
