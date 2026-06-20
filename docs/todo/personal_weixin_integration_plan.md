# Personal Weixin Integration Execution Plan

Date: 2026-06-20

## Decision

Build the personal WeChat path as `platform=weixin` through Tencent iLink Bot API.
This is intentionally not WeCom, official account, or a third-party desktop bridge.

The first shippable target is text DM receive/reply for one local operator. Group
chat support stays disabled by default until a real account proves that iLink
delivers ordinary group events for the connected bot identity.

## Current Alpha Baseline

Alpha already has the gateway foundation needed for a platform adapter:

- `src/alpha_agent/gateway/models.py` defines platform-neutral
  `ConversationSource`, `InboundMessage`, `OutboundMessage`, and `DeliveryResult`.
- `src/alpha_agent/gateway/adapters/base.py` defines a synchronous
  `PlatformAdapter` contract: `connect(handler)`, `disconnect()`, `send()`, and
  `send_typing()`.
- `src/alpha_agent/gateway/runner.py` owns normalization handoff into
  `AlphaAgent.respond()`, durable inbound dedup, cached outbound retry, runtime
  error replies, processing hooks, and redacted gateway logs.
- `src/alpha_agent/gateway/session.py` provides durable external conversation to
  Alpha session mapping and inbound dedup by platform message id or fallback text
  fingerprint.
- `src/alpha_agent/daemon/runtime.py` starts configured adapters inside the daemon,
  connects them through `GatewayRuntimeBridge`, and disconnects them on shutdown.
- `src/alpha_agent/gateway/config.py` currently returns no real adapters.
- `src/alpha_agent/config.py`, `config.example.toml`, and `.env.example` currently
  have no platform-specific Weixin config.

The current bridge is synchronous. A Weixin long-poll transport should therefore
hide its polling lifecycle inside the adapter rather than converting the daemon or
runtime to async.

## Hermes Reference Points

Use these Hermes files as reference material only; do not copy the broader Hermes
gateway runner, plugin system, slash-command surface, or media stack into Alpha:

- `gateway/platforms/weixin.py`
- `gateway/config.py`
- `tests/gateway/test_weixin.py`
- `website/docs/user-guide/messaging/weixin.md`

Relevant lessons from Hermes:

- iLink uses HTTP long polling via `ilink/bot/getupdates`.
- QR login returns `account_id`, `bot_token`, `baseurl`, and user identity, and the
  credentials are persisted for later startup.
- The poll cursor is `get_updates_buf`; it must be persisted across restarts.
- Outbound replies should include the latest per-peer `context_token`.
- Stale session responses include `errcode=-14`; Hermes also treats
  `ret=-2` / `errcode=-2` with `errmsg="unknown error"` as stale session rather
  than genuine rate limiting.
- When a send fails because the context token is stale, retry once without the
  token and clear the cached peer token.
- iLink bot identities are often `...@im.bot` accounts and usually cannot receive
  ordinary WeChat group messages. Default group policy should be `disabled`.
- Text replies need chunking, but short structured Markdown should remain in one
  bubble whenever possible.
- The media path requires encrypted CDN upload/download and `cryptography`; keep it
  out of the text MVP.

## Architecture

### Adapter Shape

Add `src/alpha_agent/gateway/adapters/weixin.py` with a small iLink client and a
sync adapter wrapper:

- `WeixinAdapter(PlatformAdapter)` exposes Alpha's sync adapter methods.
- `connect(handler)` validates config and starts one polling thread.
- The polling thread owns network clients and calls the sync `handler` for each
  accepted normalized message.
- `disconnect()` requests shutdown, interrupts the poll loop, joins the thread with
  a bounded timeout, and closes clients.
- `send(source, outbound)` sends text to `source.chat_id`, splitting long content
  into chunks and returning the last iLink client message id.
- `send_typing(source)` is best-effort and no-op when no typing ticket is cached.

Prefer existing `httpx` for the text MVP because Alpha already depends on it and
the adapter contract is synchronous. Add `aiohttp` only if a later async transport
or media path clearly needs it.

### Config

Add a Weixin config dataclass under `AlphaConfig`:

- `gateway.weixin.enabled`
- `gateway.weixin.account_id`
- `gateway.weixin.token`
- `gateway.weixin.base_url` defaulting to `https://ilinkai.weixin.qq.com`
- `gateway.weixin.dm_policy`: `open`, `allowlist`, `disabled`
- `gateway.weixin.group_policy`: `disabled`, `allowlist`, `open`
- `gateway.weixin.allowed_users`
- `gateway.weixin.allowed_groups`
- `gateway.weixin.message_max_chars`
- `gateway.weixin.send_chunk_delay_seconds`
- `gateway.weixin.send_chunk_retries`
- `gateway.weixin.send_chunk_retry_delay_seconds`
- `gateway.weixin.poll_timeout_seconds`

Environment variables should use the Alpha prefix, for example
`ALPHA_WEIXIN_ENABLED`, `ALPHA_WEIXIN_ACCOUNT_ID`, and `ALPHA_WEIXIN_TOKEN`.
Avoid unprefixed `WEIXIN_*` in Alpha's primary config path.

`configured_adapters()` should instantiate `WeixinAdapter` only when enabled and
the account id plus token can be resolved.

### Credential And Adapter State

Keep platform operational state outside cognition.

- Token credentials may come from config/env or from an Alpha-owned credential file
  under `~/.alpha-agent/weixin/accounts/`, written with owner-only permissions.
- `context_token` values are adapter state, not `ConversationSource.metadata`, not
  `source_metadata`, and not counterpart identity.
- Persist `context_token` per `(account_id, peer_id)`.
- Persist `get_updates_buf` per `account_id`.
- Persist neither raw iLink token nor raw `context_token` in gateway logs.
- If a generic state mechanism is added, prefer a narrow
  `gateway_adapter_state` table keyed by `(platform, account_id, key)` over
  Weixin-specific state scattered across files.

### Message Normalization

Normalize iLink inbound messages before they reach the bridge:

- DM:
  - `ConversationSource.platform = "weixin"`
  - `chat_id = from_user_id`
  - `chat_type = "dm"`
  - `user_id = from_user_id`
- Group, if iLink actually delivers it:
  - `chat_id = room_id` or `chat_room_id`
  - `chat_type = "group"`
  - `user_id = from_user_id`
- `platform_message_id = message_id` when present.
- `source.message_id = message_id` when present.
- `source.user_name` may be sender nickname when available; otherwise the sender id.
- `raw_metadata` must be sanitized and must not contain auth token,
  `context_token`, or large raw media payloads.

For the session mapping, update `GatewayRuntimeBridge` so session mode can be
selected per message or per adapter:

- DM should use `SessionMode.DM`.
- Group should default to `SessionMode.GROUP_PER_USER`.
- A future config can opt group chats into `GROUP_SHARED`, but not in the MVP.

### Poll Cursor Semantics

Do not advance the persisted `get_updates_buf` until all messages in the poll
response have been attempted through the gateway handler. This preserves the
current gateway cached-outbound retry behavior: if delivery fails, the same
platform message can be fetched again and retried without re-running Alpha.

For the MVP, process messages from a poll response sequentially. Add a bounded
queue only after the simple path works and tests show polling latency is a real
problem.

## Phased Work

### Phase 0: Account And Constraint Gate

Goal: prove that the chosen personal WeChat path is actually usable before
touching runtime code.

Tasks:

- Confirm the target account can obtain iLink QR credentials.
- Confirm DM events arrive through `getupdates`.
- Try one ordinary group mention and record whether iLink delivers any group event.
- Decide whether token persistence is config/env-only for the first pass or whether
  `alpha gateway weixin login` lands before adapter startup.

Acceptance criteria:

- A short note is added to this plan or TODO with the observed account constraints.
- If group events are not observed, `group_policy=disabled` remains the documented
  default and no group feature is promised.

Verification:

- Manual network probe against iLink using a throwaway script outside CI.
- No raw token, account id, or peer id is committed to the repo.

### Phase 1: Config And Adapter Wiring

Goal: Alpha can discover a configured Weixin adapter without starting a real poll.

Tasks:

- Add `WeixinGatewayConfig` to `src/alpha_agent/config.py`.
- Add config key typing, allowed values, env loading, masking, and config rendering.
- Update `config.example.toml` and `.env.example`.
- Add `src/alpha_agent/gateway/adapters/weixin.py` with config validation and a
  placeholder iLink client interface that can be faked in tests.
- Wire `configured_adapters()` and `configured_adapter_names()`.
- Extend `alpha gateway doctor` checks for enabled adapter, missing credentials,
  and optional QR dependency state.

Acceptance criteria:

- With Weixin disabled, behavior stays the same and no Weixin network dependency is
  imported at startup.
- With Weixin enabled and missing credentials, `doctor` reports a clear problem.
- With Weixin enabled and fake credentials, adapter names include `weixin`.

Verification:

- `uv run pytest tests/test_config.py tests/test_gateway_cli.py -q`
- New tests for config env overrides, allowed values, secret masking, and adapter
  discovery.

### Phase 2: iLink Text Client And State Stores

Goal: implement the protocol pieces needed for text-only operation without
coupling them to Alpha turns.

Tasks:

- Implement `_api_post()` for iLink JSON payloads with required headers:
  `AuthorizationType=ilink_bot_token`, `iLink-App-Id=bot`, and client version.
- Implement `get_updates(sync_buf)`.
- Implement `send_text(peer_id, text, context_token, client_id)`.
- Implement stale-session detection for `errcode=-14` and `-2` plus
  `unknown error`.
- Implement rate-limit retry/backoff separately from stale-session fallback.
- Implement persisted `context_token` cache.
- Implement persisted `get_updates_buf`.
- Implement atomic writes and owner-only permissions for credential/state files if
  file-backed stores are used.

Acceptance criteria:

- Unit tests can drive the iLink client with fake HTTP responses.
- Context tokens restore after process restart.
- Sync buffer restore/save behavior is deterministic.
- Stale context token send retries once without token and clears the cached token.
- Genuine rate-limit `-2` does not clear context token.

Verification:

- New `tests/test_weixin_adapter.py` protocol/state unit tests.
- `uv run pytest tests/test_weixin_adapter.py -q`

### Phase 3: Receive Loop And Normalization

Goal: Weixin inbound text reaches `GatewayRuntimeBridge` as platform-neutral
messages.

Tasks:

- Implement `connect(handler)` poll thread.
- Normalize DM text messages into `InboundMessage`.
- Default group messages to ignored unless `group_policy` allows them.
- Apply DM allowlist before invoking the gateway handler.
- Extract text from iLink `item_list`; use voice transcript text when present.
- Update per-peer context token on inbound messages.
- Fetch and cache typing ticket best-effort when supported.
- Add adapter-level 5-minute content fingerprint dedup for duplicate text with
  different iLink message ids.
- Save `get_updates_buf` only after message handling attempts complete.

Acceptance criteria:

- DM text from an allowed sender invokes the gateway handler once.
- DM text from a blocked sender is ignored and does not create an Alpha session.
- Duplicate message id is suppressed by existing gateway dedup.
- Duplicate content with different message ids is suppressed by adapter TTL dedup.
- Context token from inbound is stored but not present in `source_metadata`.
- Group events are ignored by default.

Verification:

- `uv run pytest tests/test_weixin_adapter.py tests/test_gateway_core.py -q`
- New tests for normalization, allowlist, group default, context token update, and
  sync cursor timing.

### Phase 4: Outbound Text Delivery

Goal: Alpha replies can be delivered back to Weixin reliably.

Tasks:

- Implement `send(source, outbound)` for text-only replies.
- Format text conservatively: preserve Markdown, collapse excessive blank lines,
  wrap very long plain lines if needed, and reject empty chunks.
- Split only when the message exceeds the configured character limit.
- Generate stable per-chunk client ids for retry of the same chunk.
- Include latest context token when available.
- Retry without context token on stale-session response.
- Add inter-chunk delay and bounded retry/backoff.
- Return `DeliveryResult(success=False, retryable=True)` for transient send errors.

Acceptance criteria:

- Empty reply never sends a blank WeChat bubble.
- Short multi-paragraph Markdown stays in one message.
- Oversized replies split at paragraph or fenced-code boundaries where possible.
- Multi-chunk delivery waits between chunks.
- Send failure lets `GatewayRuntimeBridge` keep cached outbound for duplicate retry.
- Stale context token fallback succeeds without re-running Alpha.

Verification:

- New outbound chunking and stale-token tests.
- `uv run pytest tests/test_weixin_adapter.py tests/test_gateway_core.py -q`

### Phase 5: QR Login Command

Goal: a user can provision credentials without manually scripting the iLink QR flow.

Tasks:

- Add `alpha gateway weixin login`.
- Request QR code via `get_bot_qrcode`.
- Render terminal QR when optional `qrcode` is installed; otherwise print the URL.
- Poll `get_qrcode_status` until confirmed, expired, denied, or timed out.
- Save `account_id`, token, base URL, and user id to the Alpha credential store.
- Print next steps using Alpha config keys, not Hermes variable names.

Acceptance criteria:

- QR timeout uses a monotonic clock.
- Expired QR is handled clearly.
- Saved credential file is owner-readable only.
- Command output never prints the raw token.

Verification:

- Unit tests with fake QR responses.
- Manual login check on a real account.

### Phase 6: Runtime And Documentation Polish

Goal: make the integration operationally inspectable and maintainable.

Tasks:

- Update README status from "no real WeChat adapter" once the text adapter works.
- Update `docs/todo/TODO.md` to mark the chosen target and completed MVP items.
- Add `alpha gateway doctor` details for credential file path, adapter enabled
  state, dependency availability, and last poll error without exposing secrets.
- Ensure gateway logs hash external chat/user ids and do not log raw token,
  `context_token`, or raw iLink payloads.
- Document group limitation and default-off group policy.
- Document the exact manual smoke test steps.

Acceptance criteria:

- New users can configure, run, and diagnose the text DM path from active docs.
- No repository file contains local machine-specific absolute paths or real
  account identifiers.
- Logs remain useful without leaking raw platform identifiers.

Verification:

- `uv run ruff check .`
- `uv run mypy src tests`
- `uv run pytest -q`
- Manual smoke test: start daemon, send DM to the iLink bot identity, receive Alpha
  reply, restart daemon, send another DM, verify context token and sync cursor were
  restored.

### Phase 7: Media Support Later

Goal: add media only after text DM is stable.

Tasks:

- Add optional media dependency set with `cryptography`.
- Implement encrypted CDN download for inbound image/file/voice/video.
- Implement outbound file upload only after URL safety and local file policy are
  reviewed against Alpha's tool sandbox model.
- Add SSRF protection for any remote media URL fetch.
- Add tests for AES key formats, upload method, media type builders, and unsafe URL
  rejection.

Acceptance criteria:

- Media support cannot compromise local file sandbox assumptions.
- Text-only deployments do not import or require media dependencies.

## Cross-Cutting Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| iLink account or bot identity cannot receive desired chats | High | Gate Phase 0 before implementation promises. Default to DM-only. |
| Raw platform identifiers leak into logs or cognition | High | Keep tokens/context tokens out of source metadata; use existing gateway log hashing; add tests. |
| Poll cursor advances before delivery retry is possible | Medium | Save cursor only after handler attempts complete; rely on existing cached outbound retry. |
| Synchronous handler blocks long polling | Medium | Accept for MVP; add bounded queue only after DM path is stable. |
| Multiple local processes use the same token | Medium | Add token-scoped lock using a hash, not the raw token. |
| Media support expands dependencies and attack surface | Medium | Defer media to Phase 7; text MVP stays dependency-light. |

## Open Questions

- Should QR login write directly to Alpha's config file, or should config point to
  the credential file and keep token rotation separate?
- Should context token and sync cursor live in SQLite `gateway_adapter_state` or in
  owner-only JSON files under `~/.alpha-agent/weixin/accounts/`?
- Is DM allowlist default `open` acceptable for the first local operator, or should
  Alpha default to `allowlist` once the operator's own sender id is known?
- If iLink group events work for a specific account type, should Alpha use
  `GROUP_PER_USER` only, or expose `GROUP_SHARED` as an explicit advanced setting?
