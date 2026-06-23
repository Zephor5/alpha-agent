# Mobile UI App Layer Plan

## Goal

Build an iOS and Android mobile app as the primary user interface for Alpha Agent.

The app uses Flutter for both platforms, communicates with the server-side Alpha Runtime through HTTPS REST and SSE, receives offline updates through APNs and FCM, and stores local UI state in SQLite.

## Delivery Scope

- Flutter mobile app for iOS and Android.
- Authenticated mobile API client.
- Chat UI with streamed assistant responses.
- Session list and session detail views.
- Local SQLite cache for sessions, messages, event cursors, pending turns, and device registration.
- Push notification registration and handling.
- Settings screen for account, runtime status, notification controls, and local cache management.
- Feedback entry points for assistant messages.
- Mobile-side observability for request failures, stream failures, cache state, and notification delivery.

## Mobile App Modules

### App Shell

- Create the Flutter application package.
- Add platform targets for iOS and Android.
- Define app routing for login, session list, chat detail, settings, and runtime status.
- Add global app state for auth status, current user, network state, and active session.
- Add light and dark themes.
- Add app lifecycle handling for foreground, background, resume, and terminated launch from notification.

### Authentication

- Implement sign in, token refresh, sign out, and session expiry handling.
- Store refresh credentials in Keychain on iOS and Android Keystore-backed secure storage on Android.
- Store short-lived access token only in memory while the app process is active.
- Attach access token through the `Authorization` header for all authenticated API calls.
- Refresh access token before opening SSE streams when the token is near expiry.
- Clear local auth state, active streams, cached account metadata, and device registration on sign out.

### API Client

- Implement a typed HTTPS client for REST endpoints.
- Implement a typed SSE client for turn event streams.
- Add retry handling for idempotent reads.
- Add request timeout handling for REST calls.
- Add stream reconnect handling for SSE using the last persisted event cursor.
- Decode all API responses into typed client models.
- Map server error responses into user-facing UI states and diagnostic records.

### Session List

- Load cached sessions from SQLite on app start.
- Refresh session list from the server after login and app resume.
- Support cursor-based pagination.
- Show session title, latest message preview, latest activity time, and pending sync status.
- Create a new session from the session list.
- Open an existing session into chat detail.
- Keep the selected session stable across app background and resume.

### Chat Detail

- Load cached messages for the selected session from SQLite.
- Fetch server messages after the cached cursor.
- Send user turns through the REST API.
- Render assistant response deltas from SSE in real time.
- Render structured runtime events for tool activity, memory activity, turn completion, cancellation, and errors.
- Persist streamed deltas and final assistant messages to SQLite.
- Support canceling an active turn.
- Support retrying failed pending turns.
- Support message feedback for useful, not useful, correction, and free-form note.
- Preserve scroll position while streaming.
- Restore active turn state after app resume.

### Local SQLite Cache

- Create local tables for:
  - `sessions`
  - `messages`
  - `turns`
  - `turn_events`
  - `stream_cursors`
  - `pending_turns`
  - `device_registration`
  - `sync_state`
- Store server IDs, local IDs, timestamps, sync status, and optimistic UI state.
- Persist the latest SSE event sequence for each active turn.
- Persist unsent and in-flight turns for retry after network recovery.
- Store enough message content to render recent conversations offline.
- Add schema migration support for app upgrades.
- Add cache compaction for old event deltas after final assistant messages are persisted.

### Streaming Event Handling

- Start the SSE stream after `POST /v1/sessions/{sessionId}/turns` returns a `turnId`.
- Persist every received event with its event ID and sequence.
- Apply events to chat UI in sequence order.
- Resume a stream from the last persisted event cursor after reconnect.
- Treat `turn_completed`, `turn_failed`, and `turn_cancelled` as terminal turn states.
- Close the active stream when the app enters background.
- Reconcile active turn state with the server when the app returns to foreground.

### Push Notifications

- Register the device with APNs on iOS.
- Register the device with FCM on Android.
- Send device token, platform, app version, locale, timezone, and notification preference to the server.
- Refresh device registration after token rotation, login, logout, and app update.
- Handle notification taps by opening the referenced session or runtime status view.
- Fetch fresh data from the server after notification open.
- Display notification content using server-provided notification type and localization key.

### Settings

- Show account identity and sign out action.
- Show server connection status.
- Show notification registration status.
- Show local cache size and last sync time.
- Provide local cache reset.
- Provide runtime diagnostics export for support.
- Provide message privacy and data handling summary.

### Runtime Status

- Show server runtime availability.
- Show active turn state for the current account.
- Show cognition worker summary returned by the server.
- Show latest sync error and retry action.
- Refresh status on app resume and manual pull-to-refresh.

### Feedback

- Add feedback controls to assistant messages.
- Send feedback through REST API.
- Cache feedback locally until acknowledged by the server.
- Show feedback submission status on the message.
- Update cached message metadata after acknowledgement.

## Server API Required By App

### Auth

- `POST /v1/auth/sign-in`
- `POST /v1/auth/refresh`
- `POST /v1/auth/sign-out`
- `GET /v1/me`

### Device Registration

- `PUT /v1/devices/current`
- `DELETE /v1/devices/current`

### Sessions

- `POST /v1/sessions`
- `GET /v1/sessions?cursor={cursor}`
- `GET /v1/sessions/{sessionId}`
- `PATCH /v1/sessions/{sessionId}`

### Messages

- `GET /v1/sessions/{sessionId}/messages?cursor={cursor}`

### Turns

- `POST /v1/sessions/{sessionId}/turns`
- `GET /v1/turns/{turnId}`
- `GET /v1/turns/{turnId}/events`
- `POST /v1/turns/{turnId}/cancel`

### Feedback

- `POST /v1/messages/{messageId}/feedback`

### Runtime

- `GET /v1/runtime/status`
- `GET /v1/memory/summary`

## API Response Contracts

### Standard Error

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request.",
    "details": []
  }
}
```

### Create Turn Response

```json
{
  "turnId": "turn_...",
  "sessionId": "session_...",
  "eventStreamUrl": "/v1/turns/turn_.../events",
  "createdAt": "2026-06-20T00:00:00Z"
}
```

### SSE Event Envelope

```json
{
  "eventId": "evt_...",
  "sequence": 1,
  "turnId": "turn_...",
  "type": "assistant_delta",
  "createdAt": "2026-06-20T00:00:00Z",
  "payload": {}
}
```

### SSE Event Types

- `turn_started`
- `assistant_delta`
- `assistant_message_committed`
- `tool_started`
- `tool_progress`
- `tool_completed`
- `memory_activity`
- `turn_completed`
- `turn_failed`
- `turn_cancelled`

## Security Tasks

- Enforce HTTPS for all API and stream traffic.
- Configure iOS App Transport Security for the production API domain.
- Configure Android Network Security Config for the production API domain.
- Store refresh credentials only in platform secure storage.
- Store local SQLite cache inside the app sandbox.
- Encrypt local SQLite cache using a platform-protected key.
- Redact message content from client logs.
- Redact tokens and device identifiers from diagnostic export.
- Add biometric unlock for opening the app after a configurable idle period.

## Sync Tasks

- Run initial sync after login.
- Run session and message sync on app resume.
- Run targeted sync after notification open.
- Run pending turn retry after network recovery.
- Run feedback retry after network recovery.
- Reconcile active turn state after SSE reconnect.
- Persist sync checkpoints after every successful page fetch.

## UI States

- Logged out.
- Loading account.
- Session list loading.
- Session list empty.
- Session list loaded.
- Session list error.
- Chat loading.
- Chat offline with cached messages.
- Chat streaming.
- Chat waiting for server.
- Chat turn failed.
- Chat turn cancelled.
- Runtime unavailable.
- Token expired.
- Push registration failed.

## Verification

- iOS app builds in release mode.
- Android app builds in release mode.
- Sign in stores refresh credentials in secure storage.
- Session list renders from SQLite before network refresh completes.
- New session creation opens chat detail.
- Sending a turn persists the user message locally.
- SSE assistant deltas render incrementally.
- SSE reconnect resumes from the last persisted event sequence.
- App resume reconciles active turns.
- Push notification tap opens the referenced session.
- Pending turns retry after network recovery.
- Feedback persists locally and syncs to the server.
- Sign out clears secure credentials and account-scoped SQLite rows.
- Diagnostics export contains no tokens or message text.

## Implementation Tasks

### Phase 1: App Foundation

- Create Flutter app package.
- Add iOS and Android platform configuration.
- Add routing, theme, app lifecycle hooks, and global state.
- Add typed models for sessions, messages, turns, events, devices, feedback, and runtime status.
- Add local SQLite database and migration runner.

### Phase 2: Auth And API Client

- Implement sign in, refresh, sign out, and account fetch.
- Implement secure credential storage.
- Implement typed REST client.
- Implement standard error handling.
- Implement request tracing without sensitive payloads.

### Phase 3: Sessions And Local Cache

- Implement session list cache reads.
- Implement session list server sync.
- Implement session creation.
- Implement message cache reads.
- Implement message server sync.
- Add pagination and sync checkpoints.

### Phase 4: Chat Turn Streaming

- Implement turn creation.
- Implement SSE connection.
- Implement event persistence.
- Implement streamed assistant rendering.
- Implement stream reconnect from cursor.
- Implement turn cancellation.
- Implement active turn reconciliation.

### Phase 5: Push And Resume Flow

- Implement APNs registration.
- Implement FCM registration.
- Implement device registration API calls.
- Implement notification tap routing.
- Implement targeted sync after notification open.
- Implement token rotation handling.

### Phase 6: Feedback, Settings, And Runtime Status

- Implement message feedback controls.
- Implement feedback sync and retry.
- Implement settings screen.
- Implement cache reset.
- Implement runtime status screen.
- Implement diagnostics export.

### Phase 7: Release Verification

- Run iOS release build.
- Run Android release build.
- Run auth flow test.
- Run chat streaming test.
- Run SSE reconnect test.
- Run offline cache test.
- Run push notification test.
- Run sign out data clearing test.
- Run diagnostics redaction test.
