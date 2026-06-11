# Session Time Reminder Execution Plan

## Status

Planned.

## Date

2026-06-11

## Source Of This Plan

This plan records the target behavior for adding lightweight local-time reminder
messages to Alpha Agent sessions. It is self-contained and based on the current
runtime/session message model.

## Goal

Give each session's message stream explicit local calendar-time context by
inserting a short `<system-reminder>` message before relevant runtime input
messages.

This applies to inputs handled through `AlphaAgent.respond()`, including local
`ask`, local `chat`, gateway messages, and Drive Loop `self_signal` turns.

## Target Behavior

### Session Start Reminder

Before the first runtime input message in a session, insert:

```text
<system-reminder>started at: {local_datetime}</system-reminder>
```

`local_datetime` is local time, precise to the minute, with timezone offset.
Example:

```text
<system-reminder>started at: 2026-06-11T21:37+08:00</system-reminder>
```

### New Day Reminder

Before each later runtime input message, check whether the current local date is
a new day compared with the last inserted time reminder for that session.

If it is a new day, insert a time update reminder before that input message:

```text
<system-reminder>time update: {local_datetime}</system-reminder>
```

Example:

```text
<system-reminder>time update: 2026-06-12T09:03+08:00</system-reminder>
```

If it is still the same local day, do not insert another reminder.

## Scope

- Reminders are part of the session source message stream.
- Reminders should appear before the input message they time-anchor.
- Reminders themselves must not trigger additional reminders.
- Existing session messages should not be rewritten or backfilled.

Detailed storage and code structure should be decided during implementation.

## Acceptance Checklist

- [ ] The first runtime input in a session is preceded by a `started at`
      reminder.
- [ ] Same-day later runtime inputs do not create duplicate time reminders.
- [ ] First runtime input on a later local day is preceded by a `time update`
      reminder.
- [ ] Reminder timestamps use local time with minute precision and timezone
      offset.
- [ ] Reminder messages appear in the session stream before the input message
      they describe.
- [ ] Drive Loop `self_signal` turns follow the same reminder rules.
