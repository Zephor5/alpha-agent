# Background LLM Contract

## Background Cognition Integration

### Rule Boundary

Background cognition must not use deterministic rules for cognitive judgment.

Allowed deterministic responsibilities:

- Polling and scheduling.
- Foreground/background coordination.
- Selecting source windows and evidence sets.
- Calling the configured LLM provider.
- Validating LLM output schemas.
- Rejecting malformed, unsafe, or unsupported outputs.
- Enforcing authority and lifecycle invariants.
- Generating new ids.
- Persisting cognition entities.
- Writing audit logs for debugging and evidence inspection.
- Updating search indexes and lookup tables.
- Expiring records with explicit `valid_until`.
- Keeping checkpoints and computing idempotency keys when needed.

Not allowed as deterministic rule logic:

- Extracting memories from natural language.
- Classifying belief kind from content.
- Deciding two beliefs are semantically equivalent.
- Deciding a conflict's cognitive meaning.
- Choosing which belief is true based on text heuristics.
- Summarizing counterpart profiles.
- Synthesizing self-understanding summaries.
- Creating domain-guidance summaries from reflection patterns.

Those are LLM-mediated cognition tasks. Deterministic code validates and records the result, but does not pretend to understand the content.

### LLM Integration Stages

The background pipeline should be layered, but the cognitive work inside each layer is LLM-driven.

1. **Source Intake**

   Select raw session messages and runtime/tool traces that have not yet been integrated. Existing beliefs may be included as comparison context for consolidation. Audit logs may help troubleshoot worker behavior but are not the canonical cognition source.

   There is no dedicated `tool_traces` table. "Tool traces" maps to existing durable storage: tool results are `session_messages` with `kind=tool_message`, and runtime/tool execution events are `runtime_traces` rows. The ledger `source_type` values `tool_trace` and `runtime_trace` both resolve to `runtime_traces`; tool result content resolves to `session_messages`. Intake selects from these concrete sources, not from an abstract trace store.

2. **Memory Extraction**

   Ask the LLM to extract candidate id-less atomic belief drafts using the stable belief ontology. The output must be structured and schema-validated.

3. **Memory Consolidation**

   Ask the LLM to compare drafts with active beliefs in the same scope and domain. The LLM proposes create, strengthen, supersede, retract, or pending-confirmation operations.

4. **Conflict Review**

   Ask the LLM to classify conflicts and identify whether they require user confirmation. Deterministic code enforces authority ceilings and can refuse unsafe automatic resolution.

5. **Summary Belief Generation**

   Ask the LLM to synthesize `summary_beliefs` such as `counterpart_profile`, `project_profile`, `domain_summary`, and `self_memory_summary` from stable consolidated beliefs. Domain guidance and self-understanding are summary beliefs, not separate cognition entity types.

### Background LLM Output Contract

The background LLM contract is the shared definition of acceptable structured output for background cognition calls. It is not one identical payload for every worker. Each worker uses:

1. A common envelope.
2. A stage-specific typed payload.
3. Shared validation rules for target references, program-attached provenance, authority, ids, lifecycle, and prompt-injection resistance.

Common envelope:

```text
operation
authority
rationale
requires_confirmation
source_span_note
```

The envelope carries no `confidence` score. Source trust is expressed only through `authority`, bounded by the ceiling. Reject outputs that include a `confidence` or any other numeric belief-strength field.

Stage-specific payloads include:

```text
atomic_belief_draft
summary_belief_draft
belief_update
profile_summary_candidate
```

`atomic_belief_draft` and `summary_belief_draft` are used only for create-like operations and must not contain ids. Program logic generates new entity ids after validation.

`belief_update` is used only for update-like operations such as strengthen, supersede, retract, or archive. It may only reference belief ids that were included in the LLM input.

The LLM output must not be required to provide exact source message ids. Background prompts may render raw messages without durable ids to preserve prompt readability and provider prefix-cache reuse. `source_span_note` is optional natural-language orientation only, such as "from the recent preference discussion"; it is not authoritative provenance.

Program logic attaches evidence after validation from the selected source window:

```text
source_window_id
session_id
ordinal_start
ordinal_end
source_message_ids
source_trace_ids
extraction_run_id
input_belief_ids
```

For extraction, accepted atomic belief drafts inherit coarse evidence from the source window selected by deterministic code. For summarization, accepted summary beliefs inherit evidence from the selected belief ids and the summary target. If finer-grained evidence is later required, the renderer may add temporary stable labels inside the prompt, but that is an optional refinement and not a first-version requirement.

LLM output must not provide new ids or idempotency keys. New belief ids, summary ids, audit ids, and idempotency keys are program-level precision artifacts. If a worker needs idempotency, deterministic code should derive it from the selected source refs, operation kind, target ids, and a normalized accepted payload. If the worker does not need an idempotency key, do not invent one.

The runtime should reject outputs that:

- Invent source ids, evidence refs, or idempotency keys.
- Reference target beliefs not included in the LLM input.
- Include a generated id in any draft entity.
- Use unsupported `memory_kind` or `summary_kind`.
- Omit required scope/about fields.
- Claim higher authority than the source ceiling allows.
- Include a numeric `confidence` or any other belief-strength score.
- Attempt to inject prompt instructions into memory content.
- Produce unrelated memory outside the selected source window.
- Ask the system to treat audit logs as canonical source material.

## Compact-Adjacent Extraction Fast Path

Provider prefix caching is an internal provider optimization. The system does not call a cache API. If two LLM requests share the same provider-visible prefix, the provider may automatically reuse cached prefix computation and lower cost. If the cache does not hit, correctness must be unchanged.

The first layer of background extraction should exploit this by treating runtime handover compression as the preferred extraction trigger for active long sessions.

Fast-path trigger:

- When `handover_compression.completed` is written, enqueue a high-priority extraction source window for the raw messages covered by `compression_point_ordinal`.
- The extraction window records `session_id`, covered ordinal range, raw source message refs, provider, model, prompt prefix hash, tools schema hash, compression trace id, and extraction version.
- The `compressed_message` is a trigger and context artifact only. It is not cognition evidence.

Prefix reuse shape:

```text
[stable system message]
[stable session profile context]
[same LLM-visible session context used by compact]
[same tools schema]
--- shared provider-visible prefix ---
[different suffix instruction for extraction]
```

Tools are part of the provider-visible prefix and must remain stable for cache reuse. `tool_choice` is not part of the prefix and may differ, but compact and extraction should avoid changing tools schema. If extraction needs structured output, prefer direct JSON output or a generic structured-output mechanism that is already part of the stable tools schema. Do not generate per-worker dynamic tools schemas in the cached prefix.

The extraction worker runs after the compaction call has finished, so it cannot reuse that call's in-memory prompt. It must deterministically reconstruct the same prefix from durable inputs: the stored raw `session_messages` for the covered range, the same stable system message, and the same stable tools schema. Prefix reuse is only real if reconstruction is byte-stable, which is why the source window records the prompt prefix hash and tools schema hash. If reconstruction cannot reproduce the recorded prefix hash, the worker still extracts correctly; it just does not get the cache discount. The handover compression code currently builds its own prompt through `build_handover_compression_prompt_*`; extraction must share the same assembled session context and tools schema as compaction, not a parallel rendering that happens to look similar.

Source-window priority:

1. **Compact fast path**: process windows produced by `handover_compression.completed`.
2. **Inactive backlog path**: process sessions that did not trigger compact after they become operationally inactive.
3. **Manual force**: process a selected window for operator/debug runs.

Inactive backlog path:

- This path handles short or medium sessions that never reach compact thresholds.
- It also handles raw messages/traces that remain pending after compact-window de-duplication.
- It should not compete with active sessions that are likely to enter handover compression soon.
- A session is eligible when it has no active foreground turn, no pending handover maintenance for the same ordinal range, no active compact fast-path extraction window for that range, and is no longer in the daemon's active-session working set or has otherwise been explicitly closed/archived.
- This inactivity check is an operational scheduling gate. It is not a cognition refresh trigger and does not cause time-only extraction.

Multiple compacts in one session:

- Later compact prompts may include earlier `compressed_message` content as part of the stable LLM-visible context. This is allowed for continuity and prefix stability.
- Earlier `compressed_message` content is context-only for extraction. It must not become new evidence for new atomic beliefs.
- The eligible extraction evidence for a compact window is the raw message/trace portion not already marked extraction-processed in the ledger.
- Related existing beliefs and previous extraction coverage may be appended after the shared prefix as duplicate guards. That dynamic appendix may reduce cache coverage after the prefix, but it must not change the shared prefix itself.
- Extraction remains id-less draft generation only. It should avoid duplicate drafts when existing beliefs already cover the content, but update, strengthen, supersede, retract, and confirmation decisions belong to consolidation.

Duplicate control:

- Compact fast path and inactive backlog path share the same processing ledger and idempotency rules.
- If backlog extraction already processed an ordinal range, a later compact fast-path window for the same range is a no-op or processes only the uncovered suffix.
- If compact fast path processed an ordinal range, inactive backlog must skip that covered range.
- Program-attached evidence uses the selected raw source window, not `compressed_message` ids.

## Daemon Background Lifecycle Contract

Automatic background cognition is owned by the daemon lifecycle. It is not started by `AlphaAgent.respond()`, not hidden inside memory tools, and not implied by manual cognition CLI commands.

Target configuration:

```toml
[cognition.background]
enabled = true
interval_seconds = 300
startup_delay_seconds = 30
tick_timeout_seconds = 60
intake_min_sources = 4
source_batch_size = 20
extraction_min_sources = 4
extraction_batch_size = 12
consolidation_min_drafts = 1
consolidation_batch_size = 12
conflict_review_min_items = 1
summary_initial_min_beliefs = 5
summary_changed_min_beliefs = 3
summary_invalidated_source_min = 1
summary_batch_size = 2
```

`enabled = true` is the target default because background cognition is part of the core memory loop. This default applies to the new LLM-mediated `BackgroundCognitionService` after the shared state-write service and daemon coordinator integration exist. The existing manual command path remains available for forced operator/debug runs, but it does not define daemon automatic behavior.

Daemon lifecycle requirements:

- On daemon start, create one daemon-owned `BackgroundCognitionService`.
- If `[cognition.background].enabled = false`, the service remains disabled and does not schedule ticks.
- If `[cognition.background].enabled = true`, the service starts after `startup_delay_seconds`, then attempts ticks at `interval_seconds`.
- Each tick runs bounded worker chunks and must respect `tick_timeout_seconds`.
- Each tick uses the daemon-owned shared `LoopCoordinator`.
- Foreground user turns have priority over background chunks.
- Graceful shutdown stops scheduling new ticks and waits for the current chunk to finish or yield.
- Immediate shutdown stops scheduling new ticks and asks the current chunk to yield as soon as possible.
- Background failures are recorded in daemon status and audit logs, but do not crash foreground turn handling unless they corrupt shared infrastructure.

Tick semantics:

- `interval_seconds` is only a scheduling cadence for checking eligible work.
- A tick does not have to run the full pipeline from source intake through summary generation.
- A tick runs bounded eligible chunks until `tick_timeout_seconds`, foreground priority, shutdown, or no eligible work stops it.
- Workers advance through persistent checkpoints and cursors.
- Manual operator/debug runs may force one pass, but normal daemon ticks must respect the same chunk boundaries and validation rules.

Gate model:

- The gate substrate is new, not the existing scheduler. The current `Scheduler` and `ScheduleTrigger.watches` gate by counting `CognitiveEventKind` events after a checkpoint over the event log. The target gates count unintegrated raw `session_messages` / traces and changed belief sets, tracked through the processing ledger. The compact fast-path trigger, `handover_compression.completed`, is written as a `runtime_traces` row, not a cognitive event, so the old event-kind watcher cannot see it. Reuse the scheduler's coordinator-acquire, yield, chunk-budget, and checkpoint primitives; replace its event-count eligibility with ledger-driven eligibility. Do not try to express raw-message gates as `CognitiveEventKind` watches.
- Every background stage is gate-driven from the raw-message layer upward.
- The daemon timer never means "refresh cognition now"; it only means "evaluate gates now".
- A stage runs only when its own source set has enough new or changed material.
- Each stage records a persistent cursor over the lower layer it consumes.
- Each stage computes eligibility from counts after that cursor, plus material invalidation counts where applicable.
- No LLM-mediated stage runs because elapsed wall-clock time alone passed.
- Time-sensitive content should be represented in atomic belief validity and handled by recall/lifecycle filtering, not by time-only background regeneration.

Layer gates:

| Layer | Source Set | Target Unit | Eligibility |
| --- | --- | --- | --- |
| Source intake | Raw `session_messages` and runtime/tool traces | Source window by session/counterpart/project | `intake_min_sources` unintegrated records, or a manual forced run |
| Memory extraction | Source windows | Extracted atomic belief draft batch | `extraction_min_sources` unextracted source records in the target window |
| Memory consolidation | Extracted drafts and active atomic beliefs | Consolidation batch by scope/about/domain | `consolidation_min_drafts` unconsolidated drafts |
| Conflict review | Queued conflicts | Conflict batch by scope/about/domain | `conflict_review_min_items` pending conflicts |
| Summary generation | Consolidated atomic beliefs | `summary_kind + scope + about + optional target_domain` | Initial, changed-source, or invalidated-source quantity gates |

Default worker order inside one tick:

1. Intake raw `session_messages` and runtime/tool traces into source windows.
2. Extract id-less `atomic_belief_draft` records from eligible source windows.
3. Consolidate extracted drafts against active atomic beliefs.
4. Review conflicts that cannot be safely applied directly.
5. Generate summary beliefs for eligible summary targets.

This order is dependency order, not a guarantee that every tick reaches every stage. If earlier stages use the tick budget, later stages wait for later ticks.

Eligibility gates:

- **Source intake** runs when at least `intake_min_sources` unintegrated raw messages or traces exist for a target window. It processes at most `source_batch_size` source records per chunk.
- **Memory extraction** runs when an unextracted source window has at least `extraction_min_sources` records, or when a manual forced run explicitly asks to process a smaller tail window. It processes at most `extraction_batch_size` source records per chunk.
- **Memory consolidation** runs when at least `consolidation_min_drafts` extracted belief drafts exist for a target scope/about/domain. It processes at most `consolidation_batch_size` drafts per chunk.
- **Conflict review** runs when at least `conflict_review_min_items` queued conflicts exist for a target scope/about/domain. It processes bounded conflict chunks and may leave unresolved conflicts pending user confirmation.
- **Summary generation** runs only for summary targets that pass the summary gates below. It processes at most `summary_batch_size` summary targets per chunk.

Summary target identity:

```text
summary_kind + scope + about + optional target_domain
```

Summary generation gates are quantity/change gates only:

- **Initial summary gate**: no active summary exists for the target, and at least `summary_initial_min_beliefs` active consolidated atomic beliefs support that target.
- **Changed-source gate**: an active summary exists, and at least `summary_changed_min_beliefs` relevant source beliefs have been created, strengthened, superseded, retracted, or otherwise materially changed since the summary's recorded source cursor.
- **Invalidated-source gate**: an active summary exists, and at least `summary_invalidated_source_min` source beliefs used by that summary have been superseded, retracted, archived, or expired.

There is no time-only gate for any LLM-mediated background stage. Time can determine when the daemon checks for work, but elapsed time alone must not extract, consolidate, resolve, or summarize cognition. If content is time-sensitive, preserve validity on the underlying atomic beliefs and handle freshness during recall or lifecycle filtering. A summary updates only when the relevant belief set changes enough to pass a quantity/change gate.

## Processing Ledger

Background checkpoints and cursors are not enough by themselves. The system must be able to answer which raw records have been processed, which stage processed them, which records are still pending, and which failures need retry or inspection.

Use a sidecar processing ledger instead of mutating raw `session_messages` or trace records. Raw records remain durable source material; processing state belongs to background cognition.

Minimum ledger entities:

```text
background_source_progress
background_source_window
background_stage_run
```

`background_source_progress` tracks per-source, per-stage progress:

```text
source_type        session_message | runtime_trace | tool_trace | atomic_belief | summary_belief | conflict
source_id
stage              intake | extraction | consolidation | conflict_review | summary
target_unit        session/counterpart/project/scope/about/domain/window/summary target
status             pending | claimed | processed | failed | skipped
attempts
last_error
claimed_by
claimed_at
processed_at
checkpoint_id
idempotency_key
```

`background_source_window` records the exact source set selected for an LLM call:

```text
window_id
stage
target_unit
source_refs
created_at
closed_at
status             pending | claimed | processed | failed | skipped
idempotency_key
```

`background_stage_run` records one worker attempt:

```text
run_id
worker_id
stage
target_unit
window_id
status             started | succeeded | failed | yielded | skipped
started_at
finished_at
input_refs
output_refs
error
```

Ledger rules:

- "Processed" is stage-specific. A message can be processed by intake while still pending extraction.
- A stage only advances its checkpoint after the ledger and accepted outputs are written atomically.
- Failed LLM output does not mark sources processed and does not advance the stage cursor.
- Retried chunks reuse deterministic `idempotency_key` values derived from source refs, stage, target unit, and normalized accepted payload where applicable.
- Program-attached provenance in accepted beliefs and summaries must point back to ledger-tracked source windows, raw source refs, or selected belief ids.
- `skipped` requires an explicit reason, such as unsupported source type, empty content, duplicate source window, or user-deleted source.
- The ledger is operational state, not cognition state. It must not be rendered into answer prompts.

Daemon status should expose the background service state, at minimum:

```text
background_enabled
background_state     disabled | starting | idle | ticking | yielding | stopping | error
background_last_tick_at
background_last_success_at
background_last_error
background_next_tick_at
```

Legacy deterministic consolidation configuration must not be treated as proof that daemon automatic background cognition is enabled. The target daemon switch is `[cognition.background].enabled`; older consolidation-specific settings are worker internals or migration leftovers until replaced by the LLM-mediated pipeline. The default-on target must not auto-start the legacy deterministic consolidation loop.
