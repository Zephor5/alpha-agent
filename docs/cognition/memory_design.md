# Memory Design

This document describes the current Alpha memory design. It is intentionally
smaller than a full layered memory system: runtime turns keep transcript state,
beliefs are direct SQLite entities, and cognition events are audit/source
records rather than the canonical belief store.

## Current Baseline

Alpha keeps three separate memory surfaces:

- `session_messages`: source transcript records used for ordinary prompt
  continuity and handover compression.
- `atomic_beliefs`: first-order long-term belief entities written by explicit
  memory tools or retained workers.
- `summary_beliefs`: compact profile and maintenance summary entities. Only
  `counterpart_profile` can become stable prompt context through the explicit
  session profile snapshot path.

`cognitive_events` remains append-only, but belief lifecycle is not rebuilt from
belief events. Memory writes mutate the belief tables directly and may emit
`memory_proposed` as an audit record for the tool decision.

## Belief Entities

Atomic beliefs carry:

- `id`
- `subject`
- `about`
- `object`
- `content`
- `memory_kind`: `fact`, `preference`, `constraint`, `procedure`, `value`, or
  `relationship`
- `scope`: `global`, `counterpart`, `self`, `project`, or `session`
- `authority`
- `lifecycle`: `pending_confirmation`, `active`, `superseded`, `retracted`, or
  `archived`
- `sources`
- `relations`
- validity and supersession metadata

Summary beliefs use the same common fields, but replace `memory_kind` with
`summary_kind` and may point back to source belief ids.

Model construction validates structured source references and typed belief
relations before records can be stored. Invalid source entries, invalid relation
targets, and atomic/summary kind mismatches fail at the model boundary.

## Scopes

Scope determines the applicability boundary:

- `global`: general project-independent memory.
- `counterpart`: memory about one counterpart and requiring a matching
  counterpart reference in `about`.
- `self`: memory about the agent subject.
- `project`: memory about a named project.
- `session`: memory about one session.

Default recall with a counterpart returns matching counterpart beliefs and, when
requested, global beliefs. Explicit scope filters are authoritative: requesting
`self`, `project`, or `session` does not also force a counterpart/global filter
just because a counterpart context exists.

## Read Path

`memory_recall` is the only model-facing dynamic lookup tool. Runtime injects a
read context into the tool executor, but the model decides when to call it.

The tool returns compact handles:

- belief id
- content
- memory kind
- result scope
- lifecycle
- held timestamp

The tool does not expose source records, relation records, or internal ranking
features. It only returns active atomic beliefs.

Stable counterpart profile context is assembled before the provider call from
summary beliefs. It is session-stable prompt context, not a dynamic lookup
result.

## Write Path

`memory_propose` is the explicit write gate. The model submits one or more
updates with:

- operation: `append_distinct`, `reinforce`, `replace`, `merge`, `correct`, or
  `retract`
- memory kind
- content and evidence when a new belief is needed
- optional target belief ids
- optional reviewed candidate ids

Accepted append operations insert active atomic beliefs. Reinforce operations
add a new source to the existing belief. Replace and merge operations insert the
new belief and mark target beliefs as superseded. Retract marks active target
beliefs as retracted. Correct writes a pending-confirmation belief and asks the
model to request user approval.

`memory_proposed` records the tool gate decision for audit and source linkage;
it is not replayed into belief state.

## Indexes

The belief projection maintains:

- `belief_about_index` for scoped `about` lookup.
- `belief_entity_index` for object/about entity lookup.
- FTS term and trigram indexes for query recall.

Inactive beliefs are removed from the FTS indexes when their lifecycle changes,
but their canonical rows remain available for direct id lookup and audit.

## Background Work

Daemon background cognition is automatic when `[cognition.background].enabled`
is `true`, which is the default. The daemon creates one subject-level
`LoopCoordinator` and shares it between foreground `AlphaAgent` instances and
the background service, so foreground turns can observe or defer background
chunks through the same priority boundary.

The background service treats `interval_seconds` as a gate-check cadence, not a
semantic refresh trigger. Each tick runs bounded eligible chunks sourced from
lower-layer material:

- source intake from raw `session_messages` and `runtime_traces`;
- LLM extraction from unprocessed raw source windows;
- LLM consolidation from extracted atomic belief drafts and active beliefs;
- conflict review from queued conflict windows;
- direct archival of expired active beliefs.

Processing state lives in the sidecar ledger tables
`background_source_progress`, `background_source_window`, and
`background_stage_run`; raw session messages and traces are not mutated to track
background progress. Extraction, consolidation, and profile/domain/self summary
outputs are cognition-maintenance artifacts, not answer-path prompt context by
default.

## Prompt Use

The prompt builder uses:

- session transcript context and handover compression for continuity.
- stable summary beliefs for counterpart profile context when available.
- explicit `memory_recall` calls for dynamic lookup.

Runtime does not silently search long-term memory from the user message and does
not inject dynamic recall results unless the model calls the tool. It also does
not inject background source windows, stage runs, audit records, extraction
outputs, consolidation outputs, or non-profile summary outputs by default.
