# Belief Ontology

## Stable Belief Ontology

### Definition

A `Belief` is a durable cognitive assertion held by the subject. It is not just a text memory. It must preserve:

- What kind of assertion it is, for atomic beliefs.
- What kind of summary it is, for summary beliefs.
- Who or what it is about.
- Where it came from.
- How authoritative it is.
- When it applies.
- How it should be updated or invalidated.
- What evidence supports it.

### Storage Shape

Use separate persisted entity types for atomic and summary beliefs:

```text
atomic_beliefs
summary_beliefs
```

They are both part of the `Belief` family, but the table split keeps invariants direct:

- Atomic beliefs always have `memory_kind` and never have `summary_kind`.
- Summary beliefs always have `summary_kind` and never have `memory_kind`.
- Recall defaults to atomic beliefs.
- Profile snapshot loading defaults to summary beliefs.

### Common Fields

Both atomic and summary beliefs share:

```text
derivation_stage   tool_written | background_extracted | background_consolidated | background_summarized | human_confirmed
scope              global | counterpart | self | project | session
about              typed references for the subject matter
content            natural-language assertion
structure          optional structured claim
authority          user_asserted | human_confirmed | system_defined | llm_interpreted | background_synthesized
sources            evidence references
confidence         numeric confidence, subordinate to authority
validity           observed_at, valid_from, valid_until, recurrence
relations          typed links to other beliefs
update_policy      how conflicts and refreshes should be handled
lifecycle          pending_confirmation | active | superseded | retracted | archived
```

### Atomic Belief Fields

```text
memory_kind        fact | preference | constraint | procedure | value | relationship
```

### Summary Belief Fields

```text
summary_kind       counterpart_profile | project_profile | domain_summary | self_memory_summary
source_belief_ids  atomic or summary beliefs summarized by this record
```

### Entity Invariants

- Raw source material is not a belief layer. Raw data belongs to `session_messages`, runtime traces, and tool traces.
- Audit logs may reference raw sources and cognition entities, but they are not the source of current cognition state.
- `derivation_stage` records how the belief was produced. It is provenance, not a content classification.
- Search indexes may be maintained for beliefs, but indexes are implementation support, not a cognition layer.

Profile is not a `memory_kind`. Profile is a summary belief in `summary_beliefs` with `summary_kind=counterpart_profile` and `derivation_stage=background_summarized`, later frozen into a `session_profile_snapshot`.

### Atomic Memory Kinds

#### fact

A descriptive assertion about the world, project, user, environment, or state.

Examples:

- The project uses `uv`.
- The user is working on `alpha-agent`.
- The daemon currently owns runtime turns.

#### preference

A preferred style, default, or choice.

Examples:

- The user prefers concise answers.
- The user prefers holistic design over local fixes.
- The user prefers direct refactors over gradual compatibility layers.

#### constraint

A rule, prohibition, requirement, or hard boundary.

Examples:

- Do not write local machine-specific absolute paths into the repository.
- Do not read historical archive docs unless explicitly requested.
- Do not preserve compatibility with existing database data during this refactor.

`constraint` must be first-class. Mapping it to `procedure` loses critical semantics.

#### procedure

A reusable method, workflow, or operational process.

Examples:

- Before implementation, inspect the current code path.
- For validation, run `ruff`, `mypy`, and `pytest`.
- For memory changes, update belief entities with source refs and write an audit record for inspection.

#### value

A stable evaluation principle or tradeoff preference.

Examples:

- Correctness is more important than speed for core cognition design.
- The main loop contract is more important than exposing every internal state.
- User authority outweighs background inference in conflict resolution.

#### relationship

A durable relation among subject, counterpart, project, role, service, trust, or ownership.

Examples:

- A session is bound to a counterpart.
- A counterpart is the project owner.
- A project uses a specific runtime owner.

### Summary Kinds

Summary beliefs summarize other beliefs. They are not ordinary facts and should not be returned as normal recall results unless the query or tool explicitly requests summary records.

#### counterpart_profile

A stable profile-level summary for one counterpart. This is the source for `session_profile_snapshot`.

#### project_profile

A stable project-level summary, useful for project-specific recall or future project session bootstrapping.

#### domain_summary

A summary of a bounded domain, such as memory behavior, interaction style, recurring implementation constraints, or guidance for a specific tool or background worker. If it affects runtime behavior, the target domain is part of the summary belief's structured claim.

#### self_memory_summary

A memory-derived summary about the agent's own behavior, capabilities, failure patterns, interaction patterns, and recurring tradeoffs. This is the self-understanding layer; do not persist a separate cognition entity for it.

### Scope

Scope controls where a belief applies:

- `global`: applies generally.
- `counterpart`: applies to a specific user/counterpart.
- `self`: applies to the agent subject.
- `project`: applies to a project or repository context.
- `session`: applies only within one session.

`scope` must be paired with `about`. A counterpart-scoped belief should reference the counterpart. A project-scoped belief should reference the project. This keeps recall and profile generation precise.

For `project` scope, the LLM may derive a project descriptor from raw context. Program logic normalizes that descriptor and generates the project reference id. The LLM does not mint project ids.

### Authority

Authority is not the same as confidence.

Recommended authority order:

1. `system_defined`
2. `human_confirmed`
3. `user_asserted`
4. `background_synthesized`
5. `llm_interpreted`

Conflict resolution should prefer higher authority before considering confidence. A low-confidence user-stated constraint still deserves stronger handling than a high-confidence background inference.

Authority meanings:

- `system_defined`: a rule from system, project, config, or other trusted program boundary.
- `human_confirmed`: a belief accepted through an explicit confirmation flow, such as the system asking whether a candidate memory is correct and the user confirming it.
- `user_asserted`: a belief derived from something the user directly said, without an additional confirmation ceremony.
- `background_synthesized`: a belief synthesized from multiple pieces of evidence or long-window context by a background LLM worker.
- `llm_interpreted`: a weak belief inferred by an LLM without a direct claim or strong evidence chain.

`human_confirmed` and `user_asserted` are intentionally separate. A normal user statement can be remembered as `user_asserted`; it only becomes `human_confirmed` after a distinct confirmation interaction.

### Authority Ceiling

Program logic must enforce an authority ceiling based on source type. The LLM may propose authority, but the accepted authority cannot exceed the ceiling.

```text
source kind                              max accepted authority
system/project rule                      system_defined
explicit confirmation flow               human_confirmed
direct user statement                    user_asserted
background synthesis over evidence       background_synthesized
LLM interpretation without direct claim   llm_interpreted
```

Rules:

- Background LLM output cannot produce `human_confirmed`.
- Only an explicit confirmation flow can produce `human_confirmed`.
- Program logic determines the maximum authority from source channel, source kind, and worker stage. The LLM may propose an authority value inside that envelope, but it does not define the envelope itself.
- If LLM output claims authority above the source ceiling, reject the output instead of silently downgrading it.
- `confidence` cannot compensate for lower authority.

### Validity

Do not use a `temporal` memory kind. Time semantics should be explicit fields:

```text
observed_at
valid_from
valid_until
recurrence
```

This lets any memory kind be time-bound.

### Relations

Do not use `causal` or `social` as top-level memory kinds. Use typed relations:

```text
supports
contradicts
supersedes
causes
caused_by
derived_from
about_same_entity_as
```

Social semantics usually belong in `relationship` plus `about`. Causal semantics belong in `relations` or structured claim fields.

### Current Enum Replacement Boundary

This is not a compatibility or old-record migration rule. Existing belief records do not need to be preserved. The current `CognitiveType` enum should be removed because its values are conceptually misplaced:

```text
factual      replace with memory_kind=fact
preference   replace with memory_kind=preference
procedural   split into memory_kind=procedure and memory_kind=constraint
value        replace with memory_kind=value
social       replace with memory_kind=relationship
temporal     remove as a type; use validity fields
causal       remove as a type; use relations or structured claims
meta         remove as a top-level type; use summary beliefs where appropriate
concept      remove as a catch-all; use summary kinds or explicit abstract facts
```

Tool-level memory types should be redefined directly against the new ontology:

```text
factual      -> fact
preference   -> preference
constraint   -> constraint
procedure    -> procedure
```
