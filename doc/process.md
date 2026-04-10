# Development Process

How we plan, build, and maintain LLM-RS. This process governs the human-Claude collaboration cycle.

---

## The Cycle

```
  Plan          Build          Close
  ----          -----          -----
  roadmap.md    TDD loop       tag version
  consensus     implement      hygiene pass
  scope phase   update docs    review & trim
      |             |              |
      v             v              v
  [phase N]    [commits]      [vN.x tag]
                                   |
                               next cycle
```

Three stages per phase, each with clear inputs and outputs.

---

## 1. Plan

**Input:** Previous phase complete, roadmap.md current.

Planning has three steps: intake review, triage, and commitment.

### 1a. Intake review

The Future Work section in roadmap.md is the intake buffer --- new ideas, discovered needs, and carry-over items land here unsorted throughout the cycle. During Plan, review everything in Future Work plus any incomplete items from the previous phase.

New items can be added to Future Work at any time (during build, between sessions, from external input). No labels needed at intake --- just capture the item so it isn't lost.

### 1b. Triage

Pull candidate items from Future Work into the phase's Remaining list and assign priority labels:

- `[must]` --- phase fails without it. These define "done" for the phase.
- `[should]` --- expected but deferrable. Will ship if time permits.
- `[could]` --- nice-to-have. First to be cut.

Order within each tier matters --- first item = highest priority within that tier.

### 1c. Commitment

- Agree on the phase goal --- one sentence describing the vertical slice this phase delivers.
- For design questions, discuss and record decisions in `doc/design/architecture.md`.
- For new specs (protocols, wire formats), write them in `doc/spec/` before implementing.

**Output:** roadmap.md updated with the phase's scope and prioritized items. Shared understanding of what "done" means.

**Rules:**
- A phase is a usable vertical slice, not a horizontal layer. It should be dogfood-able.
- Keep phase scope to what fits in a single version tag. If it feels too big, split it.
- All `[must]` items must be complete to tag the phase version. `[should]` and `[could]` items that don't make it roll back to Future Work or forward to the next phase.
- Items explicitly not wanted go to Parked with a reason.

---

## 2. Build

**Input:** Phase scope agreed in roadmap.md.

**Activities:**
- Work bottom-up through crate layers (core -> providers/store -> CLI).
- Strict TDD: failing test first, implement, refactor. `cargo test --workspace` and `cargo clippy --workspace` gate every commit.
- Update `CLAUDE.md` as implementation progresses (new types, APIs, flags, test counts).
- Record gotchas and non-obvious workarounds in `doc/implementation.md` as they arise --- don't batch these, write them while the context is fresh.

**Output:** All `[must]` items implemented, all tests green, docs current.

**Multi-session continuity:** Phases typically span multiple Claude Code sessions. At the start of each session, re-establish context by reading `CLAUDE.md`, recent `git log`, and the current phase section of `roadmap.md`. These are the continuity mechanism --- not memory or prior conversation.

**Rules:**
- One concern per commit. Commit messages explain *why*, not *what*.
- Lightweight roadmap updates are fine during build: parking items, adjusting priority labels, adding newly discovered items to Future Work (intake). Don't triage or restructure mid-phase.
- If scope changes mid-phase (item harder than expected, new insight), discuss before adjusting.

---

## 3. Close (Phase-Boundary Hygiene)

**Input:** Phase implementation complete (all `[must]` items done), ready to tag.

**Activities:**

### 3a. Tag the version

```bash
git tag vN.x <commit>
```

### 3b. Hygiene checklist

| Doc | Action |
|-----|--------|
| `doc/roadmap.md` | Update phase status table. Promote Remaining to Done. Move incomplete `[should]`/`[could]` items back to Future Work (they'll be re-triaged in the next Plan). Trim completed phase detail. |
| `CLAUDE.md` | Audit against actual code (see [audit checklist](#claude-md-audit-checklist) below). |
| `doc/design/architecture.md` | Add any new design decisions made during the phase. |
| `doc/implementation.md` | Preserve gotchas and workarounds. Trim build-log narrative that git history already covers. |
| `doc/process.md` | Review only if the cycle felt off this phase. |
| `README.md` | Update only at major milestones (human decides). |

### 3c. CLAUDE.md audit checklist

Concrete checks to verify CLAUDE.md hasn't drifted from reality:

1. **Test counts** --- run `cargo test --workspace 2>&1 | grep "test result"` and compare against the counts listed in CLAUDE.md.
2. **Implementation status** --- do the "Phase N complete" paragraphs reflect what's actually done? Are there new completions to add?
3. **CLI commands and flags** --- spot-check `llm --help` and a few subcommand `--help` outputs against the documented flags.
4. **Key types and traits** --- grep for key type names (`Provider`, `Prompt`, `Response`, `ChainEvent`, etc.) and verify the described fields/methods still match.
5. **Build commands** --- verify the documented `cargo test`, `wasm-pack`, and `maturin` commands still work.

### 3d. Roadmap size check

If `doc/roadmap.md` exceeds ~150 lines, something needs to move:
- Completed phase detail -> `doc/implementation.md`
- Stable decisions -> `doc/design/architecture.md`
- Redundant content -> delete

A pre-commit hook enforces this: warns at 150 lines, blocks at 200.

### 3e. Review parked items

Briefly revisit Parked items. Has anything changed that makes a parked item worth reconsidering? If not, leave it.

**Output:** Clean docs, version tagged, ready for next Plan stage.

---

## Versioning

Versions follow phases: `v0.1` (Phase 1), `v0.2` (Phase 2), etc.

### Tagging rules

- **Phase version** (`vN.x`): tagged when all `[must]` items for that phase are complete. This is the primary tag.
- **Patch version** (`vN.x.1`, `vN.x.2`): tagged for meaningful incremental work between phases --- bug fixes, quick additions, or follow-up work that doesn't warrant a new phase. Each patch tag still gets a lightweight hygiene pass (at minimum: update roadmap status and verify CLAUDE.md test counts).
- **Don't tag arbitrarily.** A tag marks a coherent milestone, not just "some work was done." If you can't write a one-sentence summary of what the tag delivers, it's not ready to tag.

### Reducing partial-phase tags

The v0.4 tag was applied to a partially complete phase. To avoid this:

- During Plan, scope `[must]` items tightly. It's better to have a small phase fully done than a large phase half done.
- If a phase is taking too long, split it: close the done part as the current phase, move remaining items to a new phase with its own version.
- Patch tags (`v0.4.1`) are the escape hatch for shipping incremental work without inflating phase numbers.

---

## Document Roles

Each document has one job and one rate of change:

| Doc | Job | Changes when |
|-----|-----|-------------|
| `doc/roadmap.md` | What we're building and when | Phase boundaries |
| `doc/design/architecture.md` | Why we made structural decisions | New design decisions |
| `doc/spec/*.md` | Protocol and format specifications | Spec changes |
| `doc/implementation.md` | Pitfall journal --- gotchas and workarounds | During implementation |
| `doc/process.md` | How we work | When the cycle feels off |
| `CLAUDE.md` | LLM working context --- types, APIs, conventions | During implementation |
| `README.md` | External-facing project overview | Major milestones |

**Key principle:** documents separated by rate of change don't rot. Mixing stable rationale with fast-changing status guarantees staleness.

---

## Anti-patterns

- **"Update docs later."** Write gotchas in implementation.md while the context is fresh. Batching doc updates leads to forgotten details.
- **"The doc is the source of truth for code details."** No --- the code is. CLAUDE.md is a *cache* for LLM context. When it conflicts with code, the code wins and the doc gets fixed.
- **"Add it to roadmap.md for now."** If it's a design decision, it goes in architecture.md. If it's a spec, it goes in spec/. If it's a learning, it goes in implementation.md. Roadmap.md is only for *what to build and when*.
- **"Keep it in case we need it."** Completed phase checklists, historical inner-loop tables, and workspace scaffolding instructions are dead weight. Git history preserves everything. Delete freely.
- **"Tag it now, clean up later."** A tag without a hygiene pass creates drift. Even patch tags get a lightweight check.
