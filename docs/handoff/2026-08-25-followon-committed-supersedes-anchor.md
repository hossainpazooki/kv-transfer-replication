# Handoff — correction to the 2026-08-24 cache-economics brief: the build is committed

2026-08-25 (UTC 2026-08-25T03:20Z). **Newest commit this brief describes: `f359445`**
(`docs: Runs 5-7, six learnings entries, handoff brief`). Working tree clean,
`main` in sync with `origin/main`.

supersedes: [2026-08-24-cache-economics-followon.md](2026-08-24-cache-economics-followon.md)
— **only its commit anchor and its first Open/next item.** Everything else in that brief
(current state, locked decisions, reuse map, invariants, and open items 1-3) stands unchanged
and is still the document to read. This entry exists because that brief's headline fact flipped
minutes after it was written, and a `/rigor:pickup` acting on the stale version would do
redundant work.

## Current state

**`built` — the follow-on build is COMMITTED AND PUSHED.** The superseded brief says
"Everything below that is tagged `built` is UNCOMMITTED working tree", names `8e286b2` as the
newest commit, and warns the work is "one `git checkout` from gone". All three were true when
written (UTC 2026-08-25T02:10Z) and are now false: the operator ran the emitted commit plan at
2026-08-24 23:00 local, landing **11 commits** `71cd8cb..f359445` on top of `8e286b2`, one per
concern, with the two overwrite-guard fixes and the reporting fail-open fix kept separate from
features and the new results separate from code.
`re-verify:` `git log --oneline 8e286b2..HEAD | wc -l` — expect `11`; and
`git status --short --untracked-files=all -- kvt scripts tests results` — expect empty.
(Scoped to the build's own paths on purpose: an unscoped `git status` is dirtied by THIS
brief and its index row, so the unscoped form would report a false alarm the moment it was
written. The claim being verified is that the build is committed, not that the repo is
globally pristine.)

**`built` — the superseded brief and its learnings are themselves tracked.** They rode in
`f359445`, so the record of the session is in history rather than sitting in a working tree.
`re-verify:` `git ls-files --error-unmatch docs/handoff/2026-08-24-cache-economics-followon.md`
— expect the path echoed, exit 0.

**Unchanged: the suite and every run artifact.** Committing moved no numbers.
`re-verify:` `.venv/Scripts/python.exe -m pytest -q` — expect `139 passed`.

## Locked decisions

- **The superseded brief is not edited.** Its immutability is the reason a reader can trust that
  it describes a specific moment; the drift it now shows is exactly what `/rigor:pickup`'s
  re-verification exists to surface. Correcting it in place would have hidden a real state
  change behind a document that looked like it had always been right.
- **Everything else in that brief remains authoritative.** This entry deliberately does not
  restate its locked decisions, reuse map or invariants. Two copies of a decision drift; one
  does not.

## Reuse map

- [2026-08-24-cache-economics-followon.md](2026-08-24-cache-economics-followon.md) — read this
  for state, decisions, reuse and invariants. Substitute `f359445` for `8e286b2` wherever it
  names a commit anchor, and skip its "First: commit" item.
- `docs/ledger.md` — Runs 1-7, the pre-registered hypotheses, and the `[SUPERSEDED by Run 6]`
  markers.
- `docs/learnings/` — 13 entries, all passing the form gate.

## Invariants

- **A brief's commit anchor is a claim like any other, and it can go stale between writing and
  reading.** Measure drift from the anchor named at the top of the NEWEST entry in
  `HANDOFF.md`, not from the one you happen to open first.

## Open / next

**The superseded brief's item 1 is now the first item: run the P = 2048 perplexity sweep.** It
is the only pending run that answers a question nobody has answered, and Run 5 turned it into a
crossover question rather than a widening-lead one. Command and caveats are in that brief.

Its items 2 (re-run the 420-sequence target dump; `source` is intact) and 3 (WP1, which needs
the Qwen3-4B checkpoint downloaded) follow unchanged. Its "First: commit" item is **done** and
should be skipped.
