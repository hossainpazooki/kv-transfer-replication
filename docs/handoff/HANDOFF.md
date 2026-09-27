# Handoff index

Pointers only — no evidence lives here. Entries are immutable: a later session writes a new
dated brief, never edits an old one. Run `/rigor:pickup` against the newest entry; it
re-verifies the brief's claims rather than trusting them.

| date | brief | describes commit | one-line |
|---|---|---|---|
| 2026-08-23 | [kv-transfer-replication](2026-08-23-kv-transfer-replication.md) | none (repo has zero commits) | Package + all four replication steps built and RUN; 42 tests; 3/3 load-bearing claims survived refutation; first next step is to commit, since missing history is the sole UNEVALUABLE. |
| 2026-08-24 | [cache-economics-followon](2026-08-24-cache-economics-followon.md) | none (build is uncommitted; newest commit 8e286b2 is the spec only) | WP1-WP3 apparatus built + reviewed, 139 tests; Runs 5-7 recorded (Run 6 resolved k=1 vs k=4 with no new data); WP3 killed before P=2048, WP2 dump half lost. First step is to commit. |
| 2026-08-25 | [followon-committed-supersedes-anchor](2026-08-25-followon-committed-supersedes-anchor.md) | f359445 | CORRECTION: the 2026-08-24 brief says the build is uncommitted and anchors on 8e286b2; the operator committed it as 11 commits 71cd8cb..f359445 minutes later. Supersedes ONLY that brief's commit anchor and its "First: commit" item; everything else there stands. |
| 2026-09-25 | [holdover-instrument](2026-09-25-holdover-instrument.md) | 47bdd39 (branch holdover-instrument; tasks 1+3 still uncommitted in the working tree) | Four-task instrument build (Pair revision/local_path, device()+KVT_DEVICE, --dtype, CI + entry points + --probe) done and green as a tree (165 tests); the operator's two pushed commits landed out of order and are each a broken tree (ImportError DTYPES); first step is the forward commit that completes them. |
