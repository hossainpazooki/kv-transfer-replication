ts: 2026-09-25T04:24:00Z
commit: 47bdd39 (HEAD at capture; the crash itself is a property of 063f402's scripts/dump_kv.py, which the current tree has already patched)
session: Claude Code session a0bc4703-7c59-4b7f-bcf2-e44daa6e7ffa, task 1 done-line check ("--help shows both flags")
status: verified
fact: `python scripts/dump_kv.py --help` had been crashing since the `--out` help text was written, because argparse %-formats every help string and that string contains a bare `20%,`. Nothing caught it: no test ran that script's `--help` (the other scripts' `--help` are tested in tests/test_scripts.py, this one was not), and the flag surface was only ever exercised by real runs that never pass `-h`. A done-line that says "--help shows the flag" is the first thing that would have found it. Fixed by `20%%`.
basis: In a detached worktree at 063f402, `PYTHONPATH=<wt> .venv/Scripts/python scripts/dump_kv.py --help 2>&1 | tail -1` printed `ValueError: unsupported format character ',' (0x2c) at index 317`. `git show 063f402:scripts/dump_kv.py | grep -c "20%,"` printed `1`; `git show HEAD:scripts/dump_kv.py | grep -c "20%%,"` printed `1` (HEAD = 47bdd39, whose dump_kv.py came in via e54c102).
re-verify: sh -c 'git show 063f402:scripts/dump_kv.py | grep -c "20%,"; grep -c "20%%," scripts/dump_kv.py'
