ts: 2026-09-25T04:24:48Z
commit: 47bdd39 (HEAD at capture; working tree carried the uncommitted task 1 + 3 files)
session: Claude Code session a0bc4703-7c59-4b7f-bcf2-e44daa6e7ffa, task 4 reinstall step ("reinstall and confirm kvt-dump --help")
status: verified
fact: This repo's `.venv` was created by uv 0.12.5, which ships no `pip` module inside the venv, and the `uv` binary is not on the PATH that Claude Code's Bash or PowerShell tools see. So `python -m pip install -e .` fails with `No module named pip` and `uv pip install` fails with `command not found`, and a `pyproject.toml` change that adds console scripts cannot be installed by either obvious route. `python -m ensurepip --upgrade` inside the venv bootstraps pip 25.0.1 and unblocks it; this leaves pip in the venv permanently, which uv does not mind.
basis: `grep -E "^uv" .venv/pyvenv.cfg` printed `uv = 0.12.5`; `command -v uv || echo "uv: absent from PATH"` printed `uv: absent from PATH`; after the earlier `ensurepip` (which printed `Successfully installed pip-25.0.1`), `.venv/Scripts/python -m pip --version` printed `pip 25.0.1 from C:\Users\hossa\dev\kv-tr...`. Before ensurepip the same command had printed `.venv\Scripts\python.exe: No module named pip`.
re-verify: sh -c 'grep -E "^uv" .venv/pyvenv.cfg; command -v uv || echo "uv: absent from PATH"; .venv/Scripts/python -m pip --version | cut -c1-12'
