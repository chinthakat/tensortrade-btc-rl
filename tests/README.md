# tests/

There is no automated test suite here, and no CI.

The 38 files in this directory are standalone scripts, each written to check one
behaviour by hand: it builds an environment or loads a trade log, prints what it
found, and exits. Nothing uses `pytest` — the files are named `test_*.py` and
`verify_*.py` by convention only, and most execute work at import time, so
`pytest tests/` will not do anything useful.

A further 23 files here were committed as empty placeholders and have been
removed; if a document elsewhere points you at a test script that no longer
exists, that is why.

Run one at a time, from the repository root:

```bash
python -m tests.test_system
```

`test_system.py` is the closest thing to a smoke test: it checks that the
required packages import, that the environment can be constructed, and that a
few data utilities behave. It is what `setup_windows.ps1` and
`setup_windows.bat` try to run after installing dependencies (they still invoke
it as `python test_system.py`, from when this file lived in the repository
root).

Many of the other scripts are stale. They reference CSV files under `data/` or
`episodes/` that are not committed, or constructor arguments that have since
been removed. Treat a failure as "this script has drifted" rather than "the
environment is broken", and check the code before trusting the script.
