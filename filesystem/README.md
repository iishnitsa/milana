# filesystem/ — product modules (no `__init__.py`)

Editable after freeze (loaded from `base_dir` like `default_tools`). Namespace package: import **submodules**, not package root.

| Module | Role |
|--------|------|
| `core.py` | worlds, dual-write, import-on-read, last_seen |
| `api.py` | `pipeline`, navigator helpers, lazy init, optional `@cacher` |
| `tool_arg.py` | `one_line_arg` / `clean_tool_arg` |

## Import path

`cross_gpt.initialize_work` inserts `base_dir` at the front of `sys.path`, then:

```python
from filesystem.api import pipeline, change_dir, …
from filesystem.tool_arg import one_line_arg
```

No `__init__.py` — tools use `filesystem.api` / `filesystem.tool_arg` after worker path setup.

**Do not** add `--hidden-import filesystem*` to the freeze: only `launcher.py` is in the
exe; `filesystem/` stays editable next to the binary.

## Behaviour

- **Main-first** — no auto fork on hierarchy
- **Import-on-read** — pull missing path from other world / disk
- **No auto-merge** on end_dialog
- **Session default** — when tools omit `session_id`, pipeline uses `global_state.now_try` (dialog try id) so `fs_copy_touched_on_end` sees the same session as writes
- **Optional** `fs_copy_touched_on_end` → `chat/dialog_artifacts/…` (default off)
- **Optional** `fs_use_git` (default on). Off → plain disk CRUD without git/worlds
- **Lazy init** on first API call
- **create_report** does not use this package

Legacy `filesystem_legacy.py` has been **removed**.
