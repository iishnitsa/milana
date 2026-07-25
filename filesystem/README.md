# filesystem/ — product package (replaces monolithic filesystem.py)

Editable after pyinstaller (loaded from base_dir like `default_tools`).

| Module | Role |
|--------|------|
| `core.py` | worlds, dual-write, import-on-read, last_seen |
| `api.py` | `pipeline`, navigator helpers, lazy init, optional `@cacher` |
| `__init__.py` | public exports for `cross_gpt` / tools |

## Behaviour (agreed)

- **Main-first** — no auto fork on hierarchy
- **Import-on-read** — pull missing path from other world / disk
- **No auto-merge** on end_dialog (promote-on-success = TODO)
- **Optional** `fs_copy_touched_on_end` → `chat/dialog_artifacts/…` (default off)
- **Lazy init** on first API call
- **create_report** does not use this package

Legacy code: `filesystem_legacy.py` (not imported).
