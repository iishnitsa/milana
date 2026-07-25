"""
Milana project filesystem (git-backed, path-only for agents).

Loaded from base_dir like default_tools (editable after pyinstaller).
Lazy-inits on first pipeline / tree / change_dir call.

Public surface matches former filesystem.py imports used by cross_gpt.
"""

from .api import (  # noqa: F401
    allowed_actions,
    change_dir,
    copy_touched_to_folder,
    create_experiment_branch,
    get_dependency_report,
    get_project,
    get_project_tree_json,
    make_result,
    normalize_action,
    open_project,
    pipeline,
    resolve_workspace_path,
    set_session,
    status_failed,
    status_failed_text,
    status_forbidden,
    status_forbidden_text,
    status_success,
    status_success_text,
    to_posix_rel,
    FsSession,
)

# Compat aliases some tools expect
status_success_text = status_success
status_failed_text = status_failed
status_forbidden_text = status_forbidden

__all__ = [
    "pipeline",
    "change_dir",
    "get_project_tree_json",
    "get_dependency_report",
    "create_experiment_branch",
    "resolve_workspace_path",
    "to_posix_rel",
    "normalize_action",
    "allowed_actions",
    "make_result",
    "open_project",
    "get_project",
    "set_session",
    "copy_touched_to_folder",
    "status_success",
    "status_failed",
    "status_forbidden",
    "status_success_text",
    "status_failed_text",
    "status_forbidden_text",
    "FsSession",
]
