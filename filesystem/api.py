"""
Module-facing filesystem API (Milana shape).

Product modules (default_tools) call::

    from fs_api import pipeline, status_success, status_failed, status_forbidden

    result = pipeline(actor, path, action="edit", handler_arg={...}, repo_path=workspace)

This matches ``cross_gpt.pipeline`` / root ``filesystem.pipeline`` enough that
tools can be moved to default_tools with minimal change.

Layers
------
  default_tools module  main(text) -> str
       -> pipeline / helpers here
       -> fs_core.FileSystem
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .core import (
    CONFLICT_IMPORT,
    DEFAULT_WORLD,
    FAILED,
    FORBIDDEN_PATH,
    FORBIDDEN_POLICY,
    NOT_FOUND,
    OK,
    STALE,
    STAGING_DIR,
    FileSystem,
    FsResult,
)

# ---------------------------------------------------------------------------
# Status strings (same as production filesystem / cross_gpt)
# ---------------------------------------------------------------------------

status_success = "success"
status_failed = "failed"
status_forbidden = "forbidden"

# legacy aliases used in some tools
status_success_text = status_success
status_failed_text = status_failed
status_forbidden_text = status_forbidden

allowed_actions = ["create", "read", "edit", "delete", "copy", "move"]

# extended ops (not all exposed via classic pipeline positional form)
EXTENDED_ACTIONS = frozenset(
    {
        "write",
        "tree",
        "search",
        "snippet",
        "exists",
        "stat",
        "reconcile",
        "materialize",
        "finish",
        "ingest",
        "publish",
        "publish_touched",
    }
)

ActorFn = Callable[..., Any]

# repo_path str -> FileSystem
_REGISTRY: Dict[str, FileSystem] = {}
# default session per repo (agent binding later: set_session)
_SESSIONS: Dict[str, str] = {}
_DEFAULT_STAGING = "temp"  # or "worktree"


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def make_result(status: str, reason: str, action: str, **extra: Any) -> Dict[str, Any]:
    base = {
        "status": status,
        "ok": status == status_success,
        "reason": reason,
        "action": action,
        "ts": _now_ts(),
    }
    base.update(extra)
    return base


def _code_to_status(code: str) -> str:
    if code == OK:
        return status_success
    if code in (FORBIDDEN_PATH, FORBIDDEN_POLICY):
        return status_forbidden
    return status_failed


def fs_result_to_dict(r: FsResult, action: str = "") -> Dict[str, Any]:
    d = make_result(
        _code_to_status(r.code),
        r.message or r.code,
        action or "",
        path=r.path,
        source_path=r.path,
        target_path=r.path,
        content_hash=r.hash,
        created=r.created,
        world=r.world,
    )
    if r.details:
        d.update({k: v for k, v in r.details.items() if k not in d})
        if "data" in r.details:
            d["data"] = r.details["data"]
    if r.content is not None and "content" not in d:
        # keep binary out of huge dumps; text prefer
        if r.content_text is not None:
            d["text"] = r.content_text
        d["size"] = len(r.content)
    if r.code == STALE:
        d["code"] = STALE
    if r.code == NOT_FOUND:
        d["code"] = NOT_FOUND
    if r.code == CONFLICT_IMPORT:
        d["code"] = CONFLICT_IMPORT
    d["fs_code"] = r.code
    return d


def open_project(
    root: Union[str, Path],
    *,
    session_id: str = "default",
    actor_id: str = "module",
    staging: str = "temp",
    **kwargs: Any,
) -> "FsSession":
    """Open/create project root (usually chat_path/files)."""
    root_s = str(Path(root).resolve())
    if root_s not in _REGISTRY:
        _REGISTRY[root_s] = FileSystem(root_s, **kwargs)
    global _DEFAULT_STAGING
    _DEFAULT_STAGING = staging or "temp"
    _SESSIONS[root_s] = session_id
    core = _REGISTRY[root_s]
    core.session(session_id, actor_id=actor_id)
    return FsSession(core=core, session_id=session_id, actor_id=actor_id, root=root_s)


def get_project(repo_path: Optional[Union[str, Path]] = None) -> FileSystem:
    if repo_path is None:
        if len(_REGISTRY) == 1:
            return next(iter(_REGISTRY.values()))
        raise ValueError("repo_path required when multiple projects open")
    root_s = str(Path(repo_path).resolve())
    if root_s not in _REGISTRY:
        _REGISTRY[root_s] = FileSystem(root_s)
        _SESSIONS.setdefault(root_s, "default")
    return _REGISTRY[root_s]


def set_session(repo_path: Union[str, Path], session_id: str, *, actor_id: str = "module") -> None:
    root_s = str(Path(repo_path).resolve())
    get_project(root_s).session(session_id, actor_id=actor_id)
    _SESSIONS[root_s] = session_id


def _session_id(repo_path: str, explicit: Optional[str] = None) -> str:
    """Resolve session for pipeline/copy_touched.

    Prefer explicit → agent ``global_state.now_try`` (dialog try id) → registry → default.
    Binding writes to now_try fixes fs_copy_touched_on_end empty files when tools omit session_id.
    """
    if explicit:
        return explicit
    try:
        import cross_gpt as cg  # local import: package may load before worker

        nt = getattr(getattr(cg, "global_state", None), "now_try", None)
        if nt is not None and str(nt).strip() != "":
            return str(nt)
    except Exception:
        pass
    return _SESSIONS.get(repo_path, "default")


def _to_rel(path_value: Optional[str], workspace: Path) -> Optional[str]:
    if path_value is None or path_value == "":
        return None
    p = Path(str(path_value))
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(workspace.resolve())
            return rel.as_posix()
        except ValueError:
            return None  # outside sandbox
    s = str(path_value).replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    s = s.lstrip("/")
    # keep ".." segments so fs_core._norm_path rejects escape
    return s


def _make_stage(workspace: Path, session_id: str, rel_path: str, data: bytes, staging: str) -> tuple:
    suffix = Path(rel_path).suffix or ".bin"
    mode = (staging or _DEFAULT_STAGING or "temp").lower()
    if mode == "worktree":
        base = workspace / STAGING_DIR / session_id
        base.mkdir(parents=True, exist_ok=True)
        ign = workspace / STAGING_DIR / ".gitignore"
        if not ign.exists():
            ign.write_text("*\n", encoding="utf-8")
        name = f"{uuid.uuid4().hex[:10]}_{Path(rel_path).name}"
        stage = base / name
        stage.write_bytes(data or b"")

        def cleanup():
            try:
                if stage.exists():
                    stage.unlink()
            except OSError:
                pass

        return str(stage), cleanup

    fd, temp_name = tempfile.mkstemp(prefix="fs_mod_", suffix=suffix)
    try:
        os.write(fd, data or b"")
    finally:
        os.close(fd)

    def cleanup():
        try:
            os.unlink(temp_name)
        except OSError:
            pass

    return temp_name, cleanup


def _run_actor_on_bytes(
    *,
    workspace: Path,
    session_id: str,
    rel_path: str,
    action: str,
    data: bytes,
    actor_func: Optional[ActorFn],
    handler_arg: Any,
    staging: str,
) -> tuple:
    """Returns (status_dict_or_none, new_bytes_or_none, actor_out)."""
    if not callable(actor_func):
        return None, data, {}
    stage_path, cleanup = _make_stage(workspace, session_id, rel_path, data, staging)
    try:
        payload = {
            "temp_file": stage_path,
            "stage_file": stage_path,
            "staging": staging,
            "path": rel_path,
            "action": action,
            "arg": handler_arg if handler_arg is not None else {},
            "content": data,
            "source_path": rel_path,
            "target_path": rel_path,
        }
        try:
            # production call_actor tries (payload,) then (temp, arg) etc.
            try:
                out = actor_func(payload)
            except TypeError:
                try:
                    out = actor_func(stage_path, handler_arg)
                except TypeError:
                    out = actor_func(stage_path)
        except Exception as e:
            return (
                make_result(status_failed, f"actor error: {e}", action, path=rel_path),
                None,
                {},
            )
        if not isinstance(out, dict):
            out = {"status": status_success, "content": out}
        st = str(out.get("status") or status_success).lower()
        if st in ("forbidden", status_forbidden):
            return (
                make_result(
                    status_forbidden,
                    str(out.get("reason") or "forbidden"),
                    action,
                    path=rel_path,
                    data=out.get("data"),
                ),
                None,
                out,
            )
        if st not in (status_success, "ok", "success", ""):
            return (
                make_result(
                    status_failed,
                    str(out.get("reason") or "actor failed"),
                    action,
                    path=rel_path,
                    data=out.get("data"),
                ),
                None,
                out,
            )
        if "content" in out and out["content"] is not None:
            c = out["content"]
            if isinstance(c, str):
                new_b = c.encode("utf-8")
            else:
                new_b = bytes(c)
        else:
            new_b = Path(stage_path).read_bytes()
        return None, new_b, out
    finally:
        cleanup()


def resolve_workspace_path(
    repo_path=None,
    prefer_chat_project=True,
    project_dir_name="files",
):
    """Lazy: chat_path/files or explicit repo_path. No git init here."""
    if repo_path:
        return str(Path(repo_path).resolve())
    try:
        import cross_gpt as cg

        fp = getattr(cg, "filesystem_project_path", None) or ""
        if fp:
            return str(Path(fp).resolve())
        cp = getattr(cg, "chat_path", None) or ""
        if cp:
            return str((Path(cp) / project_dir_name).resolve())
    except Exception:
        pass
    if len(_REGISTRY) == 1:
        return next(iter(_REGISTRY.keys()))
    return str(Path(".").resolve())


def to_posix_rel(path_text):
    return str(path_text or "").replace("\\", "/")


def normalize_action(action):
    a = str(action or "").strip().lower()
    return {
        "update": "edit",
        "remove": "delete",
        "rename": "move",
        "replace": "edit",
        "write": "edit",
    }.get(a, a)


def _effective_use_git(use_git_kw: bool) -> bool:
    """Setting fs_use_git=0 forces plain disk (no git worlds/branches)."""
    try:
        import cross_gpt as cg
        if int(getattr(cg.global_state, "fs_use_git", 1) or 0) == 0:
            return False
    except Exception:
        pass
    return bool(use_git_kw)


def _safe_rel(path_text, workspace: Path) -> Optional[str]:
    if path_text is None or str(path_text).strip() == "":
        return None
    try:
        p = Path(str(path_text))
        if not p.is_absolute():
            full = (workspace / p).resolve()
        else:
            full = p.resolve()
        rel = full.relative_to(workspace)
        return rel.as_posix()
    except Exception:
        return None


def _pipeline_plain_disk(
    actor_func=None,
    filename=None,
    for_actor=None,
    action: str = "create",
    handler_arg=None,
    workspace: Path = None,
    *,
    session_id: Optional[str] = None,
    staging: Optional[str] = None,
    content: Optional[Union[str, bytes]] = None,
    force: bool = False,
    query: Optional[str] = None,
    include_disk: bool = False,
    mode: str = "both",
    **_ignored: Any,
) -> Dict[str, Any]:
    """
    Plain filesystem ops without git/worlds: direct files under workspace.
    Same result shape as git pipeline for tools.
    """
    action = normalize_action(action or "create")
    if action == "write":
        action = "edit" if filename else "create"
    ws = Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    sid = _session_id(str(ws), session_id)
    staging_mode = staging or _DEFAULT_STAGING

    def _abs(rel: str) -> Path:
        return (ws / rel).resolve()

    # tree / search (disk only)
    if action == "tree":
        files = []
        for root, _dirs, names in os.walk(ws):
            for n in names:
                fp = Path(root) / n
                try:
                    files.append(fp.relative_to(ws).as_posix())
                except Exception:
                    pass
        return make_result(status_success, "ok", action, files=files)
    if action == "search":
        q = (query or "").lower()
        hits = []
        if q:
            for root, _dirs, names in os.walk(ws):
                for n in names:
                    if q in n.lower():
                        try:
                            hits.append((Path(root) / n).relative_to(ws).as_posix())
                        except Exception:
                            pass
        return make_result(status_success, "ok", action, matches=hits)

    source_rel = _safe_rel(filename, ws)
    if filename and source_rel is None:
        return make_result(status_forbidden, "path outside project", action)
    target_rel = None
    if action in ("copy", "move"):
        target_rel = _safe_rel(for_actor, ws)
        if for_actor and target_rel is None:
            return make_result(status_forbidden, "path outside project", action)
    elif action == "create":
        target_rel = _safe_rel(for_actor, ws) if for_actor else source_rel
        if for_actor and target_rel is None:
            return make_result(status_forbidden, "path outside project", action)
        source_rel = target_rel or source_rel
    elif action == "edit" and for_actor:
        alt = _safe_rel(for_actor, ws)
        if for_actor and alt is None:
            return make_result(status_forbidden, "path outside project", action)
        if alt:
            target_rel = alt

    op_path = target_rel or source_rel
    if action in ("read", "edit", "delete", "copy", "move") and not source_rel:
        return make_result(status_failed, "path not specified", action)
    if action == "create" and not op_path:
        return make_result(status_failed, "path not specified", action)
    if action in ("copy", "move") and not target_rel:
        return make_result(status_failed, "target path not specified", action)

    if action == "delete":
        p = _abs(source_rel)
        if not p.exists():
            return make_result(status_failed, "not found", action, path=source_rel)
        try:
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
            else:
                p.unlink()
            return make_result(status_success, "ok", action, path=source_rel)
        except Exception as e:
            return make_result(status_failed, str(e), action, path=source_rel)

    if action == "copy":
        import shutil
        src, dst = _abs(source_rel), _abs(target_rel)
        if not src.exists():
            return make_result(status_failed, "not found", action, path=source_rel)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            return make_result(status_success, "ok", action, source_path=source_rel, target_path=target_rel)
        except Exception as e:
            return make_result(status_failed, str(e), action)

    if action == "move":
        import shutil
        src, dst = _abs(source_rel), _abs(target_rel)
        if not src.exists():
            return make_result(status_failed, "not found", action, path=source_rel)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return make_result(status_success, "ok", action, source_path=source_rel, target_path=target_rel)
        except Exception as e:
            return make_result(status_failed, str(e), action)

    if action == "read":
        p = _abs(source_rel)
        if not p.exists() or not p.is_file():
            return make_result(status_failed, "not found", action, path=source_rel)
        try:
            data = p.read_bytes()
        except Exception as e:
            return make_result(status_failed, str(e), action, path=source_rel)
        if callable(actor_func):
            err, _, out = _run_actor_on_bytes(
                workspace=ws, session_id=sid, rel_path=source_rel, action="read",
                data=data, actor_func=actor_func, handler_arg=handler_arg, staging=staging_mode,
            )
            if err:
                return err
            d = make_result(status_success, operation_success_text(), action, path=source_rel)
            d["data"] = out.get("data")
            return d
        d = make_result(status_success, operation_success_text(), action, path=source_rel)
        try:
            d["text"] = data.decode("utf-8")
        except Exception:
            d["text"] = None
        return d

    if action in ("create", "edit"):
        path = op_path or source_rel
        p = _abs(path)
        existing = b""
        if p.exists() and p.is_file():
            try:
                existing = p.read_bytes()
            except Exception:
                existing = b""
        if content is not None:
            existing = content.encode("utf-8") if isinstance(content, str) else content
        elif isinstance(handler_arg, dict) and "content" in handler_arg and not callable(actor_func):
            c = handler_arg.get("content")
            existing = c.encode("utf-8") if isinstance(c, str) else (c or b"")
        new_bytes = existing
        actor_out: Dict[str, Any] = {}
        if callable(actor_func):
            seed = existing
            if action == "create" and not seed and isinstance(handler_arg, dict):
                c = handler_arg.get("content")
                if isinstance(c, str):
                    seed = c.encode("utf-8")
                elif c is not None:
                    seed = bytes(c)
            err, new_bytes, actor_out = _run_actor_on_bytes(
                workspace=ws, session_id=sid, rel_path=path, action=action,
                data=seed or b"", actor_func=actor_func, handler_arg=handler_arg, staging=staging_mode,
            )
            if err:
                return err
            if new_bytes is None:
                return make_result(status_failed, "actor produced no content", action, path=path)
        elif isinstance(handler_arg, dict) and "content" in handler_arg:
            c = handler_arg.get("content")
            new_bytes = c.encode("utf-8") if isinstance(c, str) else (c or b"")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(new_bytes if new_bytes is not None else b"")
        except Exception as e:
            return make_result(status_failed, str(e), action, path=path)
        d = make_result(status_success, operation_success_text(), action, path=path, source_path=path, target_path=path)
        if actor_out.get("data") is not None:
            d["data"] = actor_out["data"]
        return d

    return make_result(status_failed, f"unsupported action (plain fs): {action}", action)


def _pipeline_impl(
    actor_func=None,
    filename=None,
    for_actor=None,
    action: str = "create",
    handler_arg=None,
    repo_path=None,
    cwd=None,
    git_folder=".git",
    use_git=True,
    intention=None,
    ask_model_fn=None,
    get_embs_fn=None,
    parse_yes_no_fn=None,
    current_branch=None,
    prefer_chat_project=True,
    experimental_mode=False,
    *,
    session_id: Optional[str] = None,
    staging: Optional[str] = None,
    content: Optional[Union[str, bytes]] = None,
    force: bool = False,
    query: Optional[str] = None,
    child_session: Optional[str] = None,
    include_disk: bool = False,
    mode: str = "both",
    **_ignored: Any,
) -> Dict[str, Any]:
    """
    Production-compatible entry. Modules pass actor + paths + action.
    Lazy-inits project on first call. Intention/graph hooks accepted, unused by default.
    """
    action = normalize_action(action or "create")
    if action == "write":
        action = "edit" if filename else "create"

    if repo_path is None:
        repo_path = resolve_workspace_path(prefer_chat_project=prefer_chat_project)

    workspace = Path(repo_path).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if not _effective_use_git(use_git):
        return _pipeline_plain_disk(
            actor_func=actor_func,
            filename=filename,
            for_actor=for_actor,
            action=action,
            handler_arg=handler_arg,
            workspace=workspace,
            session_id=session_id,
            staging=staging,
            content=content,
            force=force,
            query=query,
            include_disk=include_disk,
            mode=mode,
        )

    core = get_project(workspace)
    sid = _session_id(str(workspace), session_id)
    core.session(sid)
    staging_mode = staging or _DEFAULT_STAGING

    # Extended discovery actions (not in classic allowed_actions list)
    if action == "tree":
        r = core.tree(sid, include_disk=include_disk)
        d = fs_result_to_dict(r, action)
        d["files"] = (r.details or {}).get("files", [])
        return d
    if action == "search":
        r = core.search(query or "", sid, mode=mode)
        return fs_result_to_dict(r, action)
    if action == "snippet":
        rel = _to_rel(filename, workspace)
        r = core.snippet(rel or "", query or "", sid)
        return fs_result_to_dict(r, action)
    if action == "reconcile":
        r = core.reconcile(sid, ingest=True, force=force)
        return fs_result_to_dict(r, action)
    if action == "materialize":
        r = core.materialize(sid, prune=False)
        return fs_result_to_dict(r, action)
    if action == "finish":
        r = core.finish_for_user(sid, child_session=child_session, force=True)
        return fs_result_to_dict(r, action)
    if action == "ingest":
        rel = _to_rel(filename, workspace)
        r = core.ingest_path(rel or "", sid)
        return fs_result_to_dict(r, action)
    if action == "publish_touched":
        from_s = for_actor or (handler_arg or {}).get("from_session")
        if not from_s:
            return make_result(status_failed, "from_session required", action)
        r = core.publish_touched(str(from_s), sid, force=force)
        return fs_result_to_dict(r, action)

    if action not in allowed_actions and action not in EXTENDED_ACTIONS:
        return make_result(status_failed, f"unsupported action: {action}", action)

    source_rel = _to_rel(filename, workspace)
    if filename and source_rel is None:
        return make_result(status_forbidden, "path outside project", action)
    target_rel = None
    if action in ("copy", "move"):
        target_rel = _to_rel(for_actor, workspace)
        if for_actor and target_rel is None:
            return make_result(status_forbidden, "path outside project", action)
    elif action == "create":
        target_rel = _to_rel(for_actor, workspace) if for_actor else source_rel
        if for_actor and target_rel is None:
            return make_result(status_forbidden, "path outside project", action)
        source_rel = target_rel or source_rel
    elif action == "edit" and for_actor:
        alt = _to_rel(for_actor, workspace)
        if for_actor and alt is None:
            return make_result(status_forbidden, "path outside project", action)
        if alt and ("/" in str(for_actor) or "." in Path(str(for_actor)).name):
            target_rel = alt

    op_path = target_rel or source_rel
    if action in ("read", "edit", "delete", "copy", "move") and not source_rel:
        return make_result(status_failed, "path not specified", action)
    if action == "create" and not op_path:
        return make_result(status_failed, "path not specified", action)
    if action in ("copy", "move") and not target_rel:
        return make_result(status_failed, "target path not specified", action)

    # --- delete ---
    if action == "delete":
        r = core.delete(source_rel, sid)
        return fs_result_to_dict(r, action)

    # --- copy / move ---
    if action == "copy":
        r = core.copy(source_rel, target_rel, sid)
        d = fs_result_to_dict(r, action)
        d["source_path"] = source_rel
        d["target_path"] = target_rel
        return d
    if action == "move":
        r = core.move(source_rel, target_rel, sid)
        d = fs_result_to_dict(r, action)
        d["source_path"] = source_rel
        d["target_path"] = target_rel
        return d

    # --- read ---
    if action == "read":
        r = core.read(source_rel, sid)
        if not r.ok():
            return fs_result_to_dict(r, action)
        if callable(actor_func):
            err, _, out = _run_actor_on_bytes(
                workspace=workspace,
                session_id=sid,
                rel_path=source_rel,
                action="read",
                data=r.content or b"",
                actor_func=actor_func,
                handler_arg=handler_arg,
                staging=staging_mode,
            )
            if err:
                return err
            d = fs_result_to_dict(r, action)
            d["data"] = out.get("data")
            d["reason"] = operation_success_text()
            return d
        d = fs_result_to_dict(r, action)
        d["reason"] = operation_success_text()
        if r.content_text is not None:
            d["text"] = r.content_text
        return d

    # --- create / edit ---
    if action in ("create", "edit"):
        path = op_path or source_rel
        existing = b""
        if action == "edit" or core.exists(path, sid, resolve=False):
            rr = core.read(path, sid)
            if action == "edit" and not rr.ok() and content is None and not callable(actor_func):
                return fs_result_to_dict(rr, action)
            if rr.ok():
                existing = rr.content or b""

        if content is not None:
            if isinstance(content, str):
                existing = content.encode("utf-8")
            else:
                existing = content
        elif isinstance(handler_arg, dict) and "content" in handler_arg and not callable(actor_func):
            c = handler_arg.get("content")
            existing = c.encode("utf-8") if isinstance(c, str) else (c or b"")

        new_bytes = existing
        actor_out: Dict[str, Any] = {}
        if callable(actor_func):
            # seed empty file for pure create
            seed = existing
            if action == "create" and not seed and isinstance(handler_arg, dict):
                c = handler_arg.get("content")
                if isinstance(c, str):
                    seed = c.encode("utf-8")
                elif c is not None:
                    seed = bytes(c)
            err, new_bytes, actor_out = _run_actor_on_bytes(
                workspace=workspace,
                session_id=sid,
                rel_path=path,
                action=action,
                data=seed or b"",
                actor_func=actor_func,
                handler_arg=handler_arg,
                staging=staging_mode,
            )
            if err:
                return err
            if new_bytes is None:
                return make_result(status_failed, "actor produced no content", action, path=path)
        elif isinstance(handler_arg, dict) and "content" in handler_arg:
            c = handler_arg.get("content")
            new_bytes = c.encode("utf-8") if isinstance(c, str) else (c or b"")

        w = core.write(path, new_bytes, sid, force=force)
        d = fs_result_to_dict(w, action)
        if actor_out.get("data") is not None:
            d["data"] = actor_out["data"]
        d["source_path"] = path
        d["target_path"] = path
        if w.ok():
            d["reason"] = operation_success_text()
        return d

    return make_result(status_failed, f"unhandled action: {action}", action)


# Lazy @cacher on pipeline when chat cache is active (replay-safe mutations).
# Without cache_path (tests / early import) run raw — no cache slots.
_pipeline_cached = None


def pipeline(*args, **kwargs):
    global _pipeline_cached
    if _pipeline_cached is None:
        impl = _pipeline_impl
        try:
            import cross_gpt as cg

            if getattr(cg, "cache_path", None) and getattr(cg, "cacher", None):
                impl = cg.cacher(_pipeline_impl)
        except Exception:
            impl = _pipeline_impl
        _pipeline_cached = impl
    return _pipeline_cached(*args, **kwargs)


def operation_success_text() -> str:
    return "ok"


def change_dir(current_dir, target_dir, repo_path=None):
    """Navigator-compatible helper (logical cwd only; no git words)."""
    workspace = Path(resolve_workspace_path(repo_path=repo_path)).resolve()
    cur = Path(current_dir or ".")
    if not cur.is_absolute():
        cur = workspace / cur
    tgt = Path(target_dir or ".")
    if not tgt.is_absolute():
        nxt = (cur / tgt).resolve() if str(target_dir) not in (".", "") else cur.resolve()
        if str(target_dir).startswith("/"):
            nxt = (workspace / str(target_dir).lstrip("/")).resolve()
        else:
            nxt = (cur / target_dir).resolve() if target_dir not in (".",) else cur.resolve()
    else:
        nxt = tgt.resolve()
    try:
        rel = nxt.relative_to(workspace)
    except ValueError:
        return {"status": status_forbidden, "reason": "path outside project", "cwd": str(current_dir)}
    # must exist as prefix of some file or be empty root — allow any subdir create-less
    return {"status": status_success, "cwd": rel.as_posix() or ".", "reason": "ok"}


def get_project_tree_json(repo_path=None, cwd=None, max_depth=8, include_hidden=False):
    workspace = Path(resolve_workspace_path(repo_path=repo_path)).resolve()
    core = get_project(workspace)
    sid = _session_id(str(workspace))
    r = core.tree(sid)
    files = r.details.get("files") or []
    # filter by cwd prefix
    prefix = (cwd or ".").replace("\\", "/").strip("./")
    if prefix and prefix != ".":
        files = [f for f in files if f == prefix or f.startswith(prefix + "/")]
    return {
        "status": status_success,
        "reason": "ok",
        "cwd": cwd or ".",
        "files": files,
        "tree": files,
    }


# ---------------------------------------------------------------------------
# Object-style session (optional; modules may use pipeline only)
# ---------------------------------------------------------------------------

@dataclass
class FsSession:
    core: FileSystem
    session_id: str
    actor_id: str = "module"
    root: str = ""

    def pipeline(self, *args, **kwargs):
        kwargs.setdefault("repo_path", self.root or str(self.core.root))
        kwargs.setdefault("session_id", self.session_id)
        if args:
            # actor_func, filename positional
            return pipeline(*args, **kwargs)
        return pipeline(**kwargs)

    def read(self, path: str, **kw) -> FsResult:
        return self.core.read(path, self.session_id, **kw)

    def write(self, path: str, content, **kw) -> FsResult:
        return self.core.write(path, content, self.session_id, **kw)

    def delete(self, path: str) -> FsResult:
        return self.core.delete(path, self.session_id)

    def move(self, src: str, dst: str) -> FsResult:
        return self.core.move(src, dst, self.session_id)

    def copy(self, src: str, dst: str) -> FsResult:
        return self.core.copy(src, dst, self.session_id)

    def tree(self, **kw) -> FsResult:
        return self.core.tree(self.session_id, **kw)

    def search(self, query: str, **kw) -> FsResult:
        return self.core.search(query, self.session_id, **kw)

    def exists(self, path: str, **kw) -> bool:
        return self.core.exists(path, self.session_id, **kw)

    def reconcile(self, **kw) -> FsResult:
        return self.core.reconcile(self.session_id, **kw)

    def materialize(self, **kw) -> FsResult:
        return self.core.materialize(self.session_id, **kw)

    def finish(self, **kw) -> FsResult:
        return self.core.finish_for_user(self.session_id, **kw)

    def hierarchy_down(self, child_session: str, **kw) -> str:
        return self.core.hierarchy_down(self.session_id, child_session, **kw)

    def hierarchy_up(self, child_session: str, **kw) -> FsResult:
        return self.core.hierarchy_up(child_session, self.session_id, **kw)

    def ingest(self, path: str) -> FsResult:
        return self.core.ingest_path(path, self.session_id)

    def snippet(self, path: str, query: str, **kw) -> FsResult:
        return self.core.snippet(path, query, self.session_id, **kw)


def copy_touched_to_folder(
    session_id: str = "default",
    *,
    repo_path=None,
    dest_dir: Optional[Union[str, Path]] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Optional (off by default in product): copy session touched files into a
    separate folder under the chat (not a branch merge into main).
    """
    import shutil
    from datetime import datetime

    workspace = Path(resolve_workspace_path(repo_path=repo_path)).resolve()
    core = get_project(workspace)
    paths = core.touched_paths(session_id)
    if dest_dir is None:
        try:
            import cross_gpt as cg

            chat = Path(getattr(cg, "chat_path", "") or workspace.parent)
        except Exception:
            chat = workspace.parent
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe = (label or session_id or "dialog").replace("/", "_").replace(":", "_")
        dest_dir = chat / "dialog_artifacts" / f"{stamp}__{safe}"
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in paths:
        data = core._get_blob(core.session(session_id).world, p)
        if data is None:
            disk = core._read_disk(p)
            if disk is None:
                continue
            data = disk
        out = dest / p
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        copied.append(p)
    return make_result(
        status_success,
        "copied",
        "copy_touched",
        dest=str(dest),
        copied=copied,
        paths=paths,
    )


def get_dependency_report(*a, **k):
    return make_result(status_success, "stub", "dependency_report", report={})


def create_experiment_branch(*a, **k):
    # Forks disabled by default (main-first). Explicit isolation only via API later.
    return make_result(status_success, "noop_main_first", "create_experiment_branch", branch=None)


# re-exports for tools
__all__ = [
    "status_success",
    "status_failed",
    "status_forbidden",
    "status_success_text",
    "status_failed_text",
    "status_forbidden_text",
    "pipeline",
    "make_result",
    "open_project",
    "get_project",
    "set_session",
    "FsSession",
    "change_dir",
    "get_project_tree_json",
    "resolve_workspace_path",
    "to_posix_rel",
    "normalize_action",
    "allowed_actions",
    "copy_touched_to_folder",
    "get_dependency_report",
    "create_experiment_branch",
    "OK",
    "NOT_FOUND",
    "STALE",
    "FORBIDDEN_PATH",
    "FAILED",
]
