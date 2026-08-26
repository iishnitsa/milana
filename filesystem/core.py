"""
Filesystem core v2 (concept-driven) — technical prototype.

Git (dulwich) always on; agent-facing API is path-only (no branch/commit words).
Intention/policy LLM is stubbed (inject or empty); create-fast has no LLM.

See tests/filesystem/concept.md
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dulwich import porcelain
from dulwich.objects import Blob, Commit, Tree
from dulwich.repo import Repo

# ---------------------------------------------------------------------------
# Codes (agent-facing)
# ---------------------------------------------------------------------------

OK = "ok"
NOT_FOUND = "not_found"
STALE = "stale"
FORBIDDEN_PATH = "forbidden_path"
FORBIDDEN_POLICY = "forbidden_policy"
CONFLICT_IMPORT = "conflict_import"
FAILED = "failed"

DEFAULT_WORLD = "world/main"
IMPORT_MAX_BYTES = 512 * 1024

# Disk scan: ignore VCS / venv noise (third-party may still write under project root)
DISK_IGNORE_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".fs_tmp",
        ".fs_work",  # module staging sandbox (gitignored conceptually)
    }
)
DISK_IGNORE_FILES = frozenset({".fs_meta.json", ".gitignore"})
STAGING_DIR = ".fs_work"


def _now() -> int:
    return int(time.time())


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _norm_path(path: str) -> str:
    p = str(path or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    # reject empty and parent escapes after normalize
    parts = []
    for part in p.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError("path_escape")
        parts.append(part)
    return "/".join(parts)


# Agent-facing plain messages (no git words)
AGENT_MSGS = {
    OK: "ok",
    NOT_FOUND: "File not found",
    STALE: "File changed since last read. Read again, then write.",
    FORBIDDEN_PATH: "Path not allowed",
    FORBIDDEN_POLICY: "Operation not allowed for this file",
    CONFLICT_IMPORT: "File already exists with different content",
    FAILED: "Operation failed",
}


@dataclass
class FsResult:
    code: str
    message: str = ""
    path: Optional[str] = None
    content: Optional[bytes] = None
    content_text: Optional[str] = None
    hash: Optional[str] = None
    world: Optional[str] = None
    created: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def ok(self) -> bool:
        return self.code == OK

    def text(self) -> str:
        if self.content_text is not None:
            return self.content_text
        if self.content is not None:
            try:
                return self.content.decode("utf-8")
            except UnicodeDecodeError:
                return ""
        return ""

    def agent_message(self) -> str:
        """Short tool result for weak LLMs — codes + facts, no VCS jargon."""
        base = self.message or AGENT_MSGS.get(self.code, self.code)
        parts = [f"status={self.code}", base]
        if self.path:
            parts.append(f"path={self.path}")
        if self.hash:
            parts.append(f"hash={self.hash[:12]}")
        if self.created:
            parts.append("created=true")
        if self.details.get("files") is not None:
            files = self.details["files"]
            parts.append(f"files={len(files)}")
        if self.details.get("hits") is not None:
            parts.append(f"hits={len(self.details['hits'])}")
        if self.details.get("imported"):
            parts.append("imported=true")
        if self.details.get("isolated"):
            parts.append("isolated=true")
        if self.details.get("truncated"):
            parts.append(f"truncated={self.details['truncated']}")
        if self.details.get("published") is not None:
            parts.append(f"published={self.details['published']}")
        return " | ".join(parts)


@dataclass
class LastSeen:
    content_hash: str
    rev: str  # commit id hex
    world: str


@dataclass
class Session:
    """Per hierarchy-level (or agent) session state — invisible to the model."""
    world: str = DEFAULT_WORLD
    last_seen: Dict[str, LastSeen] = field(default_factory=dict)
    actor_id: str = "agent"
    # optional policy cache path -> dict
    policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # paths deleted in this session/world — do not auto-import back on read
    tombstones: Dict[str, set] = field(default_factory=dict)  # world -> set(paths)
    # paths created/edited in this world (for promote after child work)
    touched: Dict[str, set] = field(default_factory=dict)  # world -> set(paths)


class FileSystem:
    """
    Git-backed project filesystem with working worlds (branches).

    Public methods mirror seamless tool API: read/write/delete/move/copy/tree.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        import_max_bytes: int = IMPORT_MAX_BYTES,
        isolate_low_risk_foreign_edit: bool = False,
        require_read_before_edit: bool = False,
        intention_fn: Optional[Callable[[str, str, str], Dict[str, Any]]] = None,
        dual_write: bool = True,
        ingest_disk_on_read: bool = True,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.import_max_bytes = import_max_bytes
        self.isolate_low_risk_foreign_edit = isolate_low_risk_foreign_edit
        self.require_read_before_edit = require_read_before_edit
        # intention_fn(action, path, actor) -> policy dict; None = defaults
        self.intention_fn = intention_fn
        # Write/delete also touch real files under root (user folder view).
        self.dual_write = dual_write
        # If blob missing but file exists on disk (third-party module), ingest on read.
        self.ingest_disk_on_read = ingest_disk_on_read
        self.meta_path = self.root / ".fs_meta.json"
        self._repo: Optional[Repo] = None
        self.sessions: Dict[str, Session] = {}
        self._ensure_repo()

    # ----- repo bootstrap -----

    def _ensure_repo(self) -> Repo:
        if self._repo is not None:
            return self._repo
        git_dir = self.root / ".git"
        if not git_dir.exists():
            porcelain.init(str(self.root))
        self._repo = Repo(str(self.root))
        # ensure default world branch exists with empty tree commit if needed
        try:
            self._repo.refs[b"HEAD"]
        except KeyError:
            pass
        self._ensure_world(DEFAULT_WORLD, base=None)
        self._save_meta()
        return self._repo

    def _meta(self) -> Dict[str, Any]:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_meta(self, extra: Optional[Dict[str, Any]] = None) -> None:
        data = self._meta()
        if extra:
            data.update(extra)
        data.setdefault("worlds", {})
        data.setdefault("file_owners", {})  # path -> created_by (on default lineage tip tracking simplified)
        data.setdefault("file_policy", {})  # path -> policy (last known)
        self.meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ref_name(self, world: str) -> bytes:
        w = world if world.startswith("world/") else f"world/{world}"
        return f"refs/heads/{w}".encode("utf-8")

    def _ensure_world(self, world: str, base: Optional[str] = None) -> str:
        repo = self._ensure_repo()
        ref = self._ref_name(world)
        if ref in repo.refs:
            return world
        tip = None
        if base:
            base_ref = self._ref_name(base)
            if base_ref in repo.refs:
                tip = repo.refs[base_ref]
        if tip is None:
            tree = Tree()
            repo.object_store.add_object(tree)
            commit = Commit()
            commit.tree = tree.id
            commit.parents = []
            commit.author = commit.committer = b"fs <fs@local>"
            commit.commit_time = commit.author_time = _now()
            commit.commit_timezone = commit.author_timezone = 0
            commit.message = b"init world"
            repo.object_store.add_object(commit)
            tip = commit.id
        repo.refs[ref] = tip
        # also set HEAD on first world
        try:
            repo.refs.set_symbolic_ref(b"HEAD", ref)
        except Exception:
            pass
        meta = self._meta()
        worlds = meta.setdefault("worlds", {})
        worlds[world] = {"created": _now(), "base": base}
        self._save_meta(meta)
        return world

    def session(self, session_id: str = "default", *, actor_id: str = "agent") -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(world=DEFAULT_WORLD, actor_id=actor_id)
            self._ensure_world(DEFAULT_WORLD)
        else:
            self.sessions[session_id].actor_id = actor_id
        return self.sessions[session_id]

    def fork_world(self, session_id: str, new_world: Optional[str] = None) -> str:
        """Hierarchy down: new world from current session world tip."""
        sess = self.session(session_id)
        name = new_world or f"world/L{_now()}_{session_id}"
        self._ensure_world(name, base=sess.world)
        sess.world = name
        # last_seen remains valid if content same after fork (same tip)
        return name

    def hierarchy_down(
        self,
        parent_session: str,
        child_session: str,
        *,
        child_actor: str = "child",
        world_name: Optional[str] = None,
    ) -> str:
        """
        Delegation start: child session forks from parent's current world tip.
        Returns new world name (system-only; never show to model).
        """
        parent = self.session(parent_session)
        child = self.session(child_session, actor_id=child_actor)
        child.world = parent.world
        # copy last_seen so child can edit without forced re-read of shared tip
        child.last_seen = {
            p: LastSeen(ls.content_hash, ls.rev, ls.world) for p, ls in parent.last_seen.items()
        }
        return self.fork_world(child_session, world_name)

    def hierarchy_up(
        self,
        child_session: str,
        parent_session: str,
        *,
        promote: bool = True,
        force: bool = False,
        restore_parent_world: bool = True,
    ) -> FsResult:
        """
        Delegation end: optionally publish_touched child → parent.
        Parent keeps (or restores) parent world; child world tip remains in history.
        """
        parent = self.session(parent_session)
        parent_world_before = parent.world
        # If parent was switched to child world for continuing work, restore
        child = self.session(child_session)
        if restore_parent_world:
            # parent world is base of child if stored in meta
            meta = self._meta()
            base = (meta.get("worlds") or {}).get(child.world, {}).get("base")
            if base:
                parent.world = base
            else:
                parent.world = parent_world_before
        if not promote:
            return FsResult(
                OK,
                "hierarchy_up_no_promote",
                world=parent.world,
                details={"promoted": [], "child_world": child.world},
            )
        pub = self.publish_touched(child_session, parent_session, force=force)
        pub.details = dict(pub.details or {})
        pub.details["child_world"] = child.world
        pub.details["parent_world"] = parent.world
        return pub

    def set_world(self, session_id: str, world: str) -> None:
        self._ensure_world(world)
        self.session(session_id).world = world

    def _mark_touched(self, sess: Session, path: str) -> None:
        sess.touched.setdefault(sess.world, set()).add(path)

    def touched_paths(self, session_id: str = "default", world: Optional[str] = None) -> List[str]:
        sess = self.session(session_id)
        w = world or sess.world
        return sorted(sess.touched.get(w, set()))

    def list_worlds(self) -> List[str]:
        repo = self._ensure_repo()
        names = []
        for ref in repo.refs.keys():
            if ref.startswith(b"refs/heads/world/"):
                names.append(ref.decode("utf-8").split("refs/heads/", 1)[1])
        return sorted(names)

    def publish(
        self,
        path: str,
        *,
        to_session: str,
        from_session: Optional[str] = None,
        from_world: Optional[str] = None,
        force: bool = False,
    ) -> FsResult:
        """
        Copy path from a source world into to_session's current world.

        Agent-facing name later: "make file available" / accept_work — not "merge".
        Used when child created file a that parent/agent3 needs alongside b.
        """
        n, err = self._safe_path(path)
        if err:
            return err
        to_sess = self.session(to_session)
        if from_world is None:
            if from_session is None:
                return FsResult(FAILED, "from_session or from_world required", path=n)
            from_world = self.session(from_session).world
        data = self._get_blob(from_world, n)
        if data is None:
            return FsResult(NOT_FOUND, "File not found in source", path=n, world=from_world)

        existing = self._get_blob(to_sess.world, n)
        if existing is not None and not force:
            if _hash_bytes(existing) != _hash_bytes(data):
                return FsResult(
                    CONFLICT_IMPORT,
                    "File already exists with different content",
                    path=n,
                    world=to_sess.world,
                    details={"source_world": from_world},
                )
            # same content — already there
            h = _hash_bytes(data)
            c = self._commit(to_sess.world)
            to_sess.last_seen[n] = LastSeen(h, c.id.hex() if c else "", to_sess.world)
            to_sess.tombstones.get(to_sess.world, set()).discard(n)
            return FsResult(OK, "already_present", path=n, hash=h, world=to_sess.world)

        rev = self._commit_files(
            to_sess.world,
            {n: data},
            json.dumps(
                {"op": "publish", "path": n, "from": from_world, "to": to_sess.world},
                ensure_ascii=False,
            ),
        )
        h = _hash_bytes(data)
        to_sess.last_seen[n] = LastSeen(h, rev, to_sess.world)
        to_sess.tombstones.get(to_sess.world, set()).discard(n)
        self._mark_touched(to_sess, n)
        # Dual-write only when publishing into the user canon (main) so folder updates.
        if self.dual_write and to_sess.world == DEFAULT_WORLD:
            self._write_disk(n, data)
        meta = self._meta()
        # keep original owner if set
        owners = meta.setdefault("file_owners", {})
        if n not in owners:
            owners[n] = self.session(from_session).actor_id if from_session else "publish"
            self._save_meta(meta)
        return FsResult(
            OK,
            "published",
            path=n,
            content=data,
            hash=h,
            world=to_sess.world,
            details={"from_world": from_world},
        )

    def publish_touched(
        self,
        from_session: str,
        to_session: str,
        *,
        force: bool = False,
    ) -> FsResult:
        """Publish all paths touched in from_session's current world into to_session world."""
        src = self.session(from_session)
        paths = self.touched_paths(from_session, src.world)
        if not paths:
            return FsResult(OK, "nothing_to_publish", details={"published": [], "failed": []})
        published, failed = [], []
        for p in paths:
            r = self.publish(p, to_session=to_session, from_session=from_session, force=force)
            if r.ok():
                published.append(p)
            else:
                failed.append({"path": p, "code": r.code, "message": r.message})
        code = OK if not failed else (OK if published else FAILED)
        return FsResult(
            code,
            "published" if published else "publish_failed",
            details={"published": published, "failed": failed, "from_world": src.world},
        )

    # ----- tree helpers -----

    def _commit(self, world: str) -> Optional[Commit]:
        repo = self._ensure_repo()
        ref = self._ref_name(world)
        if ref not in repo.refs:
            return None
        return repo[repo.refs[ref]]

    def _tree_map(self, tree_id: bytes, prefix: str = "") -> Dict[str, Tuple[int, bytes]]:
        """path -> (mode, blob_id)"""
        repo = self._ensure_repo()
        tree = repo[tree_id]
        out: Dict[str, Tuple[int, bytes]] = {}
        for entry in tree.items():
            # dulwich Tree items: (name, mode, sha)
            name = entry.path.decode("utf-8", errors="replace") if hasattr(entry, "path") else entry[0].decode("utf-8", errors="replace")
            mode = entry.mode if hasattr(entry, "mode") else entry[1]
            sha = entry.sha if hasattr(entry, "sha") else entry[2]
            rel = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"
            if mode & 0o170000 == 0o040000:  # dir
                out.update(self._tree_map(sha, rel))
            else:
                out[rel] = (mode, sha)
        return out

    def _get_blob(self, world: str, path: str) -> Optional[bytes]:
        c = self._commit(world)
        if c is None:
            return None
        m = self._tree_map(c.tree)
        if path not in m:
            return None
        _, blob_id = m[path]
        blob = self._repo[blob_id]
        return blob.data

    def _write_tree_from_map(self, files: Dict[str, bytes]) -> bytes:
        """Build nested trees from path->content. Returns root tree id."""
        repo = self._ensure_repo()

        # nest: segments
        def build(level: Dict[str, Any]) -> bytes:
            tree = Tree()
            for name, node in sorted(level.items()):
                if isinstance(node, dict):
                    child_id = build(node)
                    tree.add(name.encode("utf-8"), 0o040000, child_id)
                else:
                    blob = Blob.from_string(node)
                    repo.object_store.add_object(blob)
                    tree.add(name.encode("utf-8"), 0o100644, blob.id)
            repo.object_store.add_object(tree)
            return tree.id

        root: Dict[str, Any] = {}
        for path, data in files.items():
            parts = path.split("/")
            cur = root
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = data
        return build(root)

    def _snapshot_files(self, world: str) -> Dict[str, bytes]:
        c = self._commit(world)
        if c is None:
            return {}
        m = self._tree_map(c.tree)
        out = {}
        for path, (_, blob_id) in m.items():
            out[path] = self._repo[blob_id].data
        return out

    def _commit_files(
        self,
        world: str,
        files: Dict[str, bytes],
        message: str,
        *,
        delete_paths: Optional[List[str]] = None,
    ) -> str:
        """Replace full tree with files dict (apply deletes first). Returns commit hex."""
        repo = self._ensure_repo()
        current = self._snapshot_files(world)
        if delete_paths:
            for p in delete_paths:
                current.pop(p, None)
        current.update(files)
        tree_id = self._write_tree_from_map(current)
        parent = None
        ref = self._ref_name(world)
        if ref in repo.refs:
            parent = repo.refs[ref]
        commit = Commit()
        commit.tree = tree_id
        commit.parents = [parent] if parent else []
        commit.author = commit.committer = b"fs <fs@local>"
        commit.commit_time = commit.author_time = _now()
        commit.commit_timezone = commit.author_timezone = 0
        commit.message = message.encode("utf-8", errors="replace")
        repo.object_store.add_object(commit)
        repo.refs[ref] = commit.id
        return commit.id.hex()

    # ----- path safety -----

    def _safe_path(self, path: str) -> Tuple[Optional[str], Optional[FsResult]]:
        try:
            n = _norm_path(path)
        except ValueError:
            return None, FsResult(FORBIDDEN_PATH, "Path outside project", path=path)
        if not n or n.startswith(".git") or n == ".fs_meta.json" or n.startswith(".fs_meta"):
            return None, FsResult(FORBIDDEN_PATH, "Path not allowed", path=path)
        # Staging sandbox is not a project path agents may read/write via normal API
        if n == STAGING_DIR or n.startswith(STAGING_DIR + "/"):
            return None, FsResult(FORBIDDEN_PATH, "Path not allowed", path=path)
        return n, None

    # ----- disk (user folder) + bypass-git -----

    def _abs(self, rel: str) -> Path:
        return self.root / rel

    def _write_disk(self, rel: str, data: bytes) -> None:
        p = self._abs(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def _delete_disk(self, rel: str) -> None:
        p = self._abs(rel)
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass

    def _read_disk(self, rel: str) -> Optional[bytes]:
        p = self._abs(rel)
        if not p.is_file():
            return None
        try:
            return p.read_bytes()
        except OSError:
            return None

    def list_disk_files(self, *, max_files: int = 5000) -> List[str]:
        """Scan project folder on disk (not git). Third-party modules write here."""
        out: List[str] = []
        root = self.root
        if not root.is_dir():
            return out
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in DISK_IGNORE_DIRS and not d.startswith(".git")
            ]
            for name in filenames:
                if name in DISK_IGNORE_FILES or name.endswith(".pyc"):
                    continue
                full = Path(dirpath) / name
                try:
                    rel = full.relative_to(root).as_posix()
                except ValueError:
                    continue
                if rel.startswith(".git") or rel == ".fs_meta.json":
                    continue
                out.append(rel)
                if len(out) >= max_files:
                    return sorted(out)
        return sorted(out)

    def reconcile(
        self,
        session_id: str = "default",
        *,
        ingest: bool = True,
        force: bool = False,
    ) -> FsResult:
        """
        Compare disk vs current world. Optionally ingest disk-only / changed files.

        Normal: third-party modules (or user) dropped files without using FsApi.
        """
        sess = self.session(session_id)
        world_files = self._snapshot_files(sess.world)
        disk_files = self.list_disk_files()
        disk_set = set(disk_files)
        world_set = set(world_files.keys())

        only_disk = sorted(disk_set - world_set)
        only_world = sorted(world_set - disk_set)
        both = sorted(disk_set & world_set)

        changed: List[str] = []
        same: List[str] = []
        for p in both:
            d = self._read_disk(p)
            if d is None:
                continue
            if _hash_bytes(d) != _hash_bytes(world_files[p]):
                changed.append(p)
            else:
                same.append(p)

        ingested: List[str] = []
        conflicts: List[Dict[str, str]] = []
        if ingest:
            for p in only_disk + changed:
                d = self._read_disk(p)
                if d is None:
                    continue
                if p in world_files and not force:
                    if _hash_bytes(d) != _hash_bytes(world_files[p]):
                        # disk wins for bypass-ingest by default when force=False
                        # still ingest (disk is source of truth for foreign writes)
                        pass
                r = self.ingest_path(p, session_id, data=d)
                if r.ok():
                    ingested.append(p)
                else:
                    conflicts.append({"path": p, "code": r.code, "message": r.message})

        return FsResult(
            OK,
            "reconciled",
            world=sess.world,
            details={
                "only_disk": only_disk,
                "only_world": only_world,
                "changed_on_disk": changed,
                "same": same,
                "ingested": ingested,
                "conflicts": conflicts,
                "disk_count": len(disk_files),
                "world_count": len(world_files),
            },
        )

    def ingest_path(
        self,
        path: str,
        session_id: str = "default",
        *,
        data: Optional[bytes] = None,
    ) -> FsResult:
        """Pull one on-disk (or provided) file into the session world without dual-write loop."""
        n, err = self._safe_path(path)
        if err:
            return err
        sess = self.session(session_id)
        if data is None:
            data = self._read_disk(n)
        if data is None:
            return FsResult(NOT_FOUND, "No disk file to ingest", path=n)
        h = _hash_bytes(data)
        rev = self._commit_files(
            sess.world,
            {n: data},
            json.dumps(
                {"op": "ingest_disk", "path": n, "actor": sess.actor_id},
                ensure_ascii=False,
            ),
        )
        sess.last_seen[n] = LastSeen(h, rev, sess.world)
        sess.tombstones.get(sess.world, set()).discard(n)
        self._mark_touched(sess, n)
        meta = self._meta()
        owners = meta.setdefault("file_owners", {})
        owners.setdefault(n, sess.actor_id or "disk")
        self._save_meta(meta)
        return FsResult(
            OK,
            "ingested",
            path=n,
            content=data,
            hash=h,
            world=sess.world,
            details={"source": "disk"},
        )

    def materialize(
        self,
        session_id: str = "default",
        *,
        world: Optional[str] = None,
        prune: bool = False,
    ) -> FsResult:
        """
        Write world blobs onto disk so the user folder matches the canon tip.

        Default world for user view: caller's session world (usually main after promote).
        prune=True removes disk files not in world (aggressive; off by default so
        bypass files are not wiped until ingested or explicitly pruned).
        """
        sess = self.session(session_id)
        w = world or sess.world
        files = self._snapshot_files(w)
        written: List[str] = []
        for rel, data in files.items():
            self._write_disk(rel, data)
            written.append(rel)
        removed: List[str] = []
        if prune:
            for rel in self.list_disk_files():
                if rel not in files:
                    self._delete_disk(rel)
                    removed.append(rel)
        return FsResult(
            OK,
            "materialized",
            world=w,
            details={
                "written": sorted(written),
                "removed": sorted(removed),
                "count": len(written),
            },
        )

    def finish_for_user(
        self,
        parent_session: str = "default",
        *,
        child_session: Optional[str] = None,
        promote: bool = True,
        force: bool = True,
        prune: bool = False,
        reconcile_disk: bool = True,
    ) -> FsResult:
        """
        End of task: promote child work → parent world, optional disk ingest,
        materialize parent world to folder. User opens project and sees results.

        force=True by default: accepting child work is intentional (Милана закончила);
        conflict_import would leave main stale while disk may already show child dual-write.
        """
        details: Dict[str, Any] = {}
        if child_session and promote:
            up = self.hierarchy_up(
                child_session, parent_session, promote=True, force=force
            )
            details["promote"] = {
                "code": up.code,
                "published": (up.details or {}).get("published"),
                "failed": (up.details or {}).get("failed"),
            }
            if not up.ok() and (up.details or {}).get("failed"):
                return FsResult(
                    up.code,
                    "promote failed",
                    world=self.session(parent_session).world,
                    details=details,
                )
        # Ensure parent session sits on its world (main after hierarchy_up)
        parent = self.session(parent_session)
        # Prefer DEFAULT_WORLD for user-facing materialize if parent is main lineage
        target_world = parent.world
        if target_world != DEFAULT_WORLD:
            # after hierarchy_up parent should be base; if still not main, materialize parent world
            pass
        if reconcile_disk:
            rec = self.reconcile(parent_session, ingest=True)
            details["reconcile"] = {
                "ingested": (rec.details or {}).get("ingested"),
                "only_disk": (rec.details or {}).get("only_disk"),
            }
        mat = self.materialize(parent_session, world=target_world, prune=prune)
        details["materialize"] = mat.details
        details["user_world"] = target_world
        details["disk_files"] = self.list_disk_files()
        return FsResult(
            OK,
            "ready_for_user",
            world=target_world,
            details=details,
        )

    # ----- public API -----

    def tree(
        self,
        session_id: str = "default",
        *,
        include_disk: bool = False,
    ) -> FsResult:
        sess = self.session(session_id)
        files = sorted(self._snapshot_files(sess.world).keys())
        details: Dict[str, Any] = {"files": files}
        if include_disk:
            disk = self.list_disk_files()
            details["disk_files"] = disk
            details["disk_only"] = sorted(set(disk) - set(files))
        return FsResult(OK, "ok", world=sess.world, details=details)

    def exists(
        self,
        path: str,
        session_id: str = "default",
        *,
        resolve: bool = False,
    ) -> bool:
        """
        resolve=False (default): only current world (+ not tombstoned).
        resolve=True: also discoverable in other worlds (would import on read).
        """
        n, err = self._safe_path(path)
        if err or not n:
            return False
        sess = self.session(session_id)
        if n in sess.tombstones.get(sess.world, set()):
            return False
        if self._get_blob(sess.world, n) is not None:
            return True
        if not resolve:
            return False
        return self._find_in_worlds(n, prefer=sess.world) is not None

    def search(
        self,
        query: str,
        session_id: str = "default",
        *,
        mode: str = "both",
        max_hits: int = 30,
        max_snippet: int = 160,
        path_glob: Optional[str] = None,
    ) -> FsResult:
        """
        Deterministic project search — no LLM.
        mode: name | content | both
        Returns hits: [{path, kind, line?, snippet?}]
        """
        q = (query or "").strip()
        if not q:
            return FsResult(FAILED, "empty query", details={"hits": []})
        sess = self.session(session_id)
        files = self._snapshot_files(sess.world)
        q_low = q.lower()
        hits: List[Dict[str, Any]] = []
        mode = (mode or "both").lower()

        def path_ok(p: str) -> bool:
            if not path_glob:
                return True
            # simple * substring match on path
            g = path_glob.replace("*", "")
            if "*" not in path_glob:
                return path_glob in p or p.endswith(path_glob)
            return g.lower() in p.lower() if g else True

        for path in sorted(files.keys()):
            if not path_ok(path):
                continue
            if mode in ("name", "both"):
                if q_low in path.lower() or q_low in Path(path).name.lower():
                    hits.append({"path": path, "kind": "name", "snippet": path})
                    if len(hits) >= max_hits:
                        break
            if mode in ("content", "both") and len(hits) < max_hits:
                data = files[path]
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if q_low in line.lower():
                        sn = line.strip()
                        if len(sn) > max_snippet:
                            sn = sn[: max_snippet - 1] + "…"
                        hits.append(
                            {
                                "path": path,
                                "kind": "content",
                                "line": i,
                                "snippet": sn,
                            }
                        )
                        if len(hits) >= max_hits:
                            break
            if len(hits) >= max_hits:
                break

        return FsResult(
            OK,
            f"found {len(hits)}" if hits else "no matches",
            world=sess.world,
            details={"hits": hits, "query": q, "mode": mode},
        )

    def snippet(
        self,
        path: str,
        query: str,
        session_id: str = "default",
        *,
        context_lines: int = 2,
        max_chars: int = 2000,
    ) -> FsResult:
        """Read file and return lines around first match of query (for weak LLM context)."""
        r = self.read(path, session_id, max_chars=None)
        if not r.ok():
            return r
        text = r.text()
        q = (query or "").strip().lower()
        lines = text.splitlines()
        if not q:
            body = "\n".join(lines[: 40])
            if len(body) > max_chars:
                body = body[: max_chars - 1] + "…"
            return FsResult(
                OK,
                "snippet",
                path=r.path,
                content_text=body,
                hash=r.hash,
                world=r.world,
                details={"match_line": None, "truncated": len(body) < len(text)},
            )
        match_i = None
        for i, line in enumerate(lines):
            if q in line.lower():
                match_i = i
                break
        if match_i is None:
            return FsResult(
                NOT_FOUND,
                "Query not found in file",
                path=r.path,
                hash=r.hash,
                world=r.world,
                details={"query": query},
            )
        lo = max(0, match_i - context_lines)
        hi = min(len(lines), match_i + context_lines + 1)
        body = "\n".join(lines[lo:hi])
        if len(body) > max_chars:
            body = body[: max_chars - 1] + "…"
        return FsResult(
            OK,
            "snippet",
            path=r.path,
            content_text=body,
            hash=r.hash,
            world=r.world,
            details={"match_line": match_i + 1, "from_line": lo + 1, "to_line": hi},
        )

    def read(
        self,
        path: str,
        session_id: str = "default",
        *,
        max_chars: Optional[int] = None,
    ) -> FsResult:
        n, err = self._safe_path(path)
        if err:
            return err
        sess = self.session(session_id)
        data = self._get_blob(sess.world, n)
        imported = False
        src_world = sess.world
        if data is None:
            # do not resurrect files intentionally deleted in this world
            if n in sess.tombstones.get(sess.world, set()):
                return FsResult(NOT_FOUND, "File not found", path=n, world=sess.world)
            # resolve + import from other worlds
            found = self._find_in_worlds(n, prefer=sess.world)
            if found is None:
                # third-party / bypass: file may exist only on disk
                if self.ingest_disk_on_read:
                    disk = self._read_disk(n)
                    if disk is not None:
                        ing = self.ingest_path(n, session_id, data=disk)
                        if ing.ok():
                            data = disk
                            imported = True
                            src_world = sess.world
                if data is None:
                    return FsResult(NOT_FOUND, "File not found", path=n, world=sess.world)
            else:
                src_world, data, rev = found
                if len(data) <= self.import_max_bytes:
                    try:
                        data.decode("utf-8")
                        is_text = True
                    except UnicodeDecodeError:
                        is_text = False
                    if is_text or len(data) <= self.import_max_bytes:
                        self._commit_files(
                            sess.world,
                            {n: data},
                            json.dumps(
                                {"op": "import", "from": src_world, "path": n},
                                ensure_ascii=False,
                            ),
                        )
                        imported = True
                        src_world = sess.world
                else:
                    # large: do not import; still return content, bind last_seen to src
                    sess.last_seen[n] = LastSeen(_hash_bytes(data), rev, src_world)
                    text = None
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        pass
                    if text is not None and max_chars is not None and len(text) > max_chars:
                        text = text[: max_chars - 1] + "…"
                    return FsResult(
                        OK,
                        "ok",
                        path=n,
                        content=data,
                        content_text=text,
                        hash=_hash_bytes(data),
                        world=src_world,
                        details={
                            "imported": False,
                            "source_world": src_world,
                            "truncated": bool(max_chars and text and text.endswith("…")),
                        },
                    )

        data = self._get_blob(sess.world, n)
        if data is None:
            return FsResult(NOT_FOUND, "File not found", path=n, world=sess.world)
        h = _hash_bytes(data)
        c = self._commit(sess.world)
        rev = c.id.hex() if c else ""
        sess.last_seen[n] = LastSeen(h, rev, sess.world)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        truncated = False
        # max_chars=0 → metadata-oriented read (stat); body still in content bytes
        if max_chars == 0:
            text = ""
            truncated = True
        elif text is not None and max_chars is not None and len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
            truncated = True
        return FsResult(
            OK,
            "ok",
            path=n,
            content=data,
            content_text=text,
            hash=h,
            world=sess.world,
            details={"imported": imported, "truncated": truncated},
        )

    def _find_in_worlds(self, path: str, prefer: str) -> Optional[Tuple[str, bytes, str]]:
        """Search all known world branches for path. Returns (world, data, rev_hex)."""
        repo = self._ensure_repo()
        # order: prefer first, then others
        names = []
        for ref in repo.refs.keys():
            if ref.startswith(b"refs/heads/world/"):
                w = ref.decode("utf-8").split("refs/heads/", 1)[1]
                names.append(w)
        if prefer in names:
            names.remove(prefer)
            names.insert(0, prefer)
        for w in names:
            data = self._get_blob(w, path)
            if data is not None:
                c = self._commit(w)
                return w, data, c.id.hex() if c else ""
        return None

    def write(
        self,
        path: str,
        content: str | bytes,
        session_id: str = "default",
        *,
        force: bool = False,
    ) -> FsResult:
        n, err = self._safe_path(path)
        if err:
            return err
        sess = self.session(session_id)
        data = content.encode("utf-8") if isinstance(content, str) else content
        existing = self._get_blob(sess.world, n)
        h_new = _hash_bytes(data)

        if existing is None:
            # create-fast (also clears tombstone if re-creating)
            rev = self._commit_files(
                sess.world,
                {n: data},
                json.dumps(
                    {
                        "op": "create",
                        "path": n,
                        "created_by": sess.actor_id,
                        "ts": _now(),
                    },
                    ensure_ascii=False,
                ),
            )
            meta = self._meta()
            owners = meta.setdefault("file_owners", {})
            owners.setdefault(n, sess.actor_id)
            self._save_meta(meta)
            sess.last_seen[n] = LastSeen(h_new, rev, sess.world)
            sess.tombstones.get(sess.world, set()).discard(n)
            self._mark_touched(sess, n)
            if self.dual_write:
                self._write_disk(n, data)
            return FsResult(
                OK,
                "created",
                path=n,
                content=data,
                hash=h_new,
                world=sess.world,
                created=True,
            )

        # edit path
        h_cur = _hash_bytes(existing)
        if not force and n in sess.last_seen:
            if sess.last_seen[n].content_hash != h_cur:
                return FsResult(
                    STALE,
                    "File changed since last read. Read again, then write.",
                    path=n,
                    hash=h_cur,
                    world=sess.world,
                )
        if self.require_read_before_edit and n not in sess.last_seen and not force:
            return FsResult(STALE, "Read the file before editing.", path=n, world=sess.world)

        # policy on first mutation if missing
        meta = self._meta()
        policies = meta.setdefault("file_policy", {})
        if n not in policies and n not in sess.policies:
            pol = self._build_policy("edit", n, sess.actor_id)
            policies[n] = pol
            sess.policies[n] = pol
            self._save_meta(meta)
        else:
            pol = policies.get(n) or sess.policies.get(n) or {}

        denied = self._as_list(pol.get("denied_ops"))
        if "edit" in [x.lower() for x in denied]:
            return FsResult(FORBIDDEN_POLICY, "Edit not allowed for this file", path=n)

        owner = meta.get("file_owners", {}).get(n, "")
        foreign = owner and owner != sess.actor_id
        risk = float(pol.get("self_risk") or pol.get("criticality") or 0)
        isolate = False
        target_world = sess.world
        if foreign and (risk >= 70 or self.isolate_low_risk_foreign_edit):
            isolate = True
        if risk >= 80:
            isolate = True

        if isolate:
            target_world = f"world/iso_{sess.actor_id}_{_now()}"
            self._ensure_world(target_world, base=sess.world)
            # copy current file set then apply edit on isolation world
            snap = self._snapshot_files(sess.world)
            snap[n] = data
            rev = self._commit_files(
                target_world,
                snap,
                json.dumps(
                    {"op": "edit_iso", "path": n, "actor": sess.actor_id, "from": sess.world},
                    ensure_ascii=False,
                ),
            )
            # agent stays on original world for next ops unless we switch —
            # concept: isolation for safety of others; actor who edited may keep seeing new content
            # Switch session to iso world so their next read sees edit (seamless for them)
            sess.world = target_world
            sess.last_seen[n] = LastSeen(h_new, rev, target_world)
            self._mark_touched(sess, n)
            if self.dual_write:
                self._write_disk(n, data)
            return FsResult(
                OK,
                "updated",
                path=n,
                content=data,
                hash=h_new,
                world=target_world,
                details={"isolated": True},
            )

        rev = self._commit_files(
            sess.world,
            {n: data},
            json.dumps({"op": "edit", "path": n, "actor": sess.actor_id}, ensure_ascii=False),
        )
        sess.last_seen[n] = LastSeen(h_new, rev, sess.world)
        self._mark_touched(sess, n)
        if self.dual_write:
            self._write_disk(n, data)
        return FsResult(OK, "updated", path=n, content=data, hash=h_new, world=sess.world)

    def delete(self, path: str, session_id: str = "default") -> FsResult:
        n, err = self._safe_path(path)
        if err:
            return err
        sess = self.session(session_id)
        existing = self._get_blob(sess.world, n)
        if existing is None:
            return FsResult(NOT_FOUND, "File not found", path=n, world=sess.world)

        meta = self._meta()
        policies = meta.setdefault("file_policy", {})
        if n not in policies:
            pol = self._build_policy("delete", n, sess.actor_id)
            policies[n] = pol
            self._save_meta(meta)
        else:
            pol = policies[n]
        denied = [x.lower() for x in self._as_list(pol.get("denied_ops"))]
        if "delete" in denied:
            return FsResult(FORBIDDEN_POLICY, "Delete not allowed for this file", path=n)

        owner = meta.get("file_owners", {}).get(n, "")
        foreign = owner and owner != sess.actor_id
        # default isolate foreign delete
        if foreign:
            iso = f"world/iso_del_{sess.actor_id}_{_now()}"
            self._ensure_world(iso, base=sess.world)
            snap = self._snapshot_files(sess.world)
            snap.pop(n, None)
            self._commit_files(
                iso,
                snap,
                json.dumps({"op": "delete_iso", "path": n, "actor": sess.actor_id}, ensure_ascii=False),
            )
            # also remove from current world so *this* agent sees deletion
            rev = self._commit_files(
                sess.world,
                {},
                json.dumps({"op": "delete", "path": n, "actor": sess.actor_id}, ensure_ascii=False),
                delete_paths=[n],
            )
            sess.last_seen.pop(n, None)
            sess.tombstones.setdefault(sess.world, set()).add(n)
            if self.dual_write:
                self._delete_disk(n)
            return FsResult(OK, "deleted", path=n, world=sess.world, details={"isolated_backup": iso})

        rev = self._commit_files(
            sess.world,
            {},
            json.dumps({"op": "delete", "path": n, "actor": sess.actor_id}, ensure_ascii=False),
            delete_paths=[n],
        )
        sess.last_seen.pop(n, None)
        sess.tombstones.setdefault(sess.world, set()).add(n)
        if self.dual_write:
            self._delete_disk(n)
        return FsResult(OK, "deleted", path=n, world=sess.world)

    def move(self, src: str, dst: str, session_id: str = "default") -> FsResult:
        r = self.read(src, session_id)
        if not r.ok():
            return r
        w = self.write(dst, r.content or b"", session_id)
        if not w.ok():
            return w
        d = self.delete(src, session_id)
        if not d.ok():
            return d
        return FsResult(OK, "moved", path=dst, details={"from": src}, world=w.world)

    def copy(self, src: str, dst: str, session_id: str = "default") -> FsResult:
        r = self.read(src, session_id)
        if not r.ok():
            return r
        return self.write(dst, r.content or b"", session_id)

    def _build_policy(self, action: str, path: str, actor: str) -> Dict[str, Any]:
        if self.intention_fn:
            try:
                pol = self.intention_fn(action, path, actor)
                if isinstance(pol, dict):
                    return pol
            except Exception:
                pass
        return {
            "purpose": f"file {path}",
            "allowed_ops": ["read", "edit", "delete", "copy", "move"],
            "denied_ops": [],
            "criticality": 30,
            "self_risk": 20,
            "created_by": actor,
        }

    @staticmethod
    def _as_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return [str(v)]
