import os
import re
import json
import time
import math
import hashlib
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from dulwich import porcelain
from dulwich.repo import Repo
from dulwich.objects import Blob, Tree, Commit

# Импорты из cross_gpt
from cross_gpt import (
    chat_path,
    ask_model,
    get_embs,
    parse_prompt_response,
    coll_exec,
    filesystem_project_path,
    get_token_limit,
    get_text_tokens_coefficient,
    text_cutter,
    global_state,
    last_messages_marker,
    global_summary_marker,
    recent_summary_marker,
)

# -----------------------------------------------------------------------------
# Текстовые константы (для локализации)
# -----------------------------------------------------------------------------

status_success_text = "success"
status_failed_text = "failed"
status_forbidden_text = "forbidden"

file_not_found_text = "Ошибка! Файл не найден"
invalid_action_text = "Неподдерживаемое действие"
forbidden_path_text = "Путь вне рабочей директории проекта"
handler_required_text = "Для действия нужен обработчик"
dulwich_missing_text = "Dulwich не установлен, git-режим недоступлен"
repo_not_found_text = "Git-репозиторий не найден"
internal_error_text = "Внутренняя ошибка"
path_not_specified_text = "Не указан путь"
folder_not_exists_text = "Папка не существует"
cwd_changed_text = "Текущая директория изменена"
cwd_unchanged_text = "Текущая директория без изменений"
project_tree_obtained_text = "Дерево проекта получено"
project_tree_failed_text = "Стартовая папка не найдена"
operation_success_text = "Операция выполнена"
access_denied_text = "Доступ запрещён"
no_target_path_text = "Не указан путь назначения"
no_source_path_text = "Не указан исходный путь"
operation_forbidden_policy_text = "Операция запрещена политикой файла"
dependency_violation_text = "Операция нарушает зависимости"
no_data_in_graph_text = "По файлу нет данных в graph"
dependency_report_ready_text = "Отчёт зависимостей сформирован"
experiment_branch_created_text = "Экспериментальная ветка создана"
fallback_reason_text = "fallback"
git_mode_off_text = "git mode off"
plain_mode_text = "plain"

# -----------------------------------------------------------------------------
# Константы конфигурации
# -----------------------------------------------------------------------------

filesystem_collection_name = "filesystem_collection"
default_project_dir_name = "files"
default_temp_dir_name = ".filesystem_tmp"
default_graph_file_name = ".filesystem_graph.json"
default_blob_pool_name = ".filesystem_blob_pool"

self_risk_hard_block = 90.0
risk_branch_threshold = 70.0
criticality_branch_threshold = 80.0
big_file_threshold_bytes = 1024 * 512

allowed_actions = ["create", "read", "edit", "delete", "copy", "move"]

# -----------------------------------------------------------------------------
# Промпты намерения (purpose обязательно остаётся центральным полем)
# -----------------------------------------------------------------------------

intention_prompts = {
    "create": {
        "purpose": "Для чего предназначен этот файл?",
        "ttl": "До какого состояния прогресса файл актуален?",
        "allowed_ops": "Какие действия с файлом разрешены и когда?",
        "denied_ops": "Какие действия с файлом запрещены и когда?",
        "owner": "Кто может управлять этим файлом?",
        "change_recommendations": "Рекомендации по безопасным изменениям файла.",
        "criticality": "Критичность файла и риск проблем при изменении.",
        "self_risk": "Какие риски несёт создание файла?",
        "depends_on": "От каких файлов зависит этот файл? Укажи пути через запятую.",
    },
    "read": {
        "purpose": "Зачем читается файл и какая информация нужна?",
        "ttl": "Актуален ли файл для задачи и почему?",
        "allowed_ops": "Что разрешено делать после чтения файла?",
        "denied_ops": "Что запрещено делать после чтения файла?",
        "owner": "Кто должен иметь доступ к информации файла?",
        "change_recommendations": "Рекомендации по безопасному использованию информации.",
        "criticality": "Критичность содержимого и риск неверной трактовки.",
        "self_risk": "Какие риски несёт чтение файла?",
        "depends_on": "Какие зависимости файла критичны для чтения? Укажи пути через запятую.",
    },
    "edit": {
        "purpose": "Как меняется цель существования файла после редактирования?",
        "ttl": "Как изменится актуальность файла после редактирования?",
        "allowed_ops": "Что разрешено делать с файлом после редактирования?",
        "denied_ops": "Что запрещено делать с файлом после редактирования?",
        "owner": "Кто должен взаимодействовать с результатом редактирования?",
        "change_recommendations": "Рекомендации по безопасному редактированию.",
        "criticality": "Критичность редактирования и риск серьёзных проблем.",
        "self_risk": "Какие риски несёт редактирование?",
        "depends_on": "От каких файлов зависит новая версия файла? Укажи пути через запятую.",
    },
    "delete": {
        "purpose": "Почему файл можно удалить и почему он больше не нужен?",
        "ttl": "Потерял ли файл актуальность и почему?",
        "allowed_ops": "Что разрешено сделать перед удалением?",
        "denied_ops": "Что запрещено делать перед удалением?",
        "owner": "Кто может подтверждать удаление?",
        "change_recommendations": "Рекомендации по безопасному удалению.",
        "criticality": "Критичность удаления и риск серьёзных проблем.",
        "self_risk": "Какие риски несёт удаление?",
        "depends_on": "Какие зависимости нужно проверить перед удалением? Укажи пути через запятую.",
    },
    "copy": {
        "purpose": "Для чего нужна копия файла и какая у неё цель?",
        "ttl": "До какого момента копия актуальна?",
        "allowed_ops": "Что разрешено делать с копией файла?",
        "denied_ops": "Что запрещено делать с копией файла?",
        "owner": "Кто должен работать с копией файла?",
        "change_recommendations": "Рекомендации по безопасной работе с копией.",
        "criticality": "Критичность копирования и риск проблем.",
        "self_risk": "Какие риски несёт копирование?",
        "depends_on": "От каких файлов должна зависеть копия? Укажи пути через запятую.",
    },
    "move": {
        "purpose": "Для чего перемещение и какая цель у файла после перемещения?",
        "ttl": "Как меняется актуальность файла после перемещения?",
        "allowed_ops": "Что разрешено делать после перемещения?",
        "denied_ops": "Что запрещено делать после перемещения?",
        "owner": "Кто отвечает за файл после перемещения?",
        "change_recommendations": "Рекомендации по безопасному перемещению.",
        "criticality": "Критичность перемещения и риск серьёзных проблем.",
        "self_risk": "Какие риски несёт перемещение?",
        "depends_on": "Какие зависимости нужно сохранить после перемещения? Укажи пути через запятую.",
    },
}

# -----------------------------------------------------------------------------
# Базовые утилиты
# -----------------------------------------------------------------------------

def now_ts():
    return time.time()


def now_int_ts():
    return int(time.time())


def to_posix_rel(path_text):
    return PurePosixPath(str(path_text).replace("\\", "/")).as_posix()


def normalize_action(action):
    if action is None:
        return ""
    action = str(action).strip().lower()
    if action == "update":
        return "edit"
    if action == "remove":
        return "delete"
    if action == "rename":
        return "move"
    if action == "replace":
        return "edit"
    if action in ("execute", "execution", "run", "exec", "исполнение", "выполнить", "запустить"):
        return "read"
    return action


def safe_float(value, default_value=50.0):
    try:
        return float(value)
    except Exception:
        return float(default_value)


def to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]
    return [text]


def to_lower_list(value):
    return [x.lower() for x in to_list(value)]


def sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def make_result(status, reason, action, **extra):
    base = {
        "status": status,
        "ok": status == status_success_text,
        "reason": reason,
        "action": action,
        "ts": now_ts(),
    }
    for key, value in extra.items():
        base[key] = value
    return base


def ensure_dir(path_text):
    os.makedirs(path_text, exist_ok=True)


def remove_file_or_dir(path_text):
    if not os.path.lexists(path_text):
        return
    if os.path.islink(path_text):
        os.unlink(path_text)
        return
    if os.path.isdir(path_text):
        shutil.rmtree(path_text)
        return
    os.remove(path_text)


# -----------------------------------------------------------------------------
# Интеграция с cross_gpt (прямые вызовы)
# -----------------------------------------------------------------------------

def resolve_workspace_path(repo_path=None, prefer_chat_project=True, project_dir_name=default_project_dir_name):
    if repo_path:
        return str(Path(repo_path).resolve())

    if prefer_chat_project:
        filesystem_path = str(filesystem_project_path or "").strip()
        if filesystem_path:
            return str(Path(filesystem_path).resolve())
        chat_path_str = str(chat_path or "").strip()
        if chat_path_str:
            root = Path(chat_path_str).resolve()
            candidate = root / project_dir_name
            if candidate.exists() and candidate.is_dir():
                return str(candidate)
            old_candidate = root / "project"
            if old_candidate.exists() and old_candidate.is_dir():
                return str(old_candidate)
            return str(root)

    return str(Path.cwd().resolve())


def resolve_rel_path(path_value, repo_root, cwd=None):
    if not path_value:
        return None, None

    repo_root_path = Path(repo_root).resolve()
    start_path = repo_root_path
    if cwd:
        start_path = (repo_root_path / cwd).resolve()

    raw = Path(str(path_value))
    if raw.is_absolute():
        abs_path = raw.resolve()
    else:
        abs_path = (start_path / raw).resolve()

    try:
        rel = abs_path.relative_to(repo_root_path)
    except Exception:
        return None, forbidden_path_text

    rel_text = to_posix_rel(str(rel))
    if rel_text in ("", "."):
        return None, path_not_specified_text
    return rel_text, None


# -----------------------------------------------------------------------------
# Папки/дерево проекта для агентов
# -----------------------------------------------------------------------------

def change_dir(current_dir, target_dir, repo_path=None):
    repo_root = resolve_workspace_path(repo_path=repo_path)
    base = Path(repo_root).resolve()

    current_rel = str(current_dir or "").strip()
    if current_rel:
        current_abs = (base / current_rel).resolve()
    else:
        current_abs = base

    if not str(current_abs).startswith(str(base)):
        current_abs = base

    target = str(target_dir or "").strip()
    if not target:
        return make_result(status_success_text, cwd_unchanged_text, "change_dir", cwd=to_posix_rel(str(current_abs.relative_to(base))))

    if Path(target).is_absolute():
        new_abs = Path(target).resolve()
    else:
        new_abs = (current_abs / target).resolve()

    if not str(new_abs).startswith(str(base)):
        return make_result(status_forbidden_text, forbidden_path_text, "change_dir", cwd=to_posix_rel(str(current_abs.relative_to(base))))

    if not new_abs.exists() or not new_abs.is_dir():
        return make_result(status_failed_text, folder_not_exists_text, "change_dir", cwd=to_posix_rel(str(current_abs.relative_to(base))))

    rel = to_posix_rel(str(new_abs.relative_to(base)))
    return make_result(status_success_text, cwd_changed_text, "change_dir", cwd=rel)


def get_project_tree_json(repo_path=None, cwd=None, max_depth=8, include_hidden=False):
    repo_root = resolve_workspace_path(repo_path=repo_path)
    base = Path(repo_root).resolve()

    start = base
    if cwd:
        start = (base / cwd).resolve()

    if not str(start).startswith(str(base)):
        return make_result(status_forbidden_text, forbidden_path_text, "project_tree")

    if not start.exists() or not start.is_dir():
        return make_result(status_failed_text, project_tree_failed_text, "project_tree")

    def walk_dir(path_obj, depth):
        node = {
            "name": path_obj.name if depth > 0 else ".",
            "path": to_posix_rel(str(path_obj.relative_to(base))) if depth > 0 else ".",
            "type": "dir",
            "children": [],
        }
        if depth >= max_depth:
            node["truncated"] = True
            return node

        try:
            entries = sorted(list(path_obj.iterdir()), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception as error:
            node["error"] = str(error)
            return node

        for child in entries:
            name = child.name
            if not include_hidden and name.startswith("."):
                continue
            if child.is_dir():
                node["children"].append(walk_dir(child, depth + 1))
            else:
                file_node = {
                    "name": name,
                    "path": to_posix_rel(str(child.relative_to(base))),
                    "type": "file",
                }
                try:
                    file_node["size"] = child.stat().st_size
                except Exception:
                    file_node["size"] = None
                node["children"].append(file_node)
        return node

    tree = walk_dir(start, 0)
    return make_result(status_success_text, project_tree_obtained_text, "project_tree", tree=tree)


# -----------------------------------------------------------------------------
# Работа с graph состояния файлов и зависимостей
# -----------------------------------------------------------------------------

def graph_file_path(repo_root):
    return os.path.join(repo_root, default_graph_file_name)


def load_graph(repo_root):
    path_text = graph_file_path(repo_root)
    if not os.path.exists(path_text):
        return {
            "version": 1,
            "updated_at": now_ts(),
            "files": {},
            "events": [],
        }
    try:
        with open(path_text, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except Exception:
        data = {
            "version": 1,
            "updated_at": now_ts(),
            "files": {},
            "events": [],
        }

    if "files" not in data or not isinstance(data["files"], dict):
        data["files"] = {}
    if "events" not in data or not isinstance(data["events"], list):
        data["events"] = []
    data["updated_at"] = now_ts()
    return data


def save_graph(repo_root, graph):
    graph["updated_at"] = now_ts()
    path_text = graph_file_path(repo_root)
    with open(path_text, "w", encoding="utf-8") as file_obj:
        json.dump(graph, file_obj, ensure_ascii=False, indent=2)


def ensure_graph_file_node(graph, rel_path):
    files_map = graph.get("files")
    if rel_path not in files_map:
        files_map[rel_path] = {
            "path": rel_path,
            "purpose": "",
            "state": {},
            "depends_on": [],
            "required_by": [],
            "history": [],
            "updated_at": now_ts(),
        }
    return files_map[rel_path]


def pseudo_embedding(text, size=32):
    raw = hashlib.sha256(str(text).encode("utf-8", errors="replace")).digest()
    vector = []
    for index in range(size):
        b = raw[index % len(raw)]
        vector.append((float(b) / 255.0) * 2.0 - 1.0)
    return vector


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    if size <= 0:
        return 0.0
    aa = [float(x) for x in a[:size]]
    bb = [float(x) for x in b[:size]]
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(size):
        dot += aa[i] * bb[i]
        na += aa[i] * aa[i]
        nb += bb[i] * bb[i]
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


def get_cached_vector(text, collection_name=filesystem_collection_name):
    vector_key = sha256_text(text)
    vector_id = "fs_vec_" + vector_key

    # 1) пробуем вытащить из коллекции
    response = coll_exec(
        action="get",
        coll_name=collection_name,
        filters={"fs_kind": "vector_cache", "vector_key": vector_key},
        fetch=["ids", "embeddings"],
        first=False,
    )
    if isinstance(response, dict):
        embeddings = response.get("embeddings") or []
        if embeddings and isinstance(embeddings[0], list) and len(embeddings[0]) > 0:
            return embeddings[0]

    # 2) если не нашли — генерируем через get_embs (или fallback)
    vector = None
    if callable(get_embs):
        try:
            vector = get_embs(text)
        except Exception:
            vector = None

    if not vector:
        vector = pseudo_embedding(text)

    # 3) сохраняем в кэш коллекции
    meta = {
        "fs_kind": "vector_cache",
        "vector_key": vector_key,
        "timestamp": now_ts(),
    }
    coll_exec(action="delete", coll_name=collection_name, ids=[vector_id])
    coll_exec(
        action="add",
        coll_name=collection_name,
        ids=[vector_id],
        embeddings=[vector],
        metadatas=[meta],
        documents=[str(text)[:4000]],
    )

    return vector


def graph_sync_node_to_chroma(node, branch_name=None, commit_id=None, collection_name=filesystem_collection_name):
    path_text = node.get("path") or ""
    state_hash = ((node.get("state") or {}).get("hash") or "")
    purpose = node.get("purpose") or ""
    depends_on = node.get("depends_on") or []

    summary = {
        "path": path_text,
        "purpose": purpose,
        "state_hash": state_hash,
        "depends_on": depends_on,
        "branch": branch_name,
        "commit": commit_id,
        "updated_at": now_ts(),
    }
    summary_text = json.dumps(summary, ensure_ascii=False)
    vector = get_cached_vector(summary_text, collection_name=collection_name)

    node_id = "fs_node_" + sha256_text(path_text)
    meta = {
        "fs_kind": "file_node",
        "path": path_text,
        "state_hash": state_hash,
        "branch": branch_name,
        "commit": commit_id,
        "timestamp": now_ts(),
    }

    coll_exec(action="delete", coll_name=collection_name, ids=[node_id])
    coll_exec(
        action="add",
        coll_name=collection_name,
        ids=[node_id],
        embeddings=[vector],
        metadatas=[meta],
        documents=[summary_text],
    )


def graph_sync_event_to_chroma(event_obj, collection_name=filesystem_collection_name):
    text = json.dumps(event_obj, ensure_ascii=False)
    vector = get_cached_vector(text, collection_name=collection_name)

    event_id = "fs_event_" + sha256_text(text + str(event_obj.get("timestamp")))
    meta = {
        "fs_kind": "commit_event",
        "action": event_obj.get("action"),
        "source_path": event_obj.get("source_path"),
        "target_path": event_obj.get("target_path"),
        "branch": event_obj.get("branch"),
        "commit": event_obj.get("commit"),
        "timestamp": event_obj.get("timestamp", now_ts()),
    }

    coll_exec(action="delete", coll_name=collection_name, ids=[event_id])
    coll_exec(
        action="add",
        coll_name=collection_name,
        ids=[event_id],
        embeddings=[vector],
        metadatas=[meta],
        documents=[text],
    )


def get_file_state_from_disk(repo_root, rel_path):
    abs_path = os.path.join(repo_root, rel_path)
    exists = os.path.exists(abs_path)
    state = {
        "path": rel_path,
        "exists": exists,
        "hash": None,
        "size": None,
        "mtime": None,
    }
    if not exists:
        return state
    try:
        st = os.stat(abs_path)
        state["size"] = st.st_size
        state["mtime"] = st.st_mtime
        with open(abs_path, "rb") as file_obj:
            data = file_obj.read()
        state["hash"] = sha256_bytes(data)
    except Exception:
        pass
    return state


def get_file_state_from_git(repo, branch_name, rel_path):
    content = get_branch_file_content(repo, branch_name, rel_path)
    state = {
        "path": rel_path,
        "exists": content is not None,
        "hash": None,
        "size": None,
        "mtime": None,
    }
    if content is None:
        return state
    state["size"] = len(content)
    state["hash"] = sha256_bytes(content)
    return state


def generate_file_state(repo_root, rel_path, use_git=False, repo=None, branch_name=None):
    if use_git and repo is not None and branch_name:
        return get_file_state_from_git(repo, branch_name, rel_path)
    return get_file_state_from_disk(repo_root, rel_path)


def collect_dependency_paths(raw_value):
    if raw_value is None:
        return []

    items = []
    if isinstance(raw_value, str):
        items = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        items = list(raw_value)
    elif isinstance(raw_value, dict):
        if "depends_on" in raw_value:
            items = to_list(raw_value.get("depends_on"))
        elif "dependencies" in raw_value:
            items = to_list(raw_value.get("dependencies"))
        else:
            items = []
    else:
        items = [str(raw_value)]

    paths = []
    path_regex = re.compile(r"[A-Za-z0-9_\-./\\]+\.[A-Za-z0-9_\-]+")
    for item in items:
        if isinstance(item, dict):
            value = item.get("path") or item.get("file") or item.get("target") or ""
            value = str(value).strip()
            if value:
                paths.append(value)
            continue

        text = str(item).strip()
        if not text:
            continue
        if "/" in text or "\\" in text or "." in Path(text).name:
            paths.append(text)
            continue
        found = path_regex.findall(text)
        for token in found:
            paths.append(token)

    unique = []
    seen = set()
    for p in paths:
        key = p.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def extract_dependencies(repo_root, action, rel_path, intention, handler_arg, cwd=None):
    raw = None
    if isinstance(handler_arg, dict):
        raw = handler_arg.get("depends_on")
        if raw is None:
            raw = handler_arg.get("dependencies")

    if raw is None and isinstance(intention, dict):
        raw = intention.get("depends_on")
        if raw is None:
            raw = intention.get("dependencies")

    dep_paths = collect_dependency_paths(raw)

    out = []
    for dep in dep_paths:
        dep_rel, dep_err = resolve_rel_path(dep, repo_root, cwd=cwd)
        if dep_err or not dep_rel:
            continue
        state = generate_file_state(repo_root, dep_rel, use_git=False)
        out.append({
            "path": dep_rel,
            "state": state,
            "state_hash": state.get("hash"),
            "exists": state.get("exists", False),
            "updated_at": now_ts(),
        })

    filtered = []
    for item in out:
        if item.get("path") == rel_path:
            continue
        filtered.append(item)

    return filtered


def refresh_reverse_links(graph):
    files_map = graph.get("files", {})

    for path_text, node in files_map.items():
        node["required_by"] = []

    for src_path, src_node in files_map.items():
        for dep in src_node.get("depends_on", []):
            dep_path = dep.get("path")
            if dep_path in files_map:
                files_map[dep_path]["required_by"].append({
                    "path": src_path,
                    "expected_state_hash": dep.get("state_hash"),
                    "updated_at": now_ts(),
                })


def add_history_record(node, action, branch_name=None, commit_id=None, note=None):
    history = node.get("history")
    if not isinstance(history, list):
        history = []
        node["history"] = history

    state_hash = ((node.get("state") or {}).get("hash") or "")
    history.append({
        "timestamp": now_ts(),
        "action": action,
        "branch": branch_name,
        "commit": commit_id,
        "state_hash": state_hash,
        "purpose": node.get("purpose", ""),
        "note": note,
    })

    if len(history) > 500:
        node["history"] = history[-500:]


def check_dependency_constraints(repo_root, graph, action, rel_path, experimental_mode=False):
    files_map = graph.get("files", {})
    node = files_map.get(rel_path)
    if not node:
        return {
            "allowed": True,
            "warnings": [],
            "errors": [],
        }

    warnings = []
    errors = []

    for dep in node.get("depends_on", []):
        dep_path = dep.get("path")
        expected_hash = dep.get("state_hash")
        if not dep_path:
            continue
        current_state = generate_file_state(repo_root, dep_path, use_git=False)
        current_hash = current_state.get("hash")
        if expected_hash and current_hash and expected_hash != current_hash:
            warnings.append(
                "Зависимость изменилась: {} (ожидалось {}, сейчас {})".format(dep_path, expected_hash[:12], current_hash[:12])
            )

    if action in ("delete", "move", "edit") and not experimental_mode:
        required_by = node.get("required_by", [])
        blocking = []
        for back in required_by:
            other_path = back.get("path")
            if not other_path:
                continue
            other_state = generate_file_state(repo_root, other_path, use_git=False)
            if other_state.get("exists"):
                blocking.append(other_path)

        if blocking:
            errors.append(
                "Невозможно действие '{}': файл '{}' нужен зависимым файлам: {}".format(
                    action,
                    rel_path,
                    ", ".join(blocking[:20]),
                )
            )

    return {
        "allowed": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }


# -----------------------------------------------------------------------------
# batched LLM вызовы (кроме parse_prompt_response)
# -----------------------------------------------------------------------------

def get_token_limit_from_bridge():
    if callable(get_token_limit):
        try:
            return int(get_token_limit())
        except Exception:
            pass
    return 8192


def get_token_coeff_from_bridge():
    if callable(get_text_tokens_coefficient):
        try:
            return float(get_text_tokens_coefficient())
        except Exception:
            pass
    return 1.0


def create_numbered_text(items):
    lines = []
    for i, text in enumerate(items, 1):
        lines.append("{}. {}".format(i, text))
    return "\n".join(lines)


def estimate_prompt_tokens(system_prompt, user_prompt):
    coeff = get_token_coeff_from_bridge()
    return (len(system_prompt or "") + len(user_prompt or "")) * coeff


def calculate_batch_size(items, start_index, system_prompt, reserve=1000, max_batch=20):
    token_limit = get_token_limit_from_bridge()
    remain = len(items) - start_index
    if remain <= 0:
        return 0

    max_try = min(max_batch, remain)
    for size in range(max_try, 0, -1):
        piece = items[start_index:start_index + size]
        user_prompt = create_numbered_text(piece)
        estimated = estimate_prompt_tokens(system_prompt, user_prompt)
        if estimated <= (token_limit - reserve):
            return size
    return 1


def parse_numbered_answers(text, expected_count):
    lines = []
    for raw in str(text or "").splitlines():
        row = raw.strip()
        if not row:
            continue
        row = re.sub(r"^\s*\d+\s*[\).:\-]\s*", "", row)
        lines.append(row)

    if len(lines) < expected_count:
        while len(lines) < expected_count:
            lines.append("")
    if len(lines) > expected_count:
        lines = lines[:expected_count]
    return lines


def ask_llm_batch(items, system_prompt):
    if not callable(ask_model):
        return [""] * len(items)

    out = []
    index = 0
    while index < len(items):
        batch_size = calculate_batch_size(items, index, system_prompt)
        piece = items[index:index + batch_size]
        numbered = create_numbered_text(piece)

        try:
            answer = ask_model(numbered, system_prompt=system_prompt)
        except RuntimeError as error:
            if "ContextOverflowError" in str(error) and callable(text_cutter):
                numbered = text_cutter(numbered)
                answer = ask_model(numbered, system_prompt=system_prompt)
            else:
                answer = ""
        except Exception:
            answer = ""

        parsed = parse_numbered_answers(answer, len(piece))
        out.extend(parsed)
        index += batch_size

    return out


# Кэш one-shot intention/permission (опция one_shot_intention_permission)
_fs_intention_cache = {}
_fs_permission_cache = {}

def determine_intention(action, prompt_text=None, global_summary=None, recent_summary=None):
    action = normalize_action(action)
    if action not in intention_prompts:
        action = "edit"

    # Разовая генерация намерения (default off) — кэш на задачу/action
    try:
        from cross_gpt import one_shot_intention_permission
        if one_shot_intention_permission and action in _fs_intention_cache:
            return dict(_fs_intention_cache[action])
    except Exception:
        pass

    if prompt_text is None:
        full_prompt = str(global_state.current_agent_history_for_filesystem or "")
        marker = str(last_messages_marker or "")
        if marker and marker in full_prompt:
            prompt_text = full_prompt.split(marker, 1)[1]
        else:
            prompt_text = full_prompt

    if global_summary is None:
        global_summary = "Нет данных"
    if recent_summary is None:
        recent_summary = "Нет данных"

    prompts = intention_prompts.get(action) or intention_prompts["edit"]
    keys = list(prompts.keys())

    agent_intention_prompt_1 = """Ты анализируешь намерения агента для файловой операции.
Ответь для каждого пункта одной строкой и по порядку.
Без нумерации, без markdown, без лишнего текста.
Учитывай контекст:
История:
"""
    # global_summary_marker
    system_prompt = (
        "Ты анализируешь намерения агента для файловой операции.\n"
        "Ответь для каждого пункта одной строкой и по порядку.\n"
        "Без нумерации, без markdown, без лишнего текста.\n"
        "Учитывай контекст:\n"
        "История:\n{}\n\nГлобальная сводка:\n{}\n\nКраткая сводка:\n{}\n".format(
            prompt_text or "Нет данных",
            global_summary,
            recent_summary,
        )
    )

    questions = []
    for key in keys:
        questions.append(prompts[key])

    answers = ask_llm_batch(questions, system_prompt)

    result = {}
    for i, key in enumerate(keys):
        result[key] = answers[i] if i < len(answers) else ""

    if not result.get("purpose"):
        result["purpose"] = "Выполнить действие '{}' с файлом".format(action)
    if "self_risk" not in result or not str(result.get("self_risk", "")).strip():
        result["self_risk"] = "50"
    if "criticality" not in result or not str(result.get("criticality", "")).strip():
        result["criticality"] = "50"

    try:
        from cross_gpt import one_shot_intention_permission
        if one_shot_intention_permission:
            _fs_intention_cache[action] = dict(result)
    except Exception:
        pass

    return result


# -----------------------------------------------------------------------------
# Git helpers (dulwich)
# -----------------------------------------------------------------------------

def open_repo(repo_root, git_folder=".git"):
    root = Path(repo_root).resolve()
    direct = str(root)
    dot_git = str((root / git_folder).resolve())

    errors = []
    for candidate in [direct, dot_git]:
        try:
            return Repo(candidate)
        except Exception as error:
            errors.append(str(error))

    discover = getattr(Repo, "discover", None)
    if callable(discover):
        try:
            return discover(str(root))
        except Exception as error:
            errors.append(str(error))

    raise FileNotFoundError(repo_not_found_text + ": " + " | ".join(errors))


def get_current_branch(repo):
    try:
        head_ref = repo.refs.read_ref(b"HEAD")
    except Exception:
        head_ref = None
    if isinstance(head_ref, bytes) and head_ref.startswith(b"refs/heads/"):
        return head_ref.decode("utf-8", errors="replace").split("refs/heads/", 1)[1]
    return None


def get_branch_head_commit_id(repo, branch_name):
    ref = ("refs/heads/" + str(branch_name)).encode("utf-8")
    return repo.refs.get(ref)


def parse_tree_items(tree_obj):
    for item in tree_obj.items():
        if isinstance(item, tuple) and len(item) == 3:
            name, mode, sha = item
        elif isinstance(item, tuple) and len(item) == 2:
            name, value = item
            if not isinstance(value, tuple) or len(value) != 2:
                continue
            mode, sha = value
        else:
            continue

        if isinstance(name, str):
            name = name.encode("utf-8")
        yield name, int(mode), sha


def tree_to_map(tree_obj):
    out = {}
    for name, mode, sha in parse_tree_items(tree_obj):
        out[name] = (mode, sha)
    return out


def build_tree_from_map(repo, items_map):
    tree = Tree()
    for name in sorted(items_map.keys()):
        tree[name] = items_map[name]
    repo.object_store.add_object(tree)
    return tree.id


def create_empty_tree(repo):
    tree = Tree()
    repo.object_store.add_object(tree)
    return tree.id


def create_blob(repo, data):
    blob = Blob.from_string(bytes(data))
    repo.object_store.add_object(blob)
    return blob.id


def get_commit(repo, commit_id):
    return repo[commit_id]


def get_file_content_from_commit(repo, commit_obj, rel_path):
    parts = [p for p in to_posix_rel(rel_path).split("/") if p and p != "."]
    if not parts:
        return None

    tree = repo[commit_obj.tree]
    for index, part in enumerate(parts):
        key = part.encode("utf-8")
        try:
            mode, sha = tree[key]
        except Exception:
            return None

        is_last = index == len(parts) - 1
        if is_last:
            try:
                blob = repo[sha]
                return bytes(blob.data)
            except Exception:
                return None

        if int(mode) != 0o040000:
            return None
        tree = repo[sha]

    return None


def get_branch_file_content(repo, branch_name, rel_path):
    commit_id = get_branch_head_commit_id(repo, branch_name)
    if not commit_id:
        return None
    commit_obj = get_commit(repo, commit_id)
    return get_file_content_from_commit(repo, commit_obj, rel_path)


def ensure_branch(repo, branch_name, base_branch=None):
    branch_ref = ("refs/heads/" + str(branch_name)).encode("utf-8")
    if branch_ref in repo.refs:
        return branch_name

    parent_id = None
    if base_branch:
        base_ref = ("refs/heads/" + str(base_branch)).encode("utf-8")
        parent_id = repo.refs.get(base_ref)
        if parent_id is None:
            raise ValueError("Базовая ветка не найдена: {}".format(base_branch))
    else:
        current = get_current_branch(repo)
        if current:
            cur_ref = ("refs/heads/" + str(current)).encode("utf-8")
            parent_id = repo.refs.get(cur_ref)
        if parent_id is None:
            try:
                parent_id = repo.head()
            except Exception:
                parent_id = None

    if parent_id is None:
        raise ValueError("Невозможно создать ветку без родительского коммита")

    repo.refs[branch_ref] = parent_id
    return branch_name


def update_tree_with_file(repo, base_tree_id, rel_path, blob_id, file_mode=0o100644):
    parts = [p for p in to_posix_rel(rel_path).split("/") if p and p != "."]
    if not parts:
        raise ValueError("Не указан целевой путь")

    def recurse(tree_id, idx):
        if tree_id:
            items = tree_to_map(repo[tree_id])
        else:
            items = {}

        key = parts[idx].encode("utf-8")

        if idx == len(parts) - 1:
            items[key] = (file_mode, blob_id)
            return build_tree_from_map(repo, items)

        old_mode, old_child_sha = items.get(key, (0o040000, None))
        if old_child_sha is None or int(old_mode) != 0o040000:
            old_child_sha = None

        new_child_sha = recurse(old_child_sha, idx + 1)
        items[key] = (0o040000, new_child_sha)
        return build_tree_from_map(repo, items)

    return recurse(base_tree_id, 0)


def remove_path_from_tree(repo, base_tree_id, rel_path):
    parts = [p for p in to_posix_rel(rel_path).split("/") if p and p != "."]
    if not parts:
        return base_tree_id, False

    def recurse(tree_id, idx):
        if tree_id is None:
            return None, False

        items = tree_to_map(repo[tree_id])
        key = parts[idx].encode("utf-8")
        if key not in items:
            return tree_id, False

        if idx == len(parts) - 1:
            del items[key]
            if not items:
                return None, True
            return build_tree_from_map(repo, items), True

        mode, child_sha = items[key]
        if int(mode) != 0o040000:
            return tree_id, False

        new_child_sha, changed = recurse(child_sha, idx + 1)
        if not changed:
            return tree_id, False

        if new_child_sha is None:
            del items[key]
        else:
            items[key] = (0o040000, new_child_sha)

        if not items:
            return None, True
        return build_tree_from_map(repo, items), True

    return recurse(base_tree_id, 0)


def create_commit(repo, branch_name, tree_id, parents, message):
    if tree_id is None:
        tree_id = create_empty_tree(repo)

    commit = Commit()
    commit.tree = tree_id
    commit.parents = list(parents or [])
    commit.author = b"AI Filesystem <agent@local>"
    commit.committer = b"AI Filesystem <agent@local>"
    t = now_int_ts()
    commit.author_time = t
    commit.commit_time = t
    commit.author_timezone = 0
    commit.commit_timezone = 0
    commit.message = str(message).encode("utf-8", errors="replace")

    repo.object_store.add_object(commit)
    ref = ("refs/heads/" + str(branch_name)).encode("utf-8")
    repo.refs[ref] = commit.id
    return commit.id


def commit_id_to_hex(commit_id):
    if isinstance(commit_id, bytes):
        return commit_id.hex()
    if commit_id is None:
        return None
    return str(commit_id)


# -----------------------------------------------------------------------------
# История файла по веткам
# -----------------------------------------------------------------------------

def get_file_commit_history_by_branch(file_path, repo_path=".", git_folder=".git"):
    if not file_path:
        return {}

    repo = open_repo(repo_path, git_folder=git_folder)
    rel_path = to_posix_rel(file_path)
    rel_bytes = rel_path.encode("utf-8")
    out = {}

    refs = porcelain.get_refs(repo) if porcelain else {}
    for ref_name, commit_id in refs.items():
        if not isinstance(ref_name, bytes) or not ref_name.startswith(b"refs/heads/"):
            continue
        branch_name = ref_name.decode("utf-8", errors="replace").split("refs/heads/", 1)[1]
        walker = repo.get_walker(include=[commit_id], paths=[rel_bytes])

        commits = []
        for entry in walker:
            c = entry.commit
            message = c.message.decode("utf-8", errors="replace").strip()
            commits.append((c.id.hex(), message))

        if commits:
            out[branch_name] = commits

    return out


# -----------------------------------------------------------------------------
# Проверка разрешений по намерениям и purpose
# -----------------------------------------------------------------------------

def parse_commit_message(message):
    try:
        data = json.loads(str(message).strip())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def purpose_similarity(current_purpose, commit_context):
    if not current_purpose or not commit_context:
        return 0.0
    v1 = get_cached_vector(current_purpose)
    v2 = get_cached_vector(commit_context)
    return cosine_similarity(v1, v2)


def is_action_allowed(filename, current_intention, commit_intention, action, experimental_mode=False, similarity_threshold=0.72):
    action = normalize_action(action)

    self_risk = safe_float(current_intention.get("self_risk", 0), 0)
    if self_risk > self_risk_hard_block and action not in ("read", "copy"):
        return False, "Действие запрещено: self_risk={} > {} (разрешены только read/copy)".format(self_risk, self_risk_hard_block)

    denied_ops = to_lower_list(commit_intention.get("denied_ops"))
    if (not experimental_mode) and action in denied_ops:
        return False, "Действие '{}' запрещено политикой файла (denied_ops)".format(action)

    allowed_ops = to_lower_list(commit_intention.get("allowed_ops"))
    if allowed_ops and action not in allowed_ops:
        return False, "Действие '{}' не входит в allowed_ops={}".format(action, allowed_ops)

    current_purpose = str(current_intention.get("purpose") or "").strip()
    commit_context = str(commit_intention.get("purpose") or commit_intention.get("history") or "").strip()

    if not current_purpose or not commit_context:
        return True, "purpose/history не заданы, мягкое разрешение"

    sim = purpose_similarity(current_purpose, commit_context)
    if sim >= similarity_threshold:
        return True, "purpose совместим (similarity={:.3f})".format(sim)

    if callable(parse_prompt_response):
        prompt = "Совместимы ли цель действия и текущее состояние объекта?"
        info = (
            "Файл: {}\n"
            "Действие: {}\n"
            "Текущая цель: {}\n"
            "Цель/история объекта: {}\n"
            "Ответь только да или нет."
        ).format(filename, action, current_purpose, commit_context)
        try:
            verdict = parse_prompt_response(prompt, info, 0)
            if verdict == 1:
                return True, "parse_prompt_response подтвердил совместимость при low similarity={:.3f}".format(sim)
            return False, "parse_prompt_response отклонил совместимость (similarity={:.3f})".format(sim)
        except Exception:
            pass

    return True, "Низкая similarity={:.3f}, но fallback разрешил действие".format(sim)


def generate_new_branch_name(action, prefix="aifs"):
    return "{}/{}-{}".format(prefix, normalize_action(action) or "op", now_int_ts())


def select_branch_or_create(history, current_intention, action, current_branch, experimental_mode=False):
    action = normalize_action(action)
    self_risk = safe_float(current_intention.get("self_risk", 50), 50)
    criticality = safe_float(current_intention.get("criticality", 50), 50)

    best_branch = None
    best_score = -1.0
    best_reason = ""

    current_purpose = str(current_intention.get("purpose") or "")

    for branch_name, commits in history.items():
        if not commits:
            continue
        _, commit_message = commits[0]
        commit_intention = parse_commit_message(commit_message)

        allowed, reason = is_action_allowed(
            filename="",
            current_intention=current_intention,
            commit_intention=commit_intention,
            action=action,
            experimental_mode=experimental_mode,
        )
        if not allowed:
            continue

        commit_context = str(commit_intention.get("history") or commit_intention.get("purpose") or "")
        score = purpose_similarity(current_purpose, commit_context)
        if branch_name == current_branch:
            score += 0.05

        if score > best_score:
            best_score = score
            best_branch = branch_name
            best_reason = reason

    if best_branch is None:
        base_branch = current_branch or (list(history.keys())[0] if history else "main")
        source_branch = None if action == "create" else base_branch
        return {
            "decision": "create_new",
            "source_branch": source_branch,
            "target_branch": generate_new_branch_name(action),
            "base_branch": base_branch,
            "reason": "Совместимая ветка не найдена",
            "score": best_score,
        }

    if not experimental_mode:
        force_new = (
            action in ("delete", "move")
            or self_risk >= risk_branch_threshold
            or criticality >= criticality_branch_threshold
        )
        if force_new:
            return {
                "decision": "create_new",
                "source_branch": best_branch,
                "target_branch": generate_new_branch_name(action),
                "base_branch": best_branch,
                "reason": "Изоляция риска: self_risk={}, criticality={}".format(self_risk, criticality),
                "score": best_score,
            }

    if best_branch == current_branch:
        return {
            "decision": "use_current",
            "source_branch": best_branch,
            "target_branch": best_branch,
            "base_branch": best_branch,
            "reason": "Используем текущую ветку: {}".format(best_reason),
            "score": best_score,
        }

    return {
        "decision": "switch_to",
        "source_branch": best_branch,
        "target_branch": best_branch,
        "base_branch": best_branch,
        "reason": "Переключение на более подходящую ветку: {}".format(best_reason),
        "score": best_score,
    }


# -----------------------------------------------------------------------------
# Обработчик файлов и оптимизация больших одинаковых файлов через symlink/hardlink
# -----------------------------------------------------------------------------

def write_bytes_with_link_optimization(repo_root, target_abs_path, content_bytes, min_size=big_file_threshold_bytes):
    ensure_dir(os.path.dirname(target_abs_path))

    if content_bytes is None:
        content_bytes = b""
    content = bytes(content_bytes)

    if len(content) < int(min_size):
        remove_file_or_dir(target_abs_path)
        with open(target_abs_path, "wb") as file_obj:
            file_obj.write(content)
        return {"linked": False, "mode": "regular"}

    pool_dir = os.path.join(repo_root, default_blob_pool_name)
    ensure_dir(pool_dir)

    h = sha256_bytes(content)
    blob_path = os.path.join(pool_dir, h + ".blob")

    if not os.path.exists(blob_path):
        with open(blob_path, "wb") as blob_file:
            blob_file.write(content)

    remove_file_or_dir(target_abs_path)

    try:
        os.symlink(blob_path, target_abs_path)
        return {"linked": True, "mode": "symlink", "blob": blob_path}
    except Exception:
        pass

    try:
        os.link(blob_path, target_abs_path)
        return {"linked": True, "mode": "hardlink", "blob": blob_path}
    except Exception:
        pass

    with open(target_abs_path, "wb") as file_obj:
        file_obj.write(content)
    return {"linked": False, "mode": "regular", "blob": blob_path}


def call_actor(actor_func, payload, temp_file_path, handler_arg):
    if not callable(actor_func):
        return None

    attempts = []
    attempts.append((payload,))
    attempts.append((temp_file_path, handler_arg))
    attempts.append((temp_file_path,))
    attempts.append((handler_arg,))
    attempts.append(tuple())

    last_error = None
    for args in attempts:
        try:
            return actor_func(*args)
        except TypeError as error:
            last_error = error
            continue

    if last_error:
        raise last_error
    raise RuntimeError("Не удалось вызвать обработчик")


def normalize_handler_output(output):
    if output is None:
        return {
            "status": status_success_text,
            "content": None,
            "data": None,
        }

    if isinstance(output, (bytes, bytearray)):
        return {
            "status": status_success_text,
            "content": bytes(output),
            "data": output,
        }

    if isinstance(output, str):
        return {
            "status": status_success_text,
            "content": output.encode("utf-8"),
            "data": output,
        }

    if isinstance(output, dict):
        status = str(output.get("status") or status_success_text).lower()
        if output.get("forbidden") is True:
            status = status_forbidden_text
        reason = str(output.get("reason") or "")

        content = output.get("content")
        if isinstance(content, str):
            content = content.encode("utf-8")
        elif isinstance(content, (bytes, bytearray)):
            content = bytes(content)

        return {
            "status": status,
            "reason": reason,
            "content": content,
            "data": output,
        }

    return {
        "status": status_success_text,
        "content": None,
        "data": output,
    }


def produce_content_with_handler(repo_root, action, source_path, target_path, existing_content, actor_func, handler_arg, cwd=None):
    if (action in ("create", "edit")) and not callable(actor_func):
        if isinstance(handler_arg, str):
            return {"status": status_success_text, "content": handler_arg.encode("utf-8"), "data": handler_arg}
        if isinstance(handler_arg, (bytes, bytearray)):
            return {"status": status_success_text, "content": bytes(handler_arg), "data": handler_arg}
        return {"status": status_failed_text, "reason": handler_required_text}

    temp_root = os.path.join(repo_root, default_temp_dir_name)
    ensure_dir(temp_root)

    with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
        name_hint = Path(target_path or source_path or "buffer.txt").name
        temp_file = os.path.join(temp_dir, name_hint)

        if existing_content is not None:
            write_bytes_with_link_optimization(repo_root, temp_file, existing_content, min_size=big_file_threshold_bytes)

        payload = {
            "action": action,
            "temp_file": temp_file,
            "source_path": source_path,
            "target_path": target_path,
            "repo_path": repo_root,
            "cwd": cwd,
            "arg": handler_arg,
            "existing_content": existing_content,
        }

        try:
            output = call_actor(actor_func, payload, temp_file, handler_arg)
        except Exception as error:
            return {"status": status_failed_text, "reason": str(error)}

        parsed = normalize_handler_output(output)

        if parsed.get("status") in (status_failed_text, status_forbidden_text):
            return parsed

        if parsed.get("content") is not None:
            return {
                "status": status_success_text,
                "content": parsed.get("content"),
                "data": parsed.get("data"),
            }

        if os.path.exists(temp_file):
            with open(temp_file, "rb") as file_obj:
                return {
                    "status": status_success_text,
                    "content": file_obj.read(),
                    "data": parsed.get("data"),
                }

        return {
            "status": status_failed_text,
            "reason": "Обработчик не выдал контент и не создал временный файл",
            "data": parsed.get("data"),
        }


# -----------------------------------------------------------------------------
# Применение операций (plain mode)
# -----------------------------------------------------------------------------

def apply_plain_operation(repo_root, action, params, actor_func):
    source_rel = params.get("source_path")
    target_rel = params.get("target_path")
    handler_arg = params.get("handler_arg")
    cwd = params.get("cwd")

    source_abs = os.path.join(repo_root, source_rel) if source_rel else None
    target_abs = os.path.join(repo_root, target_rel) if target_rel else None

    if action == "read":
        if not source_abs or not os.path.exists(source_abs):
            return make_result(status_failed_text, file_not_found_text, action, source_path=source_rel)

        with open(source_abs, "rb") as file_obj:
            content = file_obj.read()

        if callable(actor_func):
            processed = produce_content_with_handler(
                repo_root=repo_root,
                action="read",
                source_path=source_rel,
                target_path=target_rel,
                existing_content=content,
                actor_func=actor_func,
                handler_arg=handler_arg,
                cwd=cwd,
            )
            if processed.get("status") == status_forbidden_text:
                return make_result(status_forbidden_text, processed.get("reason", access_denied_text), action, source_path=source_rel)
            if processed.get("status") == status_failed_text:
                return make_result(status_failed_text, processed.get("reason", internal_error_text), action, source_path=source_rel)
            return make_result(status_success_text, operation_success_text, action, source_path=source_rel, data=processed.get("data", content))

        return make_result(status_success_text, operation_success_text, action, source_path=source_rel, data=content)

    if action in ("create", "edit"):
        existing = None
        destination_abs = target_abs or source_abs

        if action == "edit":
            if not source_abs or not os.path.exists(source_abs):
                return make_result(status_failed_text, file_not_found_text, action, source_path=source_rel)
            with open(source_abs, "rb") as file_obj:
                existing = file_obj.read()

        processed = produce_content_with_handler(
            repo_root=repo_root,
            action=action,
            source_path=source_rel,
            target_path=target_rel,
            existing_content=existing,
            actor_func=actor_func,
            handler_arg=handler_arg,
            cwd=cwd,
        )
        if processed.get("status") == status_forbidden_text:
            return make_result(status_forbidden_text, processed.get("reason", access_denied_text), action, source_path=source_rel, target_path=target_rel)
        if processed.get("status") == status_failed_text:
            return make_result(status_failed_text, processed.get("reason", internal_error_text), action, source_path=source_rel, target_path=target_rel)

        if not destination_abs:
            return make_result(status_failed_text, no_target_path_text, action)

        info = write_bytes_with_link_optimization(repo_root, destination_abs, processed.get("content", b""))
        return make_result(status_success_text, operation_success_text, action, source_path=source_rel, target_path=target_rel, storage=info)

    if action == "delete":
        if not source_abs or not os.path.exists(source_abs):
            return make_result(status_failed_text, file_not_found_text, action, source_path=source_rel)
        remove_file_or_dir(source_abs)
        return make_result(status_success_text, operation_success_text, action, source_path=source_rel)

    if action == "copy":
        if not source_abs or not os.path.exists(source_abs):
            return make_result(status_failed_text, file_not_found_text, action, source_path=source_rel)
        if not target_abs:
            return make_result(status_failed_text, no_target_path_text, action)

        with open(source_abs, "rb") as file_obj:
            content = file_obj.read()
        info = write_bytes_with_link_optimization(repo_root, target_abs, content)
        return make_result(status_success_text, operation_success_text, action, source_path=source_rel, target_path=target_rel, storage=info)

    if action == "move":
        if not source_abs or not os.path.exists(source_abs):
            return make_result(status_failed_text, file_not_found_text, action, source_path=source_rel)
        if not target_abs:
            return make_result(status_failed_text, no_target_path_text, action)

        ensure_dir(os.path.dirname(target_abs))
        shutil.move(source_abs, target_abs)
        return make_result(status_success_text, operation_success_text, action, source_path=source_rel, target_path=target_rel)

    return make_result(status_failed_text, invalid_action_text + ": " + str(action), action)


# -----------------------------------------------------------------------------
# Применение операций (git mode)
# -----------------------------------------------------------------------------

def apply_git_operation(repo_root, action, params, actor_func, git_folder=".git"):
    action = normalize_action(action)

    repo = open_repo(repo_root, git_folder=git_folder)

    source_branch = params.get("source_branch")
    target_branch = params.get("target_branch")
    source_path = params.get("source_path")
    target_path = params.get("target_path")
    intention = dict(params.get("intention") or {})
    handler_arg = params.get("handler_arg")
    cwd = params.get("cwd")

    current_branch = get_current_branch(repo) or "main"
    if not source_branch:
        source_branch = current_branch
    if not target_branch and action != "read":
        target_branch = source_branch

    if source_branch:
        ensure_branch(repo, source_branch, base_branch=current_branch)
    if target_branch:
        ensure_branch(repo, target_branch, base_branch=source_branch or current_branch)

    if action == "read":
        if not source_path:
            return make_result(status_failed_text, no_source_path_text, action)

        data = get_branch_file_content(repo, source_branch, source_path)
        if data is None:
            return make_result(status_failed_text, file_not_found_text, action, source_path=source_path, source_branch=source_branch)

        if callable(actor_func):
            processed = produce_content_with_handler(
                repo_root=repo_root,
                action="read",
                source_path=source_path,
                target_path=target_path,
                existing_content=data,
                actor_func=actor_func,
                handler_arg=handler_arg,
                cwd=cwd,
            )
            if processed.get("status") == status_forbidden_text:
                return make_result(status_forbidden_text, processed.get("reason", access_denied_text), action)
            if processed.get("status") == status_failed_text:
                return make_result(status_failed_text, processed.get("reason", internal_error_text), action)
            return make_result(status_success_text, operation_success_text, action, source_path=source_path, source_branch=source_branch, data=processed.get("data", data))

        return make_result(status_success_text, operation_success_text, action, source_path=source_path, source_branch=source_branch, data=data)

    target_ref = ("refs/heads/" + str(target_branch)).encode("utf-8")
    parent_commit_id = repo.refs.get(target_ref)
    parent_commit = repo[parent_commit_id] if parent_commit_id else None
    base_tree_id = parent_commit.tree if parent_commit else None
    parents = [parent_commit_id] if parent_commit_id else []

    if action == "create":
        if not target_path:
            return make_result(status_failed_text, no_target_path_text, action)

        produced = produce_content_with_handler(
            repo_root=repo_root,
            action="create",
            source_path=source_path,
            target_path=target_path,
            existing_content=None,
            actor_func=actor_func,
            handler_arg=handler_arg,
            cwd=cwd,
        )
        if produced.get("status") == status_forbidden_text:
            return make_result(status_forbidden_text, produced.get("reason", access_denied_text), action)
        if produced.get("status") == status_failed_text:
            return make_result(status_failed_text, produced.get("reason", internal_error_text), action)

        blob_id = create_blob(repo, produced.get("content", b""))
        new_tree_id = update_tree_with_file(repo, base_tree_id, target_path, blob_id)

    elif action == "edit":
        if not source_path:
            return make_result(status_failed_text, no_source_path_text, action)

        current_content = get_branch_file_content(repo, source_branch, source_path)
        if current_content is None:
            return make_result(status_failed_text, file_not_found_text, action, source_path=source_path, source_branch=source_branch)

        produced = produce_content_with_handler(
            repo_root=repo_root,
            action="edit",
            source_path=source_path,
            target_path=target_path or source_path,
            existing_content=current_content,
            actor_func=actor_func,
            handler_arg=handler_arg,
            cwd=cwd,
        )
        if produced.get("status") == status_forbidden_text:
            return make_result(status_forbidden_text, produced.get("reason", access_denied_text), action)
        if produced.get("status") == status_failed_text:
            return make_result(status_failed_text, produced.get("reason", internal_error_text), action)

        destination = target_path or source_path
        blob_id = create_blob(repo, produced.get("content", b""))
        new_tree_id = update_tree_with_file(repo, base_tree_id, destination, blob_id)

    elif action == "delete":
        if not source_path:
            return make_result(status_failed_text, no_source_path_text, action)
        new_tree_id, removed = remove_path_from_tree(repo, base_tree_id, source_path)
        if not removed:
            return make_result(status_failed_text, file_not_found_text, action, source_path=source_path)

    elif action == "copy":
        if not source_path:
            return make_result(status_failed_text, no_source_path_text, action)
        if not target_path:
            return make_result(status_failed_text, no_target_path_text, action)

        source_content = get_branch_file_content(repo, source_branch, source_path)
        if source_content is None:
            return make_result(status_failed_text, file_not_found_text, action, source_path=source_path)

        blob_id = create_blob(repo, source_content)
        new_tree_id = update_tree_with_file(repo, base_tree_id, target_path, blob_id)

    elif action == "move":
        if not source_path:
            return make_result(status_failed_text, no_source_path_text, action)
        if not target_path:
            return make_result(status_failed_text, no_target_path_text, action)

        source_content = get_branch_file_content(repo, source_branch, source_path)
        if source_content is None:
            return make_result(status_failed_text, file_not_found_text, action, source_path=source_path)

        blob_id = create_blob(repo, source_content)
        moved_tree = update_tree_with_file(repo, base_tree_id, target_path, blob_id)
        new_tree_id, removed = remove_path_from_tree(repo, moved_tree, source_path)
        if not removed:
            return make_result(status_failed_text, file_not_found_text, action, source_path=source_path)

    else:
        return make_result(status_failed_text, invalid_action_text + ": " + str(action), action)

    commit_payload = dict(intention or {})
    commit_payload.setdefault("action", action)
    commit_payload.setdefault("source_path", source_path)
    commit_payload.setdefault("target_path", target_path)
    commit_payload.setdefault("history", commit_payload.get("purpose", ""))
    commit_payload["commit_ts"] = now_int_ts()

    commit_message = json.dumps(commit_payload, ensure_ascii=False)
    commit_id = create_commit(repo, target_branch, new_tree_id, parents, commit_message)
    commit_hex = commit_id_to_hex(commit_id)

    return make_result(
        status_success_text,
        operation_success_text,
        action,
        repo_path=repo_root,
        source_path=source_path,
        target_path=target_path,
        source_branch=source_branch,
        target_branch=target_branch,
        commit_id=commit_hex,
    )


def apply_file_operation(repo_path, action, params, actor_func, git_folder=".git"):
    action = normalize_action(action)
    if action not in allowed_actions:
        return make_result(status_failed_text, invalid_action_text + ": " + str(action), action)

    repo_root = resolve_workspace_path(repo_path)
    use_git = bool(params.get("use_git", True))

    if use_git:
        return apply_git_operation(repo_root, action, params, actor_func, git_folder=git_folder)
    return apply_plain_operation(repo_root, action, params, actor_func)


# -----------------------------------------------------------------------------
# Обновление graph после операции
# -----------------------------------------------------------------------------

def update_graph_after_operation(repo_root, graph, action, result_obj, intention, dependencies, experimental_mode=False):
    files_map = graph.get("files", {})

    source_path = result_obj.get("source_path")
    target_path = result_obj.get("target_path")
    branch_name = result_obj.get("target_branch") or result_obj.get("source_branch")
    commit_id = result_obj.get("commit_id")

    if action == "create":
        main_path = target_path
    elif action == "copy":
        main_path = target_path
    elif action == "move":
        main_path = target_path
    elif action == "delete":
        main_path = source_path
    else:
        main_path = target_path or source_path

    if not main_path:
        return

    node = ensure_graph_file_node(graph, main_path)

    if isinstance(intention, dict):
        new_purpose = str(intention.get("purpose") or "").strip()
        if new_purpose:
            node["purpose"] = new_purpose

    if action == "delete":
        state = {
            "path": main_path,
            "exists": False,
            "hash": None,
            "size": None,
            "mtime": None,
        }
    else:
        state = generate_file_state(repo_root, main_path, use_git=False)

    node["state"] = state
    node["updated_at"] = now_ts()

    if dependencies is not None:
        node["depends_on"] = list(dependencies)

    if action == "move" and source_path and source_path in files_map and source_path != target_path:
        old_node = files_map.get(source_path)
        if old_node and old_node is not node:
            history_to_merge = old_node.get("history", [])
            if history_to_merge:
                if not isinstance(node.get("history"), list):
                    node["history"] = []
                node["history"] = history_to_merge + node["history"]
                if len(node["history"]) > 500:
                    node["history"] = node["history"][-500:]
            files_map.pop(source_path, None)

    add_history_record(node, action, branch_name=branch_name, commit_id=commit_id)

    event_obj = {
        "timestamp": now_ts(),
        "action": action,
        "source_path": source_path,
        "target_path": target_path,
        "branch": branch_name,
        "commit": commit_id,
        "purpose": node.get("purpose"),
        "experimental_mode": bool(experimental_mode),
        "self_risk": safe_float((intention or {}).get("self_risk", 0), 0),
    }
    graph["events"].append(event_obj)
    if len(graph["events"]) > 2000:
        graph["events"] = graph["events"][-2000:]

    refresh_reverse_links(graph)

    graph_sync_node_to_chroma(node, branch_name=branch_name, commit_id=commit_id)
    graph_sync_event_to_chroma(event_obj)


# -----------------------------------------------------------------------------
# Экспериментальная ветка
# -----------------------------------------------------------------------------

def create_experiment_branch(repo_path=None, base_branch=None, branch_prefix="aifs-experiment", git_folder=".git"):
    repo_root = resolve_workspace_path(repo_path=repo_path)

    repo = open_repo(repo_root, git_folder=git_folder)

    if not base_branch:
        base_branch = get_current_branch(repo) or "main"

    new_branch = "{}/{}".format(branch_prefix, now_int_ts())
    ensure_branch(repo, base_branch, base_branch=base_branch)
    ensure_branch(repo, new_branch, base_branch=base_branch)

    return make_result(
        status_success_text,
        experiment_branch_created_text,
        "create_experiment_branch",
        repo_path=repo_root,
        base_branch=base_branch,
        branch=new_branch,
    )


# -----------------------------------------------------------------------------
# Главный pipeline
# -----------------------------------------------------------------------------

def pipeline(
    actor_func,
    filename,
    for_actor=None,
    action="create",
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
):
    action = normalize_action(action)
    if action not in allowed_actions:
        return make_result(status_failed_text, invalid_action_text + ": " + str(action), action)

    workspace = resolve_workspace_path(repo_path=repo_path, prefer_chat_project=prefer_chat_project)

    source_input = filename or None
    target_input = None

    if action in ("copy", "move"):
        target_input = for_actor
    elif action == "create":
        target_input = for_actor if for_actor else filename
    elif action == "edit" and for_actor:
        if "/" in str(for_actor) or "\\" in str(for_actor) or "." in Path(str(for_actor)).name:
            target_input = for_actor

    source_rel, source_err = resolve_rel_path(source_input, workspace, cwd=cwd)
    target_rel, target_err = resolve_rel_path(target_input, workspace, cwd=cwd)

    if source_err and action in ("read", "edit", "delete", "copy", "move"):
        return make_result(status_forbidden_text, source_err, action, repo_path=workspace)
    if target_err and target_input:
        return make_result(status_forbidden_text, target_err, action, repo_path=workspace)

    if action in ("copy", "move") and not target_rel:
        return make_result(status_failed_text, no_target_path_text, action, repo_path=workspace)

    # Намерение
    if intention is None:
        intention = determine_intention(action)
    if not isinstance(intention, dict):
        intention = {}
    if not intention.get("purpose"):
        intention["purpose"] = "Выполнить '{}' для файла".format(action)

    self_risk = safe_float(intention.get("self_risk", 0), 0)
    if self_risk > self_risk_hard_block and action not in ("read", "copy"):
        return make_result(
            status_forbidden_text,
            "Операция запрещена: self_risk={} > {} (разрешены только read/copy)".format(self_risk, self_risk_hard_block),
            action,
            repo_path=workspace,
            source_path=source_rel,
            target_path=target_rel,
            intention=intention,
        )

    graph = load_graph(workspace)

    op_path_for_check = source_rel if action in ("read", "edit", "delete", "move") else target_rel
    dep_check = check_dependency_constraints(
        repo_root=workspace,
        graph=graph,
        action=action,
        rel_path=op_path_for_check,
        experimental_mode=bool(experimental_mode),
    )

    if not dep_check.get("allowed"):
        return make_result(
            status_forbidden_text,
            " ; ".join(dep_check.get("errors", [])) or dependency_violation_text,
            action,
            repo_path=workspace,
            source_path=source_rel,
            target_path=target_rel,
            dependency_warnings=dep_check.get("warnings", []),
            dependency_errors=dep_check.get("errors", []),
            intention=intention,
        )

    decision = {
        "decision": plain_mode_text,
        "source_branch": None,
        "target_branch": None,
        "reason": git_mode_off_text,
        "score": 0.0,
    }

    if use_git:
        repo = open_repo(workspace, git_folder=git_folder)
        real_current_branch = current_branch or get_current_branch(repo) or "main"

        history_path = source_rel
        if action == "create":
            history_path = target_rel or source_rel

        history = {}
        if history_path:
            history = get_file_commit_history_by_branch(history_path, repo_path=workspace, git_folder=git_folder)

        if action in ("read", "edit", "delete", "copy", "move"):
            if not source_rel:
                return make_result(status_failed_text, no_source_path_text, action, repo_path=workspace)
            if not history:
                cur = get_branch_file_content(repo, real_current_branch, source_rel)
                if cur is None:
                    return make_result(status_failed_text, file_not_found_text, action, repo_path=workspace, source_path=source_rel)

        decision = select_branch_or_create(
            history=history,
            current_intention=intention,
            action=action,
            current_branch=real_current_branch,
            experimental_mode=bool(experimental_mode),
        )

    if use_git and action in ("read", "edit", "delete", "copy", "move") and not decision.get("source_branch"):
        decision["source_branch"] = current_branch or "main"
    if use_git and action != "read" and not decision.get("target_branch"):
        decision["target_branch"] = decision.get("source_branch") or current_branch or "main"

    if use_git and source_rel and action in ("read", "edit", "delete", "copy", "move", "create"):
        try:
            history_for_check = get_file_commit_history_by_branch(source_rel, repo_path=workspace, git_folder=git_folder)
        except Exception:
            history_for_check = {}

        src_branch = decision.get("source_branch")
        if src_branch in history_for_check and history_for_check.get(src_branch):
            last_message = history_for_check[src_branch][0][1]
            commit_int = parse_commit_message(last_message)
            allowed, reason = is_action_allowed(
                filename=source_rel,
                current_intention=intention,
                commit_intention=commit_int,
                action=action,
                experimental_mode=bool(experimental_mode),
            )
            if not allowed:
                return make_result(
                    status_forbidden_text,
                    "{}: {}".format(operation_forbidden_policy_text, reason),
                    action,
                    repo_path=workspace,
                    source_path=source_rel,
                    target_path=target_rel,
                    decision=decision,
                    intention=intention,
                )

    main_rel = source_rel if action in ("read", "edit", "delete", "move") else target_rel
    dependencies = extract_dependencies(
        repo_root=workspace,
        action=action,
        rel_path=main_rel,
        intention=intention,
        handler_arg=handler_arg,
        cwd=cwd,
    )

    params = {
        "source_branch": decision.get("source_branch"),
        "target_branch": decision.get("target_branch"),
        "source_path": source_rel,
        "target_path": target_rel,
        "intention": intention,
        "handler_arg": handler_arg,
        "cwd": cwd,
        "use_git": bool(use_git),
    }

    result_obj = apply_file_operation(
        repo_path=workspace,
        action=action,
        params=params,
        actor_func=actor_func,
        git_folder=git_folder,
    )

    result_obj["decision"] = decision.get("decision")
    result_obj["details"] = result_obj.get("details", {})
    result_obj["details"]["branch_reason"] = decision.get("reason")
    result_obj["details"]["branch_score"] = decision.get("score")
    result_obj["details"]["workspace"] = workspace
    result_obj["intention"] = intention
    result_obj["dependency_warnings"] = dep_check.get("warnings", [])

    if result_obj.get("status") == status_success_text:
        update_graph_after_operation(
            repo_root=workspace,
            graph=graph,
            action=action,
            result_obj=result_obj,
            intention=intention,
            dependencies=dependencies,
            experimental_mode=bool(experimental_mode),
        )
        save_graph(workspace, graph)

    return result_obj


# -----------------------------------------------------------------------------
# Дополнительная функция для агента: отчёт по зависимостям файла
# -----------------------------------------------------------------------------

def get_dependency_report(filename, repo_path=None, cwd=None):
    workspace = resolve_workspace_path(repo_path=repo_path)
    rel_path, err = resolve_rel_path(filename, workspace, cwd=cwd)
    if err:
        return make_result(status_forbidden_text, err, "dependency_report", repo_path=workspace)

    graph = load_graph(workspace)
    node = graph.get("files", {}).get(rel_path)
    if not node:
        return make_result(status_failed_text, no_data_in_graph_text, "dependency_report", repo_path=workspace, path=rel_path)

    depends_status = []
    for dep in node.get("depends_on", []):
        dep_path = dep.get("path")
        expected_hash = dep.get("state_hash")
        now_state = generate_file_state(workspace, dep_path, use_git=False)
        depends_status.append({
            "path": dep_path,
            "expected_hash": expected_hash,
            "current_hash": now_state.get("hash"),
            "ok": (expected_hash == now_state.get("hash")) if expected_hash and now_state.get("hash") else False,
            "exists": now_state.get("exists"),
        })

    out = {
        "path": rel_path,
        "purpose": node.get("purpose"),
        "state": node.get("state"),
        "depends_on": depends_status,
        "required_by": node.get("required_by", []),
        "history_count": len(node.get("history", [])),
    }

    return make_result(status_success_text, dependency_report_ready_text, "dependency_report", repo_path=workspace, report=out)