'''
smart_file_operation
perform create/read/edit/delete/copy/move via natural language using filesystem pipeline and model-based path/action parsing; Send a natural language request, e.g. "create file notes.txt with text 'hello'", "delete old report.pdf", "move source.txt to backup/source.txt". Several ops in one call: separate with ";" or JSON array
Smart file operation
Parses user request into file operation(s) and executes through filesystem; cascade via ";" or JSON list
'''

import os
import re
import json
import difflib
from pathlib import Path

from cross_gpt import ask_model, chat_path, filesystem_project_path
from filesystem.api import pipeline, status_success, status_forbidden
from filesystem.tool_arg import clean_tool_arg as _clean_tool_arg


def _workspace():
    if filesystem_project_path:
        return filesystem_project_path
    if chat_path:
        project_path = os.path.join(chat_path, 'project')
        if os.path.isdir(project_path):
            return project_path
        files_path = os.path.join(chat_path, 'files')
        if os.path.isdir(files_path):
            return files_path
        return chat_path
    return os.getcwd()


def _scan_files(root_path, max_files=800):
    out = []
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.venv', 'venv', '.fs_work')]
        for name in files:
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, root_path).replace('\\', '/')
            out.append({'abs': abs_path, 'rel': rel_path, 'name': name})
            if len(out) >= max_files:
                return out
    return out


def _strip_code_fences(text):
    """Убрать ```json ... ``` / ``` ... ``` — иначе json.loads падает."""
    if not text:
        return ""
    s = str(text).strip()
    s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _normalize_op(data):
    """Один dict операции → action/source/target/content/purpose."""
    if not isinstance(data, dict):
        return None
    action = str(data.get("action") or "").strip().lower()
    src = str(data.get("source") or data.get("src") or data.get("from") or data.get("path") or "").strip()
    dst = str(data.get("target") or data.get("dst") or data.get("to") or data.get("filename") or "").strip()
    content = data.get("content")
    if content is None:
        content = ""
    else:
        content = str(content)
    purpose = str(data.get("purpose") or "").strip()
    if not action:
        return None
    return {
        "action": action,
        "source": src,
        "target": dst,
        "content": content,
        "purpose": purpose,
        "experimental_mode": bool(data.get("experimental_mode")),
    }


def _normalize_ops_list(data):
    """list/dict → list of normalized op dicts (empty if nothing usable)."""
    if isinstance(data, dict):
        n = _normalize_op(data)
        return [n] if n else []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        n = _normalize_op(item)
        if n:
            out.append(n)
    return out


def _looks_like_json_payload(text):
    s = _strip_code_fences(text or "").lstrip()
    return s.startswith("{") or s.startswith("[")


def _extract_json_ops(text):
    """
    dict / list / fenced ```json → list of op dicts, or None if not JSON ops.
    """
    if not text:
        return None
    text = _strip_code_fences(text)
    candidates = [text]
    m_arr = re.search(r"\[[\s\S]*\]", text)
    if m_arr:
        candidates.append(m_arr.group(0))
    m_obj = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.S)
    if not m_obj:
        m_obj = re.search(r"\{[\s\S]*\}", text)
    if m_obj:
        candidates.append(m_obj.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        ops = _normalize_ops_list(data)
        if ops:
            return ops
    return None


# Next segment after ";" looks like a new file op (not content with ";")
_CASCADE_NEXT_OP = (
    r"(?:созда\w*|create|скопир\w*|копир\w*|copy|перемест\w*|переимен\w*|move|rename|"
    r"удали\w*|delete|remove|прочита\w*|покажи|read|show|измени|замени|"
    r"edit|update|mkdir)\b"
)


def _split_cascade_text(text):
    """
    Split NL by ';' only when the next part starts with an action verb.
    'text a; b' without action → one piece. Cascade when model wrote several ops.
    """
    if not text or ";" not in text:
        return [text]
    parts = re.split(r";\s*(?=" + _CASCADE_NEXT_OP + r")", text, flags=re.I)
    parts = [p.strip() for p in parts if p and str(p).strip()]
    return parts if len(parts) > 1 else [text]


def _path_score(hint, candidate):
    if not hint:
        return 0.0
    hint_low = hint.lower().strip()
    rel_low = candidate['rel'].lower()
    name_low = candidate['name'].lower()
    ratio_rel = difflib.SequenceMatcher(None, hint_low, rel_low).ratio()
    ratio_name = difflib.SequenceMatcher(None, hint_low, name_low).ratio()
    contains = 1.0 if (hint_low in rel_low or hint_low in name_low) else 0.0
    return max(ratio_rel, ratio_name) * 0.7 + contains * 0.3


def _pick_path(hint, files):
    if not hint:
        return None
    best = None
    best_score = -1.0
    for item in files:
        score = _path_score(hint, item)
        if score > best_score:
            best_score = score
            best = item
    if best_score < 0.25:
        return None
    return best


_PATH_TOKEN = r'(?:[\w./\\-]+\.[\w]+|[\w./\\-]+/[\w./\\-]+)'


def _heuristic_parse(text):
    """
    Regex-first: не ждём идеальный JSON от слабой модели.
    """
    t = (text or "").strip()
    low = t.lower()

    # copy / move: "скопируй A в B", "copy A to B", "A -> B"
    m = re.search(
        r'(?:скопир\w*|copy)\s+(?:файл(?:а|у)?\s+)?["«]?(%s)["»]?\s+(?:в|to|into|->)\s+["«]?(%s)["»]?'
        % (_PATH_TOKEN, _PATH_TOKEN),
        t,
        re.I,
    )
    if m:
        return "copy", {"source": m.group(1).replace("\\", "/"), "target": m.group(2).replace("\\", "/"), "content": "", "purpose": ""}

    m = re.search(
        r'(?:перемест\w*|переимен\w*|move|rename)\s+(?:файл(?:а|у)?\s+)?["«]?(%s)["»]?\s+(?:в|to|into|->)\s+["«]?(%s)["»]?'
        % (_PATH_TOKEN, _PATH_TOKEN),
        t,
        re.I,
    )
    if m:
        return "move", {"source": m.group(1).replace("\\", "/"), "target": m.group(2).replace("\\", "/"), "content": "", "purpose": ""}

    m = re.search(r'["«]?(%s)["»]?\s*->\s*["«]?(%s)["»]?' % (_PATH_TOKEN, _PATH_TOKEN), t)
    if m and any(k in low for k in ("копир", "copy", "перемест", "move")):
        act = "move" if any(k in low for k in ("перемест", "move", "rename", "переимен")) else "copy"
        return act, {"source": m.group(1).replace("\\", "/"), "target": m.group(2).replace("\\", "/"), "content": "", "purpose": ""}

    # delete
    m = re.search(r'(?:удали|delete|remove)\s+(?:файл(?:а|у)?\s+)?["«]?(%s)["»]?' % _PATH_TOKEN, t, re.I)
    if m:
        return "delete", {"source": m.group(1).replace("\\", "/"), "target": "", "content": "", "purpose": ""}

    # read
    m = re.search(r'(?:прочитай|покажи|read|show)\s+(?:файл(?:а|у)?|содержимое\s+)?["«]?(%s)["»]?' % _PATH_TOKEN, t, re.I)
    if m:
        return "read", {"source": m.group(1).replace("\\", "/"), "target": "", "content": "", "purpose": ""}

    # create folder alone — not supported as empty path; signal special
    # ru: папка / директория / каталог; en: folder / directory / dir
    _folder_word = r'(?:папк\w*|директор\w*|каталог\w*|folder|director(?:y)?|dirs?)'
    if re.search(r'(?:созда\w*|create|mkdir)\s+' + _folder_word, low) or re.search(
        r'\bmkdir\b', low
    ):
        m = re.search(_folder_word + r'\s+["«]?(%s)["»]?' % _PATH_TOKEN, t, re.I)
        if not m:
            m = re.search(
                r'(?:созда\w*|create|mkdir)\s+' + _folder_word + r'\s+["«]?([^\s"«»]+)',
                t,
                re.I,
            )
        if not m:
            m = re.search(r'\bmkdir\s+["«]?([^\s"«»]+)', t, re.I)
        folder = (m.group(1) if m else "").replace("\\", "/").rstrip("/")
        return "mkdir", {"source": "", "target": folder, "content": "", "purpose": "folder"}

    # create file with text: создай файл path с текстом ...
    m = re.search(
        r'(?:созда\w*|create)\s+(?:файл\s+)?["«]?(%s)["»]?\s*(?:с\s+текстом|with\s+text|:)\s*[«"]?([\s\S]+?)[»"]?\s*$'
        % _PATH_TOKEN,
        t,
        re.I,
    )
    if m:
        return "create", {
            "source": "",
            "target": m.group(1).replace("\\", "/"),
            "content": m.group(2).strip().strip('"').strip("'"),
            "purpose": "",
        }

    m = re.search(r'(?:созда\w*|create)\s+(?:файл\s+)?["«]?(%s)["»]?' % _PATH_TOKEN, t, re.I)
    if m and Path(m.group(1)).suffix:
        # content after "текстом"
        content = ""
        mc = re.search(r'(?:текстом|text)\s*[:=]?\s*[«"]?([\s\S]+)', t, re.I)
        if mc:
            content = mc.group(1).strip().strip('"').strip("'")
        return "create", {"source": "", "target": m.group(1).replace("\\", "/"), "content": content, "purpose": ""}

    # edit / replace content
    m = re.search(
        r'(?:измени|замени|edit|update).*?["«]?(%s)["»]?.*?(?:на|to)\s*[«"]([^"»]+)[»"]'
        % _PATH_TOKEN,
        t,
        re.I | re.S,
    )
    if m:
        return "edit", {"source": m.group(1).replace("\\", "/"), "target": m.group(1).replace("\\", "/"), "content": m.group(2).strip(), "purpose": ""}

    return None, {}


def _fix_dir_target(src, dst):
    """Если target — папка (хвост / или без расширения), дописать basename source."""
    if not src or not dst:
        return dst
    dst = dst.replace("\\", "/").strip()
    src = src.replace("\\", "/").strip()
    if dst.endswith("/"):
        return dst + Path(src).name
    # no suffix and doesn't look like file → treat as directory
    if not Path(dst).suffix and not dst.endswith(Path(src).name):
        # e.g. demo_fs/archive
        return dst.rstrip("/") + "/" + Path(src).name
    return dst


def _is_folder_only_create(action, src, dst, content, text, purpose):
    if action != "create":
        return False
    path = (dst or src or "").rstrip("/")
    if content and str(content).strip():
        return False
    if Path(path).suffix:
        return False
    blob = f"{text} {purpose} {path}".lower()
    if re.search(r"папк|директор|каталог|folder|director|mkdir", blob):
        return True
    # path without extension and empty content
    return bool(path) and not Path(path).suffix


def _parse_one_request(text, files):
    """
    One NL fragment → (action, parsed_dict) or multi: (None, list_of_ops).
    Heuristic first; LLM may return one object or a cascade array.
    """
    h_action, h_parsed = _heuristic_parse(text)
    if h_action:
        return h_action, h_parsed

    low = (text or "").lower()
    action = "read"
    if any(k in low for k in getattr(main, "keywords_create", ["созда", "create"])):
        action = "create"
    elif any(k in low for k in getattr(main, "keywords_delete", ["удали", "delete"])):
        action = "delete"
    elif any(k in low for k in getattr(main, "keywords_move", ["перемест", "move"])):
        action = "move"
    elif any(k in low for k in getattr(main, "keywords_copy", ["копир", "copy"])):
        action = "copy"
    elif any(k in low for k in getattr(main, "keywords_edit", ["замени", "edit", "replace"])):
        action = "edit"

    file_list = "\n".join(f["rel"] for f in files[:200])
    prompt = (
        getattr(main, "prompt_parse_part1", "")
        + text
        + getattr(main, "prompt_parse_part2", "\nFiles:\n")
        + file_list
    )
    raw = ask_model(prompt, all_user=True)
    ops = _extract_json_ops(raw)
    if ops and len(ops) > 1:
        return None, ops
    if ops and len(ops) == 1:
        parsed = ops[0]
        return str(parsed.get("action") or action).lower(), parsed
    # merge: heuristic paths if LLM empty
    h2_a, h2_p = _heuristic_parse(text)
    if h2_a:
        return h2_a, h2_p
    return action, {}


def _collect_ops(text, files):
    """
    Build cascade plan: list of (action, parsed_dict, fragment).
    Sources: pure JSON array/object; NL split by ';' before next action verb; else one op.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Pure / obvious JSON payload → all ops (cascade if array)
    if _looks_like_json_payload(text):
        jops = _extract_json_ops(text)
        if jops:
            return [(op["action"], op, text) for op in jops]

    # NL cascade: "op1; op2; op3"
    segs = _split_cascade_text(text)
    if len(segs) > 1:
        plan = []
        for seg in segs:
            if _looks_like_json_payload(seg):
                jops = _extract_json_ops(seg)
                if jops:
                    for op in jops:
                        plan.append((op["action"], op, seg))
                    continue
            a, p = _parse_one_request(seg, files)
            if a is None and isinstance(p, list):
                for op in p:
                    plan.append((op["action"], op, seg))
            else:
                plan.append((a, p if isinstance(p, dict) else {}, seg))
        return plan

    # Single NL / maybe LLM returns array for whole text
    a, p = _parse_one_request(text, files)
    if a is None and isinstance(p, list):
        return [(op["action"], op, text) for op in p]
    return [(a, p if isinstance(p, dict) else {}, text)]


def _use_git_flag():
    try:
        from cross_gpt import global_state as _gs

        return bool(getattr(_gs, "fs_use_git", True))
    except Exception:
        return True


def _run_one_op(workspace, files, action, parsed, fragment):
    """
    Execute one FS op. Returns (kind, payload):
      kind: 'ok' | 'fail' | 'forbidden' | 'skip' | 'need'
      payload: result dict or message string
    """
    parsed = parsed or {}
    action = str(action or parsed.get("action") or "").strip().lower()
    src_hint = str(parsed.get("source") or "").strip().replace("\\", "/")
    dst_hint = str(parsed.get("target") or "").strip().replace("\\", "/")
    content = str(parsed.get("content") or "")
    purpose = str(parsed.get("purpose") or "").strip()
    experimental_mode = bool(parsed.get("experimental_mode"))
    frag = fragment or ""

    if not action:
        return "fail", main.cannot_parse_text

    if action == "mkdir" or _is_folder_only_create(
        action, src_hint, dst_hint, content, frag, purpose
    ):
        return "skip", main.folder_hint_text

    def _resolve_hint(hint, *, must_exist=False):
        if not hint:
            return None
        hint = hint.replace("\\", "/").strip().strip('"').strip("'")
        explicit = ("/" in hint) or ("." in Path(hint).name)
        if explicit:
            abs_p = os.path.join(workspace, hint) if not os.path.isabs(hint) else hint
            if must_exist and not os.path.isfile(abs_p):
                item = _pick_path(hint, files)
                return item["rel"] if item else hint
            return hint
        item = _pick_path(hint, files)
        if item:
            return item["rel"]
        return hint

    source_path = _resolve_hint(
        src_hint, must_exist=(action in ("read", "edit", "delete", "copy", "move"))
    )
    target_path = _resolve_hint(dst_hint, must_exist=False)

    if action in ("copy", "move"):
        target_path = _fix_dir_target(source_path or src_hint, target_path or dst_hint)
        if source_path and os.path.isabs(str(source_path)):
            try:
                source_path = os.path.relpath(source_path, workspace).replace("\\", "/")
            except Exception:
                pass
        if target_path and os.path.isabs(str(target_path)):
            try:
                target_path = os.path.relpath(target_path, workspace).replace("\\", "/")
            except Exception:
                pass
        if source_path and target_path and source_path == target_path:
            return (
                "fail",
                main.failed_text
                + ": source и target совпадают — укажи разный путь назначения "
                "(например demo_fs/archive/name.txt)",
            )

    if action in ("read", "edit", "delete", "copy", "move") and not source_path:
        return "need", main.need_source_text
    if action in ("copy", "move") and not target_path:
        return "need", main.need_target_text
    if action == "create":
        if not target_path:
            target_path = source_path
        if not target_path:
            return "need", main.need_target_text

    def _to_pipeline_path(p):
        if not p:
            return p
        p = str(p).replace("\\", "/")
        if os.path.isabs(p):
            return p
        return os.path.join(workspace, p)

    pipe_src = _to_pipeline_path(source_path if source_path else target_path)
    pipe_dst = _to_pipeline_path(target_path) if target_path else None

    def write_actor(payload):
        return {"status": "success", "content": payload.get("arg", {}).get("content", "")}

    actor = None
    handler_arg = {}
    if action in ("create", "edit"):
        actor = write_actor
        handler_arg["content"] = content

    intention = {"self_risk": "50"}
    if purpose:
        intention["purpose"] = purpose

    result = pipeline(
        actor,
        pipe_src,
        for_actor=pipe_dst,
        action=action,
        handler_arg=handler_arg,
        repo_path=workspace,
        use_git=_use_git_flag(),
        intention=intention,
        experimental_mode=experimental_mode,
    )

    if result.get("status") == status_forbidden:
        return "forbidden", f"{main.forbidden_text}: {result.get('reason')}"
    if result.get("status") != status_success:
        return "fail", f"{main.failed_text}: {result.get('reason')}"
    if action in ("copy", "move"):
        sp = result.get("source_path") or source_path
        tp = result.get("target_path") or target_path
        if sp and tp and str(sp).rstrip("/") == str(tp).rstrip("/"):
            return (
                "fail",
                f"{main.failed_text}: copy/move source==target ({sp}). "
                f"Нужен другой target, напр. demo_fs/archive/{Path(str(sp)).name}",
            )
    return "ok", result


def _format_cascade_report(steps, stopped_msg=None):
    """steps: list of {i, action, status, detail}."""
    total = len(steps)
    ok_n = sum(1 for s in steps if s["status"] in ("ok", "skip"))
    body = json.dumps(steps, ensure_ascii=False)
    if stopped_msg:
        return (
            f"{main.failed_text}: cascade {ok_n}/{total} "
            f"(останов на шаге {len(steps)}). {stopped_msg}\n{body}"
        )
    if total == 1 and steps[0]["status"] == "ok":
        return f"{main.done_text}: {json.dumps(steps[0]['detail'], ensure_ascii=False)}"
    if total == 1 and steps[0]["status"] == "skip":
        return steps[0]["detail"]
    return f"{main.done_text}: cascade {ok_n}/{total}\n{body}"


def main(text):
    if not hasattr(main, "attr_names"):
        main.attr_names = (
            "empty_files_text",
            "cannot_parse_text",
            "need_source_text",
            "need_target_text",
            "done_text",
            "failed_text",
            "forbidden_text",
            "folder_hint_text",
            "prompt_parse_part1",
            "prompt_parse_part2",
            "keywords_create",
            "keywords_edit",
            "keywords_delete",
            "keywords_copy",
            "keywords_move",
            "keywords_read",
        )
        main.empty_files_text = "В рабочей папке нет файлов"
        main.cannot_parse_text = "Не удалось разобрать запрос"
        main.need_source_text = "Для операции нужен source"
        main.need_target_text = "Для операции нужен target"
        main.done_text = "Готово"
        main.failed_text = "Ошибка"
        main.forbidden_text = "Запрещено"
        main.folder_hint_text = (
            "Папки создаются сами при записи/копировании/перемещении файла. "
            "Не вызывай create для папки. Сразу: copy/move с путём "
            "demo_fs/archive/file.txt (родительский каталог появится автоматически)."
        )
        main.prompt_parse_part1 = """
Разбери пользовательский запрос на файловую операцию (или несколько).
Верни ТОЛЬКО JSON: один объект ИЛИ массив объектов (каскад по порядку). Без markdown.
{"action":"create|read|edit|delete|copy|move","source":"...","target":"...","content":"...","purpose":"...","experimental_mode":false}
Правила:
- copy/move: source и target — разные пути к ФАЙЛАМ (не к папке без имени файла).
- Не используй action create для папок. Папки не создают отдельно.
- create: target = путь файла, content = текст.
- Несколько операций: массив [...] в порядке выполнения.
Запрос:
"""
        main.prompt_parse_part2 = """
Список файлов:
"""
        main.keywords_create = ["созда", "create"]
        main.keywords_edit = ["замени", "измени", "редакт", "edit", "replace", "update"]
        main.keywords_delete = ["удали", "delete", "remove"]
        main.keywords_copy = ["копир", "copy"]
        main.keywords_move = ["перемест", "move", "rename"]
        main.keywords_read = ["прочита", "покажи", "read", "show"]
        return

    text = _clean_tool_arg(text)
    if not text:
        return main.cannot_parse_text

    workspace = _workspace()
    files = _scan_files(workspace)
    plan = _collect_ops(text, files)
    if not plan:
        return main.cannot_parse_text

    steps = []
    for i, (action, parsed, fragment) in enumerate(plan, start=1):
        # refresh file index between steps (create then copy same path)
        if i > 1:
            files = _scan_files(workspace)
        kind, payload = _run_one_op(workspace, files, action, parsed, fragment)
        if kind == "skip":
            steps.append(
                {"i": i, "action": action or "mkdir", "status": "skip", "detail": payload}
            )
            continue
        if kind == "ok":
            steps.append(
                {"i": i, "action": action, "status": "ok", "detail": payload}
            )
            continue
        # fail / forbidden / need — stop cascade
        steps.append(
            {
                "i": i,
                "action": action or "?",
                "status": kind,
                "detail": payload if isinstance(payload, str) else payload,
            }
        )
        if len(plan) == 1:
            if kind in ("fail", "forbidden", "need") and isinstance(payload, str):
                return payload
            return f"{main.failed_text}: {payload}"
        return _format_cascade_report(steps, stopped_msg=str(payload))

    return _format_cascade_report(steps)
