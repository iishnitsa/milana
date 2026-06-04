'''
smart_file_operation
perform create/read/edit/delete/copy/move via natural language using filesystem pipeline and model-based path/action parsing; Send a natural language request, e.g. "create file notes.txt with text 'hello'", "delete old report.pdf", "move source.txt to backup/source.txt". Tool detects action and paths automatically
Smart file operation
Parses user request into file operation and executes it through filesystem
'''

import os
import re
import json
import difflib
import importlib.util

from cross_gpt import ask_model, chat_path


def _load_filesystem():
    if hasattr(main, '_filesystem_mod'):
        return main._filesystem_mod
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    fs_path = os.path.join(base_dir, 'filesystem.py')
    spec = importlib.util.spec_from_file_location('tests_filesystem_runtime', fs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    main._filesystem_mod = mod
    return mod


def _workspace():
    if chat_path:
        project_path = os.path.join(chat_path, 'project')
        if os.path.isdir(project_path):
            return project_path
        return chat_path
    return os.getcwd()


def _scan_files(root_path, max_files=800):
    out = []
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.venv', 'venv')]
        for name in files:
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, root_path).replace('\\', '/')
            out.append({'abs': abs_path, 'rel': rel_path, 'name': name})
            if len(out) >= max_files:
                return out
    return out


def _extract_json(text):
    if not text:
        return None
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


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


def _parse_request(text, files):
    low = text.lower()
    default_action = 'read'
    if any(k in low for k in ('созда', 'create')):
        default_action = 'create'
    elif any(k in low for k in ('замени', 'измени', 'редакт', 'edit', 'replace', 'update')):
        default_action = 'edit'
    elif any(k in low for k in ('удали', 'delete', 'remove')):
        default_action = 'delete'
    elif any(k in low for k in ('копир', 'copy')):
        default_action = 'copy'
    elif any(k in low for k in ('перемест', 'move', 'rename')):
        default_action = 'move'
    elif any(k in low for k in ('прочита', 'покажи', 'read', 'show')):
        default_action = 'read'

    files_view = '\n'.join([f"{i+1}. {f['rel']}" for i, f in enumerate(files[:150])])
    prompt = (
        "Разбери пользовательский запрос на файловую операцию.\n"
        "Верни ТОЛЬКО JSON:\n"
        "{\"action\":\"create|read|edit|delete|copy|move\","
        "\"source\":\"...\",\"target\":\"...\",\"content\":\"...\","
        "\"purpose\":\"...\",\"experimental_mode\":false}\n"
        "Если значение неизвестно, верни пустую строку.\n"
        f"Запрос:\n{text}\n\n"
        f"Список файлов:\n{files_view}"
    )
    try:
        raw = ask_model(prompt, all_user=True)
        parsed = _extract_json(raw)
    except Exception:
        parsed = None

    if not parsed:
        parsed = {}
    parsed.setdefault('action', default_action)
    parsed.setdefault('source', '')
    parsed.setdefault('target', '')
    parsed.setdefault('content', '')
    parsed.setdefault('purpose', '')
    parsed.setdefault('experimental_mode', False)
    return parsed


def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'empty_files_text',
            'cannot_parse_text',
            'need_source_text',
            'need_target_text',
            'done_text',
            'failed_text',
            'forbidden_text',
        )
        main.empty_files_text = 'В рабочей папке нет файлов'
        main.cannot_parse_text = 'Не удалось разобрать запрос'
        main.need_source_text = 'Для операции нужен source'
        main.need_target_text = 'Для операции нужен target'
        main.done_text = 'Готово'
        main.failed_text = 'Ошибка'
        main.forbidden_text = 'Запрещено'
        return

    fs = _load_filesystem()
    workspace = _workspace()
    files = _scan_files(workspace)

    parsed = _parse_request(text, files)
    action = str(parsed.get('action') or '').strip().lower()
    if action == 'update':
        action = 'edit'
    if action == 'remove':
        action = 'delete'
    if action == 'rename':
        action = 'move'
    if action not in ('create', 'read', 'edit', 'delete', 'copy', 'move'):
        return f"{main.cannot_parse_text}: action"

    src_hint = str(parsed.get('source') or '').strip()
    dst_hint = str(parsed.get('target') or '').strip()
    content = str(parsed.get('content') or '')
    purpose = str(parsed.get('purpose') or '').strip()
    experimental_mode = bool(parsed.get('experimental_mode'))

    src_item = _pick_path(src_hint, files) if src_hint else None
    dst_item = _pick_path(dst_hint, files) if dst_hint else None

    source_path = src_item['abs'] if src_item else src_hint
    target_path = dst_item['abs'] if dst_item else dst_hint

    if action in ('read', 'edit', 'delete', 'copy', 'move') and not source_path:
        return main.need_source_text
    if action in ('copy', 'move') and not target_path:
        return main.need_target_text
    if action == 'create':
        if not target_path:
            target_path = source_path
        if not target_path:
            return main.need_target_text

    def write_actor(payload):
        return {'status': 'success', 'content': payload.get('arg', {}).get('content', '')}

    actor = None
    handler_arg = {}
    if action in ('create', 'edit'):
        actor = write_actor
        handler_arg['content'] = content

    intention = {}
    if purpose:
        intention['purpose'] = purpose
    intention['self_risk'] = '50'

    result = fs.pipeline(
        actor,
        source_path if source_path else target_path,
        for_actor=target_path,
        action=action,
        handler_arg=handler_arg,
        repo_path=workspace,
        use_git=False,
        intention=intention,
        experimental_mode=experimental_mode,
    )

    if result.get('status') == status_forbidden:
        return f"{main.forbidden_text}: {result.get('reason')}"
    if result.get('status') != status_success:
        return f"{main.failed_text}: {result.get('reason')}"
    return f"{main.done_text}: {json.dumps(result, ensure_ascii=False)}"


status_success = "success"
status_forbidden = "forbidden"

