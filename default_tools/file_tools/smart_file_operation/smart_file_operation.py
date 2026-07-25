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

from cross_gpt import ask_model, chat_path, filesystem_project_path
from filesystem import pipeline, status_success, status_forbidden


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
    low = (text or '').lower()
    action = 'read'
    if any(k in low for k in getattr(main, 'keywords_create', ['созда', 'create'])):
        action = 'create'
    elif any(k in low for k in getattr(main, 'keywords_delete', ['удали', 'delete'])):
        action = 'delete'
    elif any(k in low for k in getattr(main, 'keywords_move', ['перемест', 'move'])):
        action = 'move'
    elif any(k in low for k in getattr(main, 'keywords_copy', ['копир', 'copy'])):
        action = 'copy'
    elif any(k in low for k in getattr(main, 'keywords_edit', ['замени', 'edit', 'replace'])):
        action = 'edit'

    file_list = '\n'.join(f['rel'] for f in files[:200])
    prompt = (
        getattr(main, 'prompt_parse_part1', '')
        + text
        + getattr(main, 'prompt_parse_part2', '\nFiles:\n')
        + file_list
    )
    raw = ask_model(prompt)
    parsed = _extract_json(raw) or {}
    if parsed.get('action'):
        action = str(parsed.get('action')).lower()
    return action, parsed


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
            'prompt_parse_part1',
            'prompt_parse_part2',
            'keywords_create',
            'keywords_edit',
            'keywords_delete',
            'keywords_copy',
            'keywords_move',
            'keywords_read',
        )
        main.empty_files_text = 'В рабочей папке нет файлов'
        main.cannot_parse_text = 'Не удалось разобрать запрос'
        main.need_source_text = 'Для операции нужен source'
        main.need_target_text = 'Для операции нужен target'
        main.done_text = 'Готово'
        main.failed_text = 'Ошибка'
        main.forbidden_text = 'Запрещено'
        main.prompt_parse_part1 = """
Разбери пользовательский запрос на файловую операцию.
Верни ТОЛЬКО JSON:
{"action":"create|read|edit|delete|copy|move","source":"...","target":"...","content":"...","purpose":"...","experimental_mode":false}
Если значение неизвестно, верни пустую строку.
Запрос:
"""
        main.prompt_parse_part2 = """
Список файлов:
"""
        main.keywords_create = ['созда', 'create']
        main.keywords_edit = ['замени', 'измени', 'редакт', 'edit', 'replace', 'update']
        main.keywords_delete = ['удали', 'delete', 'remove']
        main.keywords_copy = ['копир', 'copy']
        main.keywords_move = ['перемест', 'move', 'rename']
        main.keywords_read = ['прочита', 'покажи', 'read', 'show']
        return

    workspace = _workspace()
    files = _scan_files(workspace)
    action, parsed = _parse_request(text, files)

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

    result = pipeline(
        actor,
        source_path if source_path else target_path,
        for_actor=target_path,
        action=action,
        handler_arg=handler_arg,
        repo_path=workspace,
        use_git=True,
        intention=intention,
        experimental_mode=experimental_mode,
    )

    if result.get('status') == status_forbidden:
        return f"{main.forbidden_text}: {result.get('reason')}"
    if result.get('status') != status_success:
        return f"{main.failed_text}: {result.get('reason')}"
    return f"{main.done_text}: {json.dumps(result, ensure_ascii=False)}"
