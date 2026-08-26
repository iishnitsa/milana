'''
dependency_report
analyze file dependency graph and explain whether an operation is safe or blocked with concrete dependency reasons; Send a request specifying a file and action (read/edit/delete/copy/move/create). Example: "show dependencies for main.py" or "what if I delete utils.py". Returns JSON report
Dependency report
Shows file dependencies and safety checks for requested operation
'''

import os
import re
import json
import difflib

from cross_gpt import (
    ask_model, filesystem_project_path,
    get_dependency_report, status_success, status_failed, status_forbidden,
    load_graph, check_dependency_constraints
)


def _scan_files(root_path, max_files=1000):
    out = []
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.venv', 'venv')]
        for name in files:
            rel_path = os.path.relpath(os.path.join(root, name), root_path).replace('\\', '/')
            out.append({'name': name, 'rel': rel_path, 'abs': os.path.join(root, name)})
            if len(out) >= max_files:
                return out
    return out


def _pick_path_by_hint(hint, files):
    if not hint:
        return None
    hint_low = hint.lower().strip()
    best = None
    best_score = -1.0
    for item in files:
        rel_low = item['rel'].lower()
        name_low = item['name'].lower()
        ratio = max(
            difflib.SequenceMatcher(None, hint_low, rel_low).ratio(),
            difflib.SequenceMatcher(None, hint_low, name_low).ratio(),
        )
        if hint_low in rel_low or hint_low in name_low:
            ratio += 0.2
        if ratio > best_score:
            best = item
            best_score = ratio
    if best_score < 0.2:
        return None
    return best


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


def _parse_request(text, files):
    low = text.lower()
    # Локализуемые ключевые слова для действий
    delete_keywords = main.delete_keywords
    edit_keywords = main.edit_keywords
    move_keywords = main.move_keywords
    copy_keywords = main.copy_keywords
    create_keywords = main.create_keywords
    read_keywords = main.read_keywords

    default_action = ''
    if any(k in low for k in delete_keywords):
        default_action = 'delete'
    elif any(k in low for k in edit_keywords):
        default_action = 'edit'
    elif any(k in low for k in move_keywords):
        default_action = 'move'
    elif any(k in low for k in copy_keywords):
        default_action = 'copy'
    elif any(k in low for k in create_keywords):
        default_action = 'create'
    elif any(k in low for k in read_keywords):
        default_action = 'read'

    files_view = '\n'.join([f"{i+1}. {f['rel']}" for i, f in enumerate(files[:200])])

    # Собираем промпт из частей
    prompt = main.prompt_parse_part1 + files_view + main.prompt_parse_part2

    try:
        raw = ask_model(prompt, all_user=True)
        parsed = _extract_json(raw)
    except Exception:
        parsed = None
    if not parsed:
        parsed = {}
    parsed.setdefault('file_hint', '')
    parsed.setdefault('action', default_action)
    return parsed


def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'file_not_selected_text',
            'done_text',
            'failed_text',
            'prompt_parse_part1',
            'prompt_parse_part2',
            'delete_keywords',
            'edit_keywords',
            'move_keywords',
            'copy_keywords',
            'create_keywords',
            'read_keywords',
        )
        main.file_not_selected_text = 'Не удалось выбрать файл для отчёта'
        main.done_text = 'Готово'
        main.failed_text = 'Ошибка'
        main.prompt_parse_part1 = """
Выдели из запроса путь файла и действие.
Верни ТОЛЬКО JSON:
{"file_hint":"...","action":"read|edit|delete|copy|move|create"}
Запрос:
"""
        main.prompt_parse_part2 = """

Файлы:
"""
        main.delete_keywords = ('удали', 'delete', 'remove')
        main.edit_keywords = ('измени', 'редакт', 'edit', 'replace', 'update')
        main.move_keywords = ('перемест', 'move')
        main.copy_keywords = ('копир', 'copy')
        main.create_keywords = ('созда', 'create')
        main.read_keywords = ('прочита', 'read')
        return

    workspace = filesystem_project_path
    files = _scan_files(workspace)
    parsed = _parse_request(text, files)

    file_hint = str(parsed.get('file_hint') or '').strip()
    action = str(parsed.get('action') or '').strip().lower()
    file_item = _pick_path_by_hint(file_hint, files) if file_hint else None
    if not file_item and files:
        file_item = _pick_path_by_hint(text, files)
    if not file_item:
        return main.file_not_selected_text

    rep = get_dependency_report(file_item['abs'], repo_path=workspace)
    if rep.get('status') != status_success:
        return f"{main.failed_text}: {rep.get('reason')}"

    report = rep.get('report', {})
    out = {
        'file': report.get('path'),
        'purpose': report.get('purpose'),
        'state': report.get('state'),
        'depends_on': report.get('depends_on', []),
        'required_by': report.get('required_by', []),
        'history_count': report.get('history_count', 0),
        'requested_action': action,
        'safety': {
            'allowed': None,
            'warnings': [],
            'errors': [],
        },
    }

    if action:
        graph = load_graph(workspace)
        chk = check_dependency_constraints(
            repo_root=workspace,
            graph=graph,
            action=action,
            rel_path=report.get('path'),
            experimental_mode=False,
        )
        out['safety'] = chk

    return f"{main.done_text}: {json.dumps(out, ensure_ascii=False, indent=2)}"