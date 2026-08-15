'''
project_navigator
navigate project folders for agents using cd-like command and json tree output from filesystem; Send command: "cd folder" to change current directory, "tree" for tree of current folder, "tree 3" for depth 3
Project navigator
Provides change_dir and project tree for agent-friendly navigation
'''

import json
import os

from cross_gpt import chat_path, filesystem_project_path
from filesystem.api import change_dir, get_project_tree_json
from filesystem.tool_arg import one_line_arg as _one_line_arg


def _workspace():
    if filesystem_project_path:
        return filesystem_project_path
    if chat_path:
        project_path = os.path.join(chat_path, 'files')
        if os.path.isdir(project_path):
            return project_path
        return chat_path
    return os.getcwd()


def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'done_text',
            'failed_text',
            'help_text',
            'empty_tree_text',
        )
        main.done_text = 'Готово'
        main.failed_text = 'Ошибка'
        main.empty_tree_text = 'пусто (нет файлов или папок)'
        main.help_text = (
            "Использование:\n"
            "- cd <путь>  -> смена текущей папки\n"
            "- tree        -> дерево текущей папки\n"
            "- tree <n>    -> дерево c глубиной n\n"
        )
        main.current_cwd = '.'
        return

    workspace = _workspace()
    raw = _one_line_arg(text)
    if not raw:
        return main.help_text

    low = raw.lower()
    if low.startswith('cd ') or low == 'cd':
        target = raw[3:].strip() if low.startswith('cd ') else '.'
        # path only — no leftover remarks
        target = target.split()[0] if target else '.'
        result = change_dir(main.current_cwd, target, repo_path=workspace)
        if result.get('status') != 'success':
            return f"{main.failed_text}: {result.get('reason')}"
        main.current_cwd = result.get('cwd', main.current_cwd)
        return f"{main.done_text}: {json.dumps(result, ensure_ascii=False)}"

    if low.startswith('tree'):
        parts = raw.split()
        depth = 8
        if len(parts) > 1:
            try:
                depth = int(parts[1])
            except Exception:
                depth = 8
        result = get_project_tree_json(
            repo_path=workspace,
            cwd=main.current_cwd,
            max_depth=depth,
            include_hidden=False,
        )
        if result.get('status') != 'success':
            return f"{main.failed_text}: {result.get('reason')}"
        tree = result.get('tree') if result.get('tree') is not None else result.get('files')
        if not tree:
            return f"{main.done_text}: {main.empty_tree_text}"
        return f"{main.done_text}: {json.dumps(tree, ensure_ascii=False)}"

    return main.help_text
