'''
project_navigator_tool
navigate project folders for agents using cd-like command and json tree output from filesystem; Send command: "cd folder" to change current directory, "tree" for tree of current folder, "tree 3" for depth 3
Project navigator
Provides change_dir and project tree for agent-friendly navigation
'''

import os
import json
import importlib.util

from cross_gpt import chat_path


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


def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'done_text',
            'failed_text',
            'help_text',
        )
        main.done_text = 'Готово'
        main.failed_text = 'Ошибка'
        main.help_text = (
            "Использование:\n"
            "- cd <путь>  -> смена текущей папки\n"
            "- tree        -> дерево текущей папки\n"
            "- tree <n>    -> дерево c глубиной n\n"
        )
        main.current_cwd = '.'
        return

    fs = _load_filesystem()
    workspace = _workspace()

    raw = str(text or '').strip()
    if not raw:
        return main.help_text

    low = raw.lower()
    if low.startswith('cd '):
        target = raw[3:].strip()
        result = fs.change_dir(main.current_cwd, target, repo_path=workspace)
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
        result = fs.get_project_tree_json(
            repo_path=workspace,
            cwd=main.current_cwd,
            max_depth=depth,
            include_hidden=False,
        )
        if result.get('status') != 'success':
            return f"{main.failed_text}: {result.get('reason')}"
        return f"{main.done_text}: {json.dumps(result.get('tree'), ensure_ascii=False)}"

    return main.help_text

