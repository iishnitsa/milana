'''
file_move
Move or rename a file or directory. One line: source path, then target path, separated by " -> " or newline.
Move or rename path
Simple move/rename without smart matching
'''

from pathlib import Path

from cross_gpt import chat_path, filesystem_project_path
from filesystem.api import pipeline, status_success, status_failed, status_forbidden
from filesystem.tool_arg import one_line_arg as _one_line_arg


def _workspace():
    if filesystem_project_path:
        return filesystem_project_path
    if chat_path:
        return str(Path(chat_path) / 'files')
    return str(Path('.').resolve())


def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = ('ok', 'err_args', 'err_src', 'forbidden_text')
        main.ok = 'Moved successfully.'
        main.err_args = 'Need source and target paths (use "src -> dst").'
        main.err_src = 'Source not found: '
        main.forbidden_text = 'Forbidden'
        return
    # Одна рабочая строка: отбрасываем комментарии модели после args
    raw = _one_line_arg(text)
    if not raw:
        # fallback: две строки src / dst без комментариев
        lines = [ln.strip() for ln in str(text or '').splitlines() if ln.strip()]
        if len(lines) >= 2 and '->' not in lines[0]:
            src, dst = lines[0], lines[1]
        else:
            return main.err_args
    elif '->' in raw:
        src, dst = [p.strip() for p in raw.split('->', 1)]
    else:
        # вторая строка только если это путь, не ремарка
        lines = [ln.strip() for ln in str(text or '').splitlines() if ln.strip()]
        if len(lines) >= 2 and not lines[1].startswith('(') and not lines[1].startswith('*'):
            src, dst = lines[0], _one_line_arg(lines[1]) or lines[1]
        else:
            return main.err_args
    if not src or not dst:
        return main.err_args
    workspace = _workspace()
    result = pipeline(
        None,
        src,
        for_actor=dst,
        action='move',
        repo_path=workspace,
    )
    if result.get('status') == status_forbidden:
        return f"{main.forbidden_text}: {result.get('reason')}"
    if result.get('status') != status_success:
        return f"{status_failed}: {result.get('reason')}"
    return f"{main.ok} {src} -> {dst}"
