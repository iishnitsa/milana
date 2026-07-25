'''
file_move
Move or rename a file or directory. One line: source path, then target path, separated by " -> " or newline.
Move or rename path
Simple move/rename without smart matching
'''

from pathlib import Path
from cross_gpt import chat_path, filesystem_project_path
from filesystem import pipeline, status_success, status_failed, status_forbidden


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
    raw = (text or '').strip()
    if '->' in raw:
        src, dst = [p.strip() for p in raw.split('->', 1)]
    else:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) < 2:
            return main.err_args
        src, dst = lines[0], lines[1]
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
