'''
file_move
Move or rename a file or directory. One line: source path, then target path, separated by " -> " or newline.
Move or rename path
Simple move/rename without smart matching
'''

import os
import shutil
from pathlib import Path
from cross_gpt import chat_path, status_success, status_failed

def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = ('ok', 'err_args', 'err_src')
        main.ok = 'Moved successfully.'
        main.err_args = 'Need source and target paths (use "src -> dst").'
        main.err_src = 'Source not found: '
        return
    raw = (text or '').strip()
    if '->' in raw:
        src, dst = [p.strip() for p in raw.split('->', 1)]
    else:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) < 2:
            return main.err_args
        src, dst = lines[0], lines[1]
    root = Path(chat_path or '.').resolve()
    src_p = Path(src)
    dst_p = Path(dst)
    if not src_p.is_absolute():
        src_p = (root / 'files' / src).resolve()
    if not dst_p.is_absolute():
        dst_p = (root / 'files' / dst).resolve()
    if not src_p.exists():
        return main.err_src + str(src_p)
    try:
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_p), str(dst_p))
        return main.ok + f' {src_p} -> {dst_p}'
    except Exception as e:
        return f'{status_failed}: {e}'
