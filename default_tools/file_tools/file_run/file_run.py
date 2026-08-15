'''
file_run
Run a script or command in the chat files folder. Pass the relative path and optional args.
Run file / script
Execute a file from the project workspace
'''

import os
import subprocess
import sys
from pathlib import Path

from cross_gpt import chat_path, filesystem_project_path, status_failed
from filesystem.tool_arg import one_line_arg as _one_line_arg


def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = ('ok_prefix', 'err_args', 'err_file')
        main.ok_prefix = 'Exit code '
        main.err_args = 'Provide a relative file path to run.'
        main.err_file = 'File not found: '
        return
    raw = _one_line_arg(text)
    if not raw:
        return main.err_args
    parts = raw.split()
    rel = parts[0]
    args = parts[1:]
    root = Path(filesystem_project_path or (Path(chat_path or '.') / 'files')).resolve()
    if not filesystem_project_path and chat_path:
        root = (Path(chat_path).resolve() / 'files')
    path = Path(rel)
    if not path.is_absolute():
        path = (root / rel).resolve()
    if not path.exists() or not path.is_file():
        return main.err_file + str(path)
    cmd = [sys.executable, str(path)] + args if path.suffix == '.py' else [str(path)] + args
    try:
        proc = subprocess.run(
            cmd, cwd=str(path.parent), capture_output=True, text=True, timeout=120)
        out = (proc.stdout or '') + (proc.stderr or '')
        return f"{main.ok_prefix}{proc.returncode}\n{out[:8000]}"
    except Exception as e:
        return f"{status_failed}: {e}"
