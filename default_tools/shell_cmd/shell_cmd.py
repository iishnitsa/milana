'''
run_command
Cross-platform shell (bash / cmd / zsh) in the agent workspace. May ask the user to confirm unless shell_skip_confirm is on.
Command line
Unified OS command module with optional confirmation dialog (chat setting shell_skip_confirm).
'''

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from cross_gpt import chat_path, global_state, confirm_shell_command, cacher

# --- module-level helpers (not redefined on every main call) ---

BANNED_COMMON = (
    'shutdown', 'reboot', 'mkfs', 'init', 'halt', 'poweroff',
    'kill ', 'killall', 'sudo', 'su',
    'format', 'taskkill', 'rmdir /s',
)

DANGEROUS_PATTERNS = (
    r'rm\s+-rf\s+/?$',
    r'rm\s+-rf\s+/\*',
    r'>\s*/dev/sd',
    r'dd\s+if=.*\s+of=/dev/sd',
    r'\|\s*.*sh',
    r':\(\)\s*{\s*:.*};\s*:',
    r'chmod\s+-R\s+777\s+/',
)


def get_agent_dir() -> Path:
    relative = os.path.join(chat_path, 'files', str(global_state.now_agent_id))
    return Path(relative).resolve()


def is_command_banned(cmd: str) -> bool:
    cmd_lc = cmd.lower()
    if any(bad in cmd_lc for bad in BANNED_COMMON):
        return True
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, cmd_lc):
            return True
    return False


def is_path_safe(token: str, base: Path) -> bool:
    try:
        path = (base / token).resolve()
        return str(path).startswith(str(base))
    except Exception:
        return False


def validate_paths(command: str, work_dir: Path, posix: bool = True) -> bool:
    try:
        tokens = shlex.split(command, posix=posix)
    except ValueError:
        return False
    for tok in tokens:
        if '/' in tok or tok.startswith('.') or '\\' in tok:
            if not is_path_safe(tok, work_dir):
                return False
    return True


def shell_argv(command: str):
    if sys.platform.startswith('win'):
        return ["cmd.exe", "/c", command], False
    if sys.platform == 'darwin':
        return ["/bin/zsh", "-c", command], True
    return ["/bin/bash", "-c", command], True


@cacher
def execute_command(command: str, work_dir: Path, timeout_text: str, exception_text: str) -> str:
    """Run shell once; cached so resume after confirm does not re-execute."""
    argv, _posix = shell_argv(command)
    try:
        result = subprocess.run(
            argv, cwd=str(work_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=60, encoding='utf-8', errors='replace')
        return result.stdout or ''
    except subprocess.TimeoutExpired:
        return timeout_text
    except Exception as e:
        return f"{exception_text} {e}"


def delete_console_folder():
    path = os.path.join(chat_path, 'files', str(global_state.now_agent_id))
    if os.path.exists(path):
        shutil.rmtree(path)


def main(text: str) -> str:
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'output_text',
            'forbidden_text',
            'path_error_text',
            'timeout_text',
            'exception_text',
            'denied_text',
        )
        main.output_text = 'Output'
        main.forbidden_text = 'Forbidden command detected'
        main.path_error_text = 'Access to paths outside the workspace is forbidden'
        main.timeout_text = 'Command timed out'
        main.exception_text = 'Error:'
        main.denied_text = 'Command not allowed by user'
        return
    command = (text or '').strip()
    if not command:
        return main.exception_text + ' empty command'
    work_dir = get_agent_dir()
    os.makedirs(work_dir, exist_ok=True)
    if is_command_banned(command):
        return main.forbidden_text
    _argv, posix = shell_argv(command)
    if not validate_paths(command, work_dir, posix=posix):
        return main.path_error_text
    if not confirm_shell_command(command, 'shell_cmd'):
        return main.denied_text
    result = execute_command(command, work_dir, main.timeout_text, main.exception_text)
    return f"{main.output_text} {result}"
