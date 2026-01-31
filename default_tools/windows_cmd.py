'''
run_command_windows
executes shell commands in a safe way and dedicated folder under Windows cmd.exe
Obtaining entry to the Windows command line interface
Access to cmd.exe, recommended only for large models. Unfortunately, one console folder can still be accessed by multiple agents. I don't think it's critical. I'll fix it later.
'''
import os
import shutil
import shlex
import subprocess
import re
from pathlib import Path
from cross_gpt import chat_path, now_agent_id

def delete_console_folder(): # TODO
    path = os.path.join(chat_path, 'console_folders', now_agent_id)
    if os.path.exists(path):
        shutil.rmtree(path)

def main(text: str) -> str:
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'output_text',
            'forbidden_text',
            'path_error_text',
            'timeout_text',
            'exception_text'
        )
        main.output_text = 'Output:'
        main.forbidden_text = 'Forbidden command detected'
        main.path_error_text = 'Access to paths outside the workspace is forbidden'
        main.timeout_text = 'Command timed out'
        main.exception_text = 'Error:'
        return

    BANNED = [
        'format', 'shutdown', 'rmdir', 'erase', 'taskkill',
        'net stop', 'net start', 'reg', 'sc', 'powershell'
    ]
    DANGEROUS_PATTERNS = [
        r'>\s*\\?dev',
        r'\|\s*.*sh',
    ]

    def get_agent_dir(chat_id: str, now_agent_id: str) -> Path:
        relative = os.path.join(chat_path, 'console_folders', now_agent_id)
        return Path(relative).resolve()

    def is_command_banned(cmd: str) -> bool:
        cmd_lc = cmd.lower()
        if any(bad in cmd_lc for bad in BANNED):
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

    def validate_paths(command: str, work_dir: Path) -> bool:
        tokens = shlex.split(command, posix=False)
        for tok in tokens:
            if '\\' in tok or (':' in tok and not tok.startswith('%')):
                if not is_path_safe(tok, work_dir):
                    return False
        return True

    def execute_command(command: str, work_dir: Path) -> str:
        try:
            result = subprocess.run(
                ["cmd.exe", "/c", command],
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
                encoding='utf-8'
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return main.timeout_text
        except Exception as e:
            return f"{main.exception_text} {e}"

    command = text.strip()
    work_dir = get_agent_dir()
    os.makedirs(work_dir, exist_ok=True)

    if is_command_banned(command):
        return main.forbidden_text

    if not validate_paths(command, work_dir):
        return main.path_error_text

    result = execute_command(command, work_dir)
    return f"{main.output_text}\n{result}"