"""
Scripted / stub model provider for tests and offline scenario runs.

Does not call any remote or local LLM. Responses come from:
  1) match rules (regex/substring on system + user prompt)
  2) FIFO queue
  3) built-in smart defaults (tool select, critic, librarian, GIGO stubs)

Programmatic API (tests / agent)::

    import model_providers.scripted_provider as sp
    sp.reset()
    sp.push_response("!!!create_executor!!! Do the task")
    sp.push_response("!!!end_dialog!!! Done")
    sp.add_rule(r"VERDICT|ВЕРДИКТ|decision", "VERDICT: APPROVE\\nВЕРДИКТ: ПРИНЯТЬ")

Connect string params (UI-compatible)::

    mode=mixed;token_limit=8192;chat_template=True;native_func_call=False;fail_if_empty=false;emb_dim=32
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from cross_gpt import let_log

# --- Provider module state (required by UI / initialize_work) ---
token_limit = 8192
emb_token_limit = 8192
do_chat_construct = True
native_func_call = False

_connected = False
_mode = "mixed"  # queue | match | mixed
_fail_if_empty = False
_emb_dim = 32
_lock = threading.RLock()

# queue of str | Callable[[str, Optional[str]], str]
_response_queue: List[Any] = []
# list of (compiled_regex | None, substring | None, response)
_rules: List[Tuple[Optional[re.Pattern], Optional[str], Any]] = []
_prompt_log: List[Dict[str, Any]] = []
_call_count = 0
_default_response = "OK"


def reset():
    """Clear queue, rules, logs. Safe between tests."""
    global _response_queue, _rules, _prompt_log, _call_count, _connected
    global _mode, _fail_if_empty, _emb_dim, token_limit, emb_token_limit
    global do_chat_construct, native_func_call, _default_response
    with _lock:
        _response_queue = []
        _rules = []
        _prompt_log = []
        _call_count = 0
        _connected = False
        _mode = "mixed"
        _fail_if_empty = False
        _emb_dim = 32
        token_limit = 8192
        emb_token_limit = 8192
        do_chat_construct = True
        native_func_call = False
        _default_response = "OK"


def push_response(text: Union[str, Callable]):
    """Append one response (or callable(prompt, system)->str) to the FIFO queue."""
    with _lock:
        _response_queue.append(text)


def push_responses(*texts):
    with _lock:
        _response_queue.extend(texts)


def add_rule(pattern: str, response: Union[str, Callable], *, regex: bool = True, flags: int = re.I | re.S):
    """
    If pattern matches system+user (or substring if regex=False), return response.
    Rules are checked in insertion order before the queue (in mixed/match modes).
    """
    with _lock:
        if regex:
            _rules.append((re.compile(pattern, flags), None, response))
        else:
            _rules.append((None, pattern, response))


def set_default_response(text: str):
    global _default_response
    with _lock:
        _default_response = text


def get_prompt_log() -> List[Dict[str, Any]]:
    with _lock:
        return list(_prompt_log)


def clear_prompt_log():
    with _lock:
        _prompt_log.clear()


def remaining_queue() -> int:
    with _lock:
        return len(_response_queue)


def call_count() -> int:
    with _lock:
        return _call_count


def _combine(prompt: str, system: Optional[str]) -> str:
    return f"{system or ''}\n{prompt or ''}"


def _resolve(value: Any, prompt: str, system: Optional[str]) -> str:
    if callable(value):
        out = value(prompt, system)
    else:
        out = value
    return "" if out is None else str(out)


def _smart_default(prompt: str, system: Optional[str]) -> Optional[str]:
    """Heuristic stubs so GIGO / critic / librarian / tool-select work without a full script."""
    blob = _combine(prompt, system)
    low = blob.lower()

    # Critic decision stage
    if any(x in blob for x in (
        "ВЕРДИКТ", "VERDICT", "старший системный аналитик",
        "senior system analyst",
    )) or ("decision" in low and "task" in low and system and "verdict" in low):
        return "ВЕРДИКТ: ПРИНЯТЬ\nVERDICT: APPROVE"

    # Critic criteria / evaluation
    if any(x in low for x in ("критери", "criteria", "decomposition", "оценк", "evaluation report", "evaluate")):
        return "1. Completeness\n2. Correctness\n3. Usefulness\nOK on all criteria."

    # Tool selection (operator / executor)
    if "only tool names separated by comma" in low or "output exactly: none" in low:
        return "None"
    if "list of allowed tools" in low and ("select tools" in low or "select tool" in low):
        return "None"

    # Tasks identical? (create_executor recreate check via parse_prompt_response)
    if "are the tasks the same" in low or "задачи одинаков" in low:
        return "0"

    # Librarian satisfaction checks
    if "output exactly one line containing only the digit" in low or "fully satisfies the query" in low:
        return "1"
    if "is the request fully satisfied" in low or "does the answer match the query" in low:
        return "1"
    if "most relevant fragment" in low:
        return (prompt or "")[:500] or "fragment"

    # GIGO / plan / roles
    if any(x in low for x in (
        "dreamer", "realist", "critic", "мечтател", "реалист", "критик",
        "make a plan", "составь план", "number of plan",
    )):
        return "1. Step one\n2. Step two\n3. Step three\n4. Finish"

    # Questions for librarian
    if "questions" in low and (";" in (system or "") or "separat" in low or "вопрос" in low):
        return "What is needed?; How to verify?"

    # Intention / PSM personality
    if "you are..." in low or "personality" in low or "ты..." in low:
        return "You are a careful, practical operator focused on clear outcomes."

    # Summaries
    if any(x in low for x in ("summary", "сводк", "summar", "chunk")):
        return "Summary of the dialogue: task progress is fine."

    # Annotation / short instruction for executor
    if "concise instruction" in low or "write a concise" in low or "one short paragraph" in low:
        return "Complete the assigned subtask using available tools carefully."

    return None


def _pick_response(prompt: str, system: Optional[str]) -> str:
    global _call_count
    with _lock:
        _call_count += 1
        blob = _combine(prompt, system)
        _prompt_log.append({
            "n": _call_count,
            "system": system,
            "prompt": prompt,
            "prompt_len": len(prompt or ""),
            "system_len": len(system or ""),
        })

        # 1) explicit rules (test/agent scripts)
        if _mode in ("match", "mixed"):
            for cre, substr, resp in _rules:
                if cre is not None and cre.search(blob):
                    out = _resolve(resp, prompt, system)
                    let_log(f"[scripted] rule match -> {out[:120]!r}")
                    return out
                if substr is not None and substr in blob:
                    out = _resolve(resp, prompt, system)
                    let_log(f"[scripted] substr match -> {out[:120]!r}")
                    return out

        # 2) smart defaults BEFORE queue — so GIGO dreamer/realist/critic/plan
        #    do not consume FIFO agent commands (create_executor / end_dialogue).
        if _mode in ("match", "mixed"):
            smart = _smart_default(prompt, system)
            if smart is not None:
                let_log(f"[scripted] smart default -> {smart[:120]!r}")
                return smart

        # 3) FIFO queue for free-form agent turns
        if _mode in ("queue", "mixed") and _response_queue:
            item = _response_queue.pop(0)
            out = _resolve(item, prompt, system)
            left = len(_response_queue)
            let_log(f"[scripted] queue pop ({left} left) -> {out[:120]!r}")
            return out

        if _fail_if_empty:
            raise RuntimeError(
                f"scripted_provider: no response for call #{_call_count}. "
                f"system[:80]={str(system)[:80]!r} prompt[:80]={str(prompt)[:80]!r}"
            )
        let_log(f"[scripted] fallback default -> {_default_response!r}")
        return _default_response


def connect(connection_string: str, timeout: int = 30, _decrypted_token: str = None):
    """
    Required provider entrypoint.
    params dict must exist for UI AST parsing.
    """
    global _connected, _mode, _fail_if_empty, _emb_dim
    global token_limit, emb_token_limit, do_chat_construct, native_func_call

    params = {
        "mode": "mixed",
        "token_limit": "8192",
        "emb_token_limit": "8192",
        "chat_template": "True",
        "native_func_call": "False",
        "fail_if_empty": "false",
        "emb_dim": "32",
        "default_response": "OK",
        "responses_file": "",
    }
    if connection_string:
        for part in connection_string.split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            k, v = k.strip().lower(), v.strip()
            if k in params:
                params[k] = v

    _mode = (params.get("mode") or "mixed").lower()
    if _mode not in ("queue", "match", "mixed"):
        _mode = "mixed"
    _fail_if_empty = str(params.get("fail_if_empty", "false")).lower() in ("1", "true", "yes")
    try:
        token_limit = int(params.get("token_limit") or 8192)
    except ValueError:
        token_limit = 8192
    try:
        emb_token_limit = int(params.get("emb_token_limit") or token_limit)
    except ValueError:
        emb_token_limit = token_limit
    do_chat_construct = str(params.get("chat_template", "True")).lower() in ("1", "true", "yes")
    native_func_call = str(params.get("native_func_call", "False")).lower() in ("1", "true", "yes")
    try:
        _emb_dim = max(8, int(params.get("emb_dim") or 32))
    except ValueError:
        _emb_dim = 32
    set_default_response(params.get("default_response") or "OK")

    # optional JSONL/JSON responses file
    rfile = (params.get("responses_file") or "").strip()
    if rfile:
        try:
            with open(rfile, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if raw.startswith("["):
                data = json.loads(raw)
                for item in data:
                    if isinstance(item, dict):
                        if "match" in item:
                            add_rule(item["match"], item.get("response", "OK"), regex=item.get("regex", True))
                        elif "response" in item:
                            push_response(item["response"])
                    else:
                        push_response(str(item))
            else:
                for line in raw.splitlines():
                    line = line.strip()
                    if line:
                        push_response(line)
        except Exception as e:
            return False, token_limit, {}, f"responses_file error: {e}"

    _connected = True
    tags = {
        "bos": "", "eos": "",
        "sys_start": "", "sys_end": "",
        "user_start": "", "user_end": "",
        "assist_start": "", "assist_end": "",
        "tool_def_start": "", "tool_def_end": "",
        "tool_call_start": "", "tool_call_end": "",
        "tool_result_start": "", "tool_result_end": "",
    }
    let_log(f"[scripted_provider] connected mode={_mode} token_limit={token_limit}")
    return True, token_limit, tags


def disconnect():
    global _connected
    _connected = False
    return True


def ask_model(generation_params: dict) -> str:
    """Completions-style: generation_params with 'prompt' and optional 'system'."""
    prompt = generation_params.get("prompt", "") or ""
    system = generation_params.get("system")
    return _pick_response(prompt, system)


def ask_model_chat(generation_params: dict) -> dict:
    """Chat-style OpenAI-compatible response dict."""
    messages = generation_params.get("messages") or []
    system_parts = []
    user_parts = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "system":
            system_parts.append(content)
        else:
            user_parts.append(content)
    system = "\n".join(system_parts) if system_parts else None
    prompt = "\n".join(user_parts)
    text = _pick_response(prompt, system)
    return {
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }]
    }


def create_embeddings(text: str) -> List[float]:
    """Deterministic pseudo-embedding from text hash (stable across runs)."""
    if text is None:
        text = ""
    data = str(text).encode("utf-8")
    digest = hashlib.sha256(data).digest()
    raw = digest
    while len(raw) < _emb_dim * 4:
        raw += hashlib.sha256(raw).digest()
    vec = []
    for i in range(_emb_dim):
        chunk = raw[i * 4:(i + 1) * 4]
        n = int.from_bytes(chunk, "little", signed=False)
        vec.append((n / 0xFFFFFFFF) * 2.0 - 1.0)
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]
