import re
import requests
import json
import time
import urllib.parse
from typing import Dict, Any, List, Tuple, Optional
from cross_gpt import let_log

# === Глобальные переменные ===
session = None              # llama-server (chat)
ollama_session = None       # Ollama для эмбеддингов (как openai_provider)
base_url = ""               # llama-server
ollama_base_url = ""        # ollama emb host
chat_model = None
emb_model = None            # local emb model name if ollama=False
ollama_emb_model = "all-minilm:latest"
use_ollama = True           # ollama=True → эмбеддинги через Ollama
request_timeout = 30

token_limit = 4096
emb_token_limit = 4096

do_chat_construct = True
native_func_call = False
filter_think_tag = False
_last_think_content = None

CONTEXT_ERROR_KEYWORDS = (
    'the input length exceeds the context length',
    'input length exceeds',
    'context length',
    'exceeds context',
    'token limit exceeded',
    'exceeds the context',
    'exceeds context length',
    'context size',
    'n_ctx',
    'context window',
    'too long',
    'maximum context',
)

do_chat_construct = True    # используем chat/completions по умолчанию
native_func_call = False    # зависит от модели (у Bonsai – False)

tags = {                    # теги для форматирования (можно оставить пустыми)
    "bos": "", "eos": "",
    "sys_start": "", "sys_end": "",
    "user_start": "", "user_end": "",
    "assist_start": "", "assist_end": "",
    "tool_def_start": "", "tool_def_end": "",
    "tool_call_start": "", "tool_call_end": "",
    "tool_result_start": "", "tool_result_end": "",
}

# Константы для повторных попыток
MAX_RETRIES = 10
BASE_BACKOFF = 2.0
MAX_WAIT_TOTAL = 300
MAX_QUOTA_WAIT = 5 * 60 * 60

# ============================================================
# Вспомогательные функции
# ============================================================

def normalize_url(url, default_port=None, default_scheme="http"):
    """Добавляет схему и порт по умолчанию."""
    if not url:
        return url
    if "://" not in url:
        url = default_scheme + "://" + url
    parsed = urllib.parse.urlparse(url)
    if default_port is not None and not parsed.port:
        host = parsed.hostname or ""
        new_netloc = f"{host}:{default_port}"
        parsed = parsed._replace(netloc=new_netloc)
    return parsed.geturl()

def normalize_model(model):
    """Добавляет :latest, если в имени нет двоеточия (для совместимости)."""
    if model and ":" not in model:
        return model + ":latest"
    return model


def _safe_int(val):
    try:
        n = int(val)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def find_context_size(model_data: dict) -> Optional[int]:
    """
    Как в ollama_provider.find_context_size: достаём n_ctx / context_length из meta модели.
    Важно: не брать n_embd (часто 4096) — только известные context-ключи и whitelist размеров.
    """
    if not isinstance(model_data, dict):
        return None
    meta = model_data.get("meta") if isinstance(model_data.get("meta"), dict) else {}
    model_info = model_data.get("model_info") if isinstance(model_data.get("model_info"), dict) else {}
    # runtime / train context keys (приоритет)
    candidates = []
    for src in (model_data, meta, model_info):
        if not isinstance(src, dict):
            continue
        for key in (
            "n_ctx", "context_length", "max_context_length", "max_position_embeddings",
            "max_seq_len", "n_ctx_train", "llama.context_length", "gemma2.context_length",
            "gemma3.context_length", "mistral.context_length", "qwen2.context_length",
            "phi3.context_length",
        ):
            if key in src:
                n = _safe_int(src.get(key))
                if n:
                    candidates.append(n)
        for k, v in src.items():
            kl = str(k).lower()
            if "context" in kl or kl.endswith("n_ctx") or "n_ctx_train" in kl:
                if "embd" in kl or "vocab" in kl:
                    continue
                n = _safe_int(v)
                if n:
                    candidates.append(n)
    if candidates:
        # предпочитаем разумный max (runtime часто чуть ниже train, например 65024 vs 65536)
        return max(candidates)
    # whitelist sizes in JSON (как ollama) — берём наибольший
    context_sizes = {2048, 4096, 8192, 16384, 32768, 65024, 65536, 128000, 131072, 200000, 262144}
    try:
        model_text = json.dumps(model_data)
        found = sorted(
            [int(num) for num in re.findall(r"\b\d{4,7}\b", model_text) if int(num) in context_sizes],
            reverse=True,
        )
        if found:
            return found[0]
    except Exception:
        pass
    return None


def find_runtime_n_ctx_from_props(props: dict) -> Optional[int]:
    """
    GET /props (llama-server):
      default_generation_settings.n_ctx  ← реальный runtime context (например 65024)
    """
    if not isinstance(props, dict):
        return None
    dgs = props.get("default_generation_settings") or {}
    if isinstance(dgs, dict):
        n = _safe_int(dgs.get("n_ctx"))
        if n:
            return n
        params = dgs.get("params") if isinstance(dgs.get("params"), dict) else {}
        n = _safe_int(params.get("n_ctx"))
        if n:
            return n
    n = _safe_int(props.get("n_ctx"))
    if n:
        return n
    return None


def _error_text(resp) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err).lower()
            if err:
                return str(err).lower()
            return str(data).lower()
    except Exception:
        pass
    return (resp.text or "").lower()


def _is_context_overflow(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in CONTEXT_ERROR_KEYWORDS)


def _backoff_sleep(attempt, start_time):
    wait_time = BASE_BACKOFF ** attempt
    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
    if wait_time > remaining:
        wait_time = remaining
    if wait_time < 0.1:
        wait_time = 0.1
    time.sleep(wait_time)
    return wait_time


def _request_with_backoff(method, url, json_payload=None, headers=None, sess=None):
    """
    Ретраи как у ollama (общие): 429/5xx, connection, timeout.
    Context overflow → RuntimeError('ContextOverflowError') без бесконечных ретраев.
    """
    use = sess or session
    if not use:
        raise RuntimeError("HTTP session not initialized")
    start_time = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        elapsed = time.time() - start_time
        if elapsed > MAX_WAIT_TOTAL:
            raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с)")

        try:
            resp = use.request(method, url, json=json_payload, headers=headers, timeout=request_timeout)

            # context overflow (400/413/500) — как ollama: сразу ContextOverflowError
            if resp.status_code in (400, 413, 500):
                err_t = _error_text(resp)
                if _is_context_overflow(err_t):
                    let_log(f"Context overflow HTTP {resp.status_code}: {err_t[:200]}")
                    raise RuntimeError("ContextOverflowError")

            if resp.status_code == 429:
                retry_after = resp.headers.get('Retry-After')
                wait_time = None
                if retry_after and str(retry_after).isdigit():
                    wait_time = int(retry_after)
                else:
                    try:
                        err_data = resp.json()
                        err_msg = str(err_data.get('error', '')).lower()
                        if isinstance(err_data.get('error'), dict):
                            err_msg = str(err_data['error'].get('message', '')).lower()
                        if 'session limit' in err_msg:
                            raise RuntimeError("balance end: session limit")
                        if 'weekly limit' in err_msg:
                            raise RuntimeError("balance end: weekly limit")
                        if 'insufficient_quota' in err_msg:
                            raise RuntimeError("balance end")
                    except RuntimeError:
                        raise
                    except Exception:
                        pass

                if wait_time is not None:
                    if wait_time > MAX_QUOTA_WAIT:
                        raise RuntimeError("balance end")
                    if wait_time > MAX_WAIT_TOTAL:
                        raise RuntimeError(
                            f"Лимит квоты требует ожидания {wait_time}с, что превышает общий лимит {MAX_WAIT_TOTAL}с")
                    let_log(f"API 429, ожидание {wait_time} с...")
                    time.sleep(wait_time)
                    continue

                if attempt < MAX_RETRIES:
                    w = _backoff_sleep(attempt, start_time)
                    let_log(f"HTTP 429 (Rate Limit), попытка {attempt}, ожидание {w:.2f} с...")
                    continue
                resp.raise_for_status()

            if resp.status_code == 402:
                raise RuntimeError("balance end")

            # 501/405 — эндпоинт не поддерживается (например embeddings без --embeddings): не ретраим
            if resp.status_code in (501, 405, 404):
                err_t = _error_text(resp)
                raise RuntimeError(f"HTTP {resp.status_code}: {err_t[:300] or resp.reason}")

            if resp.status_code in (500, 502, 503, 504):
                if attempt < MAX_RETRIES:
                    w = _backoff_sleep(attempt, start_time)
                    let_log(f"HTTP {resp.status_code}, попытка {attempt}, ожидание {w:.2f} с...")
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            return resp.json()

        except RuntimeError as e:
            # ContextOverflow / balance — не ретраим
            if "ContextOverflowError" in str(e) or "balance end" in str(e):
                raise
            if attempt < MAX_RETRIES:
                _backoff_sleep(attempt, start_time)
                continue
            raise

        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES:
                w = _backoff_sleep(attempt, start_time)
                let_log(f"Ошибка соединения, попытка {attempt}, ожидание {w:.2f} с...")
                continue
            raise RuntimeError(f"Ошибка соединения после {MAX_RETRIES} попыток: {e}")

        except requests.exceptions.Timeout as e:
            if attempt < MAX_RETRIES:
                w = _backoff_sleep(attempt, start_time)
                let_log(f"Таймаут, попытка {attempt}, ожидание {w:.2f} с...")
                continue
            raise RuntimeError(f"Таймаут запроса после {MAX_RETRIES} попыток: {e}")

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                w = _backoff_sleep(attempt, start_time)
                let_log(f"Сетевая ошибка, попытка {attempt}: {e}. ожидание {w:.2f} с...")
                continue
            raise RuntimeError(f"Сетевая ошибка после {MAX_RETRIES} попыток: {e}")

        except Exception as e:
            if "ContextOverflowError" in str(e) or "balance end" in str(e):
                raise
            if attempt < MAX_RETRIES:
                _backoff_sleep(attempt, start_time)
                continue
            raise RuntimeError(f"Неожиданная ошибка: {e}")

    raise RuntimeError("Превышено максимальное количество попыток")

# ============================================================
# Основные функции провайдера (обязательный интерфейс)
# ============================================================

def _extract_think(text: str):
    """Как ollama: вырезать <think>...</think>."""
    if not text:
        return text, None
    pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return pattern.sub('', text).strip(), match.group(1).strip()
    return text, None


def connect(connection_string: str, timeout=None) -> Tuple[bool, int, Dict[str, Optional[str]], Optional[str]]:
    """
    llama-server (chat) + опционально Ollama embeddings.

    Пример:
      url=http://localhost:8080; model=Bonsai-8B-Q1_0.gguf;
      use_ollama_embs=true; ollama_url=http://localhost:11434; ollama_emb_model=all-minilm:latest;
      chat_template=true; native_func_call=false; filter_think_tag=false
    """
    global session, ollama_session, base_url, ollama_base_url, chat_model, emb_model, ollama_emb_model
    global token_limit, emb_token_limit, tags, request_timeout, use_ollama
    global do_chat_construct, native_func_call, filter_think_tag

    # is_thinking removed: never sent to server; use filter_think_tag for <think> strip.
    params = {
        "url": "http://localhost:8080",
        "model": "bonsai-8b",
        "use_ollama_embs": "true",
        "ollama_url": "http://localhost:11434",
        "ollama_emb_model": "all-minilm:latest",
        "emb_url": "",
        "emb_model": "",
        "n_ctx": "",
        "timeout": "",
        "chat_template": "true",
        "native_func_call": "false",
        "filter_think_tag": "false",
    }
    for part in (connection_string or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in params:
            params[key] = value

    # === НОРМАЛИЗАЦИЯ (как ollama_provider) ===
    params["url"] = normalize_url(params["url"], default_port=8080)
    params["ollama_url"] = normalize_url(params["ollama_url"] or "http://localhost:11434", default_port=11434)
    if params["emb_url"]:
        params["emb_url"] = normalize_url(params["emb_url"], default_port=11434)
    params["ollama_emb_model"] = normalize_model(params["ollama_emb_model"])
    if params["emb_model"]:
        params["emb_model"] = normalize_model(params["emb_model"])

    base_url = params["url"].rstrip("/")
    chat_model = params["model"]
    ollama_base_url = (params["emb_url"] or params["ollama_url"]).rstrip("/")
    ollama_emb_model = params["ollama_emb_model"]
    emb_model = params["emb_model"] or ollama_emb_model

    do_chat_construct = params["chat_template"].lower().strip() == "true"
    native_func_call = params["native_func_call"].lower().strip() == "true"
    use_ollama = params["use_ollama_embs"].lower().strip() == "true"
    filter_think_tag = params["filter_think_tag"].lower().strip() == "true"

    timeout_str = params["timeout"].strip()
    if timeout_str.isdigit(): request_timeout = int(timeout_str)
    else: request_timeout = timeout

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    ollama_session = None

    # health / models availability
    try:
        resp = session.get(f"{base_url}/health", timeout=request_timeout)
        if resp.status_code != 200:
            resp = session.get(f"{base_url}/v1/models", timeout=request_timeout)
            if resp.status_code != 200:
                return False, token_limit, tags, f"Сервер {base_url} не отвечает на /health или /v1/models"
    except Exception as e:
        return False, token_limit, tags, f"Ошибка подключения к {base_url}: {e}"

    # --- 1) явный override (как num_ctx у ollama) ---
    explicit_ctx = None
    raw = (params.get("n_ctx") or "").strip()
    if raw.isdigit(): explicit_ctx = int(raw)

    # --- 2) runtime n_ctx из /props (эталон для llama-server) ---
    props_ctx = None
    try:
        props_resp = session.get(f"{base_url}/props", timeout=request_timeout)
        if props_resp.status_code == 200:
            props = props_resp.json()
            props_ctx = find_runtime_n_ctx_from_props(props)
            let_log(f"llama.cpp /props n_ctx={props_ctx}, model_alias={props.get('model_alias')}")
    except Exception as e:
        let_log(f"llama.cpp: /props недоступен: {e}")

    # --- 3) train / model meta из /v1/models ---
    model_ids = []
    models_ctx = None
    model_info = None
    try:
        models_resp = session.get(f"{base_url}/v1/models", timeout=request_timeout)
        if models_resp.status_code == 200:
            data = models_resp.json()
            models_list = data.get("data") or []
            # ollama-like "models" field sometimes present
            if not models_list and isinstance(data.get("models"), list):
                models_list = data["models"]
            for m in models_list:
                mid = m.get("id") or m.get("model") or m.get("name") or ""
                if mid:
                    model_ids.append(mid)
            if models_list:
                model_info = models_list[0]
                want = (chat_model or "").lower()
                if want:
                    for m in models_list:
                        mid = (m.get("id") or m.get("model") or m.get("name") or "").lower()
                        if want in mid or mid in want:
                            model_info = m
                            break
                models_ctx = find_context_size(model_info)
                let_log(
                    f"llama.cpp /v1/models id={model_info.get('id') or model_info.get('name')}, "
                    f"meta.n_ctx_train={ (model_info.get('meta') or {}).get('n_ctx_train') }, "
                    f"resolved={models_ctx}"
                )
    except Exception as e:
        let_log(f"llama.cpp: не удалось прочитать /v1/models: {e}")

    # приоритет: explicit → props runtime → models meta
    resolved = explicit_ctx or props_ctx or models_ctx
    if resolved:
        token_limit = int(resolved)
    else:
        token_limit = 4096
        let_log("llama.cpp: context не найден, fallback 4096")

    # если explicit не задан, runtime /props предпочтительнее train (65024 vs 65536)
    if not explicit_ctx and props_ctx:
        token_limit = int(props_ctx)

    warn = None
    if token_limit in (4095, 4096) and not explicit_ctx:
        let_log(f"[validation] llama.cpp context={token_limit} — возможно не прочитан /props")
        warn = (
            f"context={token_limit} (fallback?). "
            f"props_n_ctx={props_ctx}, models_ctx={models_ctx}, models={model_ids or 'n/a'}. "
            f"Задайте n_ctx= в connection string или проверьте GET /props"
        )

    emb_token_limit = token_limit

    # --- Ollama embeddings (как openai_provider: ollama=True + ollama_url + ollama_emb_model) ---
    if use_ollama:
        try:
            ollama_session = requests.Session()
            ollama_session.headers.update({"Content-Type": "application/json"})
            r = ollama_session.get(f"{ollama_base_url}/api/tags", timeout=request_timeout)
            r.raise_for_status()
            available = [m.get("name") for m in (r.json().get("models") or []) if m.get("name")]
            let_log(f"Ollama emb models: {available}")
            if ollama_emb_model not in available:
                let_log(f"Ollama модель '{ollama_emb_model}' не найдена, ищем embed-*")
                embed_models = [m for m in available if "embed" in m.lower() or "minilm" in m.lower()]
                if embed_models:
                    ollama_emb_model = embed_models[0]
                elif available:
                    ollama_emb_model = available[0]
                else:
                    raise RuntimeError("Нет моделей в Ollama")
                let_log(f"Выбрана ollama_emb_model={ollama_emb_model}")
            show = ollama_session.post(
                f"{ollama_base_url}/api/show",
                json={"name": ollama_emb_model},
                timeout=request_timeout)
            if show.status_code == 200:
                emb_ctx = find_context_size(show.json())
                if emb_ctx:
                    emb_token_limit = emb_ctx
            let_log(
                f"Ollama embeddings ON: url={ollama_base_url} model={ollama_emb_model} "
                f"emb_token_limit={emb_token_limit}"
            )
        except Exception as e:
            ollama_session = None
            err = (
                f"Ollama emb connect failed (use_ollama_embs=true): {e}. "
                f"Задайте рабочий ollama_url/ollama_emb_model или use_ollama_embs=false."
            )
            let_log(err)
            # D13: не продолжаем молча — embeddings обязательны при use_ollama_embs
            return False, token_limit, tags, err

    tags = {
        "bos": "", "eos": "",
        "sys_start": "", "sys_end": "",
        "user_start": "", "user_end": "",
        "assist_start": "", "assist_end": "",
        "tool_def_start": "", "tool_def_end": "",
        "tool_call_start": "", "tool_call_end": "",
        "tool_result_start": "", "tool_result_end": "",
    }
    if warn:
        tags["_validation_warning"] = warn

    let_log(
        f"llama.cpp chat={base_url} model={chat_model} ctx={token_limit} "
        f"(explicit={explicit_ctx}, props={props_ctx}, models={models_ctx}); "
        f"ollama={use_ollama} ollama_url={ollama_base_url} ollama_emb={ollama_emb_model}; "
        f"chat_template={do_chat_construct} native_func_call={native_func_call} "
        f"filter_think_tag={filter_think_tag}"
    )
    return True, token_limit, tags, warn

def disconnect() -> bool:
    """Закрыть сессии."""
    global session, ollama_session, base_url, ollama_base_url, chat_model, emb_model
    closed = False
    if ollama_session is not None and ollama_session is not session:
        try:
            ollama_session.close()
        except Exception:
            pass
        ollama_session = None
        closed = True
    if session:
        try:
            session.close()
        except Exception:
            pass
        session = None
        closed = True
    base_url = ""
    ollama_base_url = ""
    chat_model = None
    emb_model = None
    return closed

def _is_thinking_param_key(key: str) -> bool:
    """D15: never forward think/thinking/reasoning keys to the server."""
    k = (key or "").lower()
    return ("think" in k) or ("reasoning" in k)


def ask_model(generation_params: Dict[str, Any]) -> str:
    """Генерация через /v1/completions."""
    global _last_think_content
    if not session or not base_url:
        raise RuntimeError("llama.cpp клиент не инициализирован. Сначала вызовите connect().")

    url = f"{base_url}/v1/completions"
    payload = {
        "model": chat_model,
        "prompt": generation_params.get("prompt", ""),
        "max_tokens": generation_params.get("max_tokens", 200),
        "temperature": generation_params.get("temperature", 0.7),
        "top_p": generation_params.get("top_p", 0.9),
        "stop": generation_params.get("stop", None),
        "echo": False,
    }
    for key in ["frequency_penalty", "presence_penalty", "repeat_penalty"]:
        if key in generation_params and not _is_thinking_param_key(key):
            payload[key] = generation_params[key]
    # Strip any accidental think/reasoning keys from payload
    for k in list(payload.keys()):
        if _is_thinking_param_key(k):
            payload.pop(k, None)

    let_log(f"ask_model: запрос к {url}")
    data = _request_with_backoff("POST", url, json_payload=payload)
    if "choices" not in data or not data["choices"]:
        raise RuntimeError("Некорректный ответ от сервера: отсутствует choices")
    result = data["choices"][0].get("text", "").strip()
    if filter_think_tag:
        result, think = _extract_think(result)
        _last_think_content = think
    return result

def ask_model_chat(generation_params: Dict[str, Any]) -> Dict[str, Any]:
    """/v1/chat/completions; native tools если native_func_call=True."""
    global _last_think_content
    if not session or not base_url:
        raise RuntimeError("llama.cpp клиент не инициализирован. Сначала вызовите connect().")

    url = f"{base_url}/v1/chat/completions"
    messages = generation_params.get("messages", [])
    payload = {
        "model": chat_model,
        "messages": messages,
        "max_tokens": generation_params.get("max_tokens", 500),
        "temperature": generation_params.get("temperature", 0.7),
        "top_p": generation_params.get("top_p", 0.9),
        "stop": generation_params.get("stop", None),
    }
    if native_func_call:
        if "tools" in generation_params:
            payload["tools"] = generation_params["tools"]
        if "tool_choice" in generation_params:
            payload["tool_choice"] = generation_params["tool_choice"]
    # D15: never send think/thinking/reasoning to server
    for k in list(payload.keys()):
        if _is_thinking_param_key(k):
            payload.pop(k, None)

    let_log(f"ask_model_chat: {url} native_func_call={native_func_call}")
    data = _request_with_backoff("POST", url, json_payload=payload)
    if "choices" not in data or not data["choices"]:
        raise RuntimeError("Некорректный ответ от сервера: отсутствует choices")
    if filter_think_tag:
        try:
            msg = data["choices"][0].get("message") or {}
            content = msg.get("content") or ""
            cleaned, think = _extract_think(content)
            data["choices"][0]["message"]["content"] = cleaned
            data["think"] = think
            _last_think_content = think
        except Exception:
            pass
    return data


def get_last_think():
    return _last_think_content


def _parse_embedding_vector(data: dict) -> Optional[List[float]]:
    if not isinstance(data, dict):
        return None
    if "data" in data and data["data"]:
        first = data["data"][0]
        if isinstance(first, dict) and first.get("embedding"):
            return first["embedding"]
        if isinstance(first, (list, tuple)):
            return list(first)
    if data.get("embedding"):
        return data["embedding"]
    embs = data.get("embeddings")
    if isinstance(embs, list) and embs:
        return list(embs[0]) if isinstance(embs[0], (list, tuple)) else list(embs)
    return None


def create_embeddings(text: str) -> List[float]:
    """
    Как openai_provider:
      1) если use_ollama_embs=True — только Ollama /api/embeddings (без silent fallback)
      2) иначе: llama-server /v1/embeddings
    """
    if not session:
        raise RuntimeError("llama.cpp клиент не инициализирован. Сначала вызовите connect().")

    text = (text or "").strip()
    if not text:
        raise RuntimeError("Пустой текст для эмбеддингов")

    last_err = None

    # --- 1) Ollama required path when use_ollama_embs=True (D13: no silent llama.cpp fallback) ---
    if use_ollama:
        if not (ollama_session and ollama_base_url):
            raise RuntimeError(
                "use_ollama_embs=true, но Ollama emb session не инициализирована. "
                "Переподключите модель или выставьте use_ollama_embs=false."
            )
        try:
            t = text
            if len(t) > 2000:
                let_log(f"create_embeddings (Ollama): обрезаем {len(t)} → 2000")
                t = t[:2000]
            model = ollama_emb_model or "all-minilm:latest"
            url = f"{ollama_base_url.rstrip('/')}/api/embeddings"
            let_log(f"create_embeddings [ollama]: {url} model={model}")
            data = _request_with_backoff(
                "POST", url, json_payload={"model": model, "prompt": t}, sess=ollama_session)
            emb = _parse_embedding_vector(data)
            if emb:
                let_log(f"create_embeddings: ollama ok dim={len(emb)}")
                return emb
            last_err = "empty ollama embedding"
        except RuntimeError as e:
            if "ContextOverflowError" in str(e) or "balance end" in str(e):
                raise
            last_err = e
            let_log(f"create_embeddings ollama fail: {e}")
        except Exception as e:
            last_err = e
            let_log(f"create_embeddings ollama fail: {e}")
        raise RuntimeError(
            f"Ollama embeddings failed (no fallback to llama.cpp): {last_err}. "
            f"Пример: use_ollama_embs=true;ollama_url=http://localhost:11434;ollama_emb_model=all-minilm:latest"
        )

    # --- 2) llama.cpp local embeddings (only when use_ollama_embs=false) ---
    try:
        model = emb_model or chat_model
        url = f"{base_url.rstrip('/')}/v1/embeddings"
        let_log(f"create_embeddings [llama.cpp]: {url} model={model}")
        data = _request_with_backoff(
            "POST", url, json_payload={"model": model, "input": text}, sess=session)
        emb = _parse_embedding_vector(data)
        if emb:
            let_log(f"create_embeddings: llama.cpp ok dim={len(emb)}")
            return emb
        last_err = "empty llama.cpp embedding"
    except RuntimeError as e:
        if "ContextOverflowError" in str(e) or "balance end" in str(e):
            raise
        last_err = e
        let_log(f"create_embeddings llama.cpp fail: {e}")
    except Exception as e:
        last_err = e
        let_log(f"create_embeddings llama.cpp fail: {e}")

    raise RuntimeError(
        f"Не удалось получить эмбеддинг: {last_err}. "
        f"Пример: use_ollama_embs=false + llama-server --embeddings, "
        f"или use_ollama_embs=true;ollama_url=http://localhost:11434;ollama_emb_model=all-minilm:latest"
    )

