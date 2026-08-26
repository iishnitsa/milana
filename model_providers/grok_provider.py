"""
Grok (xAI) provider — OpenAI-compatible Chat Completions API.

Base URL: https://api.x.ai/v1
Endpoints: POST /chat/completions, GET /models (validate only, no browser).
SDK-equivalent: OpenAI client with base_url=https://api.x.ai/v1 and XAI_API_KEY.

Connection example:
  url=https://api.x.ai/v1;model=grok-4-1-fast-reasoning;token=xai-...;password=;token_limit=131072;ollama=True;ollama_url=http://localhost:11434

Token: plain `token=` / `api_token=` or encrypted with `password=set` (UI encryption_utils).
Also: env XAI_API_KEY / GROK_API_KEY; worker `_decrypted_token`; UI `_decrypted_password`.
Embeddings: optional local Ollama (xAI has no public emb API); set ollama=False if RAG off.

Note: model_providers/xai_provider.py is a thin import alias only (not listed in UI).
"""
import requests
import re
import os
import json
import time
import random
import urllib.parse
from cross_gpt import let_log

xai_session = None
ollama_session = None

xai_base_url = ""
ollama_base_url = ""
xai_timeout = 0

default_chat_model = "grok-4-1-fast-reasoning"
emb_model = "all-minilm:latest"
ollama_emb_model = "all-minilm:latest"

token_limit = 131072
emb_token_limit = 4095

do_chat_construct = True
native_func_call = False

tags = {
    "bos": "", "eos": "",
    "sys_start": "", "sys_end": "",
    "user_start": "", "user_end": "",
    "assist_start": "", "assist_end": "",
    "tool_def_start": "", "tool_def_end": "",
    "tool_call_start": "", "tool_call_end": "",
    "tool_result_start": "", "tool_result_end": "",
}

MAX_RETRIES = 12
BASE_BACKOFF = 2.0
last_wait_time = 0.0

OLLAMA_MAX_RETRIES = 20
OLLAMA_MAX_WAIT_TOTAL = 420
OLLAMA_BASE_BACKOFF = 2.0
MAX_QUOTA_WAIT = 5 * 60 * 60


def normalize_url(url, default_port=None, default_scheme="https"):
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
    if model and ":" not in model:
        return model + ":latest"
    return model


def find_ollama_context_size(model_data, model_name="unknown"):
    let_log(f"Определение контекста для Ollama модели '{model_name}'")
    model_info = model_data.get('model_info', {})
    direct_keys = [
        'context_length', 'max_position_embeddings', 'max_seq_len', 'n_ctx',
        'llama.context_length', 'gemma2.context_length', 'gemma3.context_length',
        'mistral.context_length', 'qwen2.context_length', 'phi3.context_length',
    ]
    for key in direct_keys:
        if key in model_info:
            try:
                val = int(model_info[key])
                if val > 0:
                    return val
            except (ValueError, TypeError):
                continue
    for key, value in model_info.items():
        if 'context_length' in key.lower() and isinstance(value, (int, float)):
            return int(value)
    parameters = model_data.get('parameters', '')
    if parameters:
        match = re.search(r'num_ctx\s*[=: ]\s*(\d+)', parameters, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 4095


def _get_timeout_value():
    global xai_timeout
    return None if xai_timeout == 0 else xai_timeout


def _make_request(api_url, payload):
    global last_wait_time
    if not xai_session:
        raise RuntimeError("xAI client not connected")
    if last_wait_time > 0:
        preemptive_pause = min(5.0, last_wait_time * 0.5)
        if preemptive_pause > 0.1:
            time.sleep(preemptive_pause)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            let_log(f"xai request attempt {attempt}: {api_url}")
            timeout_value = _get_timeout_value()
            r = xai_session.post(api_url, json=payload, timeout=timeout_value)
            if r.status_code == 429:
                try:
                    error_data = r.json().get("error", {})
                    error_code = error_data.get("code")
                    error_msg = error_data.get("message", "").lower()
                except Exception:
                    error_code = None
                    error_msg = r.text.lower()
                if error_code == "insufficient_quota" or "insufficient_quota" in error_msg:
                    raise RuntimeError("balance end")
                retry_after = r.headers.get('Retry-After')
                if retry_after and retry_after.isdigit():
                    wait_time = min(60, int(retry_after))
                    time.sleep(wait_time)
                    continue
                if attempt < MAX_RETRIES:
                    wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                    last_wait_time = wait_time
                    time.sleep(wait_time)
                    continue
                raise RuntimeError("Rate limit exceeded after maximum retries")
            if r.status_code == 402:
                raise RuntimeError("balance end")
            if r.status_code == 400:
                try:
                    error_data = r.json().get("error", {})
                    error_code = error_data.get("code")
                    error_msg = error_data.get("message", "").lower()
                    if error_code == "context_length_exceeded" or "context_length" in error_msg:
                        raise RuntimeError("ContextOverflowError")
                except RuntimeError:
                    raise
                except Exception:
                    pass
                r.raise_for_status()
            if 500 <= r.status_code < 600:
                if attempt < MAX_RETRIES:
                    wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                    last_wait_time = wait_time
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"Server error after maximum retries: {r.text}")
            r.raise_for_status()
            resp_data = r.json()
            if "error" in resp_data:
                err_msg = resp_data["error"].get("message", "Unknown error") if isinstance(resp_data["error"], dict) else str(resp_data["error"])
                if "insufficient_quota" in err_msg.lower():
                    raise RuntimeError("balance end")
                if attempt < MAX_RETRIES:
                    time.sleep(min(60, (BASE_BACKOFF ** attempt) + random.random()))
                    continue
                raise RuntimeError(f"API Provider Error: {err_msg}")
            last_wait_time *= 0.7
            if last_wait_time < 0.1:
                last_wait_time = 0
            return resp_data
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(min(60, (BASE_BACKOFF ** attempt) + random.random()))
                continue
            raise RuntimeError("Request timed out after maximum retries")
        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2)
        except Exception as e:
            if "balance end" in str(e) or "ContextOverflowError" in str(e):
                raise
            raise RuntimeError(f"xAI request error: {e}")
    raise RuntimeError("xAI: max retries exceeded")


def _ollama_request_with_backoff(api_url, json_payload):
    start_time = time.time()
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        elapsed = time.time() - start_time
        if elapsed > OLLAMA_MAX_WAIT_TOTAL:
            raise RuntimeError(f"Превышено общее время ожидания ({OLLAMA_MAX_WAIT_TOTAL} с)")
        try:
            response = ollama_session.post(api_url, json=json_payload)
            if response.status_code == 429:
                if attempt < OLLAMA_MAX_RETRIES:
                    wait_time = min(OLLAMA_BASE_BACKOFF ** attempt, OLLAMA_MAX_WAIT_TOTAL - elapsed)
                    time.sleep(max(0.1, wait_time))
                    continue
                response.raise_for_status()
            if response.status_code in (500, 502, 503, 504):
                if attempt < OLLAMA_MAX_RETRIES:
                    time.sleep(max(0.1, OLLAMA_BASE_BACKOFF ** attempt))
                    continue
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(max(0.1, OLLAMA_BASE_BACKOFF ** attempt))
                continue
            raise RuntimeError(f"Ollama error after retries: {e}")
    raise RuntimeError("Ollama: max retries exceeded")


def connect(connection_string, timeout=30, _decrypted_token=None, _decrypted_password=None):
    """
    Worker: _decrypted_token after encryption_utils.
    UI validate: may pass _decrypted_password with plain API key path.
    Returns [ok, token_limit, tags] or [False, 0, tags, error].
    """
    global xai_session, ollama_session
    global xai_base_url, ollama_base_url, xai_timeout
    global default_chat_model, emb_model, ollama_emb_model
    global token_limit, emb_token_limit
    global do_chat_construct, native_func_call, tags
    params = {
        "url": "https://api.x.ai/v1",
        "model": "grok-4-1-fast-reasoning",
        "emb_model": "all-minilm:latest",
        "token": "",
        "api_token": "",
        "password": "",
        "token_limit": "131072",
        "chat_template": "True",
        "native_func_call": "False",
        "ollama_url": "http://localhost:11434",
        "ollama_emb_model": "all-minilm:latest",
        "ollama": "True",
        "timeout": "0",
    }
    for part in (connection_string or "").split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        kl = k.strip().lower()
        if kl in params:
            params[kl] = v.strip()
        elif kl == "api_key":
            params["token"] = v.strip()
    params["url"] = normalize_url(params["url"], default_port=None, default_scheme="https")
    params["ollama_url"] = normalize_url(params["ollama_url"], default_port=11434)
    params["ollama_emb_model"] = normalize_model(params["ollama_emb_model"])
    xai_base_url = params["url"].rstrip("/")
    ollama_base_url = params["ollama_url"].rstrip("/")
    default_chat_model = params["model"]
    emb_model = params["emb_model"]
    ollama_emb_model = params["ollama_emb_model"]
    try:
        timeout_val = int(params["timeout"])
        xai_timeout = timeout_val if timeout_val >= 0 else 0
    except ValueError:
        xai_timeout = 0
    try:
        token_limit = int(params["token_limit"])
    except ValueError:
        token_limit = 131072
    emb_token_limit = 4095
    do_chat_construct = params["chat_template"].lower() == "true"
    native_func_call = params["native_func_call"].lower() == "true"
    use_ollama = params["ollama"].lower() == "true"
    # token priority: explicit decrypt args → plain token/api_token in string → env
    if _decrypted_token is not None:
        api_key = _decrypted_token
    elif _decrypted_password is not None and params.get("password") in ("", "empty"):
        # UI sometimes passes password field as plain when no encryption
        api_key = params.get("token") or params.get("api_token") or _decrypted_password
    else:
        api_key = (
            params.get("token")
            or params.get("api_token")
            or os.getenv("XAI_API_KEY")
            or os.getenv("GROK_API_KEY")
            or ""
        )
    if params.get("password") == "empty":
        api_key = params.get("token") or params.get("api_token") or api_key
    if not api_key or api_key in ("set",):
        return [False, 0, tags, "No API key (token= / api_token= / XAI_API_KEY / GROK_API_KEY)"]
    xai_session = requests.Session()
    xai_session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "milana-grok-provider",
    })
    xai_session.emb_model = emb_model
    # Validate via HTTP only (no webbrowser.open)
    try:
        timeout_value = _get_timeout_value()
        r = xai_session.get(f"{xai_base_url}/models", timeout=timeout_value or 30)
        r.raise_for_status()
        let_log(f"Grok/xAI connect OK, models status={r.status_code}")
    except Exception as e:
        return [False, 0, tags, f"Grok/xAI connect error: {e}"]
    ollama_session = None
    if use_ollama:
        try:
            ollama_session = requests.Session()
            ollama_session.headers.update({"Content-Type": "application/json"})
            timeout_value = _get_timeout_value()
            r = ollama_session.get(f"{ollama_base_url}/api/tags", timeout=timeout_value or 30)
            r.raise_for_status()
            models_data = r.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            if ollama_emb_model not in available_models:
                embed_models = [m for m in available_models if 'embed' in m.lower()]
                if embed_models:
                    ollama_emb_model = embed_models[0]
                elif available_models:
                    ollama_emb_model = available_models[0]
                else:
                    raise RuntimeError("No Ollama models for embeddings")
            show_response = ollama_session.post(
                f"{ollama_base_url}/api/show",
                json={"name": ollama_emb_model},
                timeout=timeout_value or 30)
            if show_response.status_code == 200:
                emb_token_limit = find_ollama_context_size(show_response.json(), ollama_emb_model)
            let_log(f"Grok+Ollama emb: {ollama_emb_model}, emb_limit={emb_token_limit}")
        except Exception as e:
            ollama_session = None
            let_log(f"Ollama emb optional failed (RAG may need ollama=True): {e}")
    return [True, token_limit, tags]


def disconnect():
    global xai_session, ollama_session
    if xai_session:
        xai_session.close()
        xai_session = None
    if ollama_session:
        ollama_session.close()
        ollama_session = None
    return True


def ask_model(generation_params):
    """Completions-style: build chat messages from prompt+system (xAI prefers chat)."""
    prompt = generation_params.get("prompt", "")
    system = generation_params.get("system", "")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": generation_params.get("model", default_chat_model),
        "messages": messages,
        "max_tokens": generation_params.get("max_tokens", 1024),
        "temperature": generation_params.get("temperature", 0.7),
        "stream": False,
    }
    data = _make_request(f"{xai_base_url}/chat/completions", payload)
    return data["choices"][0]["message"]["content"].strip()


def ask_model_chat(generation_params):
    payload = {
        "model": generation_params.get("model", default_chat_model),
        "messages": generation_params.get("messages", []),
        "max_tokens": generation_params.get("max_tokens", 1024),
        "temperature": generation_params.get("temperature", 0.7),
        "stream": False,
    }
    return _make_request(f"{xai_base_url}/chat/completions", payload)


def create_embeddings(text):
    text = text.strip()
    if not ollama_session:
        raise RuntimeError(
            "xAI provider has no embeddings API; enable ollama=True and a local emb model")
    try:
        if len(text) > 2000:
            text = text[:2000]
        payload = {"model": ollama_emb_model, "prompt": text}
        data = _ollama_request_with_backoff(f"{ollama_base_url}/api/embeddings", payload)
        return data["embedding"]
    except Exception as e:
        raise RuntimeError(f"Ollama embeddings error: {e}")
