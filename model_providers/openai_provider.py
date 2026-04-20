import requests
import re
import os
import json
import time
import random
import urllib.parse
from cross_gpt import let_log

# === Глобальные переменные ===
openai_session = None
ollama_session = None

openai_base_url = ""
ollama_base_url = ""
openai_timeout = 0

default_chat_model = "gpt-5-nano"
emb_model = "text-embedding-3-small"
ollama_emb_model = "all-minilm:latest"

token_limit = 4095
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
    "tool_result_start": "", "tool_result_end": "",}

# Настройки адаптивности (ТОЛЬКО ДЛЯ OPENAI)
MAX_RETRIES = 12          # Максимум попыток
BASE_BACKOFF = 2.0        # Основание экспоненты (2, 4, 8, 16...)
last_wait_time = 0.0      # "Память" о задержке между вызовами

# Константы для прогрессивной задержки Ollama (fallback)
OLLAMA_MAX_RETRIES = 20
OLLAMA_MAX_WAIT_TOTAL = 420      # 7 минут для обычных rate limit'ов
OLLAMA_BASE_BACKOFF = 2.0
MAX_QUOTA_WAIT = 5 * 60 * 60      # 5 часов – максимальное время ожидания для квотных ошибок

def normalize_url(url, default_port=None, default_scheme="http"):
    """Добавляет схему по умолчанию, если отсутствует. Порт добавляется только если указан default_port."""
    if not url: return url
    if "://" not in url: url = default_scheme + "://" + url
    parsed = urllib.parse.urlparse(url)
    if default_port is not None and not parsed.port:
        host = parsed.hostname or ""
        new_netloc = f"{host}:{default_port}"
        parsed = parsed._replace(netloc=new_netloc)
    return parsed.geturl()

def normalize_model(model):
    """Добавляет :latest, если в имени модели нет двоеточия."""
    if model and ":" not in model: return model + ":latest"
    return model

def find_ollama_context_size(model_data, model_name="unknown"):
    """
    Автоматическое определение лимита контекста для Ollama модели.
    Использует данные из ответа /api/show (model_data).
    """
    let_log(f"Определение контекста для Ollama модели '{model_name}'")
    model_info = model_data.get('model_info', {})
    # 1. Прямые ключи для разных архитектур
    direct_keys = [
        'context_length',
        'max_position_embeddings',
        'max_seq_len',
        'n_ctx',
        'llama.context_length',
        'gemma2.context_length',
        'gemma3.context_length',
        'mistral.context_length',
        'qwen2.context_length',
        'phi3.context_length',]
    for key in direct_keys:
        if key in model_info:
            try:
                val = int(model_info[key])
                if val > 0: let_log(f"Найден контекст по ключу '{key}': {val}"); return val
            except (ValueError, TypeError): continue
    # 2. Любой ключ, содержащий 'context_length'
    for key, value in model_info.items():
        if 'context_length' in key.lower() and isinstance(value, (int, float)): let_log(f"Найден контекст по ключу '{key}': {int(value)}"); return int(value)
    # 3. Поиск в parameters (num_ctx)
    parameters = model_data.get('parameters', '')
    if parameters:
        match = re.search(r'num_ctx\s*[=: ]\s*(\d+)', parameters, re.IGNORECASE)
        if match:
            ctx = int(match.group(1))
            let_log(f"Найден num_ctx в parameters: {ctx}")
            return ctx
    # 4. Жадный поиск типичных контекстов в JSON
    context_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 128000, 200000, 262144]
    model_text = json.dumps(model_data)
    found_sizes = sorted([int(num) for num in re.findall(r'\b\d{4,7}\b', model_text)
                          if int(num) in context_sizes], reverse=True)
    if found_sizes: let_log(f"Жадный поиск нашёл контекст: {found_sizes[0]}"); return found_sizes[0]
    # Значение по умолчанию
    let_log(f"Не удалось определить контекст для {model_name}, используется 4095")
    return 4095

def _get_timeout_value():
    global openai_timeout
    return None if openai_timeout == 0 else openai_timeout

def _make_request(api_url, payload):
    """
    Универсальный метод для всех OpenAI-совместимых запросов.
    Реализует адаптивный RPM, экспоненциальный кулдаун и память.
    """
    global last_wait_time
    if not openai_session: raise RuntimeError("OpenAI client not connected")
    # Превентивная пауза на основе памяти о прошлых 429
    if last_wait_time > 0:
        preemptive_pause = min(5.0, last_wait_time * 0.5)
        if preemptive_pause > 0.1: time.sleep(preemptive_pause)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            let_log(f"openai request attempt {attempt}: {api_url}")
            timeout_value = _get_timeout_value()
            r = openai_session.post(api_url, json=payload, timeout=timeout_value)
            # --- Обработка 429 Too Many Requests ---
            if r.status_code == 429:
                try:
                    error_data = r.json().get("error", {})
                    error_code = error_data.get("code")
                    error_msg = error_data.get("message", "").lower()
                except: error_code = None; error_msg = r.text.lower()
                # Недостаток средств (квота)
                if error_code == "insufficient_quota" or "insufficient_quota" in error_msg: let_log(f"CRITICAL: insufficient quota - {r.text}"); raise RuntimeError("balance end")
                # Проверяем Retry-After
                retry_after = r.headers.get('Retry-After')
                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                    if wait_time > 60: wait_time = 60
                    let_log(f"Status 429 with Retry-After={wait_time}s. Waiting...")
                    time.sleep(wait_time)
                    continue
                # Обычный rate limit – экспоненциальная задержка с памятью
                if attempt < MAX_RETRIES:
                    wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                    last_wait_time = wait_time
                    let_log(f"Status 429 (Rate Limit). Waiting {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    continue
                else: raise RuntimeError("Rate limit exceeded after maximum retries")
            # --- Обработка 402 Payment Required ---
            if r.status_code == 402: let_log(f"Payment Required (402): {r.text}"); raise RuntimeError("balance end")
            # --- Обработка 400 Bad Request (в т.ч. context length) ---
            if r.status_code == 400:
                try:
                    error_data = r.json().get("error", {})
                    error_code = error_data.get("code")
                    error_msg = error_data.get("message", "").lower()
                    if error_code == "context_length_exceeded" or "context_length" in error_msg: raise RuntimeError("ContextOverflowError")
                except: pass
                r.raise_for_status()  # выбросит исключение с деталями
            # --- Обработка 5xx Server Errors ---
            if 500 <= r.status_code < 600:
                if attempt < MAX_RETRIES:
                    retry_after = r.headers.get('Retry-After')
                    if retry_after and retry_after.isdigit(): wait_time = int(retry_after)
                    else: wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                    last_wait_time = wait_time
                    let_log(f"Status {r.status_code} (Server Error). Waiting {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    continue
                else: raise RuntimeError(f"Server error after maximum retries: {r.text}")
            r.raise_for_status()
            resp_data = r.json()
            # Проверка на скрытые ошибки в JSON (прокси)
            if "error" in resp_data:
                err_info = resp_data["error"]
                err_msg = err_info.get("message", "Unknown error")
                let_log(f"Provider returned error in JSON: {err_msg}")
                if "insufficient_quota" in err_msg.lower(): raise RuntimeError("balance end")
                if attempt < MAX_RETRIES:
                    wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                    let_log(f"Retrying in {wait_time:.2f}s due to provider glitch...")
                    time.sleep(wait_time)
                    continue
                else: raise RuntimeError(f"API Provider Error: {err_msg}")
            # Успех – уменьшаем память о задержках
            last_wait_time *= 0.7
            if last_wait_time < 0.1: last_wait_time = 0
            return resp_data
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                last_wait_time = wait_time
                let_log(f"Request timeout on attempt {attempt}. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                continue
            else: raise RuntimeError("Request timed out after maximum retries")
        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES: raise
            time.sleep(2)
        except Exception as e:
            if "balance end" in str(e) or "ContextOverflowError" in str(e): raise
            raise RuntimeError(f"OpenAI request error: {e}")

def _ollama_request_with_backoff(api_url, json_payload):
    """
    Для запросов к Ollama (эмбеддинги) с прогрессивной задержкой.
    Обрабатывает квотные ошибки (session limit / weekly limit) с ограничением ожидания 5 часов.
    """
    start_time = time.time()
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        elapsed = time.time() - start_time
        if elapsed > OLLAMA_MAX_WAIT_TOTAL: raise RuntimeError(f"Превышено общее время ожидания ({OLLAMA_MAX_WAIT_TOTAL} с)")
        try:
            response = ollama_session.post(api_url, json=json_payload)
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                wait_time = None
                if retry_after and retry_after.isdigit(): wait_time = int(retry_after)
                else:
                    try:
                        err_data = response.json()
                        err_msg = err_data.get('error', '').lower()
                        if 'session limit' in err_msg: wait_time = 5 * 60 * 60
                        elif 'weekly limit' in err_msg: wait_time = 7 * 24 * 60 * 60
                        elif 'insufficient_quota' in err_msg: raise RuntimeError("balance end")
                    except: pass
                if wait_time is not None:
                    # НОВОЕ: если время ожидания превышает 5 часов – сразу исключение
                    if wait_time > MAX_QUOTA_WAIT: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                    if wait_time > OLLAMA_MAX_WAIT_TOTAL: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает общий лимит {OLLAMA_MAX_WAIT_TOTAL}с")
                    let_log(f"Ollama: квотный лимит (429). Ожидание {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue
                # Обычный rate limit
                if attempt < OLLAMA_MAX_RETRIES:
                    wait_time = OLLAMA_BASE_BACKOFF ** attempt
                    remaining = OLLAMA_MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining: wait_time = remaining
                    if wait_time < 0.1: wait_time = 0.1
                    let_log(f"Ollama HTTP 429 (Rate Limit) на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue
                else: response.raise_for_status()
            if response.status_code in (500, 502, 503, 504):
                if attempt < OLLAMA_MAX_RETRIES:
                    wait_time = OLLAMA_BASE_BACKOFF ** attempt
                    remaining = OLLAMA_MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining: wait_time = remaining
                    if wait_time < 0.1: wait_time = 0.1
                    let_log(f"Ollama HTTP {response.status_code} на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue
                else: response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            if attempt < OLLAMA_MAX_RETRIES:
                wait_time = OLLAMA_BASE_BACKOFF ** attempt
                remaining = OLLAMA_MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining: wait_time = remaining
                if wait_time < 0.1: wait_time = 0.1
                let_log(f"Ollama ошибка соединения на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                time.sleep(wait_time)
                continue
            else: raise RuntimeError(f"Ollama ошибка соединения после {OLLAMA_MAX_RETRIES} попыток: {e}")
        except requests.exceptions.Timeout as e:
            if attempt < OLLAMA_MAX_RETRIES:
                wait_time = OLLAMA_BASE_BACKOFF ** attempt
                remaining = OLLAMA_MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining: wait_time = remaining
                if wait_time < 0.1: wait_time = 0.1
                let_log(f"Ollama таймаут на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                time.sleep(wait_time)
                continue
            else: raise RuntimeError(f"Ollama таймаут запроса после {OLLAMA_MAX_RETRIES} попыток: {e}")
        except requests.exceptions.RequestException as e:
            if attempt < OLLAMA_MAX_RETRIES:
                wait_time = OLLAMA_BASE_BACKOFF ** attempt
                remaining = OLLAMA_MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining: wait_time = remaining
                if wait_time < 0.1: wait_time = 0.1
                let_log(f"Ollama сетевая ошибка на попытке {attempt}: {e}. Ожидание {wait_time:.2f} с...")
                time.sleep(wait_time)
                continue
            else: raise RuntimeError(f"Ollama сетевая ошибка после {OLLAMA_MAX_RETRIES} попыток: {e}")
    raise RuntimeError("Ollama: превышено максимальное количество попыток")

def connect(connection_string, timeout=30, _decrypted_token=None):
    global openai_session, ollama_session
    global openai_base_url, ollama_base_url, openai_timeout
    global default_chat_model, emb_model, ollama_emb_model
    global token_limit, emb_token_limit
    global do_chat_construct, native_func_call, tags
    params = {
        "url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "emb_model": "text-embedding-3-small",
        "token": '',
        "password": '',
        "token_limit": "32768",
        "chat_template": "True",
        "native_func_call": "False",
        "ollama_url": "http://localhost:11434",
        "ollama_emb_model": "all-minilm:latest",
        "ollama": "False",
        "timeout": "0",}
    for part in connection_string.split(";"):
        if "=" not in part: continue
        k, v = part.split("=", 1)
        if k.strip().lower() in params: params[k.strip().lower()] = v.strip()
    params["url"] = normalize_url(params["url"], default_port=None, default_scheme="https")
    params["ollama_url"] = normalize_url(params["ollama_url"], default_port=11434)
    params["ollama_emb_model"] = normalize_model(params["ollama_emb_model"])
    openai_base_url = params["url"].rstrip("/")
    ollama_base_url = params["ollama_url"].rstrip("/")
    default_chat_model = params["model"]
    emb_model = params["emb_model"]
    ollama_emb_model = params["ollama_emb_model"]
    try:
        timeout_val = int(params["timeout"])
        openai_timeout = timeout_val if timeout_val >= 0 else 0
    except ValueError: openai_timeout = 0
    token_limit = int(params["token_limit"])
    emb_token_limit = token_limit   # временно, потом переопределим для Ollama
    do_chat_construct = params["chat_template"].lower() == "true"
    native_func_call = params["native_func_call"].lower() == "true"
    use_ollama = params["ollama"].lower() == "true"
    # Приоритет: если передан расшифрованный токен, используем его
    if _decrypted_token is not None: api_key = _decrypted_token
    else: api_key = params["token"] or os.getenv("OPENAI_API_KEY")
    if not api_key: return [False, 0, tags, "No API key"]
    openai_session = requests.Session()
    openai_session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "cross-gpt-openai-provider",})
    openai_session.emb_model = emb_model
    try:
        timeout_value = _get_timeout_value()
        r = openai_session.get(f"{openai_base_url}/models", timeout=timeout_value)
        r.raise_for_status()
    except Exception as e: return [False, 0, tags, f"OpenAI connect error: {e}"]
    # --- Настройка Ollama для эмбеддингов ---
    ollama_session = None
    if use_ollama:
        try:
            ollama_session = requests.Session()
            ollama_session.headers.update({"Content-Type": "application/json"})
            timeout_value = _get_timeout_value()
            r = ollama_session.get(f"{ollama_base_url}/api/tags", timeout=timeout_value)
            r.raise_for_status()
            models_data = r.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            let_log(f"Доступные модели Ollama: {available_models}")
            # Проверяем, существует ли запрошенная модель
            if ollama_emb_model not in available_models:
                let_log(f"Ollama модель '{ollama_emb_model}' не найдена. Ищем альтернативу...")
                embed_models = [m for m in available_models if 'embed' in m.lower()]
                if embed_models: ollama_emb_model = embed_models[0]; let_log(f"Выбрана модель для эмбеддингов: {ollama_emb_model}")
                else:
                    if available_models: ollama_emb_model = available_models[0]; let_log(f"Модель для эмбеддингов не найдена. Используем первую: {ollama_emb_model}")
                    else: raise RuntimeError("Нет доступных моделей в Ollama")
            # Получаем детали модели для определения контекста
            show_payload = {"name": ollama_emb_model}
            show_response = ollama_session.post(f"{ollama_base_url}/api/show", json=show_payload, timeout=timeout_value)
            if show_response.status_code == 200: model_details = show_response.json(); emb_token_limit = find_ollama_context_size(model_details, ollama_emb_model)
            else: emb_token_limit = 4095
            let_log(f"Ollama embeddings включены. Модель: {ollama_emb_model}, лимит контекста: {emb_token_limit}")
        except Exception as e:
            ollama_session = None
            let_log(f"Ollama connect failed (fallback to OpenAI embeddings): {e}")
            # Если Ollama не работает, эмбеддинги будут через OpenAI
            emb_token_limit = token_limit
    return [True, token_limit, tags]

def disconnect():
    global openai_session, ollama_session
    if openai_session: openai_session.close(); openai_session = None
    if ollama_session: ollama_session.close(); ollama_session = None
    return True

def ask_model(generation_params):
    prompt = generation_params.get("prompt", "")
    system = generation_params.get("system", "")
    full_prompt = (
        tags["bos"] +
        tags["sys_start"] + system + tags["sys_end"] +
        tags["user_start"] + prompt + tags["user_end"] +
        tags["assist_start"])
    payload = {
        "model": generation_params.get("model", default_chat_model),
        "prompt": full_prompt,
        "max_tokens": generation_params.get("max_tokens", 1024),
        "temperature": generation_params.get("temperature", 0.7),}
    data = _make_request(f"{openai_base_url}/completions", payload)
    return data["choices"][0]["text"].strip()

def ask_model_chat(generation_params):
    payload = {
        "model": generation_params.get("model", default_chat_model),
        "messages": generation_params.get("messages", []),
        "max_tokens": generation_params.get("max_tokens", 1024),
        "temperature": generation_params.get("temperature", 0.7),
        "stream": False,}
    return _make_request(f"{openai_base_url}/chat/completions", payload)

def create_embeddings(text):
    text = text.strip()
    # Приоритет – Ollama, если он подключён
    if ollama_session:
        try:
            if len(text) > 2000:
                let_log(f"create_embeddings (Ollama): обрезаем текст с {len(text)} до 2000 символов")
                text = text[:2000]
            payload = {"model": ollama_emb_model, "prompt": text}
            data = _ollama_request_with_backoff(f"{ollama_base_url}/api/embeddings", payload)
            return data["embedding"]
        except Exception as e: raise RuntimeError(f"Ollama embeddings error: {e}")
    if not openai_session: raise RuntimeError("No embeddings backend")
    payload = {"model": openai_session.emb_model, "input": text}
    data = _make_request(f"{openai_base_url}/embeddings", payload)
    if "data" in data and len(data["data"]) > 0: return data["data"][0]["embedding"]
    else: raise RuntimeError(f"Could not find embeddings in response: {data}")