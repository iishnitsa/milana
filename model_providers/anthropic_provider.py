# anthropic_provider.py
import requests
import json
import traceback
import time
import random
import re
from typing import Optional, Dict, Any, List, Tuple
from cross_gpt import let_log

# Глобальные переменные состояния
session = None
api_key = None
base_url = "https://api.anthropic.com/v1"
chat_model = "claude-3-haiku-20240307"
timeout = 30
token_limit = 200000

tags = {
    "system": None,
    "user": "Human:",
    "assistant": "Assistant:",
    "end": None}

# Константы для повторных попыток Anthropic
MAX_RETRIES = 12
BASE_BACKOFF = 2.0
MAX_WAIT_TOTAL = 300
MAX_QUOTA_WAIT = 5 * 60 * 60  # 5 часов

# Константы для Ollama (эмбеддинги)
OLLAMA_MAX_RETRIES = 20
OLLAMA_MAX_WAIT_TOTAL = 420
OLLAMA_BASE_BACKOFF = 2.0

# Ollama настройки (обязательны для эмбеддингов)
ollama_session = None
ollama_base_url = "http://localhost:11434"
ollama_emb_model = "all-minilm:latest"
emb_token_limit = 4095

MODEL_LIMITS = {
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-2.1": 200000,
    "claude-2.0": 100000,
    "claude-instant-1.2": 100000,}

def _normalize_url(url, default_port=None, default_scheme="https"):
    """Добавляет схему по умолчанию, если отсутствует."""
    if not url: return url
    if "://" not in url: url = default_scheme + "://" + url
    return url

def _normalize_model(model):
    """Добавляет :latest, если в имени модели нет двоеточия."""
    if model and ":" not in model: return model + ":latest"
    return model

def _find_ollama_context_size(model_data, model_name="unknown"):
    """Определяет лимит контекста для Ollama модели"""
    let_log(f"Определение контекста для Ollama модели '{model_name}'")
    model_info = model_data.get('model_info', {})
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
                if val > 0: return val
            except (ValueError, TypeError): continue
    for key, value in model_info.items():
        if 'context_length' in key.lower() and isinstance(value, (int, float)): return int(value)
    parameters = model_data.get('parameters', '')
    if parameters:
        match = re.search(r'num_ctx\s*[=: ]\s*(\d+)', parameters, re.IGNORECASE)
        if match: return int(match.group(1))
    model_file = model_data.get('model_file', '')
    if model_file:
        match = re.search(r'num_ctx\s*[=: ]\s*(\d+)', model_file, re.IGNORECASE)
        if match: return int(match.group(1))
    context_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 128000, 200000, 262144]
    model_text = json.dumps(model_data)
    found_sizes = sorted([int(num) for num in re.findall(r'\b\d{4,7}\b', model_text)
                          if int(num) in context_sizes], reverse=True)
    if found_sizes: return found_sizes[0]
    return 4095

def _handle_http_error(resp, attempt):
    """Обработка HTTP ошибок Anthropic."""
    if resp.status_code == 429:
        retry_after = resp.headers.get('retry-after') or resp.headers.get('Retry-After')
        wait_time = None
        if retry_after and retry_after.isdigit():
            wait_time = int(retry_after)
            if wait_time > MAX_QUOTA_WAIT: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
            let_log(f"Anthropic 429 with Retry-After={wait_time}s. Waiting...")
            time.sleep(wait_time)
            return True
        try:
            err_data = resp.json()
            err_msg = str(err_data).lower()
            if 'insufficient_quota' in err_msg or 'balance' in err_msg: raise RuntimeError("balance end")
            if 'session limit' in err_msg: wait_time = 5 * 60 * 60
            elif 'weekly limit' in err_msg: wait_time = 7 * 24 * 60 * 60
            if wait_time is not None:
                if wait_time > MAX_QUOTA_WAIT: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                let_log(f"Anthropic квотная ошибка: ожидание {wait_time}с...")
                time.sleep(wait_time)
                return True
        except: pass
        if attempt < MAX_RETRIES:
            wait_time = min(60, BASE_BACKOFF ** attempt + random.random())
            let_log(f"Anthropic 429 (Rate Limit). Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
            return True
        else: resp.raise_for_status()
    elif resp.status_code == 402: raise RuntimeError("balance end")
    elif resp.status_code == 400:
        try:
            err_data = resp.json()
            err_msg = str(err_data).lower()
            if 'context' in err_msg or 'token' in err_msg or 'length' in err_msg: raise RuntimeError("ContextOverflowError")
        except: pass
        resp.raise_for_status()
    elif 500 <= resp.status_code < 600:
        if attempt < MAX_RETRIES:
            wait_time = min(60, BASE_BACKOFF ** attempt + random.random())
            let_log(f"Anthropic {resp.status_code} Server Error. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
            return True
        else: resp.raise_for_status()
    return False

def _make_request(payload):
    """Универсальный метод для запросов к Anthropic API."""
    global session, timeout
    if not session: raise RuntimeError("Anthropic client not connected")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(f"{base_url}/messages", json=payload, timeout=timeout)
            if _handle_http_error(resp, attempt): continue
            resp.raise_for_status()
            return resp.json()
        except RuntimeError as e:
            if "balance end" in str(e) or "ContextOverflowError" in str(e): raise
            if attempt == MAX_RETRIES: raise
            time.sleep(BASE_BACKOFF ** attempt)
        except requests.RequestException as e:
            msg = str(e).lower()
            if "context" in msg or "token" in msg or "limit" in msg: raise RuntimeError("ContextOverflowError")
            if attempt == MAX_RETRIES: raise ConnectionError(f"Ошибка генерации: {str(e)}")
            time.sleep(BASE_BACKOFF ** attempt)
        except Exception as e:
            if attempt == MAX_RETRIES: raise RuntimeError(f"Ошибка обработки ответа: {str(e)}")
            time.sleep(BASE_BACKOFF ** attempt)
    raise RuntimeError("Max retries exceeded")

def connect(connection_string: str, timeout: int = 30, _decrypted_token: str = None) -> Tuple[bool, int, Dict[str, Optional[str]]]:
    global session, api_key, chat_model, token_limit, emb_token_limit, tags, timeout
    global ollama_base_url, ollama_emb_model, ollama_session
    params = {
        'chat': 'claude-3-haiku-20240307',
        'token': '',
        'password': '',
        'ollama_url': 'http://localhost:11434',
        'ollama_emb_model': 'all-minilm:latest'}
    for part in connection_string.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            if key in params: params[key] = value
    timeout = timeout if timeout > 0 else 30
    ollama_base_url = _normalize_url(params['ollama_url'], default_port=11434)
    ollama_emb_model = _normalize_model(params['ollama_emb_model'])
    if _decrypted_token is not None: api_key = _decrypted_token; let_log("Anthropic: using decrypted token from _decrypted_token")
    else:
        api_key = params['token']
        if api_key and params.get('password') == 'set': let_log("Anthropic: password=set but no _decrypted_token provided, will try to use token as is")
    if not api_key: return False, token_limit, tags, "No API key provided"
    if params['chat'] not in MODEL_LIMITS: return False, token_limit, tags, f"Неподдерживаемая модель: {params['chat']}. Доступные: {list(MODEL_LIMITS.keys())}"
    chat_model = params['chat']
    token_limit = MODEL_LIMITS[chat_model]
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": api_key})
    # Обязательная настройка Ollama для эмбеддингов
    try:
        ollama_session = requests.Session()
        ollama_session.headers.update({"Content-Type": "application/json"})
        tags_url = f"{ollama_base_url}/api/tags"
        resp = ollama_session.get(tags_url, timeout=timeout)
        resp.raise_for_status()
        models_data = resp.json()
        available_models = [model['name'] for model in models_data.get('models', [])]
        let_log(f"Доступные модели Ollama: {available_models}")
        if ollama_emb_model not in available_models:
            let_log(f"Ollama модель '{ollama_emb_model}' не найдена. Ищем альтернативу...")
            embed_models = [m for m in available_models if 'embed' in m.lower()]
            if embed_models: ollama_emb_model = embed_models[0]; let_log(f"Выбрана модель для эмбеддингов: {ollama_emb_model}")
            else:
                if available_models: ollama_emb_model = available_models[0]; let_log(f"Модель для эмбеддингов не найдена. Используем первую: {ollama_emb_model}")
                else: raise RuntimeError("Нет доступных моделей в Ollama")
        show_url = f"{ollama_base_url}/api/show"
        show_payload = {"name": ollama_emb_model}
        show_resp = ollama_session.post(show_url, json=show_payload, timeout=timeout)
        if show_resp.status_code == 200:
            model_details = show_resp.json()
            emb_token_limit = _find_ollama_context_size(model_details, ollama_emb_model)
        else: emb_token_limit = 4095
        let_log(f"Ollama для эмбеддингов успешно настроен. Модель: {ollama_emb_model}, лимит контекста: {emb_token_limit}")
    except Exception as e:
        session = None
        ollama_session = None
        return False, token_limit, tags, f"Ollama connection failed (required for embeddings): {e}"
    # Проверка Anthropic
    try:
        test_payload = {"model": chat_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
        r = session.post(f"{base_url}/messages", json=test_payload, timeout=timeout)
        if r.status_code >= 400:
            session = None
            ollama_session = None
            return False, token_limit, tags, f"Ошибка при проверке модели Anthropic: {r.status_code}"
        return True, token_limit, tags
    except Exception as e:
        traceback.print_exc()
        session = None
        ollama_session = None
        return False, token_limit, tags, str(e)

def disconnect() -> bool:
    global session, ollama_session
    if session is not None: session.close(); session = None
    if ollama_session is not None: ollama_session.close(); ollama_session = None
    return True

def ask_model(generation_params: Dict[str, Any]) -> str:
    global session, chat_model, timeout
    if session is None: raise RuntimeError("Клиент Anthropic не инициализирован")
    try:
        prompt = generation_params["prompt"]
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": chat_model,
            "messages": messages,
            "max_tokens": generation_params.get("max_tokens", 1024),
            "temperature": generation_params.get("temperature", 1.0),
            "top_p": generation_params.get("top_p", None),
            "stream": False}
        if payload["top_p"] is None: del payload["top_p"]
        data = _make_request(payload)
        return data["content"][0]["text"]
    except RuntimeError as e: raise
    except Exception as e: traceback.print_exc(); raise RuntimeError(f"Ошибка генерации: {str(e)}")

def ask_model_chat(generation_params: Dict[str, Any]) -> Dict[str, Any]:
    global session, chat_model, timeout
    if session is None: raise RuntimeError("Клиент Anthropic не инициализирован")
    try:
        messages = generation_params.get("messages", [])
        if not messages: raise ValueError("Нет сообщений для чата")
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        dialog_messages = [m for m in messages if m["role"] != "system"]
        payload = {
            "model": chat_model,
            "messages": dialog_messages,
            "max_tokens": generation_params.get("max_tokens", 1024),
            "temperature": generation_params.get("temperature", 1.0),
            "top_p": generation_params.get("top_p", None),
            "stream": False}
        if system_messages: payload["system"] = "\n".join(system_messages)
        if payload["top_p"] is None: del payload["top_p"]
        data = _make_request(payload)
        response_text = data["content"][0]["text"]
        return {
            "id": f"chatcmpl-{chat_model}",
            "choices": [{"message": {"role": "assistant", "content": response_text}, "finish_reason": "stop", "index": 0}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "model": chat_model}
    except RuntimeError as e: raise
    except Exception as e: traceback.print_exc(); raise RuntimeError(f"Ошибка чата: {str(e)}")

def create_embeddings(text: str) -> List[float]:
    """Создание эмбеддингов через Ollama (без отдельной вспомогательной функции)"""
    global ollama_session, ollama_base_url, ollama_emb_model
    if ollama_session is None: raise RuntimeError("Ollama для эмбеддингов не инициализирован. Проверьте подключение.")
    url = f"{ollama_base_url}/api/embeddings"
    payload = {"model": ollama_emb_model, "prompt": text.strip()}
    start_time = time.time()
    attempt = 1
    while True:
        try:
            elapsed = time.time() - start_time
            if elapsed > OLLAMA_MAX_WAIT_TOTAL: raise RuntimeError(f"Превышено общее время ожидания ({OLLAMA_MAX_WAIT_TOTAL} с) для эмбеддингов")
            response = ollama_session.post(url, json=payload, timeout=30)
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
                    if wait_time > MAX_QUOTA_WAIT: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                    if wait_time > OLLAMA_MAX_WAIT_TOTAL: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает общий лимит {OLLAMA_MAX_WAIT_TOTAL}с")
                    let_log(f"Ollama эмбеддинги: квотная 429, ожидание {wait_time} с")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                if attempt < OLLAMA_MAX_RETRIES:
                    wait_time = OLLAMA_BASE_BACKOFF ** attempt
                    remaining = OLLAMA_MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining: wait_time = remaining
                    if wait_time < 0.1: wait_time = 0.1
                    let_log(f"Ollama эмбеддинги: 429 rate limit, повтор через {wait_time:.2f} с")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
            if response.status_code == 500:
                error_text = response.text.lower()
                if any(keyword in error_text for keyword in ['context length', 'exceeds context', 'token limit']): raise RuntimeError('ContextOverflowError')
            response.raise_for_status()
            data = response.json()
            embedding = data.get('embedding', [])
            if not embedding: raise ValueError("Пустой вектор эмбеддингов в ответе от Ollama")
            return embedding
        except requests.exceptions.ConnectionError as e:
            if attempt < OLLAMA_MAX_RETRIES:
                wait_time = OLLAMA_BASE_BACKOFF ** attempt
                remaining = OLLAMA_MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining: wait_time = remaining
                if wait_time < 0.1: wait_time = 0.1
                let_log(f"Ollama эмбеддинги: Ошибка соединения, повтор через {wait_time:.2f} с")
                time.sleep(wait_time)
                attempt += 1
                continue
            else: raise RuntimeError(f"Ошибка соединения после {OLLAMA_MAX_RETRIES} попыток: {e}")
        except requests.exceptions.Timeout as e:
            if attempt < OLLAMA_MAX_RETRIES:
                wait_time = OLLAMA_BASE_BACKOFF ** attempt
                remaining = OLLAMA_MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining: wait_time = remaining
                if wait_time < 0.1: wait_time = 0.1
                let_log(f"Ollama эмбеддинги: Таймаут, повтор через {wait_time:.2f} с")
                time.sleep(wait_time)
                attempt += 1
                continue
            else: raise RuntimeError(f"Таймаут запроса после {OLLAMA_MAX_RETRIES} попыток: {e}")
        except RuntimeError as e:
            if "balance end" in str(e) or "ContextOverflowError" in str(e): raise
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(OLLAMA_BASE_BACKOFF ** attempt)
                attempt += 1
                continue
            raise
        except Exception as e:
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(OLLAMA_BASE_BACKOFF ** attempt)
                attempt += 1
                continue
            raise RuntimeError(f"Ошибка API эмбеддингов Ollama: {str(e)}")