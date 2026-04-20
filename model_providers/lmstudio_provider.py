import requests
import traceback
import time
import random
from typing import Optional, Dict, Any, List, Tuple
from cross_gpt import let_log

# Глобальные переменные состояния
session = None
base_url = ""
chat_model = None
emb_model = None
auth = None
headers = {}
timeout_value = 30
token_limit = 4096
emb_token_limit = 4096

# Настройки для Ollama fallback
use_ollama = False
ollama_url = ""
ollama_emb_model = ""
ollama_session = None

tags = {
    "system": None,
    "user": None,
    "assistant": None,
    "end": None}

# Константы для повторных попыток
MAX_RETRIES = 10
BASE_BACKOFF = 2.0
MAX_WAIT_TOTAL = 300
MAX_QUOTA_WAIT = 5 * 60 * 60  # 5 часов в секундах

def _request_with_retry(method: str, url: str, **kwargs):
    """Универсальный метод с экспоненциальным backoff и обработкой 429/402"""
    global session, headers, auth, timeout_value
    start_time = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.request(method, url, headers=headers, auth=auth, timeout=timeout_value, **kwargs)
            if resp.status_code == 429:
                retry_after = resp.headers.get('Retry-After')
                wait_time = None
                if retry_after and retry_after.isdigit(): wait_time = int(retry_after)
                else:
                    try:
                        err_data = resp.json()
                        err_msg = err_data.get('error', {}).get('message', '').lower()
                        if 'session limit' in err_msg: wait_time = 5 * 60 * 60 # 5 часов
                        elif 'weekly limit' in err_msg: wait_time = 7 * 24 * 60 * 60 # 7 дней
                        elif 'insufficient_quota' in err_msg: raise RuntimeError("balance end")
                    except: pass
                if wait_time is not None:
                    if wait_time > MAX_QUOTA_WAIT: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                    let_log(f"LM Studio 429 квота, ожидание {wait_time} с")
                    time.sleep(wait_time)
                    continue
                # Обычный rate limit
                if attempt < MAX_RETRIES:
                    wait_time = min(60, BASE_BACKOFF ** attempt + random.random())
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining: wait_time = max(remaining, 0.1)
                    let_log(f"LM Studio 429 rate limit, повтор через {wait_time:.2f} с")
                    time.sleep(wait_time)
                    continue
            if resp.status_code == 402: raise RuntimeError("balance end")
            if resp.status_code >= 500:
                if attempt < MAX_RETRIES:
                    wait_time = min(60, BASE_BACKOFF ** attempt)
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining: wait_time = max(remaining, 0.1)
                    time.sleep(wait_time)
                    continue
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            err_str = str(e).lower()
            if any(phrase in err_str for phrase in ["context length", "exceeds context", "token limit"]): raise RuntimeError("ContextOverflowError")
            if attempt == MAX_RETRIES: raise RuntimeError(f"HTTP error: {e}")
            wait_time = BASE_BACKOFF ** attempt
            remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
            if wait_time > remaining: wait_time = max(remaining, 0.1)
            time.sleep(wait_time)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES: raise RuntimeError(f"Connection error: {e}")
            wait_time = BASE_BACKOFF ** attempt
            remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
            if wait_time > remaining: wait_time = max(remaining, 0.1)
            time.sleep(wait_time)
        except RuntimeError: raise
        except Exception as e:
            if attempt == MAX_RETRIES: raise
            wait_time = BASE_BACKOFF ** attempt
            remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
            if wait_time > remaining: wait_time = max(remaining, 0.1)
            time.sleep(wait_time)
    raise RuntimeError("Max retries exceeded")

def _ollama_embeddings(text: str) -> List[float]:
    """Получение эмбеддингов через Ollama"""
    global ollama_session, ollama_url, ollama_emb_model, timeout_value
    url = f"{ollama_url}/api/embeddings"
    payload = {"model": ollama_emb_model, "prompt": text}
    start_time = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            elapsed = time.time() - start_time
            if elapsed > MAX_WAIT_TOTAL: raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с)")
            resp = ollama_session.post(url, json=payload, timeout=timeout_value)
            if resp.status_code == 429:
                retry_after = resp.headers.get('Retry-After')
                wait_time = None
                if retry_after and retry_after.isdigit(): wait_time = int(retry_after)
                else:
                    try:
                        err_data = resp.json()
                        err_msg = err_data.get('error', '').lower()
                        if 'session limit' in err_msg: wait_time = 5 * 60 * 60
                        elif 'weekly limit' in err_msg: wait_time = 7 * 24 * 60 * 60
                        elif 'insufficient_quota' in err_msg: raise RuntimeError("balance end")
                    except: pass
                if wait_time is not None:
                    if wait_time > MAX_QUOTA_WAIT: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                    let_log(f"Ollama эмбеддинги: квотная 429, ожидание {wait_time} с")
                    time.sleep(wait_time)
                    continue
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining: wait_time = max(remaining, 0.1)
                    let_log(f"Ollama эмбеддинги: 429 rate limit, повтор через {wait_time:.2f} с")
                    time.sleep(wait_time)
                    continue
            resp.raise_for_status()
            return resp.json()['embedding']
        except requests.HTTPError as e:
            msg = resp.text.lower() if 'resp' in locals() else str(e).lower()
            if any(phrase in msg for phrase in ["exceeds context", "context length", "token limit"]): raise RuntimeError("ContextOverflowError")
            if attempt == MAX_RETRIES: raise
            wait_time = BASE_BACKOFF ** attempt
            remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
            if wait_time > remaining: wait_time = max(remaining, 0.1)
            time.sleep(wait_time)
        except requests.exceptions.ConnectionError as e:
            if attempt == MAX_RETRIES: raise RuntimeError(f"Ошибка соединения: {e}")
            wait_time = BASE_BACKOFF ** attempt
            remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
            if wait_time > remaining: wait_time = max(remaining, 0.1)
            time.sleep(wait_time)
    raise RuntimeError("Не удалось получить эмбеддинги от Ollama")

def connect(connection_string: str, timeout: int = 30, _decrypted_token: str = None) -> Tuple[bool, int, Dict[str, Optional[str]]]:
    global session, base_url, chat_model, emb_model, auth, headers, timeout_value
    global token_limit, emb_token_limit, tags, use_ollama, ollama_url, ollama_emb_model, ollama_session
    params = {
        'host': 'http://localhost',
        'port': 1234,
        'chat': '',
        'emb': '',
        'user': '',
        'token': '',
        'password': '',
        'timeout': str(timeout),
        'ollama': 'false',
        'ollama_url': 'http://localhost:11434',
        'ollama_emb_model': 'all-minilm:latest'}
    try:
        for part in connection_string.split(';'):
            if '=' in part:
                key, value = part.split('=', 1)
                key, value = key.strip().lower(), value.strip()
                if key == 'host': params['host'] = value if '://' in value else f'http://{value}'
                elif key == 'port': params['port'] = int(value)
                elif key in ('chat', 'emb', 'user', 'password', 'token', 'timeout', 'ollama', 'ollama_url', 'ollama_emb_model'): params[key] = value
        timeout_value = int(params['timeout'])
        # Настройка аутентификации
        actual_token = _decrypted_token if _decrypted_token is not None else params['token']
        auth = (params['user'], params['password']) if params['user'] and params['password'] else None
        headers = {"Content-Type": "application/json"}
        if actual_token: headers["Authorization"] = f"Bearer {actual_token}"
        # Настройка URL
        base_url = f"{params['host']}:{params['port']}"
        chat_model = params['chat']
        emb_model = params['emb']
        # Настройка Ollama fallback
        use_ollama = params['ollama'].lower() == 'true'
        ollama_url = params['ollama_url'].rstrip('/')
        ollama_emb_model = params['ollama_emb_model']
        # Создаем сессию
        session = requests.Session()
        # Получаем список моделей
        models_info = _list_models()
        if models_info:
            # Проверка существования чат-модели
            if chat_model:
                found = False
                for model in models_info:
                    if model["id"] == chat_model: token_limit = model.get("context_length", token_limit); found = True; break
                if not found: available_ids = [m["id"] for m in models_info]; return False, token_limit, tags, f"Запрошенная чат-модель '{chat_model}' не найдена. Доступные: {available_ids}"
            # Проверка существования эмбеддинг-модели
            if emb_model:
                found = False
                for model in models_info:
                    if model["id"] == emb_model: emb_token_limit = model.get("context_length", emb_token_limit); found = True; break
                if not found and emb_model: let_log(f"Модель для эмбеддингов '{emb_model}' не найдена, будет использована чат-модель"); emb_model = chat_model; emb_token_limit = token_limit
        # Настройка Ollama сессии если нужно
        if use_ollama:
            ollama_session = requests.Session()
            ollama_session.headers.update({"Content-Type": "application/json"})
            # Проверяем доступность Ollama
            try: r = ollama_session.get(f"{ollama_url}/api/tags", timeout=timeout_value); r.raise_for_status(); let_log(f"Ollama доступен, URL: {ollama_url}")
            except Exception as e: let_log(f"Ollama недоступен: {e}"); ollama_session = None; use_ollama = False
        return True, token_limit, tags
    except Exception as e: traceback.print_exc(); return False, token_limit, tags, str(e)

def disconnect() -> bool:
    global session, ollama_session, base_url, chat_model, emb_model, headers, auth
    if session: session.close(); session = None
    if ollama_session: ollama_session.close(); ollama_session = None
    base_url = ""
    chat_model = None
    emb_model = None
    headers = {}
    auth = None
    return True

def _list_models() -> List[Dict]:
    """Получение списка доступных моделей"""
    global session, base_url, headers, auth, timeout_value
    try:
        resp = requests.get(f"{base_url}/v1/models", headers=headers, auth=auth, timeout=timeout_value)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except: return []

def ask_model(generation_params: Dict[str, Any]) -> str:
    global session, base_url, chat_model, headers, auth, timeout_value
    if session is None: raise RuntimeError("LM Studio клиент не инициализирован")
    if "prompt" not in generation_params: raise ValueError("'prompt' is required")
    if not chat_model: raise ValueError("Chat model not specified")
    url = f"{base_url}/v1/completions"
    payload = {
        "model": chat_model,
        "prompt": generation_params["prompt"],
        "max_tokens": generation_params.get("max_tokens", 200),
        "temperature": generation_params.get("temperature", 0.7),
        "top_p": generation_params.get("top_p", 0.95),
        "repetition_penalty": generation_params.get("repeat_penalty", 1.1),
        "stream": False}
    data = _request_with_retry("POST", url, json=payload)
    return data["choices"][0]["text"]

def ask_model_chat(generation_params: Dict[str, Any]) -> Dict[str, Any]:
    global session, base_url, chat_model, headers, auth, timeout_value
    if session is None: raise RuntimeError("LM Studio клиент не инициализирован")
    if "messages" not in generation_params: raise ValueError("'messages' is required for chat")
    if not chat_model: raise ValueError("Chat model not specified")
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": generation_params.get("model", chat_model),
        "messages": generation_params["messages"],
        "max_tokens": generation_params.get("max_tokens", 200),
        "temperature": generation_params.get("temperature", 0.7),
        "top_p": generation_params.get("top_p", 0.95),
        "frequency_penalty": generation_params.get("frequency_penalty", 0.0),
        "presence_penalty": generation_params.get("presence_penalty", 0.0),
        "stop": generation_params.get("stop", None),
        "stream": False}
    if "repetition_penalty" in generation_params: payload["repetition_penalty"] = generation_params["repetition_penalty"]
    if "logit_bias" in generation_params: payload["logit_bias"] = generation_params["logit_bias"]
    if "user" in generation_params: payload["user"] = generation_params["user"]
    return _request_with_retry("POST", url, json=payload)

def create_embeddings(text: str) -> List[float]:
    global session, base_url, emb_model, headers, auth, timeout_value
    global use_ollama, ollama_session
    if session is None: raise RuntimeError("LM Studio клиент не инициализирован")
    if use_ollama and ollama_session:
        try: return _ollama_embeddings(text)
        except Exception as e: raise RuntimeError(f"Ollama embeddings error: {e}")
    if not emb_model: raise ValueError("Embedding model not specified")
    url = f"{base_url}/v1/embeddings"
    payload = {"model": emb_model, "input": text}
    data = _request_with_retry("POST", url, json=payload)
    return data["data"][0]["embedding"]