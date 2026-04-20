import requests
import re
import json
import time
import random
import urllib.parse
from typing import Optional, Dict, Any, List, Tuple
from huggingface_hub import InferenceClient
from cross_gpt import let_log

# === Глобальные переменные ===
hf_client = None          # клиент для чата/генерации
hf_emb_client = None      # клиент для эмбеддингов (если не Ollama)
chat_model = None
emb_model = None
timeout = 30

# Ollama fallback
use_ollama = False
ollama_session = None
ollama_url = ""
ollama_emb_model = "all-minilm:latest"

# Лимиты
token_limit = 32768
emb_token_limit = 225

# Теги для форматирования (как в оригинале)
tags = {
    "system": "[INST] <<SYS>>\n",
    "user": "\n<</SYS>>\n[/INST] ",
    "assistant": " ",
    "end": " [/INST]"}

# Константы для повторных попыток
MAX_RETRIES = 10
BASE_BACKOFF = 2.0
MAX_WAIT_TOTAL = 300
MAX_QUOTA_WAIT = 5 * 60 * 60  # 5 часов

# === Вспомогательные функции ===
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
    """Автоматическое определение лимита контекста для Ollama модели."""
    let_log(f"Определение контекста для Ollama модели '{model_name}'")
    model_info = model_data.get('model_info', {})
    direct_keys = [
        'context_length', 'max_position_embeddings', 'max_seq_len', 'n_ctx',
        'llama.context_length', 'gemma2.context_length', 'gemma3.context_length',
        'mistral.context_length', 'qwen2.context_length', 'phi3.context_length']
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
    context_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 128000, 200000, 262144]
    model_text = json.dumps(model_data)
    found_sizes = sorted([int(num) for num in re.findall(r'\b\d{4,7}\b', model_text)
                          if int(num) in context_sizes], reverse=True)
    if found_sizes: return found_sizes[0]
    return 4095

def _call_with_retry(func, *args, **kwargs):
    """Универсальный метод с экспоненциальным backoff и обработкой 429/402"""
    start_time = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try: return func(*args, **kwargs)
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed > MAX_WAIT_TOTAL: raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с)")
            err_str = str(e).lower()
            # Проверка на ошибку контекста
            if any(phrase in err_str for phrase in ["context", "token length", "max_length", "context_length_exceeded"]): raise RuntimeError("ContextOverflowError")
            # Проверка на ошибку баланса (402 или insufficient_quota)
            if "402" in err_str or "insufficient_quota" in err_str or "payment required" in err_str: raise RuntimeError("balance end")
            # Проверка на 429 (rate limit или quota)
            if "429" in err_str:
                retry_after = None
                if hasattr(e, 'response') and e.response is not None: retry_after = e.response.headers.get('Retry-After')
                if retry_after and retry_after.isdigit(): wait_time = int(retry_after)
                elif "session limit" in err_str: wait_time = 5 * 60 * 60
                elif "weekly limit" in err_str: wait_time = 7 * 24 * 60 * 60
                else: wait_time = BASE_BACKOFF ** attempt
                # Если время ожидания превышает MAX_QUOTA_WAIT (5 часов) -> balance end
                if wait_time > MAX_QUOTA_WAIT: raise RuntimeError("balance end")
                # Если ожидание > 5 минут -> balance end (как в старом коде)
                if wait_time > 300: raise RuntimeError("balance end")
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining: wait_time = remaining
                if wait_time < 0.1: wait_time = 0.1
                let_log(f"HF API 429, попытка {attempt}, ожидание {wait_time:.2f} с...")
                time.sleep(wait_time)
                continue
            # Остальные ошибки — пробуем повторить с экспоненциальной задержкой
            if attempt < MAX_RETRIES:
                wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                let_log(f"Ошибка HF API: {e}, повтор через {wait_time:.2f} с...")
                time.sleep(wait_time)
                continue
            else: raise RuntimeError(f"Ошибка после {MAX_RETRIES} попыток: {e}")
    raise RuntimeError("Не удалось выполнить запрос")

def _ollama_embeddings(text: str) -> List[float]:
    """Запрос эмбеддингов через Ollama с обработкой ошибок"""
    global ollama_session, ollama_url, ollama_emb_model, timeout
    url = f"{ollama_url}/api/embeddings"
    payload = {"model": ollama_emb_model, "prompt": text}
    start_time = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            elapsed = time.time() - start_time
            if elapsed > MAX_WAIT_TOTAL: raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с)")
            resp = ollama_session.post(url, json=payload, timeout=timeout)
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
                    if wait_time > MAX_QUOTA_WAIT: raise RuntimeError("balance end")
                    if wait_time > MAX_WAIT_TOTAL: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает общий лимит {MAX_WAIT_TOTAL}с")
                    let_log(f"Ollama эмбеддинги: квотная 429, ожидание {wait_time} с")
                    time.sleep(wait_time)
                    continue
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining: wait_time = remaining
                    if wait_time < 0.1: wait_time = 0.1
                    let_log(f"Ollama эмбеддинги: 429 rate limit, повтор через {wait_time} с")
                    time.sleep(wait_time)
                    continue
            resp.raise_for_status()
            return resp.json()['embedding']
        except requests.HTTPError as e:
            msg = resp.text.lower()
            if any(phrase in msg for phrase in ["exceeds context", "context length", "token limit"]): raise RuntimeError("ContextOverflowError")
            if attempt == MAX_RETRIES: raise
            wait_time = BASE_BACKOFF ** attempt
            time.sleep(min(wait_time, 60))
        except requests.exceptions.ConnectionError as e:
            if attempt == MAX_RETRIES: raise RuntimeError(f"Ошибка соединения: {e}")
            time.sleep(BASE_BACKOFF ** attempt)
    raise RuntimeError("Не удалось получить эмбеддинги от Ollama")

# === Основные функции провайдера ===
def connect(connection_string: str, timeout: int = 30, _decrypted_token: str = None) -> Tuple[bool, int, Dict[str, Optional[str]]]:
    global hf_client, hf_emb_client, chat_model, emb_model
    global use_ollama, ollama_session, ollama_url, ollama_emb_model
    global token_limit, emb_token_limit, tags
    params = {
        'token': None,
        'chat': None,
        'emb': None,
        'timeout': str(timeout),
        'ollama': 'false',
        'ollama_url': 'http://localhost:11434',
        'ollama_emb_model': 'all-minilm:latest'}
    for part in connection_string.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            key, value = key.strip().lower(), value.strip()
            if key in params and value: params[key] = value
    # Нормализация URL и моделей
    params["ollama_url"] = normalize_url(params["ollama_url"], default_port=11434)
    params["ollama_emb_model"] = normalize_model(params["ollama_emb_model"])
    # API токен
    api_token = _decrypted_token if _decrypted_token is not None else params['token']
    if not api_token: return False, token_limit, tags, "No API token provided"
    timeout = int(params['timeout'])
    globals()['timeout'] = timeout
    # Настройка тегов на основе имени модели (если указана)
    if params['chat']:
        model_lower = params['chat'].lower()
        if 'mistral' in model_lower: tags.update({"system": "<s>[INST] ", "user": " [/INST]", "assistant": " ", "end": "</s>"})
        elif 'gemma' in model_lower: tags.update({"system": "<start_of_turn>model\n", "user": "<start_of_turn>user\n", "assistant": "<start_of_turn>model\n", "end": "<end_of_turn>\n"})
    # Проверка существования модели через HEAD-запрос к HF API
    if params['chat']:
        try:
            api_url = f"https://huggingface.co/api/models/{params['chat']}"
            resp = requests.head(api_url, timeout=timeout)
            if resp.status_code != 200: return False, token_limit, tags, f"Модель '{params['chat']}' не найдена на Hugging Face"
        except Exception as e: return False, token_limit, tags, f"Ошибка проверки модели: {e}"
    # Инициализация основного клиента для чата
    chat_model = params['chat']
    hf_client = InferenceClient(model=chat_model, token=api_token, timeout=timeout) if chat_model else None
    # Инициализация клиента для эмбеддингов (если не используется Ollama)
    use_ollama = (params['ollama'].lower() == 'true')
    ollama_url = params['ollama_url'].rstrip('/')
    ollama_emb_model = params['ollama_emb_model']
    if not use_ollama and params['emb']: emb_model = params['emb']; hf_emb_client = InferenceClient(model=emb_model, token=api_token, timeout=timeout)
    else: emb_model = None; hf_emb_client = None
    # Настройка Ollama для эмбеддингов (если включён)
    if use_ollama:
        try:
            ollama_session = requests.Session()
            ollama_session.headers.update({"Content-Type": "application/json"})
            tags_url = f"{ollama_url}/api/tags"
            resp = ollama_session.get(tags_url, timeout=timeout)
            resp.raise_for_status()
            models_data = resp.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            if ollama_emb_model not in available_models:
                let_log(f"Ollama модель '{ollama_emb_model}' не найдена. Доступные: {available_models}")
                embed_models = [m for m in available_models if 'embed' in m.lower()]
                if embed_models: ollama_emb_model = embed_models[0]; let_log(f"Выбрана модель для эмбеддингов: {ollama_emb_model}")
                else: let_log("Не найдено подходящей embed-модели, эмбеддинги через Ollama будут недоступны"); use_ollama = False
            if use_ollama:
                show_payload = {"name": ollama_emb_model}
                show_resp = ollama_session.post(f"{ollama_url}/api/show", json=show_payload, timeout=timeout)
                if show_resp.status_code == 200: model_details = show_resp.json(); emb_token_limit = find_ollama_context_size(model_details, ollama_emb_model)
                else: emb_token_limit = 4095
        except Exception as e: let_log(f"Ошибка при проверке Ollama: {e}"); use_ollama = False; emb_token_limit = 4096
    # Определение token_limit для чат-модели (из конфига HF)
    if params['chat']:
        try:
            api_url = f"https://huggingface.co/api/models/{params['chat']}"
            resp = requests.get(api_url, timeout=timeout)
            if resp.status_code == 200:
                model_info = resp.json()
                config = model_info.get('config', {})
                token_limit = config.get('max_position_embeddings', 32768)
                let_log(f"Модель '{params['chat']}': контекст {token_limit} (из max_position_embeddings)")
            else: token_limit = 32768; let_log(f"Не удалось получить конфиг модели '{params['chat']}', контекст по умолчанию {token_limit}")
        except Exception as e: let_log(f"Не удалось получить контекст модели HF: {e}"); token_limit = 32768
    return True, token_limit, tags

def disconnect() -> bool:
    global hf_client, hf_emb_client, ollama_session, chat_model, emb_model
    global use_ollama, ollama_url, ollama_emb_model
    hf_client = None
    hf_emb_client = None
    if ollama_session: ollama_session.close(); ollama_session = None
    chat_model = None
    emb_model = None
    use_ollama = False
    ollama_url = ""
    ollama_emb_model = "all-minilm:latest"
    return True

def ask_model(generation_params: Dict[str, Any]) -> str:
    global hf_client, chat_model, timeout
    if hf_client is None or chat_model is None: raise RuntimeError("Hugging Face client not initialized")
    if "prompt" not in generation_params: raise ValueError("'prompt' is required")
    prompt = generation_params["prompt"]
    gen_kwargs = {
        "max_new_tokens": generation_params.get("max_tokens", 200),
        "temperature": generation_params.get("temperature", 0.7),
        "top_p": generation_params.get("top_p", 0.9),
        "repetition_penalty": generation_params.get("repeat_penalty", 1.1),
        "do_sample": True}
    def _generate(): return hf_client.text_generation(prompt, **gen_kwargs).strip()
    return _call_with_retry(_generate)

def ask_model_chat(generation_params: Dict[str, Any]) -> Dict[str, Any]:
    global hf_client, chat_model, timeout
    if hf_client is None or chat_model is None: raise RuntimeError("Hugging Face client not initialized")
    if "messages" not in generation_params: raise ValueError("'messages' is required for chat")
    messages = generation_params["messages"]
    model = generation_params.get("model", chat_model)
    chat_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": generation_params.get("max_tokens", 500),
        "temperature": generation_params.get("temperature", 0.7),
        "top_p": generation_params.get("top_p", 0.9),
        "stop": generation_params.get("stop", None),}
    if "frequency_penalty" in generation_params: chat_kwargs["frequency_penalty"] = generation_params["frequency_penalty"]
    if "presence_penalty" in generation_params: chat_kwargs["presence_penalty"] = generation_params["presence_penalty"]
    if "seed" in generation_params: chat_kwargs["seed"] = generation_params["seed"]
    def _chat():
        response = hf_client.chat.completions.create(**chat_kwargs)
        return {
            "id": response.id,
            "choices": [
                {
                    "message": {
                        "content": choice.message.content,
                        "role": choice.message.role
                    },
                    "finish_reason": choice.finish_reason,
                    "index": choice.index
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            } if response.usage else {},
            "model": response.model,
            "created": response.created}
    return _call_with_retry(_chat)

def create_embeddings(text: str) -> List[float]:
    global use_ollama, ollama_session, hf_emb_client, emb_model
    text = text.strip()
    if use_ollama and ollama_session:
        try: return _ollama_embeddings(text)
        except Exception as e: raise RuntimeError(f"Ollama embeddings error: {e}")
    if hf_emb_client is None or emb_model is None: raise RuntimeError("No embeddings backend available")
    def _embed():
        output = hf_emb_client.feature_extraction(text)
        if isinstance(output, list):
            if output and isinstance(output[0], list): return [float(x) for x in output[0]]
            return [float(x) for x in output]
        return output.tolist()
    return _call_with_retry(_embed)
