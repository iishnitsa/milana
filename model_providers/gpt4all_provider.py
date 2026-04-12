import requests
import re
import json
import time
import urllib.parse
from cross_gpt import let_log

# === Глобальные переменные состояния ===
session = None
base_url: str = ""
default_chat_model = None
ollama_base_url: str = "http://localhost:11434"
ollama_emb_model: str = "all-minilm:latest"

# Значения по умолчанию
token_limit = 4095
emb_token_limit = 4095
do_chat_construct = True
native_func_call = False

# Новые глобальные переменные для фильтрации think-части
filter_think_enabled: bool = False
filter_start_tag: str = "</think>"
filter_end_tag: str = ""

# Константы для прогрессивной задержки
MAX_RETRIES = 20          # Максимальное количество попыток
MAX_WAIT_TOTAL = 420      # 7 минут в секундах
BASE_BACKOFF = 2.0        # Основание экспоненты
MAX_QUOTA_WAIT = 5 * 60 * 60  # 5 часов — максимальное допустимое ожидание для квотных ошибок

def normalize_url(url, default_port, default_scheme="http"):
    """Добавляет схему и порт по умолчанию, если они отсутствуют."""
    if not url:
        return url
    if "://" not in url:
        url = default_scheme + "://" + url
    parsed = urllib.parse.urlparse(url)
    if not parsed.port:  # порт не указан
        host = parsed.hostname or ""
        new_netloc = f"{host}:{default_port}"
        parsed = parsed._replace(netloc=new_netloc)
    return parsed.geturl()

def normalize_model(model):
    """Добавляет :latest, если в имени модели нет двоеточия."""
    if model and ":" not in model:
        return model + ":latest"
    return model

def find_context_size(model_data, base_url, headers):
    """
    Автоматическое определение лимита контекста для модели Ollama.
    Использует данные из ответа /api/show (model_data).
    """
    print('получение контекста')
    model_info = model_data.get('model_info', {})

    # Сначала проверяем точные ключи для разных архитектур
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
        'phi3.context_length',
    ]
    for key in direct_keys:
        if key in model_info:
            try:
                val = int(model_info[key])
                if val > 0:
                    return val
            except (ValueError, TypeError):
                continue

    # Если точных ключей нет, ищем любой ключ, содержащий 'context_length' (регистронезависимо)
    for key, value in model_info.items():
        if 'context_length' in key.lower() and isinstance(value, (int, float)):
            return int(value)

    parameters = model_data.get('parameters', '')
    if parameters:
        match = re.search(r'num_ctx\s*[=: ]\s*(\d+)', parameters, re.IGNORECASE)
        if match:
            return int(match.group(1))

    model_file = model_data.get('model_file', '')
    if model_file:
        match = re.search(r'num_ctx\s*[=: ]\s*(\d+)', model_file, re.IGNORECASE)
        if match:
            return int(match.group(1))

    possible_paths = [
        ['parameters', 'num_ctx'],
        ['parameters', 'context_length'],
        ['model_info', 'context_length'],
        ['model_info', 'max_seq_len'],
        ['model_info', 'n_ctx'],
        ['model_info', 'gemma3.context_length'],
        ['model_info', 'llama.context_length'],
        ['model_info', 'mistral.context_length'],
        ['details', 'context_length'],
    ]
    for path in possible_paths:
        try:
            value = model_data
            for key in path:
                value = value[key]
            if isinstance(value, (int, float)):
                return int(value)
        except (KeyError, TypeError):
            continue

    # Fallback: ищем числа 2048, 4096, 8192 и т.д. в сериализованных данных (расширенный список)
    context_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 128000, 200000, 262144]
    model_text = json.dumps(model_data)
    found_sizes = sorted([int(num) for num in re.findall(r'\b\d{4,7}\b', model_text)
                          if int(num) in context_sizes], reverse=True)
    if found_sizes:
        return found_sizes[0]

    return 4095

def connect(connection_string, timeout=30):
    """
    Подключение к серверу GPT4All API с поддержкой Ollama для эмбеддингов.
    Формат строки подключения:
    "url=http://localhost:4891; model=mistral-7b-instruct-v0.1.Q4_0.gguf; ollama_url=http://localhost:11434; emb_model=all-minilm:latest"
    """
    global session, base_url, default_chat_model, token_limit, emb_token_limit
    global ollama_base_url, ollama_emb_model, do_chat_construct, native_func_call
    global filter_think_enabled, filter_start_tag, filter_end_tag

    params = {
        "url": "http://localhost:4891",
        "model": "ministral",
        "token_limit": 32768,
        "chat_template": "True",
        "native_func_call": "False",
        "ollama_url": "http://localhost:11434",
        "emb_model": "all-minilm:latest",
        "bos": "",
        "eos": "",
        "sys_start": "",
        "sys_end": "",
        "user_start": "",
        "user_end": "",
        "assist_start": "",
        "assist_end": "",
        "tool_def_start": "",
        "tool_def_end": "",
        "tool_call_start": "",
        "tool_call_end": "",
        "tool_result_start": "",
        "tool_result_end": "",
        "filter_think": "False",
        "filter_start": "</think>",
        "filter_end": "",
    }
    tags = None

    for part in connection_string.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in params:
            params[key] = value

    params["url"] = normalize_url(params["url"], default_port=4891)
    params["ollama_url"] = normalize_url(params["ollama_url"], default_port=11434)
    params["model"] = normalize_model(params["model"])
    params["emb_model"] = normalize_model(params["emb_model"])

    base_url = params["url"].strip('/')
    ollama_base_url = params["ollama_url"].strip('/')
    ollama_emb_model = params["emb_model"]
    do_chat_construct = params["chat_template"].lower().strip() == "true"
    native_func_call = params["native_func_call"].lower().strip() == "true"
    filter_think_enabled = params["filter_think"].lower().strip() == "true"
    filter_start_tag = params["filter_start"]
    filter_end_tag = params["filter_end"]

    tags = {
        "bos": params["bos"],
        "eos": params["eos"],
        "sys_start": params["sys_start"],
        "sys_end": params["sys_end"],
        "user_start": params["user_start"],
        "user_end": params["user_end"],
        "assist_start": params["assist_start"],
        "assist_end": params["assist_end"],
        "tool_def_start": params["tool_def_start"],
        "tool_def_end": params["tool_def_end"],
        "tool_call_start": params["tool_call_start"],
        "tool_call_end": params["tool_call_end"],
        "tool_result_start": params["tool_result_start"],
        "tool_result_end": params["tool_result_end"],
    }

    try:
        # === Подключение к GPT4All ===
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        api_url = f"{base_url}/v1/models"
        response = session.get(api_url, timeout=timeout)
        response.raise_for_status()
        models_data = response.json()
        available_models = [model['id'] for model in models_data.get('data', [])]
        if not available_models:
            return [False, 0, tags, "Не удалось получить список моделей с сервера GPT4All."]

        requested_model = params.get("model")
        if requested_model and requested_model in available_models:
            default_chat_model = requested_model
        else:
            # Не найдена модель GPT4All — ошибка, не подставляем первую
            if requested_model:
                error_msg = f"Запрошенная модель GPT4All '{requested_model}' не найдена. Доступные: {available_models}"
            else:
                error_msg = f"Модель GPT4All не указана. Доступные: {available_models}"
            return [False, 0, tags, error_msg]

        token_limit = int(params["token_limit"])

        # === Подключение к Ollama для эмбеддингов ===
        ollama_models_url = f"{ollama_base_url}/api/tags"
        ollama_response = requests.get(ollama_models_url, timeout=timeout)
        ollama_response.raise_for_status()
        ollama_models_data = ollama_response.json()
        ollama_available_models = [model['name'] for model in ollama_models_data.get('models', [])]
        if not ollama_available_models:
            return [False, 0, tags, "Не удалось получить список моделей с сервера Ollama."]

        # Проверяем, существует ли запрошенная эмбеддинг-модель
        if ollama_emb_model not in ollama_available_models:
            error_msg = f"Запрошенная модель эмбеддингов Ollama '{ollama_emb_model}' не найдена. Доступные: {ollama_available_models}"
            return [False, 0, tags, error_msg]

        # Определяем контекст для эмбеддинг-модели
        try:
            show_url = f"{ollama_base_url}/api/show"
            show_payload = {"name": ollama_emb_model}
            show_response = requests.post(show_url, json=show_payload, timeout=timeout)
            if show_response.status_code == 200:
                model_details = show_response.json()
                emb_token_limit = find_context_size(model_details, ollama_base_url, {})
            else:
                emb_token_limit = 4095
        except Exception as e:
            let_log(f"Ошибка при получении деталей модели эмбеддингов: {e}")
            emb_token_limit = 4095

        return [True, token_limit, tags]

    except requests.exceptions.RequestException as e:
        session = None
        return [False, 0, tags, f"Ошибка подключения: {e}"]
    except Exception as e:
        session = None
        return [False, 0, tags, f"Непредвиденная ошибка: {e}"]

def disconnect() -> bool:
    global session, base_url, default_chat_model
    if session:
        session.close()
        session = None
        base_url = ""
        default_chat_model = None
        return True
    return False

def apply_think_filter(text: str) -> str:
    global filter_think_enabled, filter_start_tag, filter_end_tag
    if not filter_think_enabled:
        return text
    start_pos = text.find(filter_start_tag)
    if start_pos == -1:
        return text
    filtered_text = text[start_pos + len(filter_start_tag):]
    if filter_end_tag and filter_end_tag.strip():
        end_pos = filtered_text.find(filter_end_tag)
        if end_pos != -1:
            filtered_text = filtered_text[:end_pos]
    return filtered_text.strip()

def _request_with_backoff(api_url, json_payload):
    """
    Универсальный метод для запросов к GPT4All с прогрессивной задержкой.
    Обрабатывает 429 с различением квотных ошибок.
    """
    start_time = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        elapsed = time.time() - start_time
        if elapsed > MAX_WAIT_TOTAL:
            raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с)")
        try:
            response = session.post(api_url, json=json_payload)
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                wait_time = None
                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                else:
                    try:
                        err_data = response.json()
                        err_msg = str(err_data).lower()
                        if 'session limit' in err_msg:
                            wait_time = 5 * 60 * 60
                        elif 'weekly limit' in err_msg:
                            wait_time = 7 * 24 * 60 * 60
                        elif 'insufficient_quota' in err_msg or 'balance' in err_msg:
                            raise RuntimeError("balance end")
                    except:
                        pass

                if wait_time is not None:
                    # Если время ожидания превышает MAX_QUOTA_WAIT (5 часов) — сразу исключение
                    if wait_time > MAX_QUOTA_WAIT:
                        raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                    # Если превышает общий лимит MAX_WAIT_TOTAL (7 минут) — тоже исключение
                    if wait_time > MAX_WAIT_TOTAL:
                        raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает общий лимит {MAX_WAIT_TOTAL}с")
                    let_log(f"GPT4All API 429 квота, ожидание {wait_time} с")
                    time.sleep(wait_time)
                    continue

                # Обычный rate limit — экспоненциальный backoff
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining:
                        wait_time = remaining
                    if wait_time < 0.1:
                        wait_time = 0.1
                    let_log(f"HTTP 429 (Rate Limit) на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()

            if response.status_code == 402:
                raise RuntimeError("balance end")

            if response.status_code in (500, 502, 503, 504):
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining:
                        wait_time = remaining
                    if wait_time < 0.1:
                        wait_time = 0.1
                    let_log(f"HTTP {response.status_code} на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()

            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES:
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining:
                    wait_time = remaining
                if wait_time < 0.1:
                    wait_time = 0.1
                let_log(f"Ошибка соединения на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                time.sleep(wait_time)
                continue
            else:
                raise RuntimeError(f"Ошибка соединения после {MAX_RETRIES} попыток: {e}")

        except requests.exceptions.Timeout as e:
            if attempt < MAX_RETRIES:
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining:
                    wait_time = remaining
                if wait_time < 0.1:
                    wait_time = 0.1
                let_log(f"Таймаут на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                time.sleep(wait_time)
                continue
            else:
                raise RuntimeError(f"Таймаут запроса после {MAX_RETRIES} попыток: {e}")

        except RuntimeError as e:
            if "balance end" in str(e):
                raise
            if attempt < MAX_RETRIES:
                time.sleep(BASE_BACKOFF ** attempt)
                continue
            raise

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(BASE_BACKOFF ** attempt)
                continue
            raise RuntimeError(f"Неожиданная ошибка: {e}")

    raise RuntimeError("Превышено максимальное количество попыток")

def ask_model(generation_params):
    if not session or not base_url:
        raise RuntimeError("GPT4All клиент не инициализирован. Сначала вызовите connect().")
    api_url = f"{base_url}/v1/completions"
    try:
        let_log(f"ask_model: Отправка запроса на {api_url}")
        if 'model' not in generation_params and default_chat_model:
            generation_params['model'] = default_chat_model
        data = _request_with_backoff(api_url, generation_params)
        let_log(f"ask_model: Получен ответ: {data}")
        if "choices" not in data or not data["choices"]:
            raise RuntimeError("Некорректный формат ответа - нет choices")
        choice = data["choices"][0]
        result = choice.get("text", "").strip()
        result = apply_think_filter(result)
        let_log(f"ask_model: Результат после фильтрации: '{result}'")
        return result
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ошибка сети: {e}")
    except Exception as e:
        raise RuntimeError(f"Неожиданная ошибка: {e}")

def ask_model_chat(generation_params):
    if not session or not base_url:
        raise RuntimeError("GPT4All клиент не инициализирован. Сначала вызовите connect().")
    api_url = f"{base_url}/v1/chat/completions"
    try:
        let_log(f"ask_model_chat: Отправка запроса на {api_url}")
        if 'model' not in generation_params and default_chat_model:
            generation_params['model'] = default_chat_model
        data = _request_with_backoff(api_url, generation_params)
        let_log(f"ask_model_chat: Получен ответ: {data}")
        if filter_think_enabled and "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                original_content = choice["message"]["content"]
                filtered_content = apply_think_filter(original_content)
                data["choices"][0]["message"]["content"] = filtered_content
                if "text" in data["choices"][0]:
                    data["choices"][0]["text"] = filtered_content
                let_log(f"ask_model_chat: Применил фильтр think-части.")
        return data
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ошибка сети: {e}")
    except Exception as e:
        raise RuntimeError(f"Неожиданная ошибка: {e}")

def create_embeddings(text):
    text = text.strip()
    global ollama_base_url, ollama_emb_model
    if not ollama_base_url or not ollama_emb_model:
        raise RuntimeError("Клиент Ollama для эмбеддингов не инициализирован. Вызовите connect() сначала.")
    api_url = f"{ollama_base_url}/api/embeddings"
    payload = {"model": ollama_emb_model, "prompt": text}
    start_time = time.time()
    attempt = 1
    while True:
        try:
            elapsed = time.time() - start_time
            if elapsed > MAX_WAIT_TOTAL:
                raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с) для эмбеддингов")
            response = requests.post(api_url, json=payload, timeout=30)

            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                wait_time = None
                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                else:
                    try:
                        err_data = response.json()
                        err_msg = str(err_data).lower()
                        if 'session limit' in err_msg:
                            wait_time = 5 * 60 * 60
                        elif 'weekly limit' in err_msg:
                            wait_time = 7 * 24 * 60 * 60
                        elif 'insufficient_quota' in err_msg:
                            raise RuntimeError("balance end")
                    except:
                        pass

                if wait_time is not None:
                    # Если время ожидания превышает MAX_QUOTA_WAIT (5 часов) — сразу исключение
                    if wait_time > MAX_QUOTA_WAIT:
                        raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                    if wait_time > MAX_WAIT_TOTAL:
                        raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает общий лимит {MAX_WAIT_TOTAL}с")
                    let_log(f"create_embeddings: квотная 429, ожидание {wait_time} с")
                    time.sleep(wait_time)
                    attempt += 1
                    continue

                # Обычный rate limit
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining:
                        wait_time = remaining
                    if wait_time < 0.1:
                        wait_time = 0.1
                    let_log(f"create_embeddings: 429 rate limit, повтор через {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue

            if response.status_code == 500:
                error_text = response.text.lower()
                if any(keyword in error_text for keyword in ['context length', 'exceeds context', 'token limit']):
                    raise RuntimeError('ContextOverflowError')

            response.raise_for_status()
            data = response.json()
            embedding = data.get('embedding', [])
            if not embedding:
                raise ValueError("Пустой вектор эмбеддингов в ответе от Ollama")
            return embedding

        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES:
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining:
                    wait_time = remaining
                if wait_time < 0.1:
                    wait_time = 0.1
                let_log(f"create_embeddings: Ошибка соединения, повтор через {wait_time:.2f} с...")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                raise RuntimeError(f"Ошибка соединения после {MAX_RETRIES} попыток: {e}")

        except requests.exceptions.Timeout as e:
            if attempt < MAX_RETRIES:
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining:
                    wait_time = remaining
                if wait_time < 0.1:
                    wait_time = 0.1
                let_log(f"create_embeddings: Таймаут, повтор через {wait_time:.2f} с...")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                raise RuntimeError(f"Таймаут запроса после {MAX_RETRIES} попыток: {e}")

        except RuntimeError as e:
            if "balance end" in str(e) or "ContextOverflowError" in str(e):
                raise
            if attempt < MAX_RETRIES:
                time.sleep(BASE_BACKOFF ** attempt)
                attempt += 1
                continue
            raise

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(BASE_BACKOFF ** attempt)
                attempt += 1
                continue
            raise RuntimeError(f"Ошибка API эмбеддингов Ollama: {str(e)}")