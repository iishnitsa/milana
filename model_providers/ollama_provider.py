import requests
import re
import json
import time
from cross_gpt import let_log

# === Глобальные переменные состояния ===
session = None
base_url = ""
default_chat_model = None
emb_model = ""
model_template_info = {}

# Значения по умолчанию
token_limit = 4095
emb_token_limit = 4095
do_chat_construct = True
native_func_call = False
tags = {}

# Константы для прогрессивной задержки
MAX_RETRIES = 20          # Максимальное количество попыток
MAX_WAIT_TOTAL = 420      # 7 минут в секундах
BASE_BACKOFF = 2.0        # Основание экспоненты

def find_context_size(model_data, base_url, headers):
    """
    Автоматическое определение лимита контекста для модели Ollama.
    """
    try:
        config_url = f"{base_url}/api/config"
        config_resp = requests.get(config_url, headers=headers, timeout=30)
        if config_resp.status_code == 200:
            server_config = config_resp.json()
            server_limit = server_config.get('max_context_length')
            if server_limit is not None: return int(server_limit)
    except Exception: pass
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
            for key in path: value = value[key]
            if isinstance(value, (int, float)): return int(value)
        except (KeyError, TypeError): continue
    if isinstance(model_data.get('parameters'), str):
        param_str = model_data['parameters']
        for line in param_str.split('\n'):
            if any(kw in line.lower() for kw in ['num_ctx', 'context', 'n_ctx', 'max_tokens']):
                numbers = re.findall(r'\b\d{3,5}\b', line)
                if numbers: return int(numbers[-1])
    context_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 128000, 200000]
    model_text = json.dumps(model_data)
    found_sizes = sorted([int(num) for num in re.findall(r'\b\d{4,6}\b', model_text)
                          if int(num) in context_sizes], reverse=True)
    if found_sizes: return found_sizes[0]
    return 4095

def _parse_template_info(template_info):
    """
    Извлекает теги из шаблона модели Ollama.
    Возвращает словарь с тегами в формате совместимом с GPT4All.
    """
    template = template_info.get("template", "")
    system_msg = template_info.get("system", "")
    # Начинаем с пустых значений
    parsed_tags = {
        "bos": "", "eos": "",
        "sys_start": "", "sys_end": "",
        "user_start": "", "user_end": "",
        "assist_start": "", "assist_end": "",
        "tool_def_start": "", "tool_def_end": "",
        "tool_call_start": "", "tool_call_end": "",
        "tool_result_start": "", "tool_result_end": "",
    }
    # Пытаемся извлечь системный тег
    if system_msg:
        # Ищем паттерны типа "<|im_start|>system" или "system:"
        system_patterns = [r'(<\|im_start\|>system)', r'(system:)', r'(\[system\])', r'(<system>)']
        for pattern in system_patterns:
            match = re.search(pattern, system_msg, re.IGNORECASE)
            if match:
                parsed_tags["sys_start"] = match.group(1)
                # Ищем соответствующий закрывающий тег
                end_pattern = re.sub(r'system', r'end', pattern, flags=re.IGNORECASE)
                end_match = re.search(end_pattern, system_msg, re.IGNORECASE)
                if end_match: parsed_tags["sys_end"] = end_match.group(1)
                break
    # Анализируем шаблон для поиска тегов
    if template:
        # Ищем специальные теги
        special_tags = {
            "bos": [r'<s>', r'<\|start\|>', r'\[start\]', r'bos'],
            "eos": [r'</s>', r'<\|end\|>', r'\[end\]', r'eos'],
            "sys_start": [r'<\|im_start\|>system', r'\[system\]', r'<system>'],
            "sys_end": [r'<\|im_end\|>', r'\[/system\]', r'</system>'],
            "user_start": [r'<\|im_start\|>user', r'\[user\]', r'<user>', r'user:'],
            "user_end": [r'<\|im_end\|>', r'\[/user\]', r'</user>'],
            "assist_start": [r'<\|im_start\|>assistant', r'\[assistant\]', r'<assistant>', r'assistant:'],
            "assist_end": [r'<\|im_end\|>', r'\[/assistant\]', r'</assistant>'],
        }
        for tag_name, patterns in special_tags.items():
            for pattern in patterns:
                matches = re.findall(pattern, template, re.IGNORECASE)
                if matches:
                    parsed_tags[tag_name] = matches[0]
                    break
    return parsed_tags

def connect(connection_string, timeout=30):
    """
    Подключение к серверу Ollama API.
    Формат строки подключения:
    "url=http://localhost:11434; model=mistral:latest; emb_model=all-minilm:latest"
    """
    global session, base_url, default_chat_model, token_limit, emb_token_limit
    global emb_model, do_chat_construct, native_func_call, tags, model_template_info
    # Параметры по умолчанию (только необходимые)
    params = {
        "url": "http://localhost:11434",
        "model": "ministral-3:latest",
        "emb_model": "all-minilm:latest",
        "chat_template": "True",
        "native_func_call": "False",
    }
    # --- Разбор строки подключения ---
    for part in connection_string.split(";"):
        part = part.strip()
        if not part or "=" not in part: continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in params: params[key] = value
    base_url = params["url"].strip('/')
    emb_model = params["emb_model"]
    do_chat_construct = params["chat_template"].lower().strip() == "true"
    native_func_call = params["native_func_call"].lower().strip() == "true"
    try:
        # === Подключение к Ollama ===
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        api_url = f"{base_url}/api/tags"
        response = session.get(api_url, timeout=timeout)
        response.raise_for_status()
        models_data = response.json()
        available_models = [model['name'] for model in models_data.get('models', [])]
        if not available_models: return [False, 0, tags, "Не удалось получить список моделей с сервера Ollama."]
        # Устанавливаем модель для чата
        requested_model = params.get("model")
        if requested_model and requested_model in available_models: default_chat_model = requested_model
        else: default_chat_model = available_models[0]
        # Получаем информацию о модели для извлечения тегов и контекста
        try:
            show_url = f"{base_url}/api/show"
            show_payload = {"name": default_chat_model}
            show_response = session.post(show_url, json=show_payload, timeout=timeout)
            if show_response.status_code == 200:
                model_details = show_response.json()
                model_template_info = model_details
                # Определяем лимит контекста
                token_limit = find_context_size(model_details, base_url, {})
                # Извлекаем теги из шаблона модели
                tags = _parse_template_info(model_details)
        except Exception as e:
            let_log(f"Ошибка при получении деталей модели: {e}")
            tags = {
                "bos": "", "eos": "",
                "sys_start": "", "sys_end": "",
                "user_start": "", "user_end": "",
                "assist_start": "", "assist_end": "",
                "tool_def_start": "", "tool_def_end": "",
                "tool_call_start": "", "tool_call_end": "",
                "tool_result_start": "", "tool_result_end": "",
            }
        # Проверяем и устанавливаем модель для эмбеддингов
        if emb_model not in available_models:
            let_log(f"Модель для эмбеддингов '{emb_model}' не найдена. Доступные модели: {available_models}")
            # Пробуем найти любую модель с 'embed' в названии
            embed_models = [m for m in available_models if 'embed' in m.lower()]
            if embed_models:
                emb_model = embed_models[0]
                let_log(f"Выбрана модель для эмбеддингов: {emb_model}")
            else:
                # Если нет моделей для эмбеддингов, используем чат-модель
                emb_model = default_chat_model
                let_log(f"Модель для эмбеддингов не найдена. Используем чат-модель: {emb_model}")
        # Автоматическое определение лимита токенов для эмбеддингов
        try:
            if emb_model != default_chat_model:  # Если модели разные, получаем детали для эмбеддинг-модели
                show_url = f"{base_url}/api/show"
                show_payload = {"name": emb_model}
                show_response = session.post(show_url, json=show_payload, timeout=timeout)
                if show_response.status_code == 200:
                    model_details = show_response.json()
                    emb_token_limit = find_context_size(model_details, base_url, {})
                else: emb_token_limit = 4095
            else: emb_token_limit = token_limit
        except Exception as e:
            let_log(f"Не удалось определить лимит токенов для эмбеддингов: {e}")
            emb_token_limit = 4095
        return [True, token_limit, tags]
    except requests.exceptions.RequestException as e:
        session = None
        return [False, 0, tags, f"Ошибка подключения: {e}"]
    except Exception as e:
        session = None
        return [False, 0, tags, f"Непредвиденная ошибка: {e}"]

def disconnect() -> bool:
    """Закрыть HTTP сессию"""
    global session, base_url, default_chat_model, model_template_info
    if session:
        session.close()
        session = None
        base_url = ""
        default_chat_model = None
        model_template_info = {}
        return True
    return False

def _request_with_backoff(api_url, json_payload):
    """
    Универсальный метод для запросов к Ollama с прогрессивной задержкой.
    Экспоненциальный backoff, ограничение по времени 7 минут.
    """
    start_time = time.time()
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        # Проверка общего времени выполнения
        elapsed = time.time() - start_time
        if elapsed > MAX_WAIT_TOTAL: raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с)")
        try:
            response = session.post(api_url, json=json_payload)
            # Обработка HTTP ошибок с повторными попытками
            if response.status_code in (429, 500, 502, 503, 504):
                # 429 – слишком много запросов, 5xx – временные ошибки сервера
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    # Не превышаем оставшееся время
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining: wait_time = remaining
                    if wait_time < 0.1: wait_time = 0.1
                    let_log(f"HTTP {response.status_code} на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue
                else: response.raise_for_status()
            # Для других статусов сразу вызываем исключение, если код не 2xx
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            # Ошибка соединения – повторяем с экспоненциальной задержкой
            if attempt < MAX_RETRIES:
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining: wait_time = remaining
                if wait_time < 0.1: wait_time = 0.1
                let_log(f"Ошибка соединения на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                time.sleep(wait_time)
                continue
            else: raise RuntimeError(f"Ошибка соединения после {MAX_RETRIES} попыток: {e}")
        except requests.exceptions.Timeout as e:
            # Таймаут запроса – повторяем
            if attempt < MAX_RETRIES:
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining: wait_time = remaining
                if wait_time < 0.1: wait_time = 0.1
                let_log(f"Таймаут на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                time.sleep(wait_time)
                continue
            else: raise RuntimeError(f"Таймаут запроса после {MAX_RETRIES} попыток: {e}")
        except requests.exceptions.RequestException as e:
            # Другие ошибки сети
            if attempt < MAX_RETRIES:
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining: wait_time = remaining
                if wait_time < 0.1: wait_time = 0.1
                let_log(f"Сетевая ошибка на попытке {attempt}: {e}. Ожидание {wait_time:.2f} с...")
                time.sleep(wait_time)
                continue
            else: raise RuntimeError(f"Сетевая ошибка после {MAX_RETRIES} попыток: {e}")
    raise RuntimeError("Превышено максимальное количество попыток")

def ask_model(generation_params):
    if not session or not base_url or not default_chat_model: raise RuntimeError("Ollama клиент не инициализирован. Сначала вызовите connect().")
    api_url = f"{base_url}/api/generate"
    try:
        let_log(f"ask_model: Отправка запроса на {api_url}")
        ollama_params = {
            "model": default_chat_model,
            "prompt": generation_params.get("prompt", ""),
            "stream": False,
            "options": {}}
        param_mapping = {
            "max_tokens": "num_predict",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "repeat_penalty": "repeat_penalty",
            "stop": "stop"}
        for param, value in generation_params.items():
            if param == "prompt": continue
            if param in param_mapping: ollama_params["options"][param_mapping[param]] = value
            else: ollama_params["options"][param] = value
        data = _request_with_backoff(api_url, ollama_params)
        let_log(f"ask_model: Получен ответ, длина: {len(str(data))} символов")
        result = data.get("response", "").strip()
        let_log(f"ask_model: Результат: '{result[:100]}...'")
        return result
    except requests.exceptions.RequestException as e: raise RuntimeError(f"Ошибка сети: {e}")
    except Exception as e: raise RuntimeError(f"Неожиданная ошибка: {e}")

def ask_model_chat(generation_params):
    if not session or not base_url or not default_chat_model: raise RuntimeError("Ollama клиент не инициализирован. Сначала вызовите connect().")
    api_url = f"{base_url}/api/chat"
    try:
        let_log(f"ask_model_chat: Отправка запроса на {api_url}")
        # Подготовка параметров для Ollama Chat API
        ollama_params = {
            "model": default_chat_model,
            "messages": generation_params.get("messages", []),
            "stream": False,
            "options": {}}
        # Маппинг параметров
        param_mapping = {
            "max_tokens": "num_predict",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "repeat_penalty": "repeat_penalty",
            "stop": "stop"}
        for param, value in generation_params.items():
            if param in ["messages", "model"]: continue
            if param in param_mapping: ollama_params["options"][param_mapping[param]] = value
            else: ollama_params["options"][param] = value
        data = _request_with_backoff(api_url, ollama_params)
        let_log(f"ask_model_chat: Получен ответ, длина: {len(str(data))} символов")
        return data
    except requests.exceptions.RequestException as e: raise RuntimeError(f"Ошибка сети: {e}")
    except Exception as e: raise RuntimeError(f"Неожиданная ошибка: {e}")

def create_embeddings(text):
    global base_url, emb_model, session
    if not base_url or not emb_model or not session: raise RuntimeError("Клиент Ollama для эмбеддингов не инициализирован. Вызовите connect() сначала.")
    api_url = f"{base_url}/api/embeddings"
    text = text.strip()
    payload = {"model": emb_model, "prompt": text}
    try:
        let_log(f"create_embeddings: Отправка запроса с моделью {emb_model}")
        let_log(f"create_embeddings: Текст: {text[:200]}...")
        start_time = time.time()
        attempt = 1
        while True:
            try:
                elapsed = time.time() - start_time
                if elapsed > MAX_WAIT_TOTAL: raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с) для эмбеддингов")
                response = session.post(api_url, json=payload, timeout=30)
                # Обработка ошибок контекста
                if response.status_code == 500:
                    error_text = response.text.lower()
                    context_error_keywords = [
                        'the input length exceeds the context length',
                        'input length exceeds',
                        'context length',
                        'exceeds context',
                        'token limit exceeded',
                        'exceeds the context',
                        'exceeds context length']
                    if any(keyword in error_text for keyword in context_error_keywords): raise RuntimeError('ContextOverflowError')
                if response.status_code in (429, 500, 502, 503, 504):
                    # Временные ошибки - повторяем с экспоненциальной задержкой
                    if attempt < MAX_RETRIES:
                        wait_time = BASE_BACKOFF ** attempt
                        remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                        if wait_time > remaining: wait_time = remaining
                        if wait_time < 0.1: wait_time = 0.1
                        let_log(f"create_embeddings: HTTP {response.status_code}, повтор через {wait_time:.2f} с...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                response.raise_for_status()
                data = response.json()
                embedding = data.get('embedding', [])
                if not embedding: raise ValueError("Пустой вектор эмбеддингов в ответе от Ollama")
                let_log(f"create_embeddings: Получен вектор размером {len(embedding)}")
                return embedding
            except requests.exceptions.ConnectionError as e:
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining: wait_time = remaining
                    if wait_time < 0.1: wait_time = 0.1
                    let_log(f"create_embeddings: Ошибка соединения, повтор через {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                else: raise RuntimeError(f"Ошибка соединения после {MAX_RETRIES} попыток: {e}")
            except requests.exceptions.Timeout as e:
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining: wait_time = remaining
                    if wait_time < 0.1: wait_time = 0.1
                    let_log(f"create_embeddings: Таймаут, повтор через {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                else: raise RuntimeError(f"Таймаут запроса после {MAX_RETRIES} попыток: {e}")
    except requests.exceptions.RequestException as e: raise RuntimeError(f"Ошибка API эмбеддингов Ollama: {str(e)}")
    except (KeyError, IndexError, ValueError) as e: raise RuntimeError(f"Некорректный формат ответа от API эмбеддингов Ollama: {e}")