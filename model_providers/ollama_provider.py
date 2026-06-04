import requests
import re
import json
import time
import urllib.parse
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

# Флаг автоматического пропуска num_predict (костыль для 400 ошибок)
_skip_num_predict = False

# Константы для прогрессивной задержки
MAX_RETRIES = 20          # Максимальное количество попыток
MAX_WAIT_TOTAL = 420      # 7 минут в секундах
BASE_BACKOFF = 2.0        # Основание экспоненты

# === НОВЫЕ ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ===
llm_cpu = False
emb_cpu = True
offload_on_boot = False
is_thinking = False
default_num_ctx = None     # заданный в строке подключения контекст (если есть)
filter_think_tag = False   # флаг фильтрации тегов <think>
_last_think_content = None # последнее извлечённое содержимое think

# Прошлые состояния для детекта изменений
_last_llm_num_ctx = None
_last_think_value = None
_last_llm_cpu = None

def normalize_url(url, default_port, default_scheme="http"):
    """Добавляет схему и порт по умолчанию, если они отсутствуют."""
    if not url: return url
    if "://" not in url: url = default_scheme + "://" + url
    parsed = urllib.parse.urlparse(url)
    if not parsed.port:  # порт не указан
        # пересобираем netloc с портом
        host = parsed.hostname or ""
        new_netloc = f"{host}:{default_port}"
        # если есть username:password, сохраняем (но у нас их нет)
        parsed = parsed._replace(netloc=new_netloc)
    return parsed.geturl()

def normalize_model(model):
    """Добавляет :latest, если в имени модели нет двоеточия."""
    if model and ":" not in model: return model + ":latest"
    return model

def find_context_size(model_data, base_url, headers):
    """
    Автоматическое определение лимита контекста для модели Ollama.
    Использует данные из ответа /api/show (model_data).
    """
    print('получение контекста')
    # Поиск в model_info (наиболее надёжное место)
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
        'phi3.context_length',]
    for key in direct_keys:
        if key in model_info:
            try:
                val = int(model_info[key])
                if val > 0: return val
            except (ValueError, TypeError): continue
    # Если точных ключей нет, ищем любой ключ, содержащий 'context_length' (регистронезависимо)
    for key, value in model_info.items():
        if 'context_length' in key.lower() and isinstance(value, (int, float)): return int(value)
    # Поиск в parameters (например, "num_ctx 4096" или "num_ctx=4096")
    parameters = model_data.get('parameters', '')
    if parameters:
        # Ищем num_ctx с пробелом или знаком равенства
        match = re.search(r'num_ctx\s*[=: ]\s*(\d+)', parameters, re.IGNORECASE)
        if match: return int(match.group(1))
        # Альтернативный поиск: просто слово num_ctx и число рядом
        match = re.search(r'num_ctx\s+(\d+)', parameters, re.IGNORECASE)
        if match: return int(match.group(1))
    # Поиск в model_file (редко, но может быть)
    model_file = model_data.get('model_file', '')
    if model_file:
        match = re.search(r'num_ctx\s*[=: ]\s*(\d+)', model_file, re.IGNORECASE)
        if match: return int(match.group(1))
    # Попытка найти в любом текстовом представлении модели (запасной вариант)
    possible_paths = [
        ['parameters', 'num_ctx'],
        ['parameters', 'context_length'],
        ['model_info', 'context_length'],
        ['model_info', 'max_seq_len'],
        ['model_info', 'n_ctx'],
        ['model_info', 'gemma3.context_length'],
        ['model_info', 'llama.context_length'],
        ['model_info', 'mistral.context_length'],
        ['details', 'context_length']]
    for path in possible_paths:
        try:
            value = model_data
            for key in path: value = value[key]
            if isinstance(value, (int, float)): return int(value)
        except (KeyError, TypeError): continue
    # Fallback: ищем числа 2048, 4096, 8192 и т.д. в сериализованных данных
    context_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 128000, 200000, 262144]
    model_text = json.dumps(model_data)
    found_sizes = sorted([int(num) for num in re.findall(r'\b\d{4,7}\b', model_text)
                          if int(num) in context_sizes], reverse=True)
    if found_sizes: return found_sizes[0]
    # Значение по умолчанию
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
        "tool_result_start": "", "tool_result_end": "",}
    # Пытаемся извлечь BOS/EOS из model_info, если доступно
    model_info = template_info.get('model_info', {})
    if 'tokenizer.ggml.bos_token_id' in model_info:
        # Преобразуем ID в строковый токен, если возможно (упрощённо)
        # В реальности нужно было бы обращаться к токенизатору, но для совместимости оставляем как есть
        bos_id = model_info.get('tokenizer.ggml.bos_token_id')
        if bos_id is not None: parsed_tags["bos"] = f"<0x{bos_id:02X}>"  # заглушка
    if 'tokenizer.ggml.eos_token_id' in model_info:
        eos_id = model_info.get('tokenizer.ggml.eos_token_id')
        if eos_id is not None: parsed_tags["eos"] = f"<0x{eos_id:02X}>"
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
                if matches: parsed_tags[tag_name] = matches[0]; break
    return parsed_tags

def connect(connection_string, timeout=30):
    """
    Подключение к серверу Ollama API.
    Формат строки подключения:
    "url=http://localhost:11434; model=mistral:latest; emb_model=all-minilm:latest;
     llm_cpu=false; emb_cpu=true; offload_on_boot=false; is_thinking=false; num_ctx=...; filter_think_tag=false"
    """
    global session, base_url, default_chat_model, token_limit, emb_token_limit
    global emb_model, do_chat_construct, native_func_call, tags, model_template_info
    global llm_cpu, emb_cpu, offload_on_boot, is_thinking, default_num_ctx
    global _last_llm_num_ctx, _last_think_value, _last_llm_cpu
    global filter_think_tag

    # Параметры по умолчанию (только необходимые)
    params = {
        "url": "http://localhost:11434",
        "model": "qwen3.5:2b",
        "emb_model": "all-minilm:latest",
        "chat_template": "True",
        "native_func_call": "False",
        "num_ctx": "",
        "llm_cpu": "false",
        "emb_cpu": "true",
        "offload_on_boot": "false",
        "is_thinking": "false",
        "filter_think_tag": "false",
    }
    # --- Разбор строки подключения ---
    for part in connection_string.split(";"):
        part = part.strip()
        if not part or "=" not in part: continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in params: params[key] = value
    # === НОРМАЛИЗАЦИЯ ===
    params["url"] = normalize_url(params["url"], default_port=11434)
    params["model"] = normalize_model(params["model"])
    params["emb_model"] = normalize_model(params["emb_model"])
    base_url = params["url"].strip('/')
    emb_model = params["emb_model"]
    do_chat_construct = params["chat_template"].lower().strip() == "true"
    native_func_call = params["native_func_call"].lower().strip() == "true"

    # Новые параметры
    llm_cpu = params["llm_cpu"].lower().strip() == "true"
    emb_cpu = params["emb_cpu"].lower().strip() == "true"
    offload_on_boot = params["offload_on_boot"].lower().strip() == "true"
    is_thinking = params["is_thinking"].lower().strip() == "true"
    filter_think_tag = params["filter_think_tag"].lower().strip() == "true"

    # Обработка num_ctx
    num_ctx_str = params.get("num_ctx", "").strip()
    if num_ctx_str.isdigit():
        default_num_ctx = int(num_ctx_str)
    else:
        default_num_ctx = None

    print('подключение')
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
        # ИЗМЕНЕНИЕ: если запрошенная модель не найдена, возвращаем ошибку, а не берём первую попавшуюся
        if requested_model and requested_model in available_models: default_chat_model = requested_model
        else:
            # Собираем понятное сообщение об ошибке
            if requested_model: error_msg = f"Запрошенная модель '{requested_model}' не найдена. Доступные модели: {available_models}"
            else: error_msg = f"Модель не указана в строке подключения. Доступные модели: {available_models}"
            return [False, 0, tags, error_msg]
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
                "tool_result_start": "", "tool_result_end": "",}
        # Проверяем и устанавливаем модель для эмбеддингов
        if emb_model not in available_models:
            let_log(f"Модель для эмбеддингов '{emb_model}' не найдена. Доступные модели: {available_models}")
            # Пробуем найти любую модель с 'embed' в названии
            embed_models = [m for m in available_models if 'embed' in m.lower()]
            if embed_models: emb_model = embed_models[0]; let_log(f"Выбрана модель для эмбеддингов: {emb_model}")
            else: emb_model = default_chat_model; let_log(f"Модель для эмбеддингов не найдена. Используем чат-модель: {emb_model}")
        # Автоматическое определение лимита токенов для эмбеддингов
        try:
            if emb_model != default_chat_model:  # Если модели разные, получаем детали для эмбеддинг-модели
                show_url = f"{base_url}/api/show"
                show_payload = {"name": emb_model}
                show_response = session.post(show_url, json=show_payload, timeout=timeout)
                if show_response.status_code == 200: model_details = show_response.json(); emb_token_limit = find_context_size(model_details, base_url, {})
                else: emb_token_limit = 4095
            else: emb_token_limit = token_limit
        except Exception as e: let_log(f"Не удалось определить лимит токенов для эмбеддингов: {e}"); emb_token_limit = 4095

        # === ВЫГРУЗКА МОДЕЛИ ПРИ ПОДКЛЮЧЕНИИ (если offload_on_boot) ===
        if offload_on_boot and session:
            try:
                session.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": default_chat_model,
                        "prompt": "",
                        "keep_alive": 0,
                        "stream": False
                    },
                    timeout=timeout
                )
                let_log("Модель выгружена (offload_on_boot=true).")
            except Exception as e:
                let_log(f"Предупреждение: не удалось выгрузить модель при старте: {e}")

        # Сохраняем начальные состояния
        _last_llm_num_ctx = default_num_ctx
        _last_think_value = is_thinking
        _last_llm_cpu = llm_cpu

        return [True, token_limit, tags]
    except requests.exceptions.RequestException as e: session = None; return [False, 0, tags, f"Ошибка подключения: {e}"]
    except Exception as e: session = None; return [False, 0, tags, f"Непредвиденная ошибка: {e}"]

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

def _check_and_reload_model():
    """
    Сравнивает текущие настройки с сохранёнными.
    Если что-то изменилось — выгружает модель (keep_alive=0),
    чтобы следующий запрос подхватил новые параметры.
    """
    global _last_llm_num_ctx, _last_think_value, _last_llm_cpu
    need_reload = False

    if _last_llm_num_ctx != default_num_ctx:
        need_reload = True
    if is_thinking and _last_think_value != is_thinking:
        need_reload = True
    if _last_llm_cpu != llm_cpu:
        need_reload = True

    if need_reload:
        let_log("Обнаружены изменения настроек. Выгружаю модель...")
        try:
            session.post(
                f"{base_url}/api/generate",
                json={
                    "model": default_chat_model,
                    "prompt": "",
                    "keep_alive": 0,
                    "stream": False
                },
                timeout=10
            )
        except Exception as e:
            let_log(f"Ошибка при выгрузке модели: {e}")
        finally:
            # Обновляем сохранённые состояния в любом случае
            _last_llm_num_ctx = default_num_ctx
            _last_think_value = is_thinking
            _last_llm_cpu = llm_cpu

def _extract_think(text):
    """
    Извлекает содержимое между <think> и </think> и возвращает (очищенный_текст, think_содержимое).
    Если теги не найдены, возвращает (исходный_текст, None).
    """
    if not text:
        return text, None
    pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        think_content = match.group(1).strip()
        cleaned = pattern.sub('', text).strip()
        return cleaned, think_content
    return text, None

def _request_with_backoff(api_url, json_payload):
    let_log(json_payload)
    """
    Универсальный метод для запросов к Ollama с прогрессивной задержкой.
    Для облачного режима отдельно обрабатывает 429 с лимитами сессии/недели.
    """
    global _skip_num_predict
    start_time = time.time()
    last_exception = None
    # Если флаг уже установлен, сразу удаляем num_predict, чтобы не тратить попытки
    if _skip_num_predict and 'options' in json_payload and 'num_predict' in json_payload['options']:
        del json_payload['options']['num_predict']
        let_log("Костыль: num_predict удалён из запроса (по флагу _skip_num_predict).")
    for attempt in range(1, MAX_RETRIES + 1):
        # Проверка общего времени выполнения
        elapsed = time.time() - start_time
        if elapsed > MAX_WAIT_TOTAL: raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с)")
        try:
            response = session.post(api_url, json=json_payload) # Обработка HTTP 400 — отдельно, для возможного исключения num_predict
            if response.status_code == 400:
                if 'options' in json_payload and 'num_predict' in json_payload['options']:# Убираем num_predict и пробуем снова
                    del json_payload['options']['num_predict']
                    let_log("Обнаружена ошибка 400. Убираем num_predict и пробуем снова.")
                    _skip_num_predict = True
                    continue  # повторить запрос (счётчик попыток не сбрасываем, но прогресс идёт)
                else: response.raise_for_status()
            # Обработка HTTP ошибок с повторными попытками
            if response.status_code == 429:  # Пытаемся определить, является ли ошибка квотной (сессионный/недельный лимит)
                retry_after = response.headers.get('Retry-After')
                wait_time = None
                if retry_after and retry_after.isdigit(): wait_time = int(retry_after)
                else: # Пытаемся извлечь из текста ошибки
                    try:
                        err_data = response.json()
                        err_msg = err_data.get('error', '').lower()
                        if 'session limit' in err_msg: wait_time = 5 * 60 * 60 # 5 часов
                        elif 'weekly limit' in err_msg: wait_time = 7 * 24 * 60 * 60  # 7 дней
                        elif 'insufficient_quota' in err_msg: raise RuntimeError("balance end")
                    except: pass
                if wait_time is not None:
                    # ИЗМЕНЕНИЕ: если время ожидания больше 5 часов (18000 секунд) – сразу исключение
                    MAX_QUOTA_WAIT = 5 * 60 * 60  # 5 часов
                    if wait_time > MAX_QUOTA_WAIT: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                    # Дополнительная проверка на общий лимит MAX_WAIT_TOTAL (420с) – для квотных ошибок она обычно не сработает,
                    # но оставим для безопасности
                    if wait_time > MAX_WAIT_TOTAL: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает общий лимит {MAX_WAIT_TOTAL}с")
                    let_log(f"Обнаружен лимит квоты (429). Ожидание {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue # повторяем запрос после ожидания
                # Иначе это обычный rate limit - используем экспоненциальный backoff
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining: wait_time = remaining
                    if wait_time < 0.1: wait_time = 0.1
                    let_log(f"HTTP 429 (Rate Limit) на попытке {attempt}. Ожидание {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue
                else: response.raise_for_status()
            if response.status_code in (500, 502, 503, 504):
                # Временные серверные ошибки - используем экспоненциальный backoff
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
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
    global _last_think_content, filter_think_tag
    if not session or not base_url or not default_chat_model: raise RuntimeError("Ollama клиент не инициализирован. Сначала вызовите connect().")
    api_url = f"{base_url}/api/generate"
    try:
        let_log(f"ask_model: Отправка запроса на {api_url}")

        # Проверка настроек и перезагрузка модели при необходимости
        _check_and_reload_model()

        ollama_params = {
            "model": default_chat_model,
            "prompt": generation_params.get("prompt", ""),
            "stream": False,
            "keep_alive": "1.5m",  # время жизни модели 1.5 минуты
            "options": {}
        }

        # Устанавливаем контекст
        ctx = default_num_ctx if default_num_ctx else token_limit
        ollama_params["options"]["num_ctx"] = ctx

        # Пробрасываем think, если модель это поддерживает
        if is_thinking:
            think_value = generation_params.get("think", False)  # По умолчанию запрещаем думать
            ollama_params["think"] = think_value

        # Пробрасываем max_tokens -> num_predict (если не запрещён флагом)
        if not _skip_num_predict and "max_tokens" in generation_params:
            ollama_params["options"]["num_predict"] = generation_params["max_tokens"]

        # Маппинг остальных параметров
        param_mapping = {
            "max_tokens": "num_predict",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "repeat_penalty": "repeat_penalty",
            "stop": "stop"}
        for param, value in generation_params.items():
            if param in ["prompt", "max_tokens", "think"]:
                continue
            if param in param_mapping:
                ollama_params["options"][param_mapping[param]] = value
            else:
                ollama_params["options"][param] = value

        data = _request_with_backoff(api_url, ollama_params)
        let_log(f"ask_model: Получен ответ, длина: {len(str(data))} символов")
        result = data.get("response", "").strip()
        let_log(f"ask_model: Результат: '{result[:100]}...'")

        # Обработка think-тегов (фильтрация)
        if filter_think_tag:
            cleaned, think = _extract_think(result)
            _last_think_content = think
            result = cleaned

        return result
    except requests.exceptions.RequestException as e: raise RuntimeError(f"Ошибка сети: {e}")
    except Exception as e: raise RuntimeError(f"Неожиданная ошибка: {e}")

def ask_model_chat(generation_params):
    global filter_think_tag
    if not session or not base_url or not default_chat_model: raise RuntimeError("Ollama клиент не инициализирован. Сначала вызовите connect().")
    api_url = f"{base_url}/api/chat"
    try:
        let_log(f"ask_model_chat: Отправка запроса на {api_url}")

        # Проверка настроек и перезагрузка модели при необходимости
        _check_and_reload_model()

        # Подготовка параметров для Ollama Chat API
        ollama_params = {
            "model": default_chat_model,
            "messages": generation_params.get("messages", []),
            "stream": False,
            "keep_alive": "1.5m",  # время жизни модели 1.5 минуты
            "options": {}}

        # Устанавливаем контекст
        ctx = default_num_ctx if default_num_ctx else token_limit
        ollama_params["options"]["num_ctx"] = ctx

        # Пробрасываем think, если модель это поддерживает
        if is_thinking:
            think_value = generation_params.get("think", False)  # По умолчанию запрещаем думать
            ollama_params["think"] = think_value

        # Пробрасываем max_tokens -> num_predict (если не запрещён флагом)
        if not _skip_num_predict and "max_tokens" in generation_params:
            ollama_params["options"]["num_predict"] = generation_params["max_tokens"]

        # Маппинг параметров
        param_mapping = {
            "max_tokens": "num_predict",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "repeat_penalty": "repeat_penalty",
            "stop": "stop"}
        for param, value in generation_params.items():
            if param in ["messages", "model", "max_tokens", "think"]:
                continue
            if param in param_mapping:
                ollama_params["options"][param_mapping[param]] = value
            else:
                ollama_params["options"][param] = value

        data = _request_with_backoff(api_url, ollama_params)
        let_log(f"ask_model_chat: Получен ответ, длина: {len(str(data))} символов")

        # Обработка think-тегов (фильтрация) для ответа чата
        if filter_think_tag and 'message' in data and 'content' in data['message']:
            original_content = data['message']['content']
            cleaned_content, think_content = _extract_think(original_content)
            data['message']['content'] = cleaned_content
            data['think'] = think_content
        elif filter_think_tag:
            data['think'] = None

        return data
    except requests.exceptions.RequestException as e: raise RuntimeError(f"Ошибка сети: {e}")
    except Exception as e: raise RuntimeError(f"Неожиданная ошибка: {e}")

def get_last_think():
    """Возвращает последнее извлечённое think-содержимое (после вызова ask_model)."""
    return _last_think_content

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
                        MAX_QUOTA_WAIT = 5 * 60 * 60  # 5 часов
                        if wait_time > MAX_QUOTA_WAIT: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                        if wait_time > MAX_WAIT_TOTAL: raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает общий лимит {MAX_WAIT_TOTAL}с")
                        let_log(f"create_embeddings: Обнаружен лимит квоты (429). Ожидание {wait_time:.2f} с...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                    # Обычный rate limit
                    if attempt < MAX_RETRIES:
                        wait_time = BASE_BACKOFF ** attempt
                        remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                        if wait_time > remaining: wait_time = remaining
                        if wait_time < 0.1: wait_time = 0.1
                        let_log(f"create_embeddings: HTTP 429 (Rate Limit), повтор через {wait_time:.2f} с...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
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