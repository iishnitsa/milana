import requests
import json
import time
import urllib.parse
from typing import Dict, Any, List, Tuple, Optional
from cross_gpt import let_log

# === Глобальные переменные ===
session = None
base_url = ""
chat_model = None
emb_model = None
timeout = 30

token_limit = 4096          # будет переопределено из ответа модели или параметров
emb_token_limit = 4096

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

def _request_with_backoff(method, url, json_payload=None, headers=None):
    """Универсальный метод с экспоненциальной задержкой и обработкой ошибок."""
    start_time = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        elapsed = time.time() - start_time
        if elapsed > MAX_WAIT_TOTAL:
            raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с)")

        try:
            resp = session.request(method, url, json=json_payload, headers=headers, timeout=timeout)

            if resp.status_code == 429:
                retry_after = resp.headers.get('Retry-After')
                wait_time = None
                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                else:
                    try:
                        err_data = resp.json()
                        err_msg = err_data.get('error', '').lower()
                        if 'session limit' in err_msg:
                            wait_time = 5 * 60 * 60
                        elif 'weekly limit' in err_msg:
                            wait_time = 7 * 24 * 60 * 60
                        elif 'insufficient_quota' in err_msg:
                            raise RuntimeError("balance end")
                    except:
                        pass

                if wait_time is not None:
                    if wait_time > MAX_QUOTA_WAIT:
                        raise RuntimeError("balance end")
                    if wait_time > MAX_WAIT_TOTAL:
                        raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает общий лимит {MAX_WAIT_TOTAL}с")
                    let_log(f"API 429, ожидание {wait_time} с...")
                    time.sleep(wait_time)
                    continue

                # Обычный rate limit – экспоненциальный backoff
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining:
                        wait_time = remaining
                    if wait_time < 0.1:
                        wait_time = 0.1
                    let_log(f"HTTP 429 (Rate Limit), попытка {attempt}, ожидание {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue
                else:
                    resp.raise_for_status()

            if resp.status_code == 402:
                raise RuntimeError("balance end")

            if resp.status_code in (500, 502, 503, 504):
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining:
                        wait_time = remaining
                    if wait_time < 0.1:
                        wait_time = 0.1
                    let_log(f"HTTP {resp.status_code}, попытка {attempt}, ожидание {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue
                else:
                    resp.raise_for_status()

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES:
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining:
                    wait_time = remaining
                if wait_time < 0.1:
                    wait_time = 0.1
                let_log(f"Ошибка соединения, попытка {attempt}, ожидание {wait_time:.2f} с...")
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
                let_log(f"Таймаут, попытка {attempt}, ожидание {wait_time:.2f} с...")
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

# ============================================================
# Основные функции провайдера (обязательный интерфейс)
# ============================================================

def connect(connection_string: str, timeout: int = 30, _decrypted_token: str = None) -> Tuple[bool, int, Dict[str, Optional[str]], Optional[str]]:
    """
    Подключение к серверу llama.cpp.
    Формат строки подключения:
        "url=http://localhost:8080; model=model_name.gguf; emb_model=all-minilm:latest; emb_url=http://localhost:11434"
    Параметры:
        url          – адрес сервера llama.cpp (по умолчанию http://localhost:8080)
        model        – имя модели (используется только для логирования, не влияет на запросы)
        emb_model    – модель для эмбеддингов (если используется отдельный сервер)
        emb_url      – URL для эмбеддингов (если не указан, используются эмбеддинги через тот же сервер)
    Возвращает: (success, token_limit, tags_dict, error_message)
    """
    global session, base_url, chat_model, emb_model, token_limit, emb_token_limit, tags

    params = {
        "url": "http://localhost:8080",
        "model": "bonsai-8b",
        "emb_model": "",
        "emb_url": "",
    }
    # Разбор строки подключения
    for part in connection_string.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in params:
            params[key] = value

    # Нормализация URL
    base_url = normalize_url(params["url"], default_port=8080)
    emb_url = normalize_url(params["emb_url"], default_port=8080) if params["emb_url"] else base_url

    chat_model = params["model"] if params["model"] else "llama-model"
    emb_model = params["emb_model"] if params["emb_model"] else ""

    # Создаём сессию
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    # Проверка доступности сервера (опционально – можно пропустить)
    try:
        resp = session.get(f"{base_url}/health", timeout=timeout)
        if resp.status_code != 200:
            # Некоторые версии llama.cpp не имеют /health – пробуем /v1/models
            resp = session.get(f"{base_url}/v1/models", timeout=timeout)
            if resp.status_code != 200:
                return False, token_limit, tags, f"Сервер {base_url} не отвечает на /health или /v1/models"
    except Exception as e:
        return False, token_limit, tags, f"Ошибка подключения к {base_url}: {e}"

    # Определяем лимит контекста (можно через /v1/models или оставить по умолчанию)
    try:
        models_resp = session.get(f"{base_url}/v1/models", timeout=timeout)
        if models_resp.status_code == 200:
            data = models_resp.json()
            if data.get("data"):
                # Берём первую модель (обычно одна)
                model_info = data["data"][0]
                # Пытаемся извлечь context_length из разных полей
                ctx = model_info.get("context_length") or model_info.get("max_context_length")
                if ctx:
                    token_limit = int(ctx)
    except:
        pass  # оставляем значение по умолчанию

    # Если эмбеддинги будут через тот же сервер – устанавливаем лимит такой же
    emb_token_limit = token_limit

    # Теги оставляем пустыми – они не используются в llama.cpp (всё форматируется на стороне модели)
    tags = {
        "bos": "", "eos": "",
        "sys_start": "", "sys_end": "",
        "user_start": "", "user_end": "",
        "assist_start": "", "assist_end": "",
        "tool_def_start": "", "tool_def_end": "",
        "tool_call_start": "", "tool_call_end": "",
        "tool_result_start": "", "tool_result_end": "",
    }

    # Устанавливаем глобальные флаги (как в других провайдерах)
    global do_chat_construct, native_func_call
    do_chat_construct = True       # используем chat/completions
    native_func_call = False       # инструменты вызываются через маркеры, не через native

    let_log(f"Провайдер llama.cpp подключен к {base_url}, модель: {chat_model}, контекст: {token_limit}")
    return True, token_limit, tags, None

def disconnect() -> bool:
    """Закрыть сессию."""
    global session, base_url, chat_model
    if session:
        session.close()
        session = None
        base_url = ""
        chat_model = None
        return True
    return False

def ask_model(generation_params: Dict[str, Any]) -> str:
    """
    Генерация через /v1/completions (для режима completions).
    """
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
    # Дополнительные параметры
    for key in ["frequency_penalty", "presence_penalty", "repeat_penalty"]:
        if key in generation_params:
            payload[key] = generation_params[key]

    let_log(f"ask_model: запрос к {url}")
    data = _request_with_backoff("POST", url, json_payload=payload)
    if "choices" not in data or not data["choices"]:
        raise RuntimeError("Некорректный ответ от сервера: отсутствует choices")
    result = data["choices"][0].get("text", "").strip()
    return result

def ask_model_chat(generation_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Генерация через /v1/chat/completions (для чата).
    Возвращает словарь в формате, совместимом с cross_gpt.
    """
    if not session or not base_url:
        raise RuntimeError("llama.cpp клиент не инициализирован. Сначала вызовите connect().")

    url = f"{base_url}/v1/chat/completions"
    messages = generation_params.get("messages", [])
    # Преобразуем сообщения в формат, ожидаемый llama.cpp (уже готово)
    payload = {
        "model": chat_model,
        "messages": messages,
        "max_tokens": generation_params.get("max_tokens", 500),
        "temperature": generation_params.get("temperature", 0.7),
        "top_p": generation_params.get("top_p", 0.9),
        "stop": generation_params.get("stop", None),
    }
    # Добавляем инструменты, если есть (если модель их поддерживает, но мы отключили native)
    if "tools" in generation_params:
        payload["tools"] = generation_params["tools"]
    if "tool_choice" in generation_params:
        payload["tool_choice"] = generation_params["tool_choice"]

    let_log(f"ask_model_chat: запрос к {url}")
    data = _request_with_backoff("POST", url, json_payload=payload)
    # Ожидаем, что ответ в формате OpenAI (выборка первого сообщения)
    # В некоторых версиях llama.cpp может быть поле "message" сразу в choices[0]
    # Проверяем структуру
    if "choices" in data and data["choices"]:
        # Можно просто вернуть как есть
        return data
    else:
        raise RuntimeError("Некорректный ответ от сервера: отсутствует choices")

def create_embeddings(text: str) -> List[float]:
    """
    Получение эмбеддингов через /v1/embeddings.
    Если emb_model не указан, используем ту же модель (если она поддерживает эмбеддинги).
    """
    if not session or not base_url:
        raise RuntimeError("llama.cpp клиент не инициализирован. Сначала вызовите connect().")

    # Если задана отдельная модель для эмбеддингов – используем её, иначе chat_model
    model_for_emb = emb_model if emb_model else chat_model

    url = f"{base_url}/v1/embeddings"
    payload = {
        "model": model_for_emb,
        "input": text,
    }

    let_log(f"create_embeddings: запрос к {url} с моделью {model_for_emb}")
    data = _request_with_backoff("POST", url, json_payload=payload)
    if "data" in data and data["data"]:
        embedding = data["data"][0].get("embedding", [])
        if embedding:
            return embedding
    raise RuntimeError("Не удалось получить эмбеддинг от сервера")