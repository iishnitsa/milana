import requests
import re
import json
import time
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
            if server_limit is not None:
                return int(server_limit)
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
            for key in path:
                value = value[key]
            if isinstance(value, (int, float)):
                return int(value)
        except (KeyError, TypeError):
            continue

    if isinstance(model_data.get('parameters'), str):
        param_str = model_data['parameters']
        for line in param_str.split('\n'):
            if any(kw in line.lower() for kw in ['num_ctx', 'context', 'n_ctx', 'max_tokens']):
                numbers = re.findall(r'\b\d{3,5}\b', line)
                if numbers:
                    return int(numbers[-1])

    context_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 128000, 200000]
    model_text = json.dumps(model_data)
    found_sizes = sorted([int(num) for num in re.findall(r'\b\d{4,6}\b', model_text)
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
    global ollama_base_url, ollama_emb_model, do_chat_construct, native_func_call, unified_tags
    global filter_think_enabled, filter_start_tag, filter_end_tag

    # Параметры по умолчанию
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

    # --- Разбор строки подключения ---
    for part in connection_string.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in params:
            params[key] = value

    base_url = params["url"].strip('/')
    ollama_base_url = params["ollama_url"].strip('/')
    ollama_emb_model = params["emb_model"]
    do_chat_construct = params["chat_template"].lower().strip() == "true"
    native_func_call = params["native_func_call"].lower().strip() == "true"
    
    # Инициализация переменных фильтрации
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

        # Устанавливаем модель для чата
        requested_model = params.get("model")
        if requested_model and requested_model in available_models:
            default_chat_model = requested_model
        else:
            default_chat_model = available_models[0]

        token_limit = int(params["token_limit"])

        # === Подключение к Ollama для эмбеддингов ===
        ollama_models_url = f"{ollama_base_url}/api/tags"
        ollama_response = requests.get(ollama_models_url, timeout=timeout)
        ollama_response.raise_for_status()

        ollama_models_data = ollama_response.json()
        ollama_available_models = [model['name'] for model in ollama_models_data.get('models', [])]

        if not ollama_available_models:
            return [False, 0, tags, "Не удалось получить список моделей с сервера Ollama."]

        # Проверяем, доступна ли модель для эмбеддингов
        if ollama_emb_model not in ollama_available_models:
            ollama_emb_model = ollama_available_models[0]

        # Автоматическое определение лимита токенов для Ollama эмбеддингов
        try:
            show_url = f"{ollama_base_url}/api/show"
            show_payload = {"name": ollama_emb_model}
            show_response = requests.post(show_url, json=show_payload, timeout=timeout)
            if show_response.status_code == 200:
                model_details = show_response.json()
                emb_token_limit = find_context_size(model_details, ollama_base_url, {})
        except Exception as e: pass

        return [True, token_limit, tags]

    except requests.exceptions.RequestException as e:
        session = None
        return [False, 0, tags, f"Ошибка подключения: {e}"]
    except Exception as e:
        session = None
        return [False, 0, tags, f"Непредвиденная ошибка: {e}"]


def disconnect() -> bool:
    """Закрыть HTTP сессию"""
    global session, base_url, default_chat_model
    if session:
        session.close()
        session = None
        base_url = ""
        default_chat_model = None
        return True
    return False


def apply_think_filter(text: str) -> str:
    """
    Применяет фильтрацию think-части к тексту ответа.
    Если filter_think_enabled = True:
      1. Ищет filter_start_tag в тексте
      2. Если находит - берет текст после него
      3. Если filter_end_tag не пуст - удаляет все до filter_end_tag
    Иначе возвращает оригинальный текст.
    """
    global filter_think_enabled, filter_start_tag, filter_end_tag
    
    if not filter_think_enabled: return text

    start_pos = text.find(filter_start_tag)
    
    if start_pos == -1: return text
    
    # Берем текст после стартового тега
    filtered_text = text[start_pos + len(filter_start_tag):]
    
    # Если указан конечный тег - удаляем все до него
    if filter_end_tag and filter_end_tag.strip():
        end_pos = filtered_text.find(filter_end_tag)
        if end_pos != -1:
            filtered_text = filtered_text[:end_pos]
    return filtered_text.strip()


def ask_model(generation_params):
    """
    Простая функция для API v1/completions
    """
    
    if not session or not base_url:
        raise RuntimeError("GPT4All клиент не инициализирован. Сначала вызовите connect().")

    api_url = f"{base_url}/v1/completions"
    
    try:
        let_log(f"ask_model: Отправка запроса на {api_url}")
        
        # ДОБАВЛЯЕМ МОДЕЛЬ В ПАРАМЕТРЫ, ЕСЛИ ЕЁ НЕТ
        if 'model' not in generation_params and default_chat_model:
            generation_params['model'] = default_chat_model
        
        response = session.post(api_url, json=generation_params)
        
        if response.status_code != 200:
            error_text = response.text
            let_log(f"ask_model: Ошибка HTTP {response.status_code}: {error_text}")
            
            # ТАКАЯ ЖЕ УЛУЧШЕННАЯ ОБРАБОТКА ОШИБОК
            if response.status_code in [413]:
                raise RuntimeError('ContextOverflowError')
            elif response.status_code == 400:
                if any(keyword in error_text.lower() for keyword in ['context', 'length', 'token', 'exceed']):
                    let_log("ask_model: Ошибка 400 связана с контекстом -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                else:
                    let_log(f"ask_model: Ошибка 400 не связана с контекстом: {error_text}")
                    raise RuntimeError(f"Ошибка запроса: {error_text}")
            elif response.status_code == 500:
                if 'context' in error_text.lower():
                    let_log("ask_model: Ошибка 500 связана с контекстом -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                else:
                    let_log(f"ask_model: Серверная ошибка 500: {error_text}")
                    raise RuntimeError(f"Серверная ошибка: {error_text}")
            elif response.status_code == 429:
                raise RuntimeError(f"Слишком много запросов (429): {error_text}")
            else:
                raise RuntimeError(f"HTTP ошибка {response.status_code}: {error_text}")
        
        data = response.json()
        let_log(f"ask_model: Получен ответ: {data}")
        
        if "choices" not in data or not data["choices"]:
            let_log("ask_model: Некорректный формат ответа - нет choices")
            raise RuntimeError("Некорректный формат ответа - нет choices")

        choice = data["choices"][0]
        result = choice.get("text", "").strip()
        
        # ПРИМЕНЯЕМ ФИЛЬТРАЦИЮ THINK-ЧАСТИ
        result = apply_think_filter(result)
        let_log(f"ask_model: Результат после фильтрации: '{result}'")
        
        # Сбрасываем флаг рекурсии при успешном выполнении
        if hasattr(ask_model, 'was_recursion_ask'):
            ask_model.was_recursion_ask = False
            
        return result

    except requests.exceptions.ConnectionError as e:
        let_log(f"ask_model: Ошибка соединения с GPT4All API: {e}")
        
        # Проверяем, была ли уже рекурсия
        if not hasattr(ask_model, 'was_recursion_ask'):
            ask_model.was_recursion_ask = False
            
        if ask_model.was_recursion_ask:
            # Если уже была рекурсия - пробрасываем ошибку
            let_log("ask_model: Рекурсия уже была, пробрасываем ошибку")
            raise RuntimeError("Ошибка соединения с GPT4All API")
        else:
            # Первая ошибка подключения - пытаемся повторить
            ask_model.was_recursion_ask = True
            let_log("ask_model: Первая ошибка подключения, повторяем запрос...")
            
            while True:
                try:
                    time.sleep(60)  # Ждем минуту
                    result = ask_model(generation_params)
                    # Сбрасываем флаг при успешном повторном вызове
                    ask_model.was_recursion_ask = False
                    return result
                except requests.exceptions.ConnectionError:
                    let_log("ask_model: Повторная ошибка подключения, ждем еще минуту...")
                    continue
                except Exception as retry_e:
                    # Другие ошибки пробрасываем
                    ask_model.was_recursion_ask = False
                    raise retry_e
                    
    except requests.exceptions.RequestException as e:
        let_log(f"ask_model: Ошибка сети: {e}")
        raise RuntimeError(f"Ошибка сети: {e}")
    except Exception as e:
        let_log(f"ask_model: Неожиданная ошибка: {e}")
        raise RuntimeError(f"Неожиданная ошибка: {e}")


def ask_model_chat(generation_params):
    """
    Простая функция для API v1/chat/completions
    Просто передает параметры в API без какой-либо обработки
    Возвращает полный ответ API как есть (словарь)
    """
    
    if not session or not base_url:
        raise RuntimeError("GPT4All клиент не инициализирован. Сначала вызовите connect().")

    api_url = f"{base_url}/v1/chat/completions"
    
    try:
        let_log(f"ask_model_chat: Отправка запроса на {api_url}")
        
        # ДОБАВЛЯЕМ МОДЕЛЬ В ПАРАМЕТРЫ, ЕСЛИ ЕЁ НЕТ
        if 'model' not in generation_params and default_chat_model: generation_params['model'] = default_chat_model
        response = session.post(api_url, json=generation_params)
        
        if response.status_code != 200:
            error_text = response.text
            let_log(f"ask_model_chat: Ошибка HTTP {response.status_code}: {error_text}")
            
            # УЛУЧШЕННАЯ ОБРАБОТКА ОШИБОК
            if response.status_code in [413]:
                # 413 - точно переполнение
                raise RuntimeError('ContextOverflowError')
            elif response.status_code == 400:
                # 400 может быть разным - анализируем текст ошибки
                if any(keyword in error_text.lower() for keyword in ['context', 'length', 'token', 'exceed']):
                    let_log("ask_model_chat: Ошибка 400 связана с контекстом -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                else:
                    # Другие ошибки 400 (как "model parameter required")
                    let_log(f"ask_model_chat: Ошибка 400 не связана с контекстом: {error_text}")
                    raise RuntimeError(f"Ошибка запроса: {error_text}")
            elif response.status_code == 500:
                # 500 может быть переполнением или серверной ошибкой
                if 'context' in error_text.lower():
                    let_log("ask_model_chat: Ошибка 500 связана с контекстом -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                else:
                    let_log(f"ask_model_chat: Серверная ошибка 500: {error_text}")
                    raise RuntimeError(f"Серверная ошибка: {error_text}")
            elif response.status_code == 429:
                raise RuntimeError(f"Слишком много запросов (429): {error_text}")
            else:
                raise RuntimeError(f"HTTP ошибка {response.status_code}: {error_text}")
        
        data = response.json()
        let_log(f"ask_model_chat: Получен ответ: {data}")
        
        # Сбрасываем флаг рекурсии при успешном выполнении
        if hasattr(ask_model_chat, 'was_recursion_chat'):
            ask_model_chat.was_recursion_chat = False
        
        # ПРИМЕНЯЕМ ФИЛЬТРАЦИЮ THINK-ЧАСТИ К СООБЩЕНИЮ АССИСТЕНТА
        if filter_think_enabled and "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                original_content = choice["message"]["content"]
                filtered_content = apply_think_filter(original_content)
                
                # Обновляем содержимое в ответе
                data["choices"][0]["message"]["content"] = filtered_content
                
                # Также обновляем текст в корне ответа, если он есть
                if "text" in data["choices"][0]:
                    data["choices"][0]["text"] = filtered_content
                
                let_log(f"ask_model_chat: Применил фильтр think-части. Было: '{original_content[:100]}...', Стало: '{filtered_content[:100]}...'")
        
        # ВОЗВРАЩАЕМ ПОЛНЫЙ ОТВЕТ КАК СЛОВАРЬ, БЕЗ ИЗВЛЕЧЕНИЯ ТЕКСТА
        return data

    except requests.exceptions.ConnectionError as e:
        let_log(f"ask_model_chat: Ошибка соединения с GPT4All API: {e}")
        
        # Проверяем, была ли уже рекурсия
        if not hasattr(ask_model_chat, 'was_recursion_chat'):
            ask_model_chat.was_recursion_chat = False
            
        if ask_model_chat.was_recursion_chat:
            # Если уже была рекурсия - пробрасываем ошибку
            let_log("ask_model_chat: Рекурсия уже была, пробрасываем ошибку")
            raise RuntimeError("Ошибка соединения с GPT4All API")
        else:
            # Первая ошибка подключения - пытаемся повторить
            ask_model_chat.was_recursion_chat = True
            let_log("ask_model_chat: Первая ошибка подключения, повторяем запрос...")
            
            while True:
                try:
                    time.sleep(60)  # Ждем минуту
                    result = ask_model_chat(generation_params)
                    # Сбрасываем флаг при успешном повторном вызове
                    ask_model_chat.was_recursion_chat = False
                    return result
                except requests.exceptions.ConnectionError:
                    let_log("ask_model_chat: Повторная ошибка подключения, ждем еще минуту...")
                    continue
                except Exception as retry_e:
                    # Другие ошибки пробрасываем
                    ask_model_chat.was_recursion_chat = False
                    raise retry_e
                    
    except requests.exceptions.RequestException as e:
        let_log(f"ask_model_chat: Ошибка сети: {e}")
        raise RuntimeError(f"Ошибка сети: {e}")
    except Exception as e:
        let_log(f"ask_model_chat: Неожиданная ошибка: {e}")
        raise RuntimeError(f"Неожиданная ошибка: {e}")

def create_embeddings(text):
    """
    Создание эмбеддингов для текста через Ollama API.
    """
    global ollama_base_url, ollama_emb_model
    
    if not ollama_base_url or not ollama_emb_model:
        raise RuntimeError("Клиент Ollama для эмбеддингов не инициализирован. Вызовите connect() сначала.")
        
    api_url = f"{ollama_base_url}/api/embeddings"
    
    payload = {
        "model": ollama_emb_model,
        "prompt": text
    }

    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        embedding = data.get('embedding', [])
        
        if not embedding:
            raise ValueError("Пустой вектор эмбеддингов в ответе от Ollama")
        
        # Сбрасываем флаг рекурсии при успешном выполнении
        if hasattr(create_embeddings, 'was_recursion_emb'):
            create_embeddings.was_recursion_emb = False
            
        return embedding

    except requests.exceptions.ConnectionError as e:
        
        # Проверяем, была ли уже рекурсия
        if not hasattr(create_embeddings, 'was_recursion_emb'):
            create_embeddings.was_recursion_emb = False
            
        if create_embeddings.was_recursion_emb:
            # Если уже была рекурсия - пробрасываем ошибку
            raise RuntimeError(f"Ошибка API эмбеддингов Ollama: {str(e)}")
        else:
            # Первая ошибка подключения - пытаемся повторить
            create_embeddings.was_recursion_emb = True
            
            while True:
                try:
                    time.sleep(60)  # Ждем минуту
                    result = create_embeddings(text)
                    # Сбрасываем флаг при успешном повторном вызове
                    create_embeddings.was_recursion_emb = False
                    return result
                except requests.exceptions.ConnectionError:
                    continue
                except Exception as retry_e:
                    # Другие ошибки пробрасываем
                    create_embeddings.was_recursion_emb = False
                    raise retry_e
                    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ошибка API эмбеддингов Ollama: {str(e)}")
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Некорректный формат ответа от API эмбеддингов Ollama: {e}")