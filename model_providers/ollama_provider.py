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
    except Exception:
        pass

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
                if end_match:
                    parsed_tags["sys_end"] = end_match.group(1)
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
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in params:
            params[key] = value

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

        if not available_models:
            return [False, 0, tags, "Не удалось получить список моделей с сервера Ollama."]

        # Устанавливаем модель для чата
        requested_model = params.get("model")
        if requested_model and requested_model in available_models:
            default_chat_model = requested_model
        else:
            default_chat_model = available_models[0]

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
                else:
                    # Если не удалось получить детали, устанавливаем значение по умолчанию
                    emb_token_limit = 4095
            else:
                emb_token_limit = token_limit
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


def ask_model(generation_params):
    """
    Простая функция для API /api/generate (аналог completions)
    С автоматическим определением thinking-моделей по структуре ответа
    И рекурсивным восстановлением при потере соединения
    """
    if not session or not base_url or not default_chat_model:
        raise RuntimeError("Ollama клиент не инициализирован. Сначала вызовите connect().")

    api_url = f"{base_url}/api/generate"
    
    try:
        let_log(f"ask_model: Отправка запроса на {api_url}")
        
        # Подготовка параметров для Ollama API
        ollama_params = {
            "model": default_chat_model,
            "prompt": generation_params.get("prompt", ""),
            "stream": False,
            "options": {}
        }
        
        # НЕ добавляем "think": true - пусть модель решает сама
        
        # Маппинг параметров
        param_mapping = {
            "max_tokens": "num_predict",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "repeat_penalty": "repeat_penalty",
            "stop": "stop"
        }
        
        for param, value in generation_params.items():
            if param == "prompt":
                continue
            if param in param_mapping:
                ollama_params["options"][param_mapping[param]] = value
            else:
                ollama_params["options"][param] = value
        
        response = session.post(api_url, json=ollama_params)
        
        if response.status_code != 200:
            error_text = response.text
            let_log(f"ask_model: Ошибка HTTP {response.status_code}: {error_text}")
            
            # Обработка ошибок контекста
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
                # Ключевые фразы, указывающие на переполнение контекста
                context_error_keywords = [
                    'the input length exceeds the context length',
                    'input length exceeds',
                    'context length',
                    'exceeds context',
                    'token limit exceeded',
                    'exceeds the context',
                    'exceeds context length'
                ]
                
                # Проверяем, есть ли в тексте ошибки указание на переполнение контекста
                if any(keyword in error_text.lower() for keyword in context_error_keywords):
                    let_log("ask_model: Ошибка 500 связана с контекстом -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                elif 'context' in error_text.lower():
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
        let_log(f"ask_model: Получен ответ, длина: {len(str(data))} символов")
        
        # АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ THINKING-МОДЕЛИ
        # Если в ответе есть поле "thinking" - это thinking-модель
        # В любом случае финальный ответ берём из поля "response"
        result = data.get("response", "").strip()
        let_log(f"ask_model: Результат: '{result[:100]}...'")
        
        # Сбрасываем флаг рекурсии при успешном выполнении
        if hasattr(ask_model, 'was_recursion'):
            ask_model.was_recursion = False
            
        return result

    except requests.exceptions.ConnectionError as e:
        let_log(f"ask_model: Ошибка соединения с Ollama API: {e}")
        
        # Проверяем, была ли уже рекурсия
        if not hasattr(ask_model, 'was_recursion'):
            ask_model.was_recursion = False
            
        if ask_model.was_recursion:
            # Если уже была рекурсия - пробрасываем ошибку
            let_log("ask_model: Рекурсия уже была, пробрасываем ошибку")
            raise RuntimeError("Ошибка соединения с Ollama API")
        else:
            # Первая ошибка подключения - пытаемся повторить
            ask_model.was_recursion = True
            let_log("ask_model: Первая ошибка подключения, повторяем запрос через 60 секунд...")
            
            
            while True:
                try:
                    time.sleep(60)  # Ждем минуту
                    result = ask_model(generation_params)
                    # Сбрасываем флаг при успешном повторном вызове
                    ask_model.was_recursion = False
                    return result
                except requests.exceptions.ConnectionError:
                    let_log("ask_model: Повторная ошибка подключения, ждем еще минуту...")
                    continue
                except Exception as retry_e:
                    # Другие ошибки пробрасываем
                    ask_model.was_recursion = False
                    raise retry_e
                    
    except requests.exceptions.RequestException as e:
        let_log(f"ask_model: Ошибка сети: {e}")
        raise RuntimeError(f"Ошибка сети: {e}")
    except Exception as e:
        let_log(f"ask_model: Неожиданная ошибка: {e}")
        raise RuntimeError(f"Неожиданная ошибка: {e}")

def ask_model_chat(generation_params):
    """
    Функция для API /api/chat (аналог chat/completions)
    Возвращает полный ответ API как есть (словарь)
    С автоматическим определением thinking-моделей
    И рекурсивным восстановлением при потере соединения
    """
    if not session or not base_url or not default_chat_model:
        raise RuntimeError("Ollama клиент не инициализирован. Сначала вызовите connect().")

    api_url = f"{base_url}/api/chat"
    
    try:
        let_log(f"ask_model_chat: Отправка запроса на {api_url}")
        
        # Подготовка параметров для Ollama Chat API
        ollama_params = {
            "model": default_chat_model,
            "messages": generation_params.get("messages", []),
            "stream": False,
            "options": {}
        }
        
        # НЕ добавляем "think": true - пусть модель решает сама
        
        # Маппинг параметров
        param_mapping = {
            "max_tokens": "num_predict",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "repeat_penalty": "repeat_penalty",
            "stop": "stop"
        }
        
        for param, value in generation_params.items():
            if param in ["messages", "model"]:
                continue
            if param in param_mapping:
                ollama_params["options"][param_mapping[param]] = value
            else:
                ollama_params["options"][param] = value
        
        response = session.post(api_url, json=ollama_params)
        
        if response.status_code != 200:
            error_text = response.text
            let_log(f"ask_model_chat: Ошибка HTTP {response.status_code}: {error_text}")
            
            # Обработка ошибок контекста
            if response.status_code in [413]:
                raise RuntimeError('ContextOverflowError')
            elif response.status_code == 400:
                if any(keyword in error_text.lower() for keyword in ['context', 'length', 'token', 'exceed']):
                    let_log("ask_model_chat: Ошибка 400 связана с контекстом -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                else:
                    let_log(f"ask_model_chat: Ошибка 400 не связана с контекстом: {error_text}")
                    raise RuntimeError(f"Ошибка запроса: {error_text}")
            elif response.status_code == 500:
                # Ключевые фразы, указывающие на переполнение контекста
                context_error_keywords = [
                    'the input length exceeds the context length',
                    'input length exceeds',
                    'context length',
                    'exceeds context',
                    'token limit exceeded',
                    'exceeds the context',
                    'exceeds context length'
                ]
                
                # Проверяем, есть ли в тексте ошибки указание на переполнение контекста
                if any(keyword in error_text.lower() for keyword in context_error_keywords):
                    let_log("ask_model_chat: Ошибка 500 связана с контекстом -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                elif 'context' in error_text.lower():
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
        let_log(f"ask_model_chat: Получен ответ, длина: {len(str(data))} символов")
        
        # АВТОМАТИЧЕСКАЯ ОБРАБОТКА ДЛЯ THINKING-МОДЕЛЕЙ В ЧАТЕ
        # Для thinking-моделей поле thinking будет внутри message.thinking
        # Фильтрация не нужна - API уже возвращает чистый ответ в message.content
        # Просто оставляем структуру как есть
        
        # Сбрасываем флаг рекурсии при успешном выполнении
        if hasattr(ask_model_chat, 'was_recursion_chat'):
            ask_model_chat.was_recursion_chat = False
            
        # Возвращаем полный ответ как словарь
        return data

    except requests.exceptions.ConnectionError as e:
        let_log(f"ask_model_chat: Ошибка соединения с Ollama API: {e}")
        
        # Проверяем, была ли уже рекурсия
        if not hasattr(ask_model_chat, 'was_recursion_chat'):
            ask_model_chat.was_recursion_chat = False
            
        if ask_model_chat.was_recursion_chat:
            # Если уже была рекурсия - пробрасываем ошибку
            let_log("ask_model_chat: Рекурсия уже была, пробрасываем ошибку")
            raise RuntimeError("Ошибка соединения с Ollama API")
        else:
            # Первая ошибка подключения - пытаемся повторить
            ask_model_chat.was_recursion_chat = True
            let_log("ask_model_chat: Первая ошибка подключения, повторяем запрос через 60 секунд...")
            
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
    С рекурсивным восстановлением при потере соединения
    """
    global base_url, emb_model, session
    
    if not base_url or not emb_model or not session:
        raise RuntimeError("Клиент Ollama для эмбеддингов не инициализирован. Вызовите connect() сначала.")
        
    api_url = f"{base_url}/api/embeddings"
    
    # Подготовка текста (ограничиваем длину для избежания ошибок контекста)
    if isinstance(text, str):
        # Можно оставить только ограничение длины, если нужно
        text = text[:2000]  # Обрезаем очень длинные тексты для эмбеддингов

    payload = {
        "model": emb_model,
        "prompt": text
    }

    try:
        let_log(f"create_embeddings: Отправка запроса с моделью {emb_model}")
        let_log(f"create_embeddings: Текст: {text[:200]}...")
        
        response = session.post(api_url, json=payload, timeout=30)
        
        if response.status_code != 200:
            error_text = response.text
            let_log(f"create_embeddings: Ошибка HTTP {response.status_code}: {error_text}")
            
            # Обработка ошибок переполнения контекста
            if response.status_code == 500:
                # Ключевые фразы, указывающие на переполнение контекста
                context_error_keywords = [
                    'the input length exceeds the context length',
                    'input length exceeds',
                    'context length',
                    'exceeds context',
                    'token limit exceeded',
                    'exceeds the context',
                    'exceeds context length'
                ]
                
                # Проверяем, есть ли в тексте ошибки указание на переполнение контекста
                if any(keyword in error_text.lower() for keyword in context_error_keywords):
                    let_log("create_embeddings: Обнаружена ошибка переполнения контекста -> ContextOverflowError")
                    raise RuntimeError('ContextOverflowError')
                else:
                    # Это какая-то другая ошибка 500 (например, таймаут)
                    let_log(f"create_embeddings: Ошибка 500 не связана с контекстом: {error_text}")
                    raise RuntimeError(f"Ошибка сервера Ollama: {error_text}")
            elif response.status_code in [413, 400]:
                # Ошибки 413 (Payload Too Large) и 400 (Bad Request) тоже могут быть связаны с размером
                let_log(f"create_embeddings: Ошибка {response.status_code} -> ContextOverflowError")
                raise RuntimeError('ContextOverflowError')
            elif response.status_code == 429:
                raise RuntimeError(f"Слишком много запросов (429): {error_text}")
            else:
                # Для всех остальных ошибок HTTP вызываем стандартное исключение
                response.raise_for_status()

        data = response.json()
        embedding = data.get('embedding', [])
        
        if not embedding:
            raise ValueError("Пустой вектор эмбеддингов в ответе от Ollama")
        
        let_log(f"create_embeddings: Получен вектор размером {len(embedding)}")
        
        # Сбрасываем флаг рекурсии при успешном выполнении
        if hasattr(create_embeddings, 'was_recursion_emb'):
            create_embeddings.was_recursion_emb = False
            
        return embedding

    except requests.exceptions.ConnectionError as e:
        let_log(f"create_embeddings: Ошибка соединения с Ollama API: {e}")
        
        # Проверяем, была ли уже рекурсия
        if not hasattr(create_embeddings, 'was_recursion_emb'):
            create_embeddings.was_recursion_emb = False
            
        if create_embeddings.was_recursion_emb:
            # Если уже была рекурсия - пробрасываем ошибку
            let_log("create_embeddings: Рекурсия уже была, пробрасываем ошибку")
            raise RuntimeError(f"Ошибка соединения с Ollama API: {str(e)}")
        else:
            # Первая ошибка подключения - пытаемся повторить
            create_embeddings.was_recursion_emb = True
            let_log("create_embeddings: Первая ошибка подключения, повторяем запрос через 60 секунд...")
            
            while True:
                try:
                    time.sleep(60)  # Ждем минуту
                    result = create_embeddings(text)
                    # Сбрасываем флаг при успешном повторном вызове
                    create_embeddings.was_recursion_emb = False
                    return result
                except requests.exceptions.ConnectionError:
                    let_log("create_embeddings: Повторная ошибка подключения, ждем еще минуту...")
                    continue
                except Exception as retry_e:
                    # Другие ошибки пробрасываем
                    create_embeddings.was_recursion_emb = False
                    raise retry_e
                    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ошибка API эмбеддингов Ollama: {str(e)}")
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Некорректный формат ответа от API эмбеддингов Ollama: {e}")