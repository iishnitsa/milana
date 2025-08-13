# ollama_provider.py
import requests
import json
import traceback
import re
import tiktoken
from typing import Optional, Dict, Any, List, Union, Tuple

# Глобальные переменные состояния
client = None
token_limit = 4095  # Значение по умолчанию для чат-модели
emb_token_limit = 4095   # Значение по умолчанию для модели эмбеддингов
tags = {
    "system": None,
    "user": None,
    "assistant": None,
    "end": None
}

class OllamaError(Exception):
    """Базовая ошибка Ollama."""
    pass

class ConnectionError(OllamaError):
    """Ошибка соединения с сервером Ollama."""
    pass

class APIError(OllamaError):
    """Ошибка API Ollama."""
    pass

def find_context_size(model_data: Dict[str, Any], base_url: str, headers: Dict[str, str]) -> int:
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

def connect(connection_string: str, timeout: int = 30) -> List[Any]:
    global client, token_limit, emb_token_limit, tags

    params = {
        'host': 'http://localhost',
        'port': 11434,
        'chat': None,
        'emb': None,
        'token': None
    }

    try:
        for part in connection_string.split(';'):
            part = part.strip()
            if not part:
                continue
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip().lower()
                value = value.strip()
                if key == 'host':
                    params['host'] = value if '://' in value else f'http://{value}'
                elif key == 'port':
                    params['port'] = int(value)
                elif key in ('chat', 'emb', 'token'):
                    params[key] = value if value else None

        client = OllamaClient(
            host=params['host'],
            port=params['port'],
            api_key=params['token'],
            timeout=timeout
        )

        connected, chat_max_tokens, emb_max_tokens, model_tags = client.connect(params)

        if not connected:
            return [False, 0, "Не удалось подключиться к серверу Ollama"]

        tags = model_tags
        token_limit = chat_max_tokens
        emb_token_limit = emb_max_tokens

        available_models = [m['name'] for m in client.list_models()]

        if params['chat'] and params['chat'] not in available_models:
            print(f"Предупреждение: чат-модель '{params['chat']}' не найдена")
            return [False, 0, "Не удалось подключиться к серверу Ollama"]
        if params['emb'] and params['emb'] not in available_models:
            print(f"Предупреждение: модель эмбеддингов '{params['emb']}' не найдена")
            return [False, 0, "Не удалось подключиться к серверу Ollama"]

        return [connected, chat_max_tokens, model_tags]

    except Exception as e:
        traceback.print_exc()
        return [False, 0, 0, f"Ошибка подключения: {str(e)}"]

def disconnect() -> bool:
    global client
    if client is not None:
        client = None
        return True
    return False

def ask_model(generation_params: Dict[str, Any]) -> str:
    global client, token_limit
    if client is None:
        raise RuntimeError("Клиент Ollama не инициализирован")
    try:
        prompt = generation_params["prompt"]
        if not client.is_within_token_limit(prompt, 'chat', token_limit):
            raise RuntimeError("ContextOverflowError")
        return client.generate(generation_params)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка генерации: {str(e)}")

def create_embeddings(text: str) -> List[float]:
    global client, emb_token_limit
    if client is None:
        raise RuntimeError("Клиент Ollama не инициализирован")
    try:
        if not client.is_within_token_limit(text, 'emb', emb_token_limit):
            raise RuntimeError("ContextOverflowError")
        return client.embeddings(text)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка создания эмбеддингов: {str(e)}")

class OllamaClient:
    def __init__(self, host: str = "http://localhost", port: int = 11434,
                 api_key: Optional[str] = None, timeout: int = 30):
        self.base_url = f"{host}:{port}"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout
        self.chat_model_info = {}
        self.emb_model_info = {}
        self.template_info = {}
        self.tags = {
            "system": None,
            "user": None,
            "assistant": None,
            "end": None
        }
        self.token_limit = 4095
        self.emb_token_limit = 4095
        self.encoding_cache = {}

    def connect(self, params: Dict[str, Any]) -> Tuple[bool, int, int, Dict]:
        try:
            models_url = f"{self.base_url}/api/tags"
            models_resp = requests.get(models_url, headers=self.headers, timeout=self.timeout)
            models_resp.raise_for_status()
            models_data = models_resp.json()
            all_models = models_data.get("models", [])

            if not all_models:
                return False, 4095, 4095, self.tags

            names = {m['name']: m for m in all_models}

            chat_model_name = params.get('chat') or all_models[0]['name']
            emb_model_name = params.get('emb') or chat_model_name

            self.chat_model_info = names.get(chat_model_name, all_models[0])
            self.emb_model_info = names.get(emb_model_name, self.chat_model_info)

            # Определяем контекст для чат-модели
            chat_show_url = f"{self.base_url}/api/show"
            chat_show_payload = {"name": self.chat_model_info["name"]}
            chat_show_resp = requests.post(chat_show_url, json=chat_show_payload, headers=self.headers, timeout=self.timeout)
            chat_show_resp.raise_for_status()
            chat_model_details = chat_show_resp.json()
            self.token_limit = find_context_size(chat_model_details, self.base_url, self.headers)

            # Определяем контекст для модели эмбеддингов, если она отличается
            if emb_model_name != chat_model_name:
                emb_show_payload = {"name": self.emb_model_info["name"]}
                emb_show_resp = requests.post(chat_show_url, json=emb_show_payload, headers=self.headers, timeout=self.timeout)
                emb_show_resp.raise_for_status()
                emb_model_details = emb_show_resp.json()
                self.emb_token_limit = find_context_size(emb_model_details, self.base_url, self.headers)
            else:
                self.emb_token_limit = self.token_limit

            self.template_info = {
                "template": chat_model_details.get("template", ""),
                "system": chat_model_details.get("system", ""),
                "parameters": chat_model_details.get("parameters", {})
            }

            self._parse_template_info()

            return True, self.token_limit, self.emb_token_limit, self.tags

        except requests.RequestException as e:
            return False, 4095, 4095, str(e)
        except Exception as e:
            return False, 4095, 4095, str(e)

    def _parse_template_info(self):
        template = self.template_info.get("template", "")
        system_msg = self.template_info.get("system", "")

        if system_msg and "system" in system_msg.lower():
            parts = system_msg.split("system", 1)
            self.tags["system"] = parts[0] + "system" + parts[1].split()[0] if len(parts) > 1 else parts[0] + "system"

        if template:
            for part in template.split():
                part_lower = part.lower()
                if "user" in part_lower and not self.tags["user"]:
                    self.tags["user"] = part
                elif "assistant" in part_lower and not self.tags["assistant"]:
                    self.tags["assistant"] = part
                elif "system" in part_lower and not self.tags["system"]:
                    self.tags["system"] = part
                elif any(x in part_lower for x in ["[/", "</", "end"]) and not self.tags["end"]:
                    self.tags["end"] = part

    def normalize_model_name(self, model_name: str) -> str:
        """Приводит имя модели к формату, распознаваемому tiktoken."""
        # Удаляем версию и другие суффиксы
        base_model = model_name.split(':')[0].lower()
        
        # Маппинг базовых моделей на известные кодировки
        model_mapping = {
            "llama": "gpt-3.5-turbo",          # LLaMA использует кодировку GPT-3
            "mistral": "gpt-3.5-turbo",        # Mistral совместим
            "gemma": "gpt-3.5-turbo",          # Gemma совместим
            "command-r": "gpt-3.5-turbo",      # Cohere
            "all-minilm": "text-embedding-ada-002"  # Модели эмбеддингов
        }
        
        for key, encoding in model_mapping.items():
            if key in base_model:
                return encoding
        
        # По умолчанию для текстовых моделей
        return "gpt-3.5-turbo"

    def get_encoding(self, model_name: str) -> Any:
        """Получает кодировщик для модели с кешированием."""
        normalized_name = self.normalize_model_name(model_name)
        if normalized_name not in self.encoding_cache:
            try:
                self.encoding_cache[normalized_name] = tiktoken.encoding_for_model(normalized_name)
            except KeyError:
                # Используем кодировку по умолчанию
                self.encoding_cache[normalized_name] = tiktoken.get_encoding("cl100k_base")
        return self.encoding_cache[normalized_name]

    def count_tokens(self, text: str, model_name: str) -> int:
        """Токенизирует текст и возвращает количество токенов."""
        encoding = self.get_encoding(model_name)
        return len(encoding.encode(text))

    def is_within_token_limit(self, text: str, model_type: str, now_token_limit: int) -> bool:
        """
        Рекурсивно проверяет, можно ли разбить текст так, чтобы каждая часть 
        укладывалась в лимит токенов.
        """
        if model_type == 'chat':
            model_name = self.chat_model_info['name']
        else:
            model_name = self.emb_model_info['name']
        
        tokens_count = self.count_tokens(text, model_name)
        # или 1,4 токена на слово (максимум)
        if tokens_count < now_token_limit:
            return True
        return False

    def generate(self, generation_params: Dict[str, Any]) -> str:
        if "prompt" not in generation_params:
            raise ValueError("'prompt' обязателен в generation_params")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.chat_model_info.get("name"),
            "prompt": generation_params["prompt"],
            "stream": False,
            "options": {}
        }

        param_mapping = {
            "max_tokens": "num_predict",
            "top_p": "top_p",
            "temperature": "temperature",
            "repeat_penalty": "repeat_penalty",
            "top_k": "top_k",
            "stop": "stop"
        }

        for param, value in generation_params.items():
            if param == "prompt":
                continue
            if param in param_mapping:
                payload["options"][param_mapping[param]] = value
            else:
                payload["options"][param] = value

        try:
            resp = requests.post(url, json=payload, headers=self.headers)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.HTTPError as e:
            msg = ""
            try:
                msg = resp.json().get("error", "") or resp.text
            except Exception:
                msg = str(e)
            if "context" in msg.lower() or "max_context_length" in msg.lower() or "token_limit" in msg.lower():
                raise RuntimeError("ContextOverflowError")
            raise ConnectionError(f"Ошибка генерации: {str(e)}")
        except Exception as e:
            raise APIError(f"Ошибка обработки ответа: {str(e)}")

    def embeddings(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.emb_model_info.get("name"),
            "prompt": text
        }

        try:
            resp = requests.post(url, json=payload, headers=self.headers)
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except requests.HTTPError as e:
            msg = ""
            try:
                msg = resp.json().get("error", "") or resp.text
            except Exception:
                msg = str(e)
            if "context" in msg.lower() or "max_context_length" in msg.lower() or "token_limit" in msg.lower():
                raise RuntimeError("ContextOverflowError")
            raise ConnectionError(f"Ошибка эмбеддингов: {str(e)}")
        except Exception as e:
            raise APIError(f"Ошибка обработки эмбеддингов: {str(e)}")

    def list_models(self) -> List[Dict]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException:
            return []
