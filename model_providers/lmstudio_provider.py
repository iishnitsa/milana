# lmstudio_provider.py
import requests
import json
import traceback
import tiktoken
from typing import Optional, Dict, Any, List, Tuple

# Глобальные переменные состояния
client = None
token_limit = 4096  # Значение по умолчанию для чат-модели
emb_token_limit = 4096  # Значение по умолчанию для модели эмбеддингов
tags = {
    "system": None,
    "user": None,
    "assistant": None,
    "end": None
}

class LMStudioError(Exception): pass
class ConnectionError(LMStudioError): pass
class APIError(LMStudioError): pass

def connect(connection_string: str, timeout: int = 30) -> Tuple[bool, int, Dict[str, Optional[str]]]:
    global client, token_limit, emb_token_limit, tags

    params = {
        'host': 'http://localhost',
        'port': 1234,
        'chat': None,
        'emb': None,
        'user': None,
        'password': None
    }

    try:
        for part in connection_string.split(';'):
            if '=' in part:
                key, value = part.split('=', 1)
                key, value = key.strip().lower(), value.strip()
                if key == 'host':
                    params['host'] = value if '://' in value else f'http://{value}'
                elif key == 'port':
                    params['port'] = int(value)
                elif key in ('chat', 'emb', 'user', 'password'):
                    params[key] = value or None

        client = LMStudioClient(
            host=params['host'],
            port=params['port'],
            chat_model=params['chat'],
            emb_model=params['emb'],
            username=params['user'],
            password=params['password'],
            timeout=timeout
        )

        connected, chat_max_tokens, emb_max_tokens, model_tags = client.connect()

        if not connected:
            return False, token_limit, tags

        token_limit = chat_max_tokens
        emb_token_limit = emb_max_tokens
        tags = model_tags

        return True, token_limit, tags

    except Exception as e:
        traceback.print_exc()
        return False, token_limit, tags

def disconnect() -> bool:
    global client
    if client is not None:
        client = None
        return True
    return False

def ask_model(generation_params: Dict[str, Any]) -> str:
    global client, token_limit
    if client is None:
        raise RuntimeError("Клиент LM Studio не инициализирован")
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
        raise RuntimeError("Клиент LM Studio не инициализирован")
    try:
        if not client.is_within_token_limit(text, 'emb', emb_token_limit):
            raise RuntimeError("ContextOverflowError")
        return client.embeddings(text)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка создания эмбеддингов: {str(e)}")

class LMStudioClient:
    def __init__(self,
                 host: str = "http://localhost",
                 port: int = 1234,
                 chat_model: Optional[str] = None,
                 emb_model: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 timeout: int = 30):
        self.base_url = f"{host}:{port}"
        self.headers = {"Content-Type": "application/json"}
        self.auth = (username, password) if username and password else None
        self.timeout = timeout
        self.chat_model = chat_model
        self.emb_model = emb_model
        self.token_limit = 4096
        self.emb_token_limit = 4096
        self.tags = {
            "system": None,
            "user": None,
            "assistant": None,
            "end": None
        }
        self.encoding_cache: Dict[str, Any] = {}

    def connect(self) -> Tuple[bool, int, int, Dict[str, Optional[str]]]:
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout
            )
            resp.raise_for_status()
            models_data = resp.json()

            # Установка лимитов токенов
            for model in models_data.get("data", []):
                if self.chat_model and model["id"] == self.chat_model:
                    self.token_limit = model.get("context_length", 4096)
                if self.emb_model and model["id"] == self.emb_model:
                    self.emb_token_limit = model.get("context_length", 4096)

            return True, self.token_limit, self.emb_token_limit, self.tags

        except requests.RequestException as e:
            return False, 4096, 4096, str(e)
        except Exception as e:
            return False, 4096, 4096, str(e)

    def get_encoding(self, model_name: str):
        if model_name not in self.encoding_cache:
            try:
                self.encoding_cache[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                self.encoding_cache[model_name] = tiktoken.get_encoding("cl100k_base")
        return self.encoding_cache[model_name]

    def count_tokens(self, text: str, model_name: str) -> int:
        return len(self.get_encoding(model_name).encode(text))

    def is_within_token_limit(self, text: str, model_type: str, token_limit: int) -> bool:
        model_name = self.chat_model if model_type == 'chat' else self.emb_model
        if not model_name:
            return True  # Если модель не указана, пропускаем проверку
        return self.count_tokens(text, model_name) < token_limit

    def generate(self, generation_params: Dict[str, Any]) -> str:
        if "prompt" not in generation_params:
            raise ValueError("'prompt' обязателен в generation_params")

        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": self.chat_model,
            "prompt": generation_params["prompt"],
            "max_tokens": generation_params.get("max_tokens", 200),
            "temperature": generation_params.get("temperature", 0.7),
            "stream": False
        }

        param_mapping = {
            "top_p": "top_p",
            "repeat_penalty": "repetition_penalty",
            "stop": "stop"
        }

        for param, value in generation_params.items():
            if param in param_mapping:
                payload[param_mapping[param]] = value
            elif param not in ["prompt", "max_tokens", "temperature"]:
                payload[param] = value

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"]
        except requests.HTTPError as e:
            msg = resp.json().get("error", "") or resp.text if hasattr(resp, 'text') else str(e)
            if any(x in msg.lower() for x in ["context", "max", "token", "position"]):
                raise RuntimeError("ContextOverflowError")
            raise ConnectionError(f"Ошибка генерации: {msg}")
        except Exception as e:
            raise APIError(f"Ошибка обработки ответа: {str(e)}")

    def embeddings(self, text: str) -> List[float]:
        if not self.emb_model:
            raise ValueError("Модель для эмбеддингов не указана")

        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.emb_model, "input": text}

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except requests.HTTPError as e:
            msg = resp.json().get("error", "") or resp.text if hasattr(resp, 'text') else str(e)
            if any(x in msg.lower() for x in ["context", "max", "token", "position"]):
                raise RuntimeError("ContextOverflowError")
            raise ConnectionError(f"Ошибка эмбеддингов: {msg}")
        except Exception as e:
            raise APIError(f"Ошибка обработки эмбеддингов: {str(e)}")

    def list_models(self) -> List[Dict]:
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json().get("data", [])
        except requests.RequestException:
            return []