# anthropic_provider.py
import requests
import json
import traceback
import tiktoken
from typing import Optional, Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer

# Глобальные переменные состояния
client = None
token_limit = 200000  # Значение по умолчанию для Anthropic
emb_token_limit = 4096  # Значение по умолчанию для модели эмбеддингов
emb_model = None  # Модель для эмбеддингов
tags = {
    "system": None,
    "user": "Human:",
    "assistant": "Assistant:",
    "end": None
}

class AnthropicError(Exception): pass
class ConnectionError(AnthropicError): pass
class APIError(AnthropicError): pass

def connect(connection_string: str, timeout: int = 30) -> Tuple[bool, int, Dict[str, Optional[str]]]:
    global client, token_limit, emb_token_limit, emb_model, tags

    params = {
        'chat': 'claude-3-haiku-20240307',
        'emb': 'sentence-transformers/all-MiniLM-L6-v2',  # Модель по умолчанию
        'token': None
    }
    
    for part in connection_string.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            if key in params:
                params[key] = value

    # Инициализация клиента Anthropic
    client = AnthropicClient(
        api_key=params['token'],
        timeout=timeout,
        chat_model=params['chat']
    )

    # Установка модели для эмбеддингов
    emb_model = params['emb']

    try:
        connected, max_tokens, model_tags = client.connect()
        if not connected:
            return False, token_limit, tags
        
        token_limit = max_tokens
        tags = model_tags
        
        return True, token_limit, tags

    except Exception as e:
        traceback.print_exc()
        return False, token_limit, tags

def disconnect() -> bool:
    global client, emb_model
    if client is not None:
        client = None
        emb_model = None
        return True
    return False

def ask_model(generation_params: Dict[str, Any]) -> str:
    global client, token_limit
    if client is None:
        raise RuntimeError("Клиент Anthropic не инициализирован")
    try:
        prompt = generation_params["prompt"]
        if not client.is_within_token_limit(prompt, token_limit):
            raise RuntimeError("ContextOverflowError")
        return client.generate(generation_params)
    except RuntimeError as e:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка генерации: {str(e)}")

def chat(messages: List[Dict[str, str]], generation_params: Dict[str, Any]) -> str:
    global client, token_limit
    if client is None:
        raise RuntimeError("Клиент Anthropic не инициализирован")
    try:
        text = json.dumps(messages)
        if not client.is_within_token_limit(text, token_limit):
            raise RuntimeError("ContextOverflowError")
        return client.chat(messages, generation_params)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка чата: {str(e)}")

def create_embeddings(text: str) -> List[float]:
    global emb_model, emb_token_limit
    if emb_model is None:
        raise RuntimeError("Модель для эмбеддингов не указана")
    
    try:
        # Используем локальную модель из sentence-transformers
        model = SentenceTransformer(emb_model)
        
        # Проверка лимита токенов (примерная оценка)
        if len(text.split()) * 1.4 > emb_token_limit:  # ~1.4 токена на слово
            raise RuntimeError("ContextOverflowError")
            
        return model.encode(text, convert_to_numpy=True).tolist()
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка создания эмбеддингов: {str(e)}")

class AnthropicClient:
    MODEL_LIMITS = {
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-2.1": 200000,
        "claude-2.0": 100000,
        "claude-instant-1.2": 100000,
    }

    DEFAULT_TAGS = {
        "system": None,
        "user": "Human:",
        "assistant": "Assistant:",
        "end": None
    }

    def __init__(self, api_key: Optional[str], timeout: int = 30, chat_model: str = "claude-3-haiku-20240307"):
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        if api_key:
            self.headers["x-api-key"] = api_key
        self.timeout = timeout
        self.chat_model = chat_model
        self.tags = self.DEFAULT_TAGS.copy()
        self.encoding_cache: Dict[str, Any] = {}

    def get_encoding(self) -> Any:
        if self.chat_model not in self.encoding_cache:
            try:
                self.encoding_cache[self.chat_model] = tiktoken.encoding_for_model(self.chat_model)
            except KeyError:
                self.encoding_cache[self.chat_model] = tiktoken.get_encoding("cl100k_base")
        return self.encoding_cache[self.chat_model]

    def count_tokens(self, text: str) -> int:
        return len(self.get_encoding().encode(text))

    def is_within_token_limit(self, text: str, token_limit: int) -> bool:
        return self.count_tokens(text) < token_limit

    def connect(self) -> Tuple[bool, int, Dict[str, Optional[str]]]:
        if self.chat_model not in self.MODEL_LIMITS:
            print(f"Предупреждение: модель '{self.chat_model}' не известна")
        max_tokens = self.MODEL_LIMITS.get(self.chat_model, 100000)

        try:
            test_payload = {
                "model": self.chat_model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5
            }
            r = requests.post(
                f"{self.base_url}/messages",
                json=test_payload,
                headers=self.headers,
                timeout=self.timeout
            )
            if r.status_code >= 400:
                raise ConnectionError(f"Ошибка при проверке модели: {r.status_code}")
            return True, max_tokens, self.tags
        except requests.RequestException as e:
            raise ConnectionError(f"Ошибка подключения: {str(e)}")
        except Exception as e:
            raise APIError(f"Ошибка API: {str(e)}")

    def generate(self, generation_params: Dict[str, Any]) -> str:
        if "prompt" not in generation_params:
            raise ValueError("'prompt' обязателен")
            
        messages = [{"role": "user", "content": generation_params["prompt"]}]
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "max_tokens": generation_params.get("max_tokens", 1024),
            "temperature": generation_params.get("temperature", 1.0),
            "top_p": generation_params.get("top_p", None),
            "stream": False
        }
        if payload["top_p"] is None:
            del payload["top_p"]

        try:
            r = requests.post(
                f"{self.base_url}/messages",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
        except requests.RequestException as e:
            msg = str(e).lower()
            if "context" in msg or "token" in msg or "limit" in msg:
                raise RuntimeError("ContextOverflowError")
            raise ConnectionError(f"Ошибка генерации: {str(e)}")
        except Exception as e:
            raise APIError(f"Ошибка обработки ответа: {str(e)}")

    def chat(self, messages: List[Dict[str, str]], generation_params: Dict[str, Any]) -> str:
        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        dialog_messages = [m for m in messages if m["role"] != "system"]

        payload = {
            "model": self.chat_model,
            "messages": dialog_messages,
            "max_tokens": generation_params.get("max_tokens", 1024),
            "temperature": generation_params.get("temperature", 1.0),
            "top_p": generation_params.get("top_p", None),
            "stream": False
        }
        if system_messages:
            payload["system"] = "\n".join(system_messages)
        if payload["top_p"] is None:
            del payload["top_p"]

        try:
            r = requests.post(
                f"{self.base_url}/messages",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
        except requests.RequestException as e:
            msg = str(e).lower()
            if "context" in msg or "token" in msg or "limit" in msg:
                raise RuntimeError("ContextOverflowError")
            raise ConnectionError(f"Ошибка чата: {str(e)}")
        except Exception as e:
            raise APIError(f"Ошибка обработки ответа: {str(e)}")