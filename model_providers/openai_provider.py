# openai_provider.py
import requests
import json
import traceback
import tiktoken
from typing import Optional, Dict, Any, List, Tuple

# Глобальные переменные
client = None
model_info = {}
tags = {
    "system": "<|im_start|>system",
    "user": "<|im_start|>user",
    "assistant": "<|im_start|>assistant",
    "end": "<|im_end|>"
}
token_limit = 4096  # По умолчанию

class OpenAIError(Exception):
    pass

class ConnectionError(OpenAIError):
    pass

class APIError(OpenAIError):
    pass

def connect(connection_string: str, timeout: int = 30) -> Tuple[bool, int, Dict[str, Optional[str]]]:
    global client, model_info, token_limit
    params = {
        'chat': 'gpt-3.5-turbo',
        'emb': 'text-embedding-ada-002',
        'token': None,
        'org': None,
        'base_url': 'https://api.openai.com/v1'
    }
    for part in connection_string.split(';'):
        part = part.strip()
        if not part: continue
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            if key in params:
                params[key] = value or params[key]

    client = OpenAIClient(
        api_key=params['token'],
        organization=params['org'],
        base_url=params['base_url'],
        chat_model=params['chat'],
        emb_model=params['emb'],
        timeout=timeout
    )

    success = client.verify_connection()
    if not success:
        return False, token_limit, tags

    model_info = {"chat_model": params['chat'], "emb_model": params['emb']}
    # Устанавливаем лимит в зависимости от модели
    token_limits = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
    }
    token_limit = token_limits.get(params['chat'], 4096)
    return True, token_limit, tags

def disconnect() -> bool:
    global client
    if client is not None:
        client = None
        return True
    return False

def ask_model(generation_params: Dict[str, Any]) -> str:
    global client
    if client is None:
        raise RuntimeError("Клиент OpenAI не инициализирован")
    try:
        return client.generate(generation_params, token_limit)
    except RuntimeError as e:
        # Пробрасываем ContextOverflowError без потери типа
        raise
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка генерации: {str(e)}")

def create_embeddings(text: str) -> List[float]:
    global client
    if client is None:
        raise RuntimeError("Клиент OpenAI не инициализирован")
    try:
        return client.embeddings(text)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка создания эмбеддингов: {str(e)}")

class OpenAIClient:
    def __init__(self,
                 api_key: str,
                 chat_model: str = "gpt-4o",
                 emb_model: str = "text-embedding-3-large",
                 organization: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 timeout: int = 30):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        if organization:
            self.headers["OpenAI-Organization"] = organization
        self.timeout = timeout
        self.chat_model = chat_model
        self.emb_model = emb_model
        self.encoding_cache: Dict[str, Any] = {}

    def verify_connection(self) -> bool:
        try:
            url = f"{self.base_url}/models"
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            raise ConnectionError(f"Ошибка подключения: {str(e)}")

    def get_encoding(self, model_name: str):
        if model_name not in self.encoding_cache:
            try:
                self.encoding_cache[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                self.encoding_cache[model_name] = tiktoken.get_encoding("cl100k_base")
        return self.encoding_cache[model_name]

    def count_tokens(self, text: str, model_name: str) -> int:
        encoding = self.get_encoding(model_name)
        return len(encoding.encode(text))

    def is_within_token_limit(self, text: str, token_limit: int) -> bool:
        return self.count_tokens(text, self.chat_model) < token_limit

    def generate(self, generation_params: Dict[str, Any], token_limit: int) -> str:
        if "prompt" not in generation_params:
            raise ValueError("'prompt' обязателен")
        prompt = generation_params["prompt"]
        if not self.is_within_token_limit(prompt, token_limit):
            raise RuntimeError("ContextOverflowError")

        url = f"{self.base_url}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": self.chat_model, "messages": messages, "max_tokens": generation_params.get("max_tokens", 500),
                   "temperature": generation_params.get("temperature", 0.7), "top_p": generation_params.get("top_p", 1.0),
                   "stream": False}
        for param in ["frequency_penalty", "presence_penalty", "stop"]:
            if param in generation_params:
                payload[param] = generation_params[param]

        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            error_msg = f"Ошибка API: {str(e)}"
            try:
                error_data = resp.json()
                if "error" in error_data:
                    error_msg = error_data["error"].get("message", error_msg)
            except:
                pass
            if "context" in error_msg.lower() or "maximum context length" in error_msg.lower() or "token limit" in error_msg.lower():
                raise RuntimeError("ContextOverflowError")
            raise APIError(error_msg)
        except Exception as e:
            raise APIError(f"Ошибка обработки ответа: {str(e)}")

    def embeddings(self, text: str) -> List[float]:
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.emb_model, "input": text}
        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except requests.RequestException as e:
            error_msg = f"Ошибка API: {str(e)}"
            try:
                error_data = resp.json()
                if "error" in error_data:
                    error_msg = error_data["error"].get("message", error_msg)
            except:
                pass
            if "context" in error_msg.lower() or "maximum context length" in error_msg.lower() or "token limit" in error_msg.lower():
                raise RuntimeError("ContextOverflowError")
            raise APIError(error_msg)
        except Exception as e:
            raise APIError(f"Ошибка обработки эмбеддингов: {str(e)}")