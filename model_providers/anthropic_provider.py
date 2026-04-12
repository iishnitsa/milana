# anthropic_provider.py
import requests
import json
import traceback
import time
import tiktoken
from typing import Optional, Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
from cross_gpt import let_log  # добавлен импорт для логирования

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
        'emb': 'sentence-transformers/all-MiniLM-L6-v2',
        'token': None
    }
    
    for part in connection_string.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            if key in params:
                params[key] = value

    # Проверка: модель должна быть известна
    if params['chat'] not in AnthropicClient.MODEL_LIMITS:
        return False, token_limit, tags, f"Неподдерживаемая модель: {params['chat']}. Доступные: {list(AnthropicClient.MODEL_LIMITS.keys())}"

    client = AnthropicClient(
        api_key=params['token'],
        timeout=timeout,
        chat_model=params['chat']
    )

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
        return False, token_limit, tags, str(e)

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

def ask_model_chat(generation_params: Dict[str, Any]) -> Dict[str, Any]:
    global client, token_limit
    if client is None:
        raise RuntimeError("Клиент Anthropic не инициализирован")
    try:
        messages = generation_params.get("messages", [])
        if not messages:
            raise ValueError("Нет сообщений для чата")
        
        total_tokens = 0
        for msg in messages:
            total_tokens += client.count_tokens(msg.get("content", ""))
        if total_tokens >= token_limit:
            raise RuntimeError("ContextOverflowError")
        
        response_text = client.chat(messages, generation_params)
        
        return {
            "id": f"chatcmpl-{client.chat_model}",
            "choices": [{"message": {"role": "assistant", "content": response_text}, "finish_reason": "stop", "index": 0}],
            "usage": {
                "prompt_tokens": total_tokens,
                "completion_tokens": client.count_tokens(response_text),
                "total_tokens": total_tokens + client.count_tokens(response_text)
            },
            "model": client.chat_model
        }
    except RuntimeError as e:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка чата: {str(e)}")

def create_embeddings(text: str) -> List[float]:
    global emb_model, emb_token_limit
    if emb_model is None:
        raise RuntimeError("Модель для эмбеддингов не указана")
    
    try:
        model = SentenceTransformer(emb_model)
        if len(text.split()) * 1.4 > emb_token_limit:
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

    def _handle_http_error(self, resp, attempt, max_retries):
        """Обработка HTTP ошибок Anthropic."""
        MAX_QUOTA_WAIT = 5 * 60 * 60  # 5 часов

        if resp.status_code == 429:
            retry_after = resp.headers.get('retry-after') or resp.headers.get('Retry-After')
            wait_time = None
            if retry_after and retry_after.isdigit():
                wait_time = int(retry_after)
                # Если время ожидания больше 5 часов – исключение
                if wait_time > MAX_QUOTA_WAIT:
                    raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                let_log(f"Anthropic 429 with Retry-After={wait_time}s. Waiting...")
                time.sleep(wait_time)
                return True
            try:
                err_data = resp.json()
                err_msg = str(err_data).lower()
                if 'insufficient_quota' in err_msg or 'balance' in err_msg:
                    raise RuntimeError("balance end")
                # Некоторые квотные ошибки могут не иметь Retry-After, но содержать текст о лимите сессии/недели
                if 'session limit' in err_msg:
                    wait_time = 5 * 60 * 60  # 5 часов
                elif 'weekly limit' in err_msg:
                    wait_time = 7 * 24 * 60 * 60  # 7 дней
                if wait_time is not None:
                    if wait_time > MAX_QUOTA_WAIT:
                        raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                    let_log(f"Anthropic квотная ошибка: ожидание {wait_time}с...")
                    time.sleep(wait_time)
                    return True
            except:
                pass
            # Обычный rate limit
            if attempt < max_retries:
                wait_time = min(60, 2 ** attempt)
                let_log(f"Anthropic 429 (Rate Limit). Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                return True
            else:
                resp.raise_for_status()
        elif resp.status_code == 402:
            raise RuntimeError("balance end")
        elif resp.status_code == 400:
            try:
                err_data = resp.json()
                err_msg = str(err_data).lower()
                if 'context' in err_msg or 'token' in err_msg or 'length' in err_msg:
                    raise RuntimeError("ContextOverflowError")
            except:
                pass
            resp.raise_for_status()
        elif 500 <= resp.status_code < 600:
            if attempt < max_retries:
                wait_time = min(60, 2 ** attempt)
                let_log(f"Anthropic {resp.status_code} Server Error. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                return True
            else:
                resp.raise_for_status()
        return False

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

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                if self._handle_http_error(r, attempt, max_retries):
                    continue
                r.raise_for_status()
                return r.json()["content"][0]["text"]
            except RuntimeError as e:
                raise
            except requests.RequestException as e:
                msg = str(e).lower()
                if "context" in msg or "token" in msg or "limit" in msg:
                    raise RuntimeError("ContextOverflowError")
                if attempt == max_retries:
                    raise ConnectionError(f"Ошибка генерации: {str(e)}")
                time.sleep(2 ** attempt)
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

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                if self._handle_http_error(r, attempt, max_retries):
                    continue
                r.raise_for_status()
                return r.json()["content"][0]["text"]
            except RuntimeError as e:
                raise
            except requests.RequestException as e:
                msg = str(e).lower()
                if "context" in msg or "token" in msg or "limit" in msg:
                    raise RuntimeError("ContextOverflowError")
                if attempt == max_retries:
                    raise ConnectionError(f"Ошибка чата: {str(e)}")
                time.sleep(2 ** attempt)
            except Exception as e:
                raise APIError(f"Ошибка обработки ответа: {str(e)}")