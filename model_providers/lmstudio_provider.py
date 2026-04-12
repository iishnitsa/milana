import requests
import traceback
import time
import random
from typing import Optional, Dict, Any, List, Tuple
from cross_gpt import let_log  # добавлен импорт для логирования

# Глобальные переменные состояния
client = None
token_limit = 4096
emb_token_limit = 4096
tags = {
    "system": None,
    "user": None,
    "assistant": None,
    "end": None
}

# Константы для повторных попыток
MAX_RETRIES = 10
BASE_BACKOFF = 2.0
MAX_WAIT_TOTAL = 300
MAX_QUOTA_WAIT = 5 * 60 * 60  # 5 часов в секундах

def connect(connection_string: str, timeout: int = 30) -> Tuple[bool, int, Dict[str, Optional[str]]]:
    global client, token_limit, emb_token_limit, tags

    params = {
        'host': 'http://localhost',
        'port': 1234,
        'chat': None,
        'emb': None,
        'user': None,
        'password': None,
        'timeout': str(timeout),
        'ollama': 'false',
        'ollama_url': 'http://localhost:11434',
        'ollama_emb_model': 'all-minilm:latest'
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
                elif key in ('chat', 'emb', 'user', 'password', 'timeout', 'ollama', 'ollama_url', 'ollama_emb_model'):
                    params[key] = value

        req_timeout = int(params['timeout'])

        client = LMStudioClient(
            host=params['host'],
            port=params['port'],
            chat_model=params['chat'],
            emb_model=params['emb'],
            username=params['user'],
            password=params['password'],
            timeout=req_timeout,
            use_ollama=(params['ollama'].lower() == 'true'),
            ollama_url=params['ollama_url'],
            ollama_emb_model=params['ollama_emb_model']
        )

        models_info = client.list_models()
        if models_info:
            # Проверка существования чат-модели
            if client.chat_model:
                found = False
                for model in models_info:
                    if model["id"] == client.chat_model:
                        token_limit = model.get("context_length", token_limit)
                        found = True
                        break
                if not found:
                    available_ids = [m["id"] for m in models_info]
                    return False, token_limit, tags, f"Запрошенная чат-модель '{client.chat_model}' не найдена. Доступные: {available_ids}"
            # Проверка существования эмбеддинг-модели
            if client.emb_model:
                found = False
                for model in models_info:
                    if model["id"] == client.emb_model:
                        emb_token_limit = model.get("context_length", emb_token_limit)
                        found = True
                        break
                if not found and client.emb_model:
                    available_ids = [m["id"] for m in models_info]
                    let_log(f"Модель для эмбеддингов '{client.emb_model}' не найдена, будет использована чат-модель")
                    client.emb_model = client.chat_model  # fallback
                    emb_token_limit = token_limit

        return True, token_limit, tags

    except Exception as e:
        traceback.print_exc()
        return False, token_limit, tags, str(e)

def disconnect() -> bool:
    global client
    if client:
        client = None
        return True
    return False

def ask_model(generation_params: Dict[str, Any]) -> str:
    global client
    if client is None:
        raise RuntimeError("LM Studio client not initialized")
    try:
        return client.generate(generation_params)
    except RuntimeError as e:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Generation error: {str(e)}")

def ask_model_chat(generation_params: Dict[str, Any]) -> Dict[str, Any]:
    global client
    if client is None:
        raise RuntimeError("LM Studio client not initialized")
    try:
        return client.chat(generation_params)
    except RuntimeError as e:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Chat error: {str(e)}")

def create_embeddings(text: str) -> List[float]:
    global client
    if client is None:
        raise RuntimeError("LM Studio client not initialized")
    try:
        return client.embeddings(text)
    except RuntimeError as e:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Embeddings error: {str(e)}")

class LMStudioClient:
    def __init__(self,
                 host: str,
                 port: int,
                 chat_model: Optional[str],
                 emb_model: Optional[str],
                 username: Optional[str],
                 password: Optional[str],
                 timeout: int,
                 use_ollama: bool,
                 ollama_url: str,
                 ollama_emb_model: str):
        self.base_url = f"{host}:{port}"
        self.headers = {"Content-Type": "application/json"}
        self.auth = (username, password) if username and password else None
        self.timeout = timeout
        self.chat_model = chat_model
        self.emb_model = emb_model
        self.tags = {"system": None, "user": None, "assistant": None, "end": None}

        self.use_ollama = use_ollama
        self.ollama_url = ollama_url.rstrip('/')
        self.ollama_emb_model = ollama_emb_model
        self.ollama_session = requests.Session() if use_ollama else None

    def _request_with_retry(self, method: str, url: str, **kwargs):
        start_time = time.time()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.request(method, url, headers=self.headers, auth=self.auth, timeout=self.timeout, **kwargs)
                if resp.status_code == 429:
                    retry_after = resp.headers.get('Retry-After')
                    wait_time = None
                    if retry_after and retry_after.isdigit():
                        wait_time = int(retry_after)
                    else:
                        try:
                            err_data = resp.json()
                            err_msg = err_data.get('error', {}).get('message', '').lower()
                            if 'session limit' in err_msg:
                                wait_time = 5 * 60 * 60  # 5 часов
                            elif 'weekly limit' in err_msg:
                                wait_time = 7 * 24 * 60 * 60  # 7 дней
                            elif 'insufficient_quota' in err_msg:
                                raise RuntimeError("balance end")
                        except:
                            pass
                    if wait_time is not None:
                        # Если квота требует ожидания более 5 часов – сразу ошибка
                        if wait_time > MAX_QUOTA_WAIT:
                            raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                        # Иначе ждём указанное время (даже если оно больше MAX_WAIT_TOTAL)
                        let_log(f"LM Studio 429 квота, ожидание {wait_time} с")
                        time.sleep(wait_time)
                        continue
                    # Обычный rate limit – экспоненциальный backoff с учётом MAX_WAIT_TOTAL
                    if attempt < MAX_RETRIES:
                        wait_time = min(60, BASE_BACKOFF ** attempt + random.random())
                        remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                        if wait_time > remaining:
                            wait_time = max(remaining, 0.1)
                        let_log(f"LM Studio 429 rate limit, повтор через {wait_time:.2f} с")
                        time.sleep(wait_time)
                        continue
                if resp.status_code == 402:
                    raise RuntimeError("balance end")
                if resp.status_code >= 500:
                    if attempt < MAX_RETRIES:
                        wait_time = min(60, BASE_BACKOFF ** attempt)
                        remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                        if wait_time > remaining:
                            wait_time = max(remaining, 0.1)
                        time.sleep(wait_time)
                        continue
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                err_str = str(e).lower()
                if any(phrase in err_str for phrase in ["context length", "exceeds context", "token limit"]):
                    raise RuntimeError("ContextOverflowError")
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"HTTP error: {e}")
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining:
                    wait_time = max(remaining, 0.1)
                time.sleep(wait_time)
            except requests.RequestException as e:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Connection error: {e}")
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining:
                    wait_time = max(remaining, 0.1)
                time.sleep(wait_time)
            except RuntimeError:
                raise
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining:
                    wait_time = max(remaining, 0.1)
                time.sleep(wait_time)
        raise RuntimeError("Max retries exceeded")

    def generate(self, generation_params: Dict[str, Any]) -> str:
        if "prompt" not in generation_params:
            raise ValueError("'prompt' is required")
        if not self.chat_model:
            raise ValueError("Chat model not specified")

        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": self.chat_model,
            "prompt": generation_params["prompt"],
            "max_tokens": generation_params.get("max_tokens", 200),
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.95),
            "repetition_penalty": generation_params.get("repeat_penalty", 1.1),
            "stream": False
        }
        data = self._request_with_retry("POST", url, json=payload)
        return data["choices"][0]["text"]

    def chat(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        if "messages" not in generation_params:
            raise ValueError("'messages' is required for chat")
        if not self.chat_model:
            raise ValueError("Chat model not specified")

        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": generation_params.get("model", self.chat_model),
            "messages": generation_params["messages"],
            "max_tokens": generation_params.get("max_tokens", 200),
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.95),
            "frequency_penalty": generation_params.get("frequency_penalty", 0.0),
            "presence_penalty": generation_params.get("presence_penalty", 0.0),
            "stop": generation_params.get("stop", None),
            "stream": False
        }
        if "repetition_penalty" in generation_params:
            payload["repetition_penalty"] = generation_params["repetition_penalty"]
        if "logit_bias" in generation_params:
            payload["logit_bias"] = generation_params["logit_bias"]
        if "user" in generation_params:
            payload["user"] = generation_params["user"]

        return self._request_with_retry("POST", url, json=payload)

    def embeddings(self, text: str) -> List[float]:
        if self.use_ollama and self.ollama_session:
            try:
                return self._ollama_embeddings(text)
            except Exception as e:
                raise RuntimeError(f"Ollama embeddings error: {e}")

        if not self.emb_model:
            raise ValueError("Embedding model not specified")

        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.emb_model, "input": text}
        data = self._request_with_retry("POST", url, json=payload)
        return data["data"][0]["embedding"]

    def _ollama_embeddings(self, text: str) -> List[float]:
        url = f"{self.ollama_url}/api/embeddings"
        payload = {"model": self.ollama_emb_model, "prompt": text}
        start_time = time.time()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                elapsed = time.time() - start_time
                if elapsed > MAX_WAIT_TOTAL:
                    raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с)")
                resp = self.ollama_session.post(url, json=payload, timeout=self.timeout)
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
                        # Если квота требует ожидания более 5 часов – сразу ошибка
                        if wait_time > MAX_QUOTA_WAIT:
                            raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                        let_log(f"Ollama эмбеддинги: квотная 429, ожидание {wait_time} с")
                        time.sleep(wait_time)
                        continue
                    # Обычный rate limit
                    if attempt < MAX_RETRIES:
                        wait_time = BASE_BACKOFF ** attempt
                        remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                        if wait_time > remaining:
                            wait_time = max(remaining, 0.1)
                        let_log(f"Ollama эмбеддинги: 429 rate limit, повтор через {wait_time:.2f} с")
                        time.sleep(wait_time)
                        continue
                resp.raise_for_status()
                return resp.json()['embedding']
            except requests.HTTPError as e:
                msg = resp.text.lower()
                if any(phrase in msg for phrase in ["exceeds context", "context length", "token limit"]):
                    raise RuntimeError("ContextOverflowError")
                if attempt == MAX_RETRIES:
                    raise
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining:
                    wait_time = max(remaining, 0.1)
                time.sleep(wait_time)
            except requests.exceptions.ConnectionError as e:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Ошибка соединения: {e}")
                wait_time = BASE_BACKOFF ** attempt
                remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                if wait_time > remaining:
                    wait_time = max(remaining, 0.1)
                time.sleep(wait_time)
        raise RuntimeError("Не удалось получить эмбеддинги от Ollama")

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
        except:
            return []