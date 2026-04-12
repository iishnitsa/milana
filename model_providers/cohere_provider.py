import requests
import traceback
import time
import re
import json
from typing import Optional, Dict, Any, List, Tuple
from cross_gpt import let_log

# Глобальные переменные состояния
client = None
token_limit = 4096  # Значение по умолчанию, будет обновлено при коннекте
emb_token_limit = 512  # Значение по умолчанию для эмбеддингов (Cohere или Ollama)
tags = {
    "system": None,
    "user": "User: ",
    "assistant": "Assistant: ",
    "end": None
}


def find_context_size(model_data: dict, model_name: str = "неизвестная модель") -> int:
    """
    Автоматическое определение лимита контекста для модели Ollama.
    Скопировано из ollama_provider.py.
    """
    let_log(f"Анализируем модель '{model_name}' для определения контекста (Ollama)")

    model_info = model_data.get('model_info', {})
    direct_keys = [
        'context_length', 'max_position_embeddings', 'max_seq_len', 'n_ctx',
        'llama.context_length', 'gemma2.context_length', 'gemma3.context_length',
        'mistral.context_length', 'qwen2.context_length', 'phi3.context_length'
    ]
    for key in direct_keys:
        if key in model_info:
            try:
                val = int(model_info[key])
                if val > 0:
                    return val
            except (ValueError, TypeError):
                continue

    for key, value in model_info.items():
        if 'context_length' in key.lower() and isinstance(value, (int, float)):
            return int(value)

    parameters = model_data.get('parameters', '')
    if parameters:
        match = re.search(r'num_ctx\s*[=: ]\s*(\d+)', parameters, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r'num_ctx\s+(\d+)', parameters, re.IGNORECASE)
        if match:
            return int(match.group(1))

    model_file = model_data.get('model_file', '')
    if model_file:
        match = re.search(r'num_ctx\s*[=: ]\s*(\d+)', model_file, re.IGNORECASE)
        if match:
            return int(match.group(1))

    possible_paths = [
        ['parameters', 'num_ctx'], ['parameters', 'context_length'],
        ['model_info', 'context_length'], ['model_info', 'max_seq_len'],
        ['model_info', 'n_ctx'], ['model_info', 'gemma3.context_length'],
        ['model_info', 'llama.context_length'], ['details', 'context_length']
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

    context_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 128000, 200000, 262144]
    model_text = json.dumps(model_data)
    found_sizes = sorted([int(num) for num in re.findall(r'\b\d{4,7}\b', model_text)
                          if int(num) in context_sizes], reverse=True)
    if found_sizes:
        return found_sizes[0]

    return 4095


def connect(connection_string: str, timeout: int = 30) -> Tuple[bool, int, Dict[str, Optional[str]]]:
    """
    Подключение к Cohere API.
    Формат: "token=XXX; chat=command-r; emb=embed-english-v3.0; timeout=60; ollama=true; ollama_url=...; ollama_emb_model=..."
    """
    global client, token_limit, emb_token_limit, tags

    params = {
        'token': None,
        'chat': 'command-r',
        'emb': 'embed-english-v3.0',
        'timeout': str(timeout),
        'ollama': 'false',
        'ollama_url': 'http://localhost:11434',
        'ollama_emb_model': 'all-minilm:latest'
    }

    for part in connection_string.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            key, value = key.strip().lower(), value.strip()
            if key in params:
                params[key] = value

    if not params['token']:
        return False, token_limit, tags

    req_timeout = int(params['timeout'])
    use_ollama = params['ollama'].lower() == 'true'

    # Если используем Ollama для эмбеддингов, проверяем доступность модели
    ollama_session = None
    ollama_emb_limit = 512
    if use_ollama:
        ollama_url = params['ollama_url'].rstrip('/')
        ollama_emb_model = params['ollama_emb_model']
        try:
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
            # Проверяем список моделей
            tags_url = f"{ollama_url}/api/tags"
            resp = session.get(tags_url, timeout=req_timeout)
            resp.raise_for_status()
            models_data = resp.json()
            available_models = [m['name'] for m in models_data.get('models', [])]
            if ollama_emb_model not in available_models:
                return False, token_limit, tags, f"Ollama модель эмбеддингов '{ollama_emb_model}' не найдена. Доступные: {available_models}"
            # Получаем контекст для модели эмбеддингов
            show_url = f"{ollama_url}/api/show"
            show_payload = {"name": ollama_emb_model}
            show_resp = session.post(show_url, json=show_payload, timeout=req_timeout)
            if show_resp.status_code == 200:
                model_details = show_resp.json()
                ollama_emb_limit = find_context_size(model_details, ollama_emb_model)
            else:
                ollama_emb_limit = 4095
            session.close()
        except Exception as e:
            let_log(f"Ошибка при проверке Ollama для эмбеддингов: {e}")
            return False, token_limit, tags, f"Ошибка подключения к Ollama: {e}"

    client = CohereClient(
        api_key=params['token'],
        chat_model=params['chat'],
        emb_model=params['emb'],
        timeout=req_timeout,
        use_ollama=use_ollama,
        ollama_url=params['ollama_url'],
        ollama_emb_model=params['ollama_emb_model']
    )
    # Получаем token_limit для чат-модели (всегда из Cohere, даже если ollama)
    try:
        url = f"{client.base_url}/models"
        resp = requests.get(url, headers=client.headers, timeout=req_timeout)
        if resp.status_code == 200:
            models_data = resp.json()
            for model in models_data.get("data", []):
                if model["id"] == client.chat_model:
                    token_limit = model.get("context_length", 256000)
                    break
    except Exception as e:
        let_log(f"Не удалось получить контекст модели Cohere: {e}")
        token_limit = 256000

    # Если используем Ollama, обновляем emb_token_limit
    if use_ollama:
        emb_token_limit = ollama_emb_limit
    else:
        # Для Cohere эмбеддингов – можно попробовать получить лимит из модели, но обычно 512
        emb_token_limit = 512

    return True, token_limit, tags


def disconnect() -> bool:
    global client
    if client:
        client = None
        return True
    return False


def ask_model(generation_params: Dict[str, Any]) -> str:
    global client
    if client is None:
        raise RuntimeError("Cohere client not initialized")
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
        raise RuntimeError("Cohere client not initialized")
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
        raise RuntimeError("Cohere client not initialized")
    try:
        return client.embeddings(text)
    except RuntimeError as e:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Embeddings error: {str(e)}")


class CohereClient:
    def __init__(self,
                 api_key: str,
                 chat_model: str,
                 emb_model: str,
                 timeout: int,
                 use_ollama: bool,
                 ollama_url: str,
                 ollama_emb_model: str):
        self.base_url = "https://api.cohere.ai/compatibility/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.timeout = timeout
        self.chat_model = chat_model
        self.emb_model = emb_model

        self.use_ollama = use_ollama
        self.ollama_url = ollama_url.rstrip('/')
        self.ollama_emb_model = ollama_emb_model
        self.ollama_session = requests.Session() if use_ollama else None

    def _handle_http_error(self, resp, attempt, max_retries, base_backoff=2.0):
        """Общая обработка HTTP ошибок для Cohere API."""
        if resp.status_code == 429:
            retry_after = resp.headers.get('Retry-After')
            if retry_after and retry_after.isdigit():
                wait_time = int(retry_after)
                let_log(f"Cohere 429 with Retry-After={wait_time}s. Waiting...")
                time.sleep(wait_time)
                return True  # retry
            # Пытаемся определить квотную ошибку по тексту
            try:
                err_data = resp.json()
                err_msg = err_data.get('error', {}).get('message', '').lower()
                if 'insufficient_quota' in err_msg or 'exceeded your current quota' in err_msg:
                    raise RuntimeError("balance end")
                # Cohere не использует session/weekly limits, но оставим на будущее
                if 'session limit' in err_msg:
                    wait_time = 5 * 60 * 60
                    let_log(f"Cohere квотный лимит (сессия), ожидание {wait_time}s...")
                    time.sleep(wait_time)
                    return True
                if 'weekly limit' in err_msg:
                    wait_time = 7 * 24 * 60 * 60
                    let_log(f"Cohere квотный лимит (неделя), ожидание {wait_time}s...")
                    time.sleep(wait_time)
                    return True
            except:
                pass
            # Обычный rate limit – экспоненциальный backoff
            if attempt < max_retries:
                wait_time = min(60, (base_backoff ** attempt))
                let_log(f"Cohere 429 (Rate Limit). Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                return True
            else:
                resp.raise_for_status()
        elif resp.status_code == 402:
            raise RuntimeError("balance end")
        elif resp.status_code == 400:
            try:
                err_data = resp.json()
                err_msg = err_data.get('error', {}).get('message', '').lower()
                if 'context' in err_msg or 'token' in err_msg or 'length' in err_msg:
                    raise RuntimeError("ContextOverflowError")
            except:
                pass
            resp.raise_for_status()
        elif 500 <= resp.status_code < 600:
            if attempt < max_retries:
                wait_time = min(60, (base_backoff ** attempt))
                let_log(f"Cohere {resp.status_code} Server Error. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                return True
            else:
                resp.raise_for_status()
        return False  # не retry

    def generate(self, generation_params: Dict[str, Any]) -> str:
        if "prompt" not in generation_params:
            raise ValueError("'prompt' is required")

        url = f"{self.base_url}/completions"
        payload = {
            "model": self.chat_model,
            "prompt": generation_params["prompt"],
            "max_tokens": generation_params.get("max_tokens", 200),
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.75),
            "frequency_penalty": generation_params.get("frequency_penalty", 0.0),
            "presence_penalty": generation_params.get("presence_penalty", 0.0),
            "stop": generation_params.get("stop_sequences", None),
            "stream": False
        }

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
                if self._handle_http_error(resp, attempt, max_retries):
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data['choices'][0]['text']
            except RuntimeError as e:
                raise
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"API error: {str(e)}")
                time.sleep(2 ** attempt)

    def chat(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        if "messages" not in generation_params:
            raise ValueError("'messages' is required for chat")

        url = f"{self.base_url}/chat/completions"
        messages = generation_params["messages"]
        processed_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                processed_messages.append({"role": "developer", "content": msg["content"]})
            else:
                processed_messages.append(msg)

        payload = {
            "model": generation_params.get("model", self.chat_model),
            "messages": processed_messages,
            "max_tokens": generation_params.get("max_tokens", 200),
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.75),
            "frequency_penalty": generation_params.get("frequency_penalty", 0.0),
            "presence_penalty": generation_params.get("presence_penalty", 0.0),
            "stop": generation_params.get("stop", None),
            "stream": False
        }

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
                if self._handle_http_error(resp, attempt, max_retries):
                    continue
                resp.raise_for_status()
                return resp.json()
            except RuntimeError as e:
                raise
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Chat API error: {str(e)}")
                time.sleep(2 ** attempt)

    def embeddings(self, text: str) -> List[float]:
        if self.use_ollama and self.ollama_session:
            try:
                return self._ollama_embeddings(text)
            except Exception as e:
                raise RuntimeError(f"Ollama embeddings error: {e}")

        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.emb_model,
            "input": [text],
            "encoding_format": "float"
        }

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
                if self._handle_http_error(resp, attempt, max_retries):
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data['data'][0]['embedding']
            except RuntimeError as e:
                raise
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Embeddings API error: {str(e)}")
                time.sleep(2 ** attempt)

    def _ollama_embeddings(self, text: str) -> List[float]:
        """
        Запрос эмбеддингов через Ollama с улучшенной обработкой квот и лимитов.
        """
        url = f"{self.ollama_url}/api/embeddings"
        payload = {"model": self.ollama_emb_model, "prompt": text}
        MAX_RETRIES = 20
        MAX_WAIT_TOTAL = 420  # 7 минут
        MAX_QUOTA_WAIT = 5 * 60 * 60  # 5 часов – максимальное допустимое ожидание для квот
        BASE_BACKOFF = 2.0
        start_time = time.time()
        attempt = 1

        while True:
            try:
                elapsed = time.time() - start_time
                if elapsed > MAX_WAIT_TOTAL:
                    raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с)")

                resp = self.ollama_session.post(url, json=payload, timeout=self.timeout)

                # Обработка 429 (Rate Limit / Quota)
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
                        # Квотная ошибка: если ожидание > MAX_QUOTA_WAIT – сразу исключение
                        if wait_time > MAX_QUOTA_WAIT:
                            raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает допустимые {MAX_QUOTA_WAIT}с")
                        if wait_time > MAX_WAIT_TOTAL:
                            raise RuntimeError(f"Лимит квоты требует ожидания {wait_time}с, что превышает общий лимит {MAX_WAIT_TOTAL}с")
                        let_log(f"Ollama эмбеддинги: квотный лимит, ожидание {wait_time:.2f} с...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue

                    # Обычный rate limit – экспоненциальный backoff
                    if attempt < MAX_RETRIES:
                        wait_time = BASE_BACKOFF ** attempt
                        remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                        if wait_time > remaining:
                            wait_time = remaining
                        if wait_time < 0.1:
                            wait_time = 0.1
                        let_log(f"Ollama эмбеддинги: HTTP 429 (Rate Limit), повтор через {wait_time:.2f} с...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                    else:
                        resp.raise_for_status()

                # Обработка ошибок контекста (500 с определённым текстом)
                if resp.status_code == 500:
                    error_text = resp.text.lower()
                    context_error_keywords = [
                        'the input length exceeds the context length',
                        'input length exceeds',
                        'context length',
                        'exceeds context',
                        'token limit exceeded'
                    ]
                    if any(kw in error_text for kw in context_error_keywords):
                        raise RuntimeError("ContextOverflowError")

                # Временные серверные ошибки (5xx)
                if resp.status_code in (500, 502, 503, 504):
                    if attempt < MAX_RETRIES:
                        wait_time = BASE_BACKOFF ** attempt
                        remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                        if wait_time > remaining:
                            wait_time = remaining
                        if wait_time < 0.1:
                            wait_time = 0.1
                        let_log(f"Ollama эмбеддинги: HTTP {resp.status_code}, повтор через {wait_time:.2f} с...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                    else:
                        resp.raise_for_status()

                resp.raise_for_status()
                return resp.json()['embedding']

            except requests.HTTPError as e:
                # Дополнительная проверка на контекст в тексте ответа
                if hasattr(e.response, 'text'):
                    msg = e.response.text.lower()
                    if any(phrase in msg for phrase in ["exceeds context", "context length", "token limit"]):
                        raise RuntimeError("ContextOverflowError")
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining:
                        wait_time = remaining
                    if wait_time < 0.1:
                        wait_time = 0.1
                    let_log(f"Ollama эмбеддинги: HTTP ошибка {resp.status_code}, повтор через {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                raise
            except requests.exceptions.ConnectionError as e:
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF ** attempt
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining:
                        wait_time = remaining
                    if wait_time < 0.1:
                        wait_time = 0.1
                    let_log(f"Ollama эмбеддинги: ошибка соединения, повтор через {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                raise RuntimeError(f"Ошибка соединения: {e}")
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Ошибка запроса эмбеддингов Ollama: {e}")