import requests
import traceback
import time
import random
from typing import Optional, Dict, Any, List, Tuple
from huggingface_hub import InferenceClient

# Глобальные переменные
client = None
token_limit = 32768
emb_token_limit = 225
tags = {
    "system": "[INST] <<SYS>>\n",
    "user": "\n<</SYS>>\n[/INST] ",
    "assistant": " ",
    "end": " [/INST]"
}

# Константы для повторных попыток
MAX_RETRIES = 10
BASE_BACKOFF = 2.0
MAX_WAIT_TOTAL = 300
MAX_QUOTA_WAIT = 5 * 60 * 60  # 5 часов в секундах

def connect(connection_string: str, timeout: int = 30) -> Tuple[bool, int, Dict[str, Optional[str]]]:
    global client, token_limit, emb_token_limit, tags

    params = {
        'token': None,
        'chat': None,
        'emb': None,
        'timeout': str(timeout),
        'ollama': 'false',
        'ollama_url': 'http://localhost:11434',
        'ollama_emb_model': 'all-minilm:latest'
    }

    for part in connection_string.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            key, value = key.strip().lower(), value.strip()
            if key in params and value:
                params[key] = value

    if not params['token']:
        return False, token_limit, tags

    req_timeout = int(params['timeout'])

    # Настройка тегов на основе имени модели
    if params['chat']:
        model_lower = params['chat'].lower()
        if 'mistral' in model_lower:
            tags.update({"system": "<s>[INST] ", "user": " [/INST]", "assistant": " ", "end": "</s>"})
        elif 'gemma' in model_lower:
            tags.update({"system": "<start_of_turn>model\n", "user": "<start_of_turn>user\n",
                         "assistant": "<start_of_turn>model\n", "end": "<end_of_turn>\n"})

    # Проверка существования модели через HEAD-запрос к HF API
    if params['chat']:
        try:
            api_url = f"https://huggingface.co/api/models/{params['chat']}"
            resp = requests.head(api_url, timeout=req_timeout)
            if resp.status_code != 200:
                return False, token_limit, tags, f"Модель '{params['chat']}' не найдена на Hugging Face"
        except Exception as e:
            return False, token_limit, tags, f"Ошибка проверки модели: {e}"

    client = HuggingFaceClient(
        api_token=params['token'],
        chat_model=params['chat'],
        emb_model=params['emb'],
        timeout=req_timeout,
        use_ollama=(params['ollama'].lower() == 'true'),
        ollama_url=params['ollama_url'],
        ollama_emb_model=params['ollama_emb_model']
    )

    if params['chat']:
        try:
            api_url = f"https://huggingface.co/api/models/{params['chat']}"
            resp = requests.get(api_url, timeout=req_timeout)
            if resp.status_code == 200:
                model_info = resp.json()
                config = model_info.get('config', {})
                token_limit = config.get('max_position_embeddings', 32768)
                let_log(f"Модель '{params['chat']}': контекст {token_limit} (из max_position_embeddings)")
            else:
                token_limit = 32768
                let_log(f"Не удалось получить конфиг модели '{params['chat']}', контекст по умолчанию {token_limit}")
        except Exception as e:
            let_log(f"Не удалось получить контекст модели HF: {e}")
            token_limit = 32768
    
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
        raise RuntimeError("Hugging Face client not initialized")
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
        raise RuntimeError("Hugging Face client not initialized")
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
        raise RuntimeError("Hugging Face client not initialized")
    try:
        return client.embeddings(text)
    except RuntimeError as e:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Embeddings error: {str(e)}")

class HuggingFaceClient:
    def __init__(self,
                 api_token: str,
                 chat_model: Optional[str],
                 emb_model: Optional[str],
                 timeout: int,
                 use_ollama: bool,
                 ollama_url: str,
                 ollama_emb_model: str):
        self.chat_model = chat_model
        self.emb_model = emb_model
        self.timeout = timeout

        self.hf_client = InferenceClient(model=chat_model, token=api_token, timeout=timeout) if chat_model else None
        self.hf_emb_client = InferenceClient(model=emb_model, token=api_token, timeout=timeout) if emb_model and not use_ollama else None

        self.use_ollama = use_ollama
        self.ollama_url = ollama_url.rstrip('/')
        self.ollama_emb_model = ollama_emb_model
        self.ollama_session = requests.Session() if use_ollama else None

    def _call_with_retry(self, func, *args, **kwargs):
        """Универсальный метод с экспоненциальным backoff и обработкой 429/402"""
        start_time = time.time()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                elapsed = time.time() - start_time
                if elapsed > MAX_WAIT_TOTAL:
                    raise RuntimeError(f"Превышено общее время ожидания ({MAX_WAIT_TOTAL} с)")

                err_str = str(e).lower()
                # Проверка на ошибку контекста
                if any(phrase in err_str for phrase in ["context", "token length", "max_length", "context_length_exceeded"]):
                    raise RuntimeError("ContextOverflowError")
                # Проверка на ошибку баланса (402 или insufficient_quota)
                if "402" in err_str or "insufficient_quota" in err_str or "payment required" in err_str:
                    raise RuntimeError("balance end")
                
                # Проверка на 429 (rate limit или quota)
                if "429" in err_str:
                    # Пытаемся извлечь Retry-After
                    retry_after = None
                    if hasattr(e, 'response') and e.response is not None:
                        retry_after = e.response.headers.get('Retry-After')
                    if retry_after and retry_after.isdigit():
                        wait_time = int(retry_after)
                    elif "session limit" in err_str:
                        wait_time = 5 * 60 * 60
                    elif "weekly limit" in err_str:
                        wait_time = 7 * 24 * 60 * 60
                    else:
                        wait_time = BASE_BACKOFF ** attempt
                    
                    # ИЗМЕНЕНИЕ: если время ожидания превышает MAX_QUOTA_WAIT (5 часов), райзим balance end
                    if wait_time > MAX_QUOTA_WAIT:
                        raise RuntimeError("balance end")
                    
                    # Если это не квотная ошибка, но ожидание > 5 минут, тоже считаем balance end (как в старом коде)
                    if wait_time > 300:  # больше 5 минут
                        raise RuntimeError("balance end")
                    
                    remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                    if wait_time > remaining:
                        wait_time = remaining
                    if wait_time < 0.1:
                        wait_time = 0.1
                    let_log(f"HF API 429, попытка {attempt}, ожидание {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue
                
                # Остальные ошибки — пробуем повторить с экспоненциальной задержкой
                if attempt < MAX_RETRIES:
                    wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                    let_log(f"Ошибка HF API: {e}, повтор через {wait_time:.2f} с...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Ошибка после {MAX_RETRIES} попыток: {e}")
        raise RuntimeError("Не удалось выполнить запрос")

    def generate(self, generation_params: Dict[str, Any]) -> str:
        if not self.hf_client:
            raise RuntimeError("Chat model not specified or client not initialized")
        if "prompt" not in generation_params:
            raise ValueError("'prompt' is required")

        prompt = generation_params["prompt"]
        gen_kwargs = {
            "max_new_tokens": generation_params.get("max_tokens", 200),
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.9),
            "repetition_penalty": generation_params.get("repeat_penalty", 1.1),
            "do_sample": True
        }

        def _generate():
            return self.hf_client.text_generation(prompt, **gen_kwargs).strip()

        return self._call_with_retry(_generate)

    def chat(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.hf_client:
            raise RuntimeError("Chat model not specified or client not initialized")
        if "messages" not in generation_params:
            raise ValueError("'messages' is required for chat")

        messages = generation_params["messages"]
        model = generation_params.get("model", self.chat_model)
        
        chat_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": generation_params.get("max_tokens", 500),
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.9),
            "stop": generation_params.get("stop", None),
        }
        if "frequency_penalty" in generation_params:
            chat_kwargs["frequency_penalty"] = generation_params["frequency_penalty"]
        if "presence_penalty" in generation_params:
            chat_kwargs["presence_penalty"] = generation_params["presence_penalty"]
        if "seed" in generation_params:
            chat_kwargs["seed"] = generation_params["seed"]

        def _chat():
            response = self.hf_client.chat.completions.create(**chat_kwargs)
            return {
                "id": response.id,
                "choices": [
                    {
                        "message": {
                            "content": choice.message.content,
                            "role": choice.message.role
                        },
                        "finish_reason": choice.finish_reason,
                        "index": choice.index
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                } if response.usage else {},
                "model": response.model,
                "created": response.created
            }

        return self._call_with_retry(_chat)

    def embeddings(self, text: str) -> List[float]:
        if self.use_ollama and self.ollama_session:
            try:
                return self._ollama_embeddings(text)
            except Exception as e:
                raise RuntimeError(f"Ollama embeddings error: {e}")

        if not self.hf_emb_client:
            raise RuntimeError("No embeddings backend available")

        def _embed():
            output = self.hf_emb_client.feature_extraction(text)
            if isinstance(output, list):
                if output and isinstance(output[0], list):
                    return [float(x) for x in output[0]]
                return [float(x) for x in output]
            return output.tolist()

        return self._call_with_retry(_embed)

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
                    # ИЗМЕНЕНИЕ: если время ожидания превышает MAX_QUOTA_WAIT (5 часов), райзим balance end
                    if wait_time is not None and wait_time > MAX_QUOTA_WAIT:
                        raise RuntimeError("balance end")
                    if wait_time is not None and wait_time <= MAX_QUOTA_WAIT:
                        let_log(f"Ollama эмбеддинги: квотная 429, ожидание {wait_time} с")
                        time.sleep(wait_time)
                        continue
                    # rate limit
                    if attempt < MAX_RETRIES:
                        wait_time = BASE_BACKOFF ** attempt
                        remaining = MAX_WAIT_TOTAL - (time.time() - start_time)
                        if wait_time > remaining:
                            wait_time = remaining
                        if wait_time < 0.1:
                            wait_time = 0.1
                        let_log(f"Ollama эмбеддинги: 429 rate limit, повтор через {wait_time} с")
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
                time.sleep(min(wait_time, 60))
            except requests.exceptions.ConnectionError as e:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Ошибка соединения: {e}")
                time.sleep(BASE_BACKOFF ** attempt)
        raise RuntimeError("Не удалось получить эмбеддинги от Ollama")