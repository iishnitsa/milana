import requests
import traceback
from typing import Optional, Dict, Any, List, Tuple

# Глобальные переменные состояния
client = None
token_limit = 4096  # Значение по умолчанию, будет обновлено при коннекте
emb_token_limit = 512  # Значение по умолчанию
tags = {
    "system": None,
    "user": "User: ",
    "assistant": "Assistant: ",
    "end": None
}

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

    # Таймаут для requests
    req_timeout = int(params['timeout'])

    client = CohereClient(
        api_key=params['token'],
        chat_model=params['chat'],
        emb_model=params['emb'],
        timeout=req_timeout,
        use_ollama=(params['ollama'].lower() == 'true'),
        ollama_url=params['ollama_url'],
        ollama_emb_model=params['ollama_emb_model']
    )
    try:
        # Используем OpenAI-совместимый эндпоинт для списка моделей
        url = f"{client.base_url}/models"
        resp = requests.get(url, headers=client.headers, timeout=req_timeout)
        if resp.status_code == 200:
            models_data = resp.json()
            for model in models_data.get("data", []):
                if model["id"] == client.chat_model:
                    # В ответе OpenAI-совместимого API нет context_length, 
                    # но Cohere может отдавать его в поле "context_length"
                    token_limit = model.get("context_length", 256000)
                    break
    except Exception as e:
        let_log(f"Не удалось получить контекст модели: {e}")
        # Fallback на значение по умолчанию
        token_limit = 256000

    return True, token_limit, tags

def disconnect() -> bool:
    global client
    if client:
        client = None
        return True
    return False

def ask_model(generation_params: Dict[str, Any]) -> str:
    """Используется для completion (prompt-based)"""
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
    """
    Запрос к чат-модели Cohere через OpenAI-совместимый эндпоинт.
    Ожидает параметры:
    - messages: список сообщений [{"role": "user", "content": "..."}, ...]
    - model: опционально, модель для использования
    - temperature, max_tokens и др. параметры генерации
    """
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
        # Используем OpenAI-совместимый эндпоинт Cohere [citation:1]
        self.base_url = "https://api.cohere.ai/compatibility/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.timeout = timeout
        self.chat_model = chat_model
        self.emb_model = emb_model

        # Настройки для Ollama (фоллбэк для эмбеддингов)
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url.rstrip('/')
        self.ollama_emb_model = ollama_emb_model
        self.ollama_session = requests.Session() if use_ollama else None

    def generate(self, generation_params: Dict[str, Any]) -> str:
        """Запрос к /completions (для обратной совместимости)"""
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

        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data['choices'][0]['text']
        except requests.HTTPError as e:
            msg = ""
            try:
                error_data = resp.json()
                msg = error_data.get('error', {}).get('message', str(e))
            except:
                msg = str(e)

            if any(phrase in msg.lower() for phrase in [
                "too many tokens", "size limit exceeded", "context length"
            ]):
                raise RuntimeError("ContextOverflowError")
            raise RuntimeError(f"API error: {msg}")
        except requests.RequestException as e:
            raise RuntimeError(f"Connection error: {str(e)}")

    def chat(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Запрос к /chat/completions (OpenAI-совместимый чат-эндпоинт) [citation:1][citation:4]
        Возвращает полный ответ от API.
        """
        if "messages" not in generation_params:
            raise ValueError("'messages' is required for chat")

        url = f"{self.base_url}/chat/completions"
        messages = generation_params["messages"]

        # Преобразование system message в developer role для Cohere [citation:1]
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

        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            msg = ""
            try:
                error_data = resp.json()
                msg = error_data.get('error', {}).get('message', str(e))
            except:
                msg = str(e)

            if any(phrase in msg.lower() for phrase in [
                "too many tokens", "size limit exceeded", "context length"
            ]):
                raise RuntimeError("ContextOverflowError")
            raise RuntimeError(f"Chat API error: {msg}")
        except requests.RequestException as e:
            raise RuntimeError(f"Connection error: {str(e)}")

    def embeddings(self, text: str) -> List[float]:
        """Создание эмбеддингов (без изменений)"""
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

        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data['data'][0]['embedding']
        except requests.HTTPError as e:
            msg = ""
            try:
                error_data = resp.json()
                msg = error_data.get('error', {}).get('message', str(e))
            except:
                msg = str(e)

            if "too many tokens" in msg.lower():
                raise RuntimeError("ContextOverflowError")
            raise RuntimeError(f"Embeddings API error: {msg}")
        except requests.RequestException as e:
            raise RuntimeError(f"Connection error: {str(e)}")

    def _ollama_embeddings(self, text: str) -> List[float]:
        url = f"{self.ollama_url}/api/embeddings"
        payload = {"model": self.ollama_emb_model, "prompt": text}
        try:
            resp = self.ollama_session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()['embedding']
        except requests.HTTPError as e:
            msg = resp.text.lower()
            if any(phrase in msg for phrase in ["exceeds context", "context length", "token limit"]):
                raise RuntimeError("ContextOverflowError")
            raise