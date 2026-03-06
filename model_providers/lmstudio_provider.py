import requests
import traceback
from typing import Optional, Dict, Any, List, Tuple

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

        # Получаем лимиты от сервера LM Studio
        models_info = client.list_models()
        if models_info:
            # Поиск лимита для чат-модели
            if client.chat_model:
                for model in models_info:
                    if model["id"] == client.chat_model:
                        token_limit = model.get("context_length", token_limit)
                        break
            # Поиск лимита для модели эмбеддингов (если она отдельная)
            if client.emb_model:
                for model in models_info:
                    if model["id"] == client.emb_model:
                        emb_token_limit = model.get("context_length", emb_token_limit)
                        break

        return True, token_limit, tags

    except Exception as e:
        traceback.print_exc()
        return False, token_limit, tags

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

        # Ollama для эмбеддингов
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url.rstrip('/')
        self.ollama_emb_model = ollama_emb_model
        self.ollama_session = requests.Session() if use_ollama else None

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

        try:
            resp = requests.post(
                url, json=payload, headers=self.headers, auth=self.auth, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"]
        except requests.HTTPError as e:
            msg = ""
            try:
                msg = resp.json().get("error", {}).get("message", str(e))
            except:
                msg = resp.text

            # Проверка на ошибки контекста [citation:10]
            if any(phrase in msg.lower() for phrase in ["context length", "exceeds context", "token limit"]):
                raise RuntimeError("ContextOverflowError")
            raise RuntimeError(f"API error: {msg}")
        except requests.RequestException as e:
            raise RuntimeError(f"Connection error: {str(e)}")

    def embeddings(self, text: str) -> List[float]:
        # Приоритет: Ollama
        if self.use_ollama and self.ollama_session:
            try:
                return self._ollama_embeddings(text)
            except Exception as e:
                raise RuntimeError(f"Ollama embeddings error: {e}")

        # Затем LM Studio
        if not self.emb_model:
            raise ValueError("Embedding model not specified")

        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.emb_model, "input": text}

        try:
            resp = requests.post(
                url, json=payload, headers=self.headers, auth=self.auth, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except requests.HTTPError as e:
            msg = ""
            try:
                msg = resp.json().get("error", {}).get("message", str(e))
            except:
                msg = resp.text

            if any(phrase in msg.lower() for phrase in ["context length", "exceeds context", "token limit"]):
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

    def list_models(self) -> List[Dict]:
        """Получение списка моделей и их параметров."""
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