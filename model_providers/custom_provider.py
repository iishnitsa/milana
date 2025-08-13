import traceback
import tiktoken
from typing import Optional, Dict, Any, List, Tuple
from milana_model_client import MilanaModelClient

# Глобальные переменные
client: Optional[MilanaModelClient] = None
token_limit = 4096  # Общий лимит токенов для чата и эмбеддингов
model_name = "gpt-3.5-turbo"

tags = {
    "system": "<|im_start|>system",
    "user": "<|im_start|>user",
    "assistant": "<|im_start|>assistant",
    "end": "<|im_end|>"
}

def connect(connection_string: str, timeout: int = 30) -> Tuple[bool, int, Dict[str, str]]:
    global client, token_limit, model_name

    params = {
        'host': '127.0.0.1',
        'port': 65432,
        'token': None
    }

    try:
        for part in connection_string.split(';'):
            if '=' in part:
                key, value = part.split('=', 1)
                params[key.strip().lower()] = value.strip()

        client = MilanaModelClient(
            server_ip=params['host'],
            port=int(params['port']),
            api_key=params['token']
        )

        ok, _, code = client.check_connection()
        if ok and code == 122:
            return True, token_limit, tags
        return False, token_limit, tags
    except Exception as e:
        traceback.print_exc()
        return False, token_limit, tags

def disconnect() -> bool:
    global client
    if client is not None:
        client.close_connection()
        client = None
        return True
    return False

def ask_model(generation_params: Dict[str, Any]) -> str:
    global client, token_limit
    if client is None:
        raise RuntimeError("Клиент Milana не инициализирован")

    prompt = generation_params.get("prompt", "")
    if not _is_within_token_limit(prompt, token_limit):
        raise RuntimeError("ContextOverflowError")

    try:
        result = client.get_text_response(prompt)
        if isinstance(result, str) and result.startswith("ERROR:"):
            raise RuntimeError(result)
        return result
    except RuntimeError:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка генерации: {str(e)}")

def create_embeddings(text: str) -> List[float]:
    global client, token_limit
    if client is None:
        raise RuntimeError("Клиент Milana не инициализирован")
    
    if not _is_within_token_limit(text, token_limit):
        raise RuntimeError("ContextOverflowError")

    try:
        result = client.get_embeddings(text)
        if isinstance(result, str) and result.startswith("ERROR:"):
            raise RuntimeError(result)
        return result
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка создания эмбеддингов: {str(e)}")

# ========== Токенизация ==========
_encoding_cache: Dict[str, Any] = {}

def _get_encoding(model: str):
    if model not in _encoding_cache:
        try:
            _encoding_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
    return _encoding_cache[model]

def _count_tokens(text: str, model: str = model_name) -> int:
    return len(_get_encoding(model).encode(text))

def _is_within_token_limit(text: str, limit: int) -> bool:
    return _count_tokens(text) < limit