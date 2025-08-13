# local_provider.py
import os
import time
import traceback
import tiktoken
from llama_cpp import Llama
from typing import Any, Dict, List, Tuple

# Глобальные переменные состояния
llm = None
llm_mode = ''
token_limit = 8192  # Общий лимит токенов для чата и эмбеддингов
model_path = ''
encoding_cache = {}

tags = {
    "system": "[INST] <<SYS>>\n",
    "user": "\n<</SYS>>\n[/INST] ",
    "assistant": " ",
    "end": " [/INST]"
}

class LocalModelError(Exception): pass
class ConnectionError(LocalModelError): pass
class APIError(LocalModelError): pass

def connect(model_path_str: str, n_ctx: int = 8192) -> Tuple[bool, int, Dict[str, Any]]:
    global llm, llm_mode, token_limit, model_path

    try:
        if not os.path.isfile(model_path_str):
            return False, 0, {"error": "Файл модели не найден"}

        unload_model()
        llm = Llama(model_path=model_path_str, n_ctx=n_ctx, verbose=False)
        llm_mode = 'answer'
        token_limit = n_ctx
        model_path = model_path_str

        metadata = llm.metadata if hasattr(llm, "metadata") else {}

        info = {
            "model_path": model_path_str,
            "token_limit": token_limit,
            "metadata": metadata
        }

        return True, token_limit, info

    except Exception as e:
        traceback.print_exc()
        return False, 0, {"error": str(e)}

def disconnect() -> bool:
    return unload_model() > 0

def unload_model() -> float:
    global llm, llm_mode
    if llm is not None:
        start = time.time()
        del llm
        llm = None
        llm_mode = ''
        return time.time() - start
    return 0

def ask_model(generation_params: Dict[str, Any]) -> str:
    global llm, llm_mode, token_limit

    if llm is None:
        raise RuntimeError("Модель не загружена")

    if llm_mode != 'answer':
        unload_model()
        connect(model_path, token_limit)

    prompt = generation_params.get("prompt", "")
    if not is_within_token_limit(prompt, token_limit):
        raise RuntimeError("ContextOverflowError")

    try:
        output = llm(**generation_params)
        return output["choices"][0]["text"]
    except Exception as e:
        traceback.print_exc()
        raise APIError(f"Ошибка генерации: {str(e)}")

def create_embeddings(text: str) -> List[float]:
    global llm, llm_mode, token_limit

    if llm is None:
        raise RuntimeError("Модель не загружена")

    if not is_within_token_limit(text, token_limit):
        raise RuntimeError("ContextOverflowError")

    if llm_mode != 'embed':
        load_embed_model()

    try:
        return llm.embed(text)[0]
    except Exception as e:
        traceback.print_exc()
        raise APIError(f"Ошибка эмбеддингов: {str(e)}")

def load_embed_model():
    global llm, llm_mode, token_limit

    if llm_mode == 'embed':
        return

    if not model_path:
        raise RuntimeError("Путь к модели не задан")

    unload_model()
    try:
        llm = Llama(model_path=model_path, n_ctx=token_limit, embedding=True, verbose=False)
        llm_mode = 'embed'
    except Exception as e:
        raise ConnectionError(f"Ошибка загрузки модели в режиме эмбеддингов: {str(e)}")

# === Token Tools ===

def get_encoding(model_name: str = "cl100k_base"):
    global encoding_cache
    if model_name not in encoding_cache:
        try:
            encoding_cache[model_name] = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding_cache[model_name] = tiktoken.get_encoding("cl100k_base")
    return encoding_cache[model_name]

def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    encoding = get_encoding(model_name)
    return len(encoding.encode(text))

def is_within_token_limit(text: str, limit: int) -> bool:
    return count_tokens(text) < limit