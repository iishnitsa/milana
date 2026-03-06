import traceback
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

    # Настройка тегов на основе имени модели (упрощенно)
    if params['chat']:
        model_lower = params['chat'].lower()
        if 'mistral' in model_lower:
            tags.update({"system": "<s>[INST] ", "user": " [/INST]", "assistant": " ", "end": "</s>"})
        elif 'gemma' in model_lower:
            tags.update({"system": "<start_of_turn>model\n", "user": "<start_of_turn>user\n",
                         "assistant": "<start_of_turn>model\n", "end": "<end_of_turn>\n"})

    client = HuggingFaceClient(
        api_token=params['token'],
        chat_model=params['chat'],
        emb_model=params['emb'],
        timeout=req_timeout,
        use_ollama=(params['ollama'].lower() == 'true'),
        ollama_url=params['ollama_url'],
        ollama_emb_model=params['ollama_emb_model']
    )

    # Получение лимитов через HF API (client.get_model_token_limit) можно добавить позже
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

        # Hugging Face Inference Client
        self.hf_client = InferenceClient(model=chat_model, token=api_token, timeout=timeout) if chat_model else None
        self.hf_emb_client = InferenceClient(model=emb_model, token=api_token, timeout=timeout) if emb_model and not use_ollama else None

        # Ollama для эмбеддингов
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url.rstrip('/')
        self.ollama_emb_model = ollama_emb_model
        self.ollama_session = requests.Session() if use_ollama else None

    def generate(self, generation_params: Dict[str, Any]) -> str:
        if not self.hf_client:
            raise RuntimeError("Chat model not specified or client not initialized")
        if "prompt" not in generation_params:
            raise ValueError("'prompt' is required")

        prompt = generation_params["prompt"]
        # Формируем параметры для text_generation
        gen_kwargs = {
            "max_new_tokens": generation_params.get("max_tokens", 200),
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.9),
            "repetition_penalty": generation_params.get("repeat_penalty", 1.1),
            "do_sample": True
        }

        try:
            # Используем text_generation, так как это наиболее универсально
            return self.hf_client.text_generation(prompt, **gen_kwargs).strip()
        except Exception as e:
            err_str = str(e).lower()
            # Попытка поймать ошибку переполнения контекста
            if any(phrase in err_str for phrase in ["context", "token length", "max_length"]):
                raise RuntimeError("ContextOverflowError")
            raise

    def embeddings(self, text: str) -> List[float]:
        # Приоритет: Ollama
        if self.use_ollama and self.ollama_session:
            try:
                return self._ollama_embeddings(text)
            except Exception as e:
                raise RuntimeError(f"Ollama embeddings error: {e}")

        # Затем Hugging Face
        if not self.hf_emb_client:
            raise RuntimeError("No embeddings backend available")

        try:
            output = self.hf_emb_client.feature_extraction(text)
            if isinstance(output, list):
                if output and isinstance(output[0], list):
                    return [float(x) for x in output[0]]
                return [float(x) for x in output]
            return output.tolist()
        except Exception as e:
            err_str = str(e).lower()
            if any(phrase in err_str for phrase in ["context", "token length", "max_length"]):
                raise RuntimeError("ContextOverflowError")
            raise

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