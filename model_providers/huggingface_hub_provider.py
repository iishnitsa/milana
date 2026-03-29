import requests
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

    # Настройка тегов на основе имени модели
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

    if params['chat']:
        try:
            api_url = f"https://huggingface.co/api/models/{params['chat']}"
            resp = requests.get(api_url, timeout=req_timeout)
            if resp.status_code == 200:
                model_info = resp.json()
                # Ищем context length в разных полях
                config = model_info.get('config', {})
                token_limit = config.get('max_position_embeddings', 32768)
            else:
                token_limit = 32768
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
    """Используется для completion (prompt-based)"""
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
    """
    Запрос к чат-модели через Hugging Face Inference API.
    Ожидает параметры:
    - messages: список сообщений [{"role": "user", "content": "..."}, ...]
    - model: опционально, модель для использования
    - temperature, max_tokens и др. параметры генерации
    """
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

        # Hugging Face Inference Client
        self.hf_client = InferenceClient(model=chat_model, token=api_token, timeout=timeout) if chat_model else None
        self.hf_emb_client = InferenceClient(model=emb_model, token=api_token, timeout=timeout) if emb_model and not use_ollama else None

        # Ollama для эмбеддингов
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url.rstrip('/')
        self.ollama_emb_model = ollama_emb_model
        self.ollama_session = requests.Session() if use_ollama else None

    def generate(self, generation_params: Dict[str, Any]) -> str:
        """Текстовая генерация на основе промпта (для обратной совместимости)"""
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

        try:
            return self.hf_client.text_generation(prompt, **gen_kwargs).strip()
        except Exception as e:
            err_str = str(e).lower()
            if any(phrase in err_str for phrase in ["context", "token length", "max_length"]):
                raise RuntimeError("ContextOverflowError")
            raise

    def chat(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Чат-генерация через Hugging Face Inference API [citation:5]
        Использует метод chat.completions.create
        """
        if not self.hf_client:
            raise RuntimeError("Chat model not specified or client not initialized")
        if "messages" not in generation_params:
            raise ValueError("'messages' is required for chat")

        # Подготовка параметров для chat.completions
        messages = generation_params["messages"]
        model = generation_params.get("model", self.chat_model)
        
        # Стандартные параметры генерации [citation:5]
        chat_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": generation_params.get("max_tokens", 500),
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.9),
            "stop": generation_params.get("stop", None),
        }
        
        # Добавляем опциональные параметры
        if "frequency_penalty" in generation_params:
            chat_kwargs["frequency_penalty"] = generation_params["frequency_penalty"]
        if "presence_penalty" in generation_params:
            chat_kwargs["presence_penalty"] = generation_params["presence_penalty"]
        if "seed" in generation_params:
            chat_kwargs["seed"] = generation_params["seed"]

        try:
            # Hugging Face InferenceClient поддерживает chat.completions.create
            response = self.hf_client.chat.completions.create(**chat_kwargs)
            
            # Преобразуем ответ в словарь для совместимости
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
        except Exception as e:
            err_str = str(e).lower()
            if any(phrase in err_str for phrase in ["context", "token length", "max_length"]):
                raise RuntimeError("ContextOverflowError")
            raise RuntimeError(f"Chat API error: {str(e)}")

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