# cohere_provider.py
import requests
import json
import traceback
import tiktoken
from typing import Optional, Dict, Any, List, Tuple

client = None
token_limit = 4096
emb_token_limit = 512  # Добавлен отдельный лимит для эмбеддингов
tags = {
    "system": None,
    "user": "User: ",
    "assistant": "Assistant: ",
    "end": None
}

class CohereError(Exception): pass
class ConnectionError(CohereError): pass
class APIError(CohereError): pass

def connect(connection_string: str, timeout: int = 30) -> List[Any]:
    global client, token_limit, emb_token_limit, tags

    params = {'chat': None, 'emb': None, 'token': None}
    for part in connection_string.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            key, value = key.strip().lower(), value.strip()
            if key in params:
                params[key] = value if value else None

    if not params['token']:
        return [False, 0, tags]

    client = CohereClient(
        api_key=params['token'],
        chat_model=params['chat'],
        emb_model=params['emb'],
        timeout=timeout
    )

    token_limit = client.get_token_limit()
    emb_token_limit = client.get_emb_token_limit()
    return [True, token_limit, tags]

def disconnect() -> bool:
    global client
    if client is not None:
        client = None
        return True
    return False

def ask_model(generation_params: Dict[str, Any]) -> str:
    global client, token_limit
    if client is None:
        raise RuntimeError("Cohere client not initialized")
    try:
        prompt = generation_params["prompt"]
        if not client.is_within_token_limit(prompt, 'chat', token_limit):
            raise RuntimeError("ContextOverflowError")
        return client.generate(generation_params)
    except RuntimeError as e:
        raise
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Generation error: {str(e)}")

def create_embeddings(text: str) -> List[float]:
    global client, emb_token_limit
    if client is None:
        raise RuntimeError("Cohere client not initialized")
    try:
        if not client.is_within_token_limit(text, 'emb', emb_token_limit):
            raise RuntimeError("ContextOverflowError")
        return client.embeddings(text)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Embeddings error: {str(e)}")

class CohereClient:
    MODEL_CONTEXT_SIZES = {
        "command": 4096,
        "command-light": 4096,
        "command-r": 128000,
        "command-r-plus": 128000,
    }
    
    EMB_MODEL_CONTEXT_SIZES = {
        "embed-english-v2.0": 512,
        "embed-english-light-v2.0": 512,
        "embed-multilingual-v2.0": 512,
    }

    def __init__(self,
                 api_key: str,
                 chat_model: Optional[str] = None,
                 emb_model: Optional[str] = None,
                 timeout: int = 30):
        self.base_url = "https://api.cohere.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.timeout = timeout
        self.chat_model = chat_model or "command"
        self.emb_model = emb_model or "embed-english-v2.0"
        self.encoding_cache: Dict[str, Any] = {}

    def get_token_limit(self) -> int:
        return self.MODEL_CONTEXT_SIZES.get(self.chat_model, 4096)
        
    def get_emb_token_limit(self) -> int:
        return self.EMB_MODEL_CONTEXT_SIZES.get(self.emb_model, 512)

    def get_encoding(self, model_name: str):
        if model_name not in self.encoding_cache:
            try:
                self.encoding_cache[model_name] = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except KeyError:
                self.encoding_cache[model_name] = tiktoken.get_encoding("cl100k_base")
        return self.encoding_cache[model_name]

    def count_tokens(self, text: str, model_type: str) -> int:
        model_name = self.chat_model if model_type == 'chat' else self.emb_model
        return len(self.get_encoding(model_name).encode(text))

    def is_within_token_limit(self, text: str, model_type: str, token_limit: int) -> bool:
        return self.count_tokens(text, model_type) < token_limit

    def generate(self, generation_params: Dict[str, Any]) -> str:
        if "prompt" not in generation_params:
            raise ValueError("'prompt' is required in generation_params")

        url = f"{self.base_url}/generate"
        payload = {
            "model": self.chat_model,
            "prompt": generation_params["prompt"],
            "max_tokens": generation_params.get("max_tokens", 200),
            "temperature": generation_params.get("temperature", 0.7),
            "k": generation_params.get("top_k", 0),
            "p": generation_params.get("top_p", 0.75),
            "frequency_penalty": generation_params.get("frequency_penalty", 0.0),
            "presence_penalty": generation_params.get("presence_penalty", 0.0),
            "stop_sequences": generation_params.get("stop_sequences", []),
            "stream": False
        }

        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data['generations'][0]['text']
        except requests.HTTPError as e:
            msg = ""
            try:
                error_data = resp.json()
                msg = error_data.get('message', str(e))
            except:
                msg = str(e)

            if any(word in msg.lower() for word in ["context", "token", "max_length"]):
                raise RuntimeError("ContextOverflowError")
            raise APIError(f"API error: {msg}")
        except requests.RequestException as e:
            raise ConnectionError(f"Connection error: {str(e)}")
        except Exception as e:
            raise APIError(f"Unknown error: {str(e)}")

    def embeddings(self, text: str) -> List[float]:
        url = f"{self.base_url}/embed"
        payload = {
            "model": self.emb_model,
            "texts": [text],
            "input_type": "search_document"
        }

        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data['embeddings'][0]
        except requests.HTTPError as e:
            msg = ""
            try:
                error_data = resp.json()
                msg = error_data.get('message', str(e))
            except:
                msg = str(e)
            raise APIError(f"Embeddings API error: {msg}")
        except requests.RequestException as e:
            raise ConnectionError(f"Connection error: {str(e)}")
        except Exception as e:
            raise APIError(f"Unknown error: {str(e)}")