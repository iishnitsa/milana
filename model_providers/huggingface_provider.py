from huggingface_hub import InferenceClient, HfApi
import traceback
import tiktoken
from typing import Optional, Dict, Any, List, Tuple

# Глобальные переменные
client = None
token_limit = 32768
emb_token_limit = 225
model_mode = None
current_provider = None
provider_determined = False  # Флаг, что режим и провайдер уже определены
tags = {
    "system": "[INST] <<SYS>>\n",
    "user": "\n<</SYS>>\n[/INST] ",
    "assistant": " ",
    "end": " [/INST]"
}

class HuggingFaceError(Exception): pass
class ConnectionError(HuggingFaceError): pass
class APIError(HuggingFaceError): pass

def connect(connection_string: str, timeout: int = 30) -> Tuple[bool, int, Dict[str, Optional[str]]]:
    """
    Инициализация клиента, но без определения провайдера/режима (это делаем при первой генерации)
    """
    global client, token_limit, emb_token_limit, tags, model_mode, current_provider, provider_determined

    params = {'chat': None, 'emb': None, 'token': None}
    for part in connection_string.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            key, value = key.strip().lower(), value.strip()
            if key in params and value:
                params[key] = value

    if not params['token']:
        print("[ERROR] Token не указан")
        return False, token_limit, tags

    if params['chat']:
        model_name = params['chat'].lower()
        if 'mistral' in model_name:
            tags.update({"system": "<s>[INST] ", "user": " [/INST]", "assistant": " ", "end": "</s>"})
        elif 'gemma' in model_name:
            tags.update({"system": "<start_of_turn>model\n", "user": "<start_of_turn>user\n",
                         "assistant": "<start_of_turn>model\n", "end": "<end_of_turn>\n"})

        # Пока не определяем провайдера — это сделаем при первой генерации
        client = HuggingFaceClient(params['token'], params['chat'], params['emb'], timeout)
        provider_determined = False

    if params['emb']:
        client = client or HuggingFaceClient(params['token'], params['chat'], params['emb'], timeout)
        emb_token_limit = client.get_model_token_limit(params['emb']) or emb_token_limit

    return True, token_limit, tags

def disconnect() -> bool:
    global client
    if client is not None:
        client = None
        return True
    return False

def ask_model(generation_params: Dict[str, Any]) -> str:
    global client, token_limit, provider_determined
    if client is None:
        raise RuntimeError("Клиент Hugging Face не инициализирован")
    try:
        prompt = generation_params["prompt"]
        if not client.is_within_token_limit(prompt, 'chat', token_limit):
            raise RuntimeError("ContextOverflowError")
        return client.generate_with_auto_detect(
            prompt,
            {
                "max_new_tokens": generation_params.get("max_tokens", 200),
                "temperature": generation_params.get("temperature", 0.7),
                "top_p": generation_params.get("top_p", 0.9),
                "repetition_penalty": generation_params.get("repeat_penalty", 1.1)
            }
        )
    except RuntimeError as e:
        if str(e) == "ContextOverflowError":
            raise
        traceback.print_exc()
        raise RuntimeError(f"Ошибка генерации: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка генерации: {str(e)}")

def create_embeddings(text: str) -> List[float]:
    global client, emb_token_limit
    if client is None:
        raise RuntimeError("Клиент Hugging Face не инициализирован")
    try:
        if not client.is_within_token_limit(text, 'emb', emb_token_limit):
            raise RuntimeError("ContextOverflowError")
        return client.embeddings(text)
    except RuntimeError as e:
        if str(e) == "ContextOverflowError":
            raise
        traceback.print_exc()
        raise RuntimeError(f"Ошибка создания эмбеддингов: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка создания эмбеддингов: {str(e)}")

class HuggingFaceClient:
    def __init__(self, api_token: str, chat_model: Optional[str], emb_model: Optional[str], timeout: int = 30):
        self.chat_model = chat_model
        self.emb_model = emb_model
        self.timeout = timeout
        self.encoding_cache: Dict[str, Any] = {}
        self.api_token = api_token
        self.api = HfApi()

        self.chat_client = None
        self.emb_client = None
        if emb_model:
            self.emb_client = InferenceClient(model=emb_model, token=api_token, timeout=timeout)

    def _create_inference_client(self, model: str, provider: str):
        return InferenceClient(model=model, token=self.api_token, timeout=self.timeout, provider=provider)

    def get_encoding(self, model_name: str):
        if model_name not in self.encoding_cache:
            try:
                self.encoding_cache[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                self.encoding_cache[model_name] = tiktoken.get_encoding("cl100k_base")
        return self.encoding_cache[model_name]

    def count_tokens(self, text: str, model_name: str) -> int:
        return len(self.get_encoding(model_name).encode(text))

    def is_within_token_limit(self, text: str, model_type: str, token_limit: int) -> bool:
        model_name = self.chat_model if model_type == 'chat' else self.emb_model
        if not model_name:
            return True
        return self.count_tokens(text, model_name) < token_limit

    def get_model_token_limit(self, model_id: str) -> Optional[int]:
        try:
            info = self.api.model_info(model_id, token=self.api_token)
            cfg = info.cardData or {}
            return cfg.get("max_position_embeddings") or cfg.get("n_positions") or cfg.get("max_sequence_length")
        except Exception:
            return None

    def _detect_provider_and_mode(self):
        """Определение рабочего провайдера и режима с тестовым запросом."""
        global model_mode, current_provider, provider_determined

        providers_to_try = [
            ("hf-inference", "text-generation"),
            ("together", "text-generation"),
            ("featherless-ai", "conversational"),
            ("auto", "text-generation"),
            ("auto", "conversational")
        ]

        for provider, mode in providers_to_try:
            try:
                temp_client = self._create_inference_client(self.chat_model, provider)
                if mode == "text-generation":
                    temp_client.text_generation("Hello", max_new_tokens=1)
                else:
                    temp_client.chat_completion([{"role": "user", "content": "Hello"}], max_tokens=1)
                # Успешно — сохраняем
                model_mode = mode
                current_provider = provider
                self.chat_client = temp_client
                provider_determined = True
                print(f"[INFO] Selected provider: {provider}, mode: {mode}")
                return
            except Exception as e:
                continue

        raise RuntimeError("Не удалось подобрать рабочий провайдер/режим для модели.")

    def generate_with_auto_detect(self, prompt: str, parameters: Dict[str, Any]) -> str:
        global provider_determined
        if not provider_determined:
            self._detect_provider_and_mode()
        return self._generate(prompt, parameters)

    def _generate(self, prompt: str, parameters: Dict[str, Any]) -> str:
        if model_mode == "text-generation":
            return self.chat_client.text_generation(
                prompt,
                max_new_tokens=parameters.get("max_new_tokens", 200),
                temperature=parameters.get("temperature", 0.7),
                top_p=parameters.get("top_p", 0.9),
                repetition_penalty=parameters.get("repetition_penalty", 1.1),
                do_sample=True
            ).strip()
        else:
            messages = [{"role": "user", "content": prompt}]
            return self.chat_client.chat_completion(
                messages,
                max_tokens=parameters.get("max_new_tokens", 200),
                temperature=parameters.get("temperature", 0.7),
                top_p=parameters.get("top_p", 0.9),
            ).choices[0].message.content.strip()

    def embeddings(self, text: str) -> List[float]:
        output = self.emb_client.feature_extraction(text)
        if isinstance(output, list):
            if isinstance(output[0], list):
                return [float(x) for x in output[0]]
            return [float(x) for x in output]
        return output.tolist()
