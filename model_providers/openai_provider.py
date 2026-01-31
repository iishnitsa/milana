import requests
import re
import os
import json
import time
import random
from cross_gpt import let_log

# === Глобальные переменные ===
openai_session = None
ollama_session = None

openai_base_url = ""
ollama_base_url = ""
openai_timeout = 0

default_chat_model = "gpt-5-nano"
emb_model = "text-embedding-3-small"
ollama_emb_model = "all-minilm:latest"

token_limit = 4095
emb_token_limit = 4095

do_chat_construct = True
native_func_call = False

tags = {
    "bos": "", "eos": "",
    "sys_start": "", "sys_end": "",
    "user_start": "", "user_end": "",
    "assist_start": "", "assist_end": "",
    "tool_def_start": "", "tool_def_end": "",
    "tool_call_start": "", "tool_call_end": "",
    "tool_result_start": "", "tool_result_end": "",
}

# Настройки адаптивности
MAX_RETRIES = 12          # Максимум попыток
BASE_BACKOFF = 2.0        # Основание экспоненты (2, 4, 8, 16...)
last_wait_time = 0.0      # "Память" о задержке между вызовами

def _get_timeout_value():
    """Возвращает значение таймаута для использования в запросах"""
    global openai_timeout
    return None if openai_timeout == 0 else openai_timeout

def _make_request(api_url, payload):
    """
    Универсальный метод для всех OpenAI-совместимых запросов.
    Реализует адаптивный RPM, экспоненциальный кулдаун и память.
    """
    global last_wait_time
    
    if not openai_session:
        raise RuntimeError("OpenAI client not connected")

    # 1. ПАМЯТЬ: Если прошлый запрос вызвал 429 или ошибку, делаем превентивную паузу
    if last_wait_time > 0:
        preemptive_pause = min(5.0, last_wait_time * 0.5)
        if preemptive_pause > 0.1:
            time.sleep(preemptive_pause)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            let_log(f"openai request attempt {attempt}: {api_url}")
            
            # Используем правильное значение таймаута
            timeout_value = _get_timeout_value()
            r = openai_session.post(api_url, json=payload, timeout=timeout_value)
            
            # --- ОБРАБОТКА ОШИБОК HTTP (429 и др) ---
            if r.status_code == 429:
                try:
                    data = r.json().get("error", {})
                    err_code = data.get("code", "")
                    err_msg = data.get("message", "").lower()
                except:
                    err_code = ""
                    err_msg = r.text.lower()

                if err_code == "insufficient_quota" or "insufficient_quota" in err_msg:
                    let_log(f"CRITICAL ERROR: {r.text}")
                    raise RuntimeError("balance end")

                if attempt < MAX_RETRIES:
                    wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                    last_wait_time = wait_time
                    let_log(f"Status 429 (RPM). Waiting {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError("balance end or extremely low RPM limit")
            
            # --- ОБРАБОТКА ОШИБКИ 402 (Payment Required) ---
            elif r.status_code == 402:
                let_log(f"Payment Required (402): {r.text}")
                try:
                    data = r.json().get("error", {})
                    err_msg = data.get("message", "Payment Required")
                    let_log(f"Balance error: {err_msg}")
                except:
                    pass
                raise RuntimeError("balance end")
            
            # --- ОБРАБОТКА СЕРВЕРНЫХ ОШИБОК (5xx) ---
            elif 500 <= r.status_code < 600:
                if attempt < MAX_RETRIES:
                    wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                    last_wait_time = wait_time
                    let_log(f"Status {r.status_code} (Server Error). Waiting {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Server error after maximum retries: {r.text}")

            # Логируем статус ответа (но это еще не гарантия успеха)
            let_log(f"status={r.status_code}, text={r.text[:500]}")
            r.raise_for_status()
            
            resp_data = r.json()

            # --- ПРОВЕРКА НА СКРЫТЫЕ ОШИБКИ В JSON (для прокси-сервисов вроде Aitunnel) ---
            # Важно: проверяем ДО любых других манипуляций с resp_data
            if "error" in resp_data:
                err_info = resp_data["error"]
                err_msg = err_info.get("message", "Unknown error")
                
                # Обновляем логирование с учетом статуса
                let_log(f"Provider returned error inside JSON (status={r.status_code}): {err_msg}")
                
                if "insufficient_quota" in err_msg.lower():
                    raise RuntimeError("balance end")
                
                if attempt < MAX_RETRIES:
                    wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                    let_log(f"Retrying in {wait_time:.2f}s due to provider glitch...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"API Provider Error: {err_msg}")

            # Успех: Плавное "остывание" памяти
            last_wait_time *= 0.7
            if last_wait_time < 0.1:
                last_wait_time = 0
            
            return resp_data

        except requests.exceptions.Timeout:
            # Таймаут возможен только если openai_timeout > 0
            if attempt < MAX_RETRIES:
                wait_time = min(60, (BASE_BACKOFF ** attempt) + random.random())
                last_wait_time = wait_time
                let_log(f"Request timeout on attempt {attempt}. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                continue
            else:
                raise RuntimeError("Request timed out after maximum retries")
        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2)
        except Exception as e:
            if "balance end" in str(e) or "API Provider Error" in str(e):
                raise
            raise RuntimeError(f"OpenAI request error: {e}")


# =============================================================

def connect(connection_string, timeout=30):
    global openai_session, ollama_session
    global openai_base_url, ollama_base_url, openai_timeout
    global default_chat_model, emb_model, ollama_emb_model
    global token_limit, emb_token_limit
    global do_chat_construct, native_func_call, tags

    params = {
        "url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "emb_model": "text-embedding-3-small",
        "token": None,
        "token_limit": "32768",
        "chat_template": "True",
        "native_func_call": "False",
        "ollama_url": "http://localhost:11434",
        "ollama_emb_model": "all-minilm:latest",
        "ollama": "False",
        "timeout": "0",
    }

    for part in connection_string.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k.strip().lower() in params:
            params[k.strip().lower()] = v.strip()

    openai_base_url = params["url"].rstrip("/")
    ollama_base_url = params["ollama_url"].rstrip("/")

    default_chat_model = params["model"]
    emb_model = params["emb_model"]
    ollama_emb_model = params["ollama_emb_model"]
    
    # Парсим таймаут: 0 = бесконечный, >0 = обычный таймаут
    try:
        timeout_val = int(params["timeout"])
        openai_timeout = timeout_val if timeout_val >= 0 else 0
    except ValueError: openai_timeout = 0

    token_limit = int(params["token_limit"])
    emb_token_limit = token_limit

    do_chat_construct = params["chat_template"].lower() == "true"
    native_func_call = params["native_func_call"].lower() == "true"

    # Определяем, используем ли мы Ollama для эмбеддингов
    use_ollama = params["ollama"].lower() == "true"

    api_key = params["token"] or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return [False, 0, tags, "No API key"]

    # --- OpenAI session ---
    openai_session = requests.Session()
    openai_session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "cross-gpt-openai-provider",
    })

    openai_session.emb_model = emb_model

    try:
        # Используем правильное значение таймаута
        timeout_value = _get_timeout_value()
        r = openai_session.get(f"{openai_base_url}/models", timeout=timeout_value)
        r.raise_for_status()
    except Exception as e:
        return [False, 0, tags, f"OpenAI connect error: {e}"]

    # --- Ollama embeddings (только если явно разрешено) ---
    ollama_session = None  # Сбрасываем сессию
    
    if use_ollama:
        try:
            ollama_session = requests.Session()
            ollama_session.headers.update({"Content-Type": "application/json"})

            # Используем правильное значение таймаута
            timeout_value = _get_timeout_value()
            r = ollama_session.get(f"{ollama_base_url}/api/tags", timeout=timeout_value)
            r.raise_for_status()
            let_log("Ollama embeddings enabled successfully")
        except Exception as e:
            ollama_session = None
            let_log(f"Ollama connect failed (fallback to OpenAI embeddings): {e}")

    return [True, token_limit, tags]


def disconnect():
    global openai_session, ollama_session
    if openai_session:
        openai_session.close()
        openai_session = None
    if ollama_session:
        ollama_session.close()
        ollama_session = None
    return True

# =============================================================
# === RAW (prompt / completions) ===============================
# =============================================================

def ask_model(generation_params):
    """
    RAW-режим: ты сам формируешь prompt (как в gpt4all)
    """
    prompt = generation_params.get("prompt", "")
    system = generation_params.get("system", "")

    full_prompt = (
        tags["bos"] +
        tags["sys_start"] + system + tags["sys_end"] +
        tags["user_start"] + prompt + tags["user_end"] +
        tags["assist_start"]
    )

    payload = {
        "model": generation_params.get("model", default_chat_model),
        "prompt": full_prompt,
        "max_tokens": generation_params.get("max_tokens", 1024),
        "temperature": generation_params.get("temperature", 0.7),
    }

    data = _make_request(f"{openai_base_url}/completions", payload)
    return data["choices"][0]["text"].strip()


# =============================================================
# === CHAT (messages / templates) ==============================
# =============================================================

def ask_model_chat(generation_params):
    """
    ШАБЛОННЫЙ режим: OpenAI сам управляет ролями
    """
    payload = {
        "model": generation_params.get("model", default_chat_model),
        "messages": generation_params.get("messages", []),
        "max_tokens": generation_params.get("max_tokens", 1024),
        "temperature": generation_params.get("temperature", 0.7),
        "stream": False,
    }

    return _make_request(f"{openai_base_url}/chat/completions", payload)


# =============================================================
# === EMBEDDINGS ===============================================
# =============================================================

def create_embeddings(text):
    """
    Ollama embeddings → если явно разрешено и доступно
    иначе OpenAI embeddings
    """
    # Пытаемся использовать Ollama только если сессия создана
    if ollama_session:
        try:
            # Используем правильное значение таймаута
            timeout_value = _get_timeout_value()
            r = ollama_session.post(
                f"{ollama_base_url}/api/embeddings",
                json={"model": ollama_emb_model, "prompt": text},
                timeout=timeout_value
            )
            r.raise_for_status()
            return r.json()["embedding"]
        except Exception as e:
            raise RuntimeError(f"Ollama embeddings error: {e}")

    # Fallback на OpenAI embeddings
    if not openai_session:
        raise RuntimeError("No embeddings backend")

    payload = {
        "model": openai_session.emb_model,
        "input": text
    }

    data = _make_request(f"{openai_base_url}/embeddings", payload)
    
    # Безопасное извлечение эмбеддинга
    if "data" in data and len(data["data"]) > 0:
        return data["data"][0]["embedding"]
    else:
        raise RuntimeError(f"Could not find embeddings in response: {data}")