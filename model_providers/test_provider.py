import os
from encryption_utils import decrypt_token, ask_for_password, SESSION_PASSWORDS

token_limit = 8192
emb_token_limit = 8192
do_chat_construct = True
native_func_call = False

tags = {
    "bos": "", "eos": "", "sys_start": "", "sys_end": "",
    "user_start": "", "user_end": "", "assist_start": "", "assist_end": "",
    "tool_def_start": "", "tool_def_end": "", "tool_call_start": "", "tool_call_end": "",
    "tool_result_start": "", "tool_result_end": "",
}

def connect(connection_string, _decrypted_password=None):
    params = {
        "api_token": "",
        "password": ""
    }
    for part in connection_string.split(";"):
        if "=" not in part: continue
        k, v = part.split("=", 1)
        if k.strip().lower() in params: 
            params[k.strip().lower()] = v.strip()
    
    raw_token = params["api_token"]
    pwd_status = params["password"]

    # Дешифровка токена
    decrypted_token = raw_token
    if pwd_status == "set":
        if _decrypted_password is None:
            # Вызываем диалоговое окно
            _decrypted_password = ask_for_password()
            if not _decrypted_password:
                return [False, 0, tags, "Ввод пароля отменен или пароль пуст."]
        
        try:
            decrypted_token = decrypt_token(raw_token, _decrypted_password)
            # Сохраняем введенный пароль в ОЗУ
            SESSION_PASSWORDS['test_provider'] = _decrypted_password
        except Exception as e:
            return [False, 0, tags, f"Ошибка дешифровки: {str(e)}"]
            
    elif pwd_status == "empty":
        pass # Пользователь сохранил токен без пароля

    if not decrypted_token:
        return [False, 0, tags, "Токен пуст"]

    return [True, token_limit, tags, "Успешное подключение к TestProvider"]

def disconnect():
    return True

def ask_model(generation_params):
    return "TestProvider: Заглушка (ask_model)"

def ask_model_chat(generation_params):
    return "TestProvider: Заглушка (ask_model_chat)"

def create_embeddings(text):
    return [0.0] * 1536