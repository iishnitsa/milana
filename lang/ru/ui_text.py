# lang/ru/ui_text.py
"""
Словарь с текстовыми переменными для русского языка.
"""
TEXTS = {
    # App General
    "app_title": "Милана",
    "error": "Ошибка",
    "success": "Успех",
    "warning": "Внимание",
    "info": "Информация",
    "save": "Сохранить",
    "cancel": "Отмена",
    "close": "Закрыть",
    "browse": "Обзор...",
    "delete": "Удалить",
    "language": "Язык",
    "restart_required": "Для применения языковых настроек требуется перезапуск приложения.",

    # Language Loading
    "lang_load_error_title": "Ошибка загрузки языка",
    "lang_load_error_message": "Не найдены языковые файлы в папке 'lang'. Приложение будет закрыто.",

    # Main Window
    "new_chat": "Новый чат",
    "settings": "Настройки",
    "attachments": "Вложения:",
    
    "chat_prefix": "Чат ",

    # Chat Context Menu
    "active_chat_delete_title": "Активный чат",
    "active_chat_delete_message": "Чат '{chat_name}' активен. Как удалить?",
    "active_chat_delete_detail": "Вы можете безопасно остановить его или удалить принудительно.",
    "delete_chat_confirm_title": "Удаление чата",
    "delete_chat_confirm_message": "Вы уверены, что хотите удалить чат '{chat_name}'?",

    # App Closing
    "active_chats_on_close_title": "Активные чаты",
    "active_chats_on_close_message": "Есть {count} активных чатов. Выберите действие:",
    "active_chats_on_close_detail": "Вы можете безопасно остановить их или закрыть принудительно.",
    "stop_safely": "Безопасно остановить",
    "terminate": "Закрыть принудительно",

    # Chat Controls
    "message_from_model_requires_answer": "Требуется ваш ответ",

    # Attachment
    "attachment_open_error": "Ошибка открытия вложения",

    # Initial Settings
    "initial_settings_title": "Первоначальная настройка",
    "save_and_continue": "Сохранить и продолжить",
    "validation_error": "Ошибка валидации",
    "token_limit_info": "Лимит токенов (макс: {max_tokens})",
    "validate_model": "Проверить модель",
    "model_validated_success": "Модель прошла валидацию. Максимум токенов: {tokens}",
    "model_not_validated": "Модель не проверена",
    "model_not_validated_continue": "Модель не проверена. Продолжить?",

    # Model Types and Settings
    "model_type": "Тип модели:",
    "local_model_path": "Путь к модели:",
    "gguf_files": "GGUF файлы",
    "huggingface_model": "Модель HuggingFace:",
    "hf_token_placeholder": "chat=chat_model_repo;emb=embeddings_model_repo;token=token",
    "openai_api_key": "chat=chat_model;emb=embeddings_model;token=api_key",
    "anthropic_api_key": "chat=chat_model;emb=embeddings_model;token=api_key",
    "mistral_api_key": "token=api_key",
    "cohere_api_key": "chat=chat_model;emb=embeddings_model;token=api_key",
    "ollama_model": "host=localhost;port=11434;chat=chat_model;emb=embeddings_model;token=optional",
    "ollama_model_hint": "",
    "lmstudio_model_hint": "Введите идентификатор, например 'local-model/model-name'",
    "lmstudio_model": "host=localhost;port=8080;chat=chat_model;emb=embeddings_model;user=optional;password=optional",
    "custom_api_server_ip": "IP сервера:",
    "custom_api_port": "Порт:",
    "custom_api_key": "API ключ (если требуется):",
    "token_limit": "Лимит токенов:",

    # Create/Edit Chat Window
    "create_chat_title": "Создать новый чат",
    "chat_name": "Название чата:",
    "chat_name_exists": "Чат с таким именем уже существует.",
    "enter_chat_name": "Введите название чата",
    "tab_model": "Модель",
    "tab_chat_settings": "Настройки чата",
    "tab_modules": "Модули",
    "create": "Создать",
    "max_tasks": "Макс. количество задач:",
    "system_modules": "Системные модули",
    "global_custom_modules": "Пользовательские модули (глобальные)",
    "chat_specific_modules": "Новые модули (только для этого чата)",
    "add_module": "Добавить модуль",
    "add_new_module_button": "+",
    "module_validation_error": "Модуль не прошел проверку:\n{error_msg}",
    "python_files": "Python файлы",

    # Global Settings Window
    "settings_title": "Настройки",
    "tab_main": "Основные",
    "settings_saved": "Настройки успешно сохранены.",
    "reset_settings_button": "Сбросить настройки",
    "reset_settings_confirm_title": "Сброс настроек",
    "reset_settings_confirm_message": "Вы уверены, что хотите сбросить все настройки? Приложение будет закрыто.",
    "select_chat_to_configure": "Выберите чат для настройки.",
    "remove_module_confirm": "Удалить выбранный модуль?",
    "error_adding_module": "Ошибка добавления модуля",

    # Module Validator
    "module_err_not_found": "Файл не найден: {path}",
    "module_err_no_docstring": "Модуль должен содержать строку документации",
    "module_err_docstring_len": "Документация должна содержать минимум 4 строки",
    "module_err_main_not_found": "Модуль должен содержать функцию main",
    "module_err_main_args": "Функция main должна принимать ровно 1 аргумент",
    "module_err_syntax": "Синтаксическая ошибка: {e}",
    "module_err_generic": "Ошибка при проверке модуля: {e}",
    "module_validated": "Модуль прошел валидацию",
    "module_custom_desc": "Пользовательский модуль",
    "module_desc_missing": "Описание отсутствует",

    # Model Validator
    "model_err_path_missing": "Укажите путь к модели",
    "model_err_invalid_file": "Неверный файл модели",
    "model_err_validation_generic": "Ошибка при валидации модели: {e}",
    "custom_api_success": "Успешное подключение к API",
    "custom_api_fail": "Не удалось подключиться к API",
    "custom_api_connect_error": "Ошибка подключения: {e}",
    "model_cfg_validated": "Настройки прошли валидацию",
    "model_type_no_validation": "Тип модели не требует валидации",
    "model_err_hf_missing": "Укажите модель HuggingFace",
    "model_err_openai_key_missing": "Введите API ключ OpenAI",
    "model_err_anthropic_key_missing": "Введите API ключ Anthropic",
    "model_err_mistral_key_missing": "Введите API ключ Mistral",
    "model_err_cohere_key_missing": "Введите API ключ Cohere",
    "model_err_ollama_missing": "Укажите модель Ollama",
    "model_err_lmstudio_missing": "Укажите модель LM Studio",
    "model_err_custom_api_missing": "Укажите IP и порт сервера",
}
