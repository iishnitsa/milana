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

    # Language Loading
    "lang_load_error_title": "Ошибка загрузки языка",
    "lang_load_error_message": "Не найдены языковые файлы в папке 'lang'. Приложение будет закрыто.",

    # Main Window
    "new_chat": "Новый чат",
    "settings": "Настройки",
    "chat_prefix": "Чат ",

    # Chat Context Menu
    "active_chat_delete_title": "Активный чат",
    "active_chat_delete_message": "Чат '{chat_name}' активен. Удалить?",
    "delete_chat_confirm_title": "Удаление чата",
    "delete_chat_confirm_message": "Вы уверены, что хотите удалить чат '{chat_name}'?",

    # App Closing
    "active_chats_on_close_title": "Активные чаты",
    "active_chats_on_close_message": "Есть {count} активных чатов. Выйти?",

    # Attachment
    "attachments": "Вложения",
    "attachment_open_error": "Ошибка открытия вложения",

    # Initial Settings
    "initial_settings_title": "Первоначальная настройка",
    "save_and_continue": "Сохранить и продолжить",
    "validation_error": "Ошибка валидации",
    "token_limit_info": "Лимит токенов: {max_tokens}",
    "validate_model": "Проверить модель",
    "model_validated_success": "Модель прошла валидацию. Максимум токенов: {tokens}",
    "model_not_validated": "Модель не проверена",
    "model_not_validated_continue": "Модель не проверена. Продолжить?",

    # Create/Edit Chat Window
    "create_chat_title": "Создать новый чат",
    "chat_name": "Название чата:",
    "chat_name_exists": "Чат с таким именем уже существует.",
    "enter_chat_name": "Введите название чата",
    "tab_model": "Модель",
    "tab_chat_settings": "Настройки чата",
    "tab_modules": "Модули",
    "create": "Создать",
    "max_critic_reactions": "Макс. реакций критика:",
    "use_rag": "Использовать продвинутую память диалога",
    "filter_generations": "Очищать генерации",
    "hierarchy_limit": "Лимит ступеней иерархии",
    "use_librarian": "Использовать Библиотекаря",
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
    "model_err_validation_generic": "Ошибка при валидации модели: {e}",
    
    # Model Validator (добавленные)
    "model_err_no_provider": "Не выбран провайдер модели",
    "model_err_provider_missing": "Провайдер '{provider}' не найден",

    "write_log": "Записывать лог",
    "write_results": "Записывать результаты",

    # Text Editor Context Menu
    "cut": "Вырезать",
    "copy": "Копировать",
    "paste": "Вставить",
    "select_all": "Выделить всё",

    # File Dialog
    "all_files": "Все файлы",

    # Folder Operations
    "folder_not_found": "Папка '{folder_name}' не найдена",
    "open_folder_error": "Ошибка открытия папки: {e}",

    # Providers
    "model_type": "Провайдеры",
    "token_limit": "Лимит токенов",
    "no_providers_found": "Провайдеры не найдены",

    # Settings
    "restart_required": "Для применения изменений требуется перезапуск приложения",

    "ok": "ОК",
    "yes": "Да",
    "no": "Нет",
}