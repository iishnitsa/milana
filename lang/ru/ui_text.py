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

    # Chat Context Menu
    "active_chat_delete_title": "Активный чат",
    "active_chat_delete_message": "Чат '{chat_name}' сейчас работает. Остановить и удалить?",
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
    "token_limit_info": "Лимит токенов",
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
    "max_messages_before_answer": "Макс. сообщений до ответа (0=выкл)",
    "max_messages_before_answer_desc": "Мягкий лимит шагов цикла агента до остановки. 0 — без лимита. Связано с прогрессом (критик / иерархия / попытки).",
    "max_executor_recreates": "Макс. пересозданий исполнителя (0=выкл)",
    "max_executor_recreates_desc": "Сколько раз create_executor может пересоздать исполнителя на уровне. 0 — без лимита. Soft stop вместо пересоздания.",
    "librarian_use_web": "Библиотекарь может искать в вебе",
    "librarian_use_web_desc": "Если выкл (по умолчанию), Библиотекарь не ходит в web search и пропускает web-фрагменты. Если вкл — веб разрешён после фазы GIGO.",
    "save_emb_dialog": "Сохранять embeddings диалога (save_emb_dialog)",
    "save_emb_dialog_desc": "Если выкл — завершённые диалоги не пишутся в долгосрочное embedding-хранилище. По умолчанию вкл.",
    "tools_no_examples": "Промпт tools без примеров команд",
    "tools_no_examples_desc": "Если вкл — у исполнителя шаблон !!!команда!!!, но без конкретных примеров вызова (например внутренний_поиск).",
    "allow_command_not_at_start": "Команда не обязательно в начале сообщения",
    "allow_command_not_at_start_desc": "Выкл (по умолчанию): маркер команды должен быть в начале сообщения. Вкл: маркер может быть позже; проверка позиции и формулировка «начни сообщение с…» смягчаются.",
    "give_operator_goal_to_executor": "Давать исполнителю цель оператора (без плана)",
    "give_operator_goal_to_executor_desc": "Выкл по умолчанию. Вкл: в системный промпт Ивана добавляется цель клиента/оператора без плана GIGO, с пометкой «цель оператора» (только контекст).",
    "deliver_user_messages": "Доставлять сообщения пользователя mid-dialog",
    "deliver_user_messages_desc": "Если вкл: перед записью ответа Ивана собеседнику берётся oldest client inject, клиент пишется в БД, Иван откладывается; Милана отвечает plain text или skip, затем Иван в БД. По умолчанию выкл.",
    "model_type_large": "Большая модель (основная)",
    "small_model_section": "Маленькая модель (опционально)",
    "small_model_section_desc": "По умолчанию выкл. Можно указать только большую. Embeddings всегда с большого провайдера. Если провайдер тот же — скопируйте params большой и смените только model.",
    "small_model_type": "Провайдер маленькой",
    "small_token_limit": "Лимит контекста маленькой",
    "small_for_cutter_only": "Маленькая только для cutter / сводок / save_emb",
    "small_for_cutter_only_desc": "Вкл (по умолчанию): small для text_cutter, RAG-сводок, группировки save_emb. Выкл: small для всего кроме диалога агента (если не включён agent-on-small).",
    "small_agent_until_protocol": "Агент на small; large после нарушения протокола",
    "small_agent_until_protocol_desc": "По умолчанию выкл. Если вкл — агент на small, при protocol fail переключается на large до валидной команды или обычного сообщения.",
    "small_copy_from_large": "Скопировать params большой (кроме model)",
    "validate_small_model": "Проверить маленькую модель",
    "small_model_disabled_hint": "Сначала включите переключатель маленькой модели.",
    "use_small_model": "Включить маленькую модель",
    "use_small_model_desc": "Опциональная вторая модель. Большая — primary. По умолчанию выкл.",
    "tab_small_model": "Маленькая модель",
    "text_cutter_block_title": "TEXT CUTTER",
    "text_cutter_token_limit": "Размер куска text_cutter (токены)",
    "text_cutter_token_limit_desc": "Примерно макс. токенов на один вызов LLM в text_cutter (по умолчанию 2000). Длинный текст режется на такие куски перед суммаризацией.",
    "max_incoming_tokens": "Макс. входящих токенов в text_cutter",
    "max_incoming_tokens_desc": "Потолок на текст, подаваемый в text_cutter (по умолчанию 10000), плюс не больше context−1000. Не путать с лимитом контекста модели.",
    "small_model_section_about": "О маленькой модели",
    "small_model_type_desc": "Модуль провайдера для маленькой модели (может совпадать с большой — тогда копируются params, меняется только model).",
    "small_token_limit_desc": "Лимит контекста маленькой модели (токены). Клипается к max модели после validate.",
    "small_model_provider_params": "Параметры маленького провайдера",
    "small_model_provider_params_desc": "Строка connection params маленькой модели (служебное; обычно заполняется из UI params).",
    "small_max_token_limit": "Макс. токенов маленькой (после validate)",
    "small_max_token_limit_desc": "Верхняя граница контекста маленькой модели после проверки connect; UI не даёт поставить small_token_limit выше.",
    "shell_skip_confirm": "Shell: не спрашивать подтверждение",
    "shell_skip_confirm_desc": "Если выкл (по умолчанию), shell-команды требуют yes/no от пользователя. Если вкл — без подтверждения.",
    "mcp_url": "MCP URL (host:port или http://...)",
    "mcp_url_desc": "MCP-сервер с доп. tools. Задаётся в глобальных Настройках или в чате (вкладка «Настройки чата»). Пример: 127.0.0.1:8765 или http://127.0.0.1:8765. При старте чата tools/list подгружается в инструменты агента.",
    "release_version": "Версия релиза приложения",
    "release_version_title": "Версия релиза",
    "release_version_missing": "В чате нет версии релиза. Приложение: {app}. Продолжить? (версия будет записана)",
    "release_version_newer": "Версия чата ({chat}) новее приложения ({app}). Продолжить? (запишется версия приложения)",
    "release_version_older": "Версия чата ({chat}) старше приложения ({app}). Продолжить? (запишется версия приложения)",
    "use_rag": "Использовать продвинутую память диалога",
    "filter_generations": "Очищать генерации",
    "hierarchy_limit": "Лимит ступеней иерархии",
    "use_librarian": "Использовать Библиотекаря",
    "recreate_agents": "Пересоздавать агентов с новой задачей",
    "skip_nested_images": "Пропускать вложенные изображения",
    "allow_ocr": "Распознавать изображения",
    "cut_wrong_command_history": "Вырезать неправильные команды от агента",
    "number_of_plan_items": "Количество пунктов плана",

    "do_translate": "Использовать переводчик",
    "target_lang": "Язык модели",
    "local_and_tools_translate": "Перевод локализации и инструментов",
    "use_local_cache": "Использовать кэш перевода чата",
    "use_global_cache": "Использовать кэш перевода всех чатов",
    "use_psm": "Генерировать личности агентам",
    "use_gigo": "Использовать проработку задачи (GIGO)",
    "use_old_gigo": "Использовать классический GIGO",
    "gigo_idea_count": "Количество идей GIGO",
    "gigo_plan_items": "Пунктов плана GIGO",
    "gigo_use_entropy": "Использовать String Seed Of Thought в GIGO",
    "gigo_use_concepts": "Использовать концепты в GIGO",
    "gigo_use_filter": "Использовать фильтрацию",
    "gigo_use_librarian": "Использовать Библиотекаря в GIGO",
    "use_magical_prompt": "Использовать магическую инструкцию",
    "show_message_datetime": "Дата/время в сообщениях для агентов",
    "chats_dir_step_title": "Папка чатов",
    "chats_dir": "Папка чатов",
    "chats_dir_migrate_title": "Перенос чатов",
    "chats_dir_migrate_message": "Перенести существующие чаты из:\n{old}\nв:\n{new}?",
    "chats_dir_migrate_failed": "Не удалось перенести чаты: {e}",
    "librarian_use_models": "Библиотекарь использует LLM",
    "module_hints_for_operator": "Подсказки модулей оператору",
    "give_all_tools": "Отдавать все инструменты",
    "critic_reuse_dialog": "Критик не пересоздаёт диалог",
    "one_shot_intention_permission": "Разовая генерация намерения и разрешения",
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
    "fs_copy_touched_on_end": "Копировать затронутые файлы проекта при конце диалога",
    "fs_use_git": "Git/ветки для файловых операций",
    "copy_user_attachments_to_files": "Копировать вложения пользователя в files/",
    "ui_light_theme": "Светлая тема (бело-оранжевая)",
    "ui_scale": "Масштаб UI",
    "gigo_role_dreamer": "Роль GIGO: Мечтатель",
    "gigo_role_realist": "Роль GIGO: Реалист",
    "gigo_role_critic": "Роль GIGO: Критик",
    "small_protocol_drop_error": "Ошибку small не писать; сразу large",
    "settings_group_others": "Прочее",
    "settings_group_memory": "Память и RAG",
    "settings_group_gigo": "GIGO",
    "settings_group_agents": "Агенты и диалог",
    "settings_group_limits": "Лимиты cutter",
    "settings_group_translation": "Перевод",
    "settings_group_files": "Файлы и FS",
    "settings_group_tools": "Shell и MCP",
    "settings_group_client": "Клиент mid-dialog",
    "settings_group_ui": "Интерфейс",
    "settings_group_fs": "Файловая система",
    "settings_group_small": "Маленькая модель",
    "settings_group_librarian": "Библиотекарь",
    "settings_group_use": "Переключатели use_*",
    "settings_group_max": "Лимиты max_*",
    "settings_group_text": "Текст / cutter",
    "settings_group_shell": "Shell",
    "settings_group_mcp": "MCP",
    "settings_group_deliver": "Клиент mid-dialog",
    "settings_group_save": "Сохранение",
    "settings_group_cut": "Вырезка команд",
    "settings_group_show": "Отображение",
    "settings_group_write": "Запись логов/результатов",
    "settings_group_filter": "Фильтрация",
    "settings_group_hierarchy": "Иерархия",
    "settings_group_number": "Числовые",
    "settings_group_target": "Язык цели",
    "settings_group_local": "Локальный перевод",
    "settings_group_do": "Переводчик",
    "settings_group_recreate": "Пересоздание",
    "settings_group_skip": "Пропуски",
    "settings_group_allow": "Разрешения",
    "settings_group_critic": "Критик",
    "settings_group_module": "Модули",
    "settings_group_give": "Инструменты агенту",
    "settings_group_one": "Намерение FS",

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

    # ===== ОПИСАНИЯ ДЛЯ НАСТРОЕК ЧАТА (ключи с суффиксом _desc) =====
    
    # Параметры из ui.py, которые отображаются в настройках чата
    "use_rag_desc": "Включает продвинутую память диалога с использованием RAG (Retrieval-Augmented Generation). Позволяет модели обращаться к сохранённым ранее сообщениям для более контекстных ответов.",
    
    "filter_generations_desc": "Автоматически очищает сгенерированные сообщения от лишней информации, оставляя только основной ответ. Улучшает читаемость диалога.",
    
    "hierarchy_limit_desc": "Макс. глубина иерархии агентов. 0 = без лимита. 1 = один корневой диалог (без делегирования вниз). Первый старт в worker всегда разрешён; блок только при попытке углубиться дальше лимита.",
    
    "use_librarian_desc": "Активирует модуль Библиотекаря, который помогает структурировать и сохранять важную информацию из диалога для последующего использования.",
    
    "recreate_agents_desc": "При смене задачи пересоздавать агентов заново, а не переиспользовать существующих. Обеспечивает чистоту выполнения для каждой новой задачи.",
    
    "skip_nested_images_desc": "Пропускать обработку изображений, вложенных в другие файлы (например, в PDF или DOCX). Ускоряет работу при большом количестве вложений.",
    
    "allow_ocr_desc": "Включает распознавание текста на изображениях с помощью OCR. Позволяет модели «читать» текст с картинок. На Linux нужен AVX2. Требует моделей из установщика.",
    "allow_ocr_no_models": "Модели изображений не установлены. Переустановите Milana с опцией «модели распознавания изображений», чтобы включить.",
    
    "cut_wrong_command_history_desc": "Если включено: удаляет из истории чата реплики с нарушением протокола (неправильная команда). Не отключает обрезку второй команды в одном ответе (remove_commands_roles). По умолчанию выкл — история сохраняется целиком, модели реже путаются.",
    
    "number_of_plan_items_desc": "Количество пунктов плана, которые генерирует агент перед выполнением задачи. Влияет на детализацию и структурированность ответа.",
    
    "do_translate_desc": "Включает автоматический перевод сообщений на целевой язык. Полезно при общении с пользователями, говорящими на разных языках.",
    
    "target_lang_desc": "Язык, на который будут переводиться сообщения при включённом переводчике. Укажите код языка (например, 'en', 'ru', 'fr').",
    
    "local_and_tools_translate_desc": "Переводить не только основные сообщения, но и названия инструментов, системные сообщения и локализацию интерфейса агента.",
    
    "use_local_cache_desc": "Использовать кэш перевода только для текущего чата. Ускоряет повторные переводы одинаковых фраз в рамках одного диалога.",
    
    "use_global_cache_desc": "Использовать общий кэш переводов для всех чатов. Позволяет экономить ресурсы при повторяющихся фразах в разных диалогах.",
    
    "use_psm_desc": "Генерировать уникальные личности для агентов на основе контекста задачи. Делает ответы более персонализированными и естественными.",
    
    "use_gigo_desc": "Включает проработку задачи по принципу GIGO (Garbage In, Garbage Out). Улучшает качество ответов за счёт более тщательного анализа входных данных.",
    
    "gigo_idea_count_desc": "Количество идей, которые генерируются в рамках GIGO для проработки задачи. Больше идей — более детальный анализ.",
    
    "gigo_use_entropy_desc": "Использовать метод String Seed Of Thought в GIGO для генерации более разнообразных и нестандартных решений.",
    
    "gigo_use_concepts_desc": "Включает использование концептуального анализа в GIGO. Позволяет выявлять скрытые связи и закономерности в данных.",
    
    "gigo_use_filter_desc": "Активирует фильтрацию идей в GIGO, оставляя только наиболее релевантные и качественные варианты для дальнейшей обработки.",
    
    "gigo_use_librarian_desc": "Использовать Библиотекаря в процессе GIGO для доступа к накопленным знаниям и предыдущим решениям.",
    
    "write_log_desc": "Сохранять подробный лог всех действий и сообщений в процессе работы чата. Полезно для отладки и анализа.",
    
    "write_results_desc": "Сохранять результаты работы агентов в отдельную папку 'results' для последующего анализа или экспорта.",

    "fs_copy_touched_on_end_desc": "При завершении диалога копировать файлы, которые агенты создали или изменили в проекте, в chat/dialog_artifacts/ (только снимок, без слияния веток). По умолчанию выключено.",
    "fs_use_git_desc": "Вкл (по умолчанию): файловая система с git/мирами (ветки под капотом). Выкл: простые операции с файлами на диске без git и выбора веток.",
    "copy_user_attachments_to_files_desc": "Если вкл: при отправке сообщения файлы-вложения копируются в chat/files/ (рабочая папка агентов). В историю и worker уходят пути к копиям. По умолчанию выкл.",
    "ui_light_theme_desc": "Светлая бело-оранжевая тема интерфейса. Выкл — тёмная (чёрный + фиолетовый). Применяется после сохранения настроек.",
    "ui_scale_desc": "Размер интерфейса. Ползунок: влево меньше, вправо крупнее. Мин. — обычный размер. Применяется сразу после сохранения.",
    "gigo_role_dreamer_desc": "Старый GIGO: роль Мечтатель. Выкл — роль пропускается. По умолчанию вкл.",
    "gigo_role_realist_desc": "Старый GIGO: роль Реалист. Выкл — роль пропускается. По умолчанию вкл.",
    "gigo_role_critic_desc": "Старый GIGO: роль Критик (фаза GIGO, не end_dialog). Выкл — роль пропускается. По умолчанию вкл.",
    "small_protocol_drop_error_desc": "Только вместе с «агент на small до протокола». Если small даёт нарушение протокола — этот ход не пишется в историю (ошибка не сохраняется), подключается large и ход повторяется с тем же входом. По умолчанию выкл.",
    
    "max_critic_reactions_desc": "Максимальное количество итераций критики, которые агент может выполнить над своим ответом. Влияет на качество финального результата.",

    "gigo_plan_items_desc": "Количество пунктов плана в финальном ответе GIGO. Если 0 — используется общее число пунктов плана.",
    "use_magical_prompt_desc": "Включать принципы Magical Prompt в критика и связанные системные промпты.",
    "use_old_gigo_desc": "Если включено — классический (старый) GIGO. Если выключено — продвинутый GIGO (gigo_adv).",
    "show_message_datetime_desc": "Если вкл: в историю для агентов (full_text) перед текстом пишется [YYYY-MM-DD HH:MM:SS]; также показывается в UI. По умолчанию выкл.",
    "chats_dir_desc": "Папка хранения данных чатов. На Windows предлагается D:\\Milana\\chats, если есть диск D: и приложение не установлено туда же; иначе — data/chats рядом с программой.",
    "librarian_use_models_desc": "Если включено, Библиотекарь использует LLM для релевантности и переформулировок. По умолчанию выкл. — только эмбеддинги и фрагменты.",
    "module_hints_for_operator_desc": "Также добавлять подсказки по модулям/инструментам в промпт оператора (по умолчанию выкл.).",
    "give_all_tools_desc": "Не выбирать подмножество инструментов — отдавать агенту полный набор.",
    "critic_reuse_dialog_desc": "Критик возвращает ответ и обновляет промпт, не пересоздавая диалог целиком (по умолчанию вкл.).",
    "vectorize_on_overflow_only_desc": "При включённом RAG векторизовать сообщения только при переполнении контекста (и только общие для оператора/исполнителя).",
    "one_shot_intention_permission_desc": "Генерировать намерение и разрешение один раз на задачу, а не на каждую файловую операцию (по умолчанию выкл.).",

    # Описания параметров провайдеров
    "provider_param_url_desc": "Базовый URL API или локального сервера (схема и порт при необходимости).",
    "provider_param_model_desc": "Имя chat/completions модели у провайдера.",
    "provider_param_chat_desc": "Идентификатор chat-модели (имя или ID провайдера).",
    "provider_param_emb_desc": "Имя модели эмбеддингов, если она отдельна от chat-модели.",
    "provider_param_emb_model_desc": "Имя embedding-модели для RAG и поиска.",
    "provider_param_emb_url_desc": "Отдельный базовый URL для endpoint эмбеддингов (опционально).",
    "provider_param_token_desc": "API-токен / ключ для авторизованных провайдеров.",
    "provider_param_api_token_desc": "API-токен / ключ для авторизованных провайдеров.",
    "provider_param_password_desc": "Локальный пароль для шифрования/расшифровки сохранённого API-токена (провайдеру не отправляется).",
    "provider_param_token_limit_desc": "Желаемый размер контекста в токенах (может быть ограничен моделью).",
    "provider_param_chat_template_desc": "Использовать chat template / chat-completions (True/False).",
    "provider_param_native_func_call_desc": "Включить нативный tool/function calling провайдера, если модель поддерживает (True/False).",
    "provider_param_ollama_url_desc": "Базовый URL Ollama для эмбеддингов или бэкенда (по умолчанию http://localhost:11434).",
    "provider_param_ollama_emb_model_desc": "Имя embedding-модели Ollama (например all-minilm:latest).",
    "provider_param_ollama_desc": "Если True — использовать Ollama для эмбеддингов (или связанных путей) вместо основного API.",
    "provider_param_timeout_desc": "Таймаут HTTP-запроса в секундах (0 — значение по умолчанию провайдера).",
    "provider_param_host_desc": "Хост локального сервера.",
    "provider_param_port_desc": "TCP-порт локального сервера.",
    "provider_param_user_desc": "Опциональное имя пользователя для basic auth или multi-user серверов.",
    "provider_param_num_ctx_desc": "Переопределение размера контекста (num_ctx) для Ollama-совместимых серверов.",
    "provider_param_llm_cpu_desc": "Принудительно запускать chat-модель на CPU (true/false).",
    "provider_param_emb_cpu_desc": "Принудительно запускать embedding-модель на CPU (true/false).",
    "provider_param_offload_on_boot_desc": "Выгружать модели из VRAM при connect/boot, если поддерживается (true/false).",
    "provider_param_is_thinking_desc": "Модель поддерживает режим thinking/reasoning (true/false).",
    "provider_param_filter_think_tag_desc": "Убирать think/reasoning теги из видимого ответа (true/false).",
    "provider_param_bos_desc": "Опциональный BOS-тег для форматирования промпта.",
    "provider_param_eos_desc": "Опциональный EOS-тег для форматирования промпта.",
    "provider_param_sys_start_desc": "Опциональный стартовый тег system-сообщения.",
    "provider_param_sys_end_desc": "Опциональный конечный тег system-сообщения.",
    "provider_param_user_start_desc": "Опциональный стартовый тег user-сообщений.",
    "provider_param_user_end_desc": "Опциональный конечный тег user-сообщений.",
    "provider_param_assist_start_desc": "Опциональный стартовый тег assistant-сообщений.",
    "provider_param_assist_end_desc": "Опциональный конечный тег assistant-сообщений.",
}