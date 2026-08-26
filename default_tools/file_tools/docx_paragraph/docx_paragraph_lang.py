locales = {
    'ru': {
        'module_doc': [
            'docx_параграф',
            'Чтение или редактирование абзаца в .docx по смыслу; Отправь запрос вида: "прочитай абзац в файле document.docx про \'старый текст\'" или "замени \'старый текст\' на \'новый текст\' в файле report.docx". Режим и файл определятся автоматически.',
            'Docx параграф',
            'Читает или редактирует абзац в .docx с помощью конвейера файловой системы и проверяет намерения',
        ],
        'main.no_docx_text': 'Docx файлы не найдены',
        'main.cannot_parse_text': 'Не удалось разобрать запрос',
        'main.edit_need_replacement_text': 'Для edit нужен новый текст абзаца (replacement)',
        'main.paragraph_not_found_text': 'Абзац не найден',
        'main.done_text': 'Готово',
        'main.forbidden_text': 'Запрещено',
        'main.failed_text': 'Ошибка',
        'main.prompt_extract_part1': """
Извлеки структуру запроса для работы с docx.
Верни ТОЛЬКО JSON без пояснений с ключами:
{"mode":"read|edit","file_hint":"...","target_query":"...","replacement":"..."}
Запрос:
""",
        'main.prompt_extract_part2': """
Кандидаты файлов:
""",
        'main.prompt_pick_file_part1': """
Выбери один файл docx для запроса.
Верни только относительный путь как в списке, без комментариев.
Запрос:
""",
        'main.prompt_pick_file_part2': """
Список:
""",
        'main.keywords_edit': ['замени', 'поменя', 'измени', 'replace', 'edit'],
        'main.keywords_read': ['прочита', 'покажи', 'read', 'show'],
    }
}