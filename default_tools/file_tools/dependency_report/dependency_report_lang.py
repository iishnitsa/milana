locales = {
    'ru': {
        'module_doc': [
            'отчёт_зависимостей',
            'Анализ графа зависимостей файлов, проверка безопасности операции; Отправь запрос с указанием файла и действия (read/edit/delete/copy/move/create). Например: "покажи зависимости для main.py" или "что будет если удалить utils.py". Вернёт JSON с отчётом',
            'Отчёт зависимостей',
            'Отображает зависимости файлов и проверки безопасности для запрошенной операции',
        ],
        'main.file_not_selected_text': 'Не удалось выбрать файл для отчёта',
        'main.done_text': 'Готово',
        'main.failed_text': 'Ошибка',
        'main.prompt_parse_part1': """
Выдели из запроса путь файла и действие.
Верни ТОЛЬКО JSON:
{"file_hint":"...","action":"read|edit|delete|copy|move|create"}
Запрос:
""",
        'main.prompt_parse_part2': """
Файлы:
""",
        'main.keywords_delete': ['удали', 'delete', 'remove'],
        'main.keywords_edit': ['измени', 'редакт', 'edit', 'replace', 'update'],
        'main.keywords_move': ['перемест', 'move'],
        'main.keywords_copy': ['копир', 'copy'],
        'main.keywords_create': ['созда', 'create'],
        'main.keywords_read': ['прочита', 'read'],
    }
}