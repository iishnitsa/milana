locales = {
    'ru': {
        'module_doc': [
            'умная_файловая_операция',
            'Универсальная операция с файлами: создание, чтение, редактирование, удаление, копирование, перемещение; Отправь запрос на естественном языке, например: "создай файл notes.txt с текстом \'привет\'", "удали старый отчет.pdf", "перемести source.txt в backup/source.txt". Инструмент сам определит действие и пути',
            'Умная операция с файлом',
            'Преобразует запрос пользователя в файловую операцию и выполняет его через файловую систему',
        ],
        'main.empty_files_text': 'В рабочей папке нет файлов',
        'main.cannot_parse_text': 'Не удалось разобрать запрос',
        'main.need_source_text': 'Для операции нужен source',
        'main.need_target_text': 'Для операции нужен target',
        'main.done_text': 'Готово',
        'main.failed_text': 'Ошибка',
        'main.forbidden_text': 'Запрещено',
        'main.prompt_parse_part1': """
Разбери пользовательский запрос на файловую операцию.
Верни ТОЛЬКО JSON:
{"action":"create|read|edit|delete|copy|move","source":"...","target":"...","content":"...","purpose":"...","experimental_mode":false}
Если значение неизвестно, верни пустую строку.
Запрос:
""",
        'main.prompt_parse_part2': """
Список файлов:
""",
        'main.keywords_create': ['созда', 'create'],
        'main.keywords_edit': ['замени', 'измени', 'редакт', 'edit', 'replace', 'update'],
        'main.keywords_delete': ['удали', 'delete', 'remove'],
        'main.keywords_copy': ['копир', 'copy'],
        'main.keywords_move': ['перемест', 'move', 'rename'],
        'main.keywords_read': ['прочита', 'покажи', 'read', 'show'],
    }
}