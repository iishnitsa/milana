locales = {
    'ru': {
        'module_doc': [
            'закончить_диалог',
            'получает конечный результат и завершает диалог. ОБЯЗАТЕЛЬНО после команды напишите полный непустой ответ клиенту (текст результата). Пустое тело отклоняется.',
            'Завершить диалог',
            'Завершает диалог и передаёт результат; после команды обязателен непустой текст ответа',
        ],
        'main.end_dialog_return': 'Ответ сохранён.',
        'main.got_client_answer': 'Получен ответ клиента:',
        'main.empty_result_text': (
            'Команда завершения диалога вызвана без результата. '
            'После команды напишите полный ответ клиенту, например: '
            '!!!закончить_диалог!!! Вот итоговый результат...'
        ),
    },
    'en': {
        'module_doc': [
            'end_dialogue',
            'gets the final result and ends the dialogue. REQUIRED: after the command write the full non-empty result for the client. Empty body is rejected.',
            'End dialogue',
            'Ends the dialogue and returns the result; non-empty answer text after the command is required',
        ],
        'main.end_dialog_return': 'Response saved.',
        'main.got_client_answer': "The client's response was received:",
        'main.empty_result_text': (
            'End dialogue was called without a result. '
            'Write the final answer for the client after the command, for example: '
            '!!!end_dialogue!!! Here is the full result...'
        ),
    },
}
