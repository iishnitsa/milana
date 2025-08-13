'''
end_dialog
gets the final result and ends the dialog
'''

from cross_gpt import (
    global_state,
    sql_exec,
    save_emb_dialog,
    get_now_try,
    up_now_try,
    let_log
)

# Получаем текстовые константы из контейнера
from cross_gpt import (
    slash_token,
    slash_n,
    kv_token,
)
# TODO: проверь на удаление диалога искусственно вызвав команду
def main(text):
    # Инициализация атрибутов модуля при первом вызове
    if not hasattr(main, 'attr_names'):
        main.attr_names = ('end_dialog_return',)
        main.end_dialog_return = 'Response saved.'
        return

    let_log('ЗАВЕРШЕНИЕ ДИАЛОГА')
    let_log(text)
    
    # Устанавливаем состояние диалога
    global_state.dialog_state = False
    global_state.stop_agent = True

    # Получаем текущий try_id и parents_id
    current_try = get_now_try()
    parts = current_try.split(slash_token)
    if len(parts) > 1:
        parents_id = slash_token.join(parts[:-1])
    else:
        parents_id = slash_token
    now_try = parts[-1]

    # Получаем историю диалога
    history = sql_exec(
        'SELECT history FROM chats WHERE chat_id=?',
        (global_state.conversations,), fetchone=True)

    # Сохраняем результат в базу данных
    last_successful_id = sql_exec(
        "SELECT MAX(successful_try_id) FROM tries WHERE parents_id = ?",
        (parents_id,), fetchone=True
    ) or 0

    sql_exec(
        '''INSERT OR REPLACE INTO tries 
        (try_id, parents_id, task, result, successful_try_id) 
        VALUES (?, ?, ?, ?, ?)''',
        (
            now_try,
            parents_id,
            global_state.main_now_task,
            text,
            last_successful_id + 1
        )
    )

    # Сохраняем эмбеддинги диалога
    save_emb_dialog(history, 'correct', text) # TODO:

    # Обновляем иерархию попыток
    up_now_try()

    # Устанавливаем результат диалога
    global_state.dialog_result = text
    return main.end_dialog_return
