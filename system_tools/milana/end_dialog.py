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
    critic,
    let_log,
    slash_token,
    slash_n,
    kv_token,
)
# НОВЫЕ ИМПОРТЫ ИЗ CHAT_MANAGER
from chat_manager import get_chat_context, delete_chat

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

    # ИЗМЕНЕНИЕ: Получаем историю через менеджер
    _, history = get_chat_context(global_state.conversations)

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

    # Сохраняем эмбеддинги диалога, если история была найдена
    if history:
        save_emb_dialog(history, 'correct', text) # TODO:

    # Обновляем иерархию попыток
    up_now_try()

    # ИЗМЕНЕНИЕ: Удаляем чат(ы) через менеджер
    delete_chat(global_state.conversations)
    if global_state.conversations % 2 == 0:
        delete_chat(global_state.conversations - 1)
        global_state.conversations -= 1
        global_state.tools_commands_dict.popitem()
    else: let_log('Диалог завершается не начавшись')
    global_state.tools_commands_dict.popitem()
    print(global_state.critic_reactions)
    try: now_critic_reactions = global_state.critic_reactions[global_state.conversations] # TODO: давай ответ критика вернётся в енд диалог в качестве ответа? хотя это перегрузит контекст но идея интернесная
    except: now_critic_reactions = 0; global_state.critic_reactions[global_state.conversations] = 0
    if now_critic_reactions < global_state.max_critic_reactions:
        critic_result = critic(global_state.main_now_task, text)
        if critic_result != 1: global_state.critic_comment = critic_result
        else: del global_state.critic_reactions[global_state.conversations]
    else: del global_state.critic_reactions[global_state.conversations]

    global_state.conversations -= 1
    # Устанавливаем результат диалога
    global_state.dialog_result = text
    return main.end_dialog_return