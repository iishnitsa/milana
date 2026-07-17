'''
end_dialogue
gets the final result and ends the dialogue
'''

import time
from cross_gpt import (
    global_state,
    sql_exec,
    save_emb_dialog,
    up_hierarchy,
    critic,
    let_log,
    get_chat_context,
    delete_chat,
    chat_path,
    recreate_agents,
    send_output_message,
    get_input_message,
    critic_reuse_dialog)
import os
from datetime import datetime
base_dir = os.path.join(chat_path, "results")

def main(text):
    if not hasattr(main, 'attr_names'):
        main.attr_names = ('end_dialog_return', 'got_client_answer')
        main.end_dialog_return = 'Response saved.'
        main.got_client_answer = "The client's response was received:"
        return
    let_log('\n' + '='*60)
    let_log('ЗАВЕРШЕНИЕ ДИАЛОГА')
    let_log(f"Current hierarchy ID: {global_state.now_try}")
    let_log(f"Result length: {len(text)} characters")
    let_log(f"Conversations counter: {global_state.conversations}")
    if not recreate_agents and global_state.conversations <= 2 and len(global_state.critic_reactions.keys()) == global_state.max_critic_reactions:
        let_log('НЕ ЗАВЕРШАЕМ ДИАЛОГ')
        send_output_message(text=text, command='ask_user')
        return '\n' + main.got_client_answer + '\n' + get_input_message(command='answer_user')['text']
    # Устанавливаем состояние диалога
    global_state.dialog_state = False
    global_state.stop_agent = True
    let_log(f"Текущий hierarchy ID после up_hierarchy: {global_state.now_try}")
    global_state.dialog_result = text
    let_log(f"Установлен dialog_result длиной {len(text)} символов")
    if global_state.write_results == 1:
        filename = str(global_state.now_try)
        filename = filename.replace('/', '__')
        filename = filename.replace(':', '_')
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{filename}.txt"
        path = os.path.join(base_dir, filename)
        try:
            os.makedirs(base_dir, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f: f.write(text)
            # файл результата — для пользователя
            try:
                with open(path + '.for_user', 'w', encoding='utf-8') as mf:
                    mf.write('for_user=1\n')
            except Exception:
                pass
        except Exception as e: let_log(f"Error: {e}")
    # ИСХОДНАЯ ЛОГИКА РАБОТЫ С КРИТИКОМ
    let_log('\n--- Проверка критика ---')
    let_log(f"Реакции критика: {global_state.critic_reactions}")
    is_dialog_correct = 'incorrect' # TODO: если разрешишь отключать критика не забудь про это
    critic_result = critic(global_state.main_now_task, text)
    if critic_result == 3:
        is_dialog_correct = 'correct'
        save_emb_dialog(is_dialog_correct, result_text=text, result=True)
    elif isinstance(critic_result, str) and critic_reuse_dialog:
        # Не пересоздаём диалог: возвращаем ответ + обновлённый промпт (комментарий критика уже в global_state)
        let_log(f"[critic_reuse_dialog] доработка без пересоздания: {str(critic_result)[:120]}...")
        save_emb_dialog(is_dialog_correct, result_text=text, result=True)
        save_emb_dialog(is_dialog_correct)
        global_state.dialog_state = True
        global_state.stop_agent = True
        global_state.dialog_ended = False
        global_state.need_owerwrite_operator = True
        # Ответ агенту: исходный результат + указание критика
        return text + '\n\n' + str(critic_result)
    elif critic_result != 2 or critic != 1:
        let_log(f"Критик требует переделки: {critic_result}...")
        save_emb_dialog(is_dialog_correct, result_text=text, result=True)
    save_emb_dialog(is_dialog_correct)
    if global_state.conversations % 2 == 0:
        save_emb_dialog(is_dialog_correct, dialog_type='executor')
        # Удаляем чат исполнителя
        delete_chat(global_state.conversations)
        global_state.tools_commands_dict.pop(global_state.conversations)
        let_log("Удалены инструменты исполнителя")
        global_state.conversations -= 1
        let_log(f"Удалён чат исполнителя, conversations уменьшен до: {global_state.conversations}")
    else: let_log('Исполнитель не был создан, чат не удалён')
    delete_chat(global_state.conversations)
    let_log("Удалёны чат оператора")
    global_state.tools_commands_dict.pop(global_state.conversations)
    let_log("Удалены инструменты оператора")
    global_state.conversations -= 1
    let_log(f"Conversations после уменьшения: {global_state.conversations}")
    let_log('Возврат на уровень выше...')
    up_hierarchy()
    # Устанавливаем результат диалога
    global_state.need_owerwrite_operator = True # TODO: проверь это в разных сценариях
    global_state.need_owerwrite_executor = True
    global_state.dialog_ended = True
    return text#main.end_dialog_return # TODO: