'''
delegate_task
transfers the task for division into subtasks and execution
'''

import difflib
from cross_gpt import (
    make_exec_first,
    system_role_text,
    operator_role_text,
    func_role_text,
    wrong_command,
    start_dialog_history,
    remove_commands_roles,
    find_all_commands,
    only_one_func_text,
    what_is_func_text,
    last_messages_marker,
    native_func_call,
    let_log,
    global_state,
    ask_model,
    text_cutter,
    gigo,
    tools_selector,
    create_executor,
    down_hierarchy,
    get_level,
    save_emb_dialog,
    create_chat,
    update_history,
    get_chat_context,
)

def find_tuple_by_first_list(data, target_list):
    for list1, list2, obj in data:
        if list1 == target_list:
            return (list1, list2), obj
    return None # Если совпадение не найдено

def main(client_task):
    # Инициализация атрибутов модуля при первом вызове
    
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'start_dialog_tool_text_1',
            'start_dialog_tool_text_2',
            'milana_template',
            'command_example',
            'conversations_limit_reached_text',
        )

        main.milana_template = '''
You are an AI operator "Milana".
You are given a task plan from a client or a higher-level dialog. Create an AI executor "Ivan" for the task by writing the executor creation command and the task (detailed, with unambiguous interpretation, with explanation of abbreviations if any).
Then work with the executor, monitor task execution, sometimes recreate him if necessary, for example, when you start working on the next item of the plan, the old executor will be disconnected from the dialog.
"Ivan" can pass one of the tasks from the plan he is executing down the hierarchy if he cannot handle it. This will create a similar dialog, and you will not have access to it. Only "Ivan" has the right to do so.
When you are confident that the overall task (the entire or almost entire plan) is completed - write the dialog completion command and pass the detailed result to the function.
'''
        main.command_example = '''
Command example – "!!!create_executor!!! Write frontend for an online store"
'''
        # Уточнено: пользователь пришлёт план и задачу
        main.start_dialog_tool_text_1 = '''You will receive a plan and a task from the user. Based on the task description, select tools from the list of allowed tools.

Output data:

Only tool names separated by comma and space in a single line.
No quotes, no periods at the end, no explanations, and no additional characters.
If no tool is suitable or if at least one incorrect or extra name is found, output exactly: None

List of allowed tools:

'''
        main.start_dialog_tool_text_2 = '''

LIMITATIONS:

It is forbidden to output any other tool names except those from the list.
You cannot invent new commands.
Any typo or incorrect entry in the list is sufficient reason to output None.

'''
        main.conversations_limit_reached_text = 'The limit of delegation levels has been reached. The task has not been transferred.'
        return
    if global_state.hierarchy_limit != 0:
        current_level = get_level()
        if current_level >= global_state.hierarchy_limit: return main.conversations_limit_reached_text
    let_log('начинается диалог')
    global_state.dialog_ended = False
    if global_state.critic_wants_retry: global_state.critic_wants_retry = False
    else:
        if global_state.conversations > 0:
            let_log('СОХРАНЕНИЕ перед делегированием')
            save_emb_dialog('delegated')
            let_log('Сохранили оператора как delegated')
            if global_state.conversations % 2 == 0 and global_state.conversations != 0:
                save_emb_dialog('delegated', 'executor')
                let_log('Сохранили исполнителя как delegated')
    global_state.stop_agent = True
    if client_task == '':
        prompt = gigo(global_state.main_now_task)
        global_state.retries = False
    else:
        global_state.main_now_task = client_task
        prompt = gigo(client_task)
    if global_state.summ_attach != global_state.summ_attach:
        prompt += global_state.summ_attach
        global_state.summ_attach = ''
    # === ДЕЛЕГИРОВАНИЕ: добавляем новый уровень ===
    down_hierarchy()
    let_log(f"После делегирования: {global_state.now_try}")
    # Выбор инструментов для Миланы
    milana_tools = global_state.milana_module_tools
    if global_state.module_tools_keys:
        # Изменено: используется system_prompt + user message (план/задача) аналогично gigo
        need_tools_raw = ask_model(
            prompt,  # user message: план и задача
            system_prompt=main.start_dialog_tool_text_1 +
                          global_state.tools_str +
                          main.start_dialog_tool_text_2
        )
        let_log(need_tools_raw)
        tools_names = find_all_commands(need_tools_raw, global_state.module_tools_keys)
        for name in tools_names:
            # Ищем инструмент по точному имени в исходном списке
            for tool_tokens, tool_desc, tool_func in global_state.another_tools:
                if name == tool_tokens:
                    milana_tools[tool_tokens] = (tool_desc, tool_func)
                    break
        let_log('ошибки нет')
    # Формирование финального промпта
    full_prompt = main.milana_template
    let_log(milana_tools)
    prompt += only_one_func_text
    for tool in milana_tools: full_prompt += tool + ' (' + milana_tools[tool][0] + '), '
    let_log(full_prompt)
    let_log(global_state.another_tools)
    if full_prompt and full_prompt[-1] == ',': full_prompt = full_prompt[:-1]
    if not native_func_call: prompt += what_is_func_text + main.command_example
    full_prompt += prompt
    global_state.conversations += 1
    create_chat(global_state.conversations, system_role_text + full_prompt)
    update_history(global_state.conversations, make_exec_first, func_role_text)
    global_state.tools_commands_dict[global_state.conversations] = milana_tools
    # Генерация начального ответа
    try:
        talk_prompt = ask_model(
            system_role_text +
            full_prompt +
            last_messages_marker +
            func_role_text +
            make_exec_first +
            operator_role_text
        )
    except:
        try:
            talk_prompt = ask_model(
                system_role_text +
                text_cutter(full_prompt) +
                last_messages_marker +
                func_role_text +
                make_exec_first +
                operator_role_text
            )
        except: raise
    update_history(global_state.conversations, talk_prompt, operator_role_text)
    let_log("НАЧАЛЬНЫЙ ОТВЕТ МИЛАНЫ:")
    let_log(talk_prompt)
    talk_prompt_for_tools = remove_commands_roles(talk_prompt)
    answer = tools_selector(talk_prompt_for_tools, global_state.conversations)
    global_state.now_agent_id = global_state.conversations
    #if answer and answer != wrong_command and global_state.dialog_state: talk_prompt = answer
    if answer != wrong_command and global_state.dialog_state and answer != None: talk_prompt = answer
    elif global_state.dialog_state:
        let_log('СОЗДАНИЕ НОВОГО СПЕЦИАЛИСТА...')
        talk_prompt = create_executor(talk_prompt)
    else: return answer
    let_log("ОТВЕТ ПОСЛЕ ОБРАБОТКИ:")
    let_log(talk_prompt)
    last_talk_prompt = talk_prompt # Это ответ от функции/инструмента
    # Получаем историю для второго вызова
    _, history_for_model = get_chat_context(global_state.conversations)
    try:
        talk_prompt = ask_model(
            system_role_text +
            full_prompt +
            history_for_model +
            func_role_text +
            last_talk_prompt +
            operator_role_text
        )
    except:
        history_for_model = start_dialog_history + text_cutter(history_for_model)
        try:
            talk_prompt = ask_model(
                system_role_text +
                full_prompt +
                last_messages_marker +
                history_for_model +
                func_role_text +
                last_talk_prompt +
                operator_role_text
            )
        except Exception as e:
            try:
                talk_prompt = ask_model(
                    system_role_text +
                    full_prompt +
                    last_messages_marker +
                    history_for_model +
                    func_role_text +
                    text_cutter(last_talk_prompt) +
                    operator_role_text
                )
            except: raise
    # Записываем ответы в историю
    update_history(global_state.conversations - 1, last_talk_prompt, func_role_text)
    update_history(global_state.conversations - 1, talk_prompt, operator_role_text)
    return talk_prompt