'''
delegate_task
transfers the task for division into subtasks and execution
'''

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
    no_markdown_instruction,
    write_shortly_prompt,
)

def find_tuple_by_first_list(data, target_list):
    for list1, list2, obj in data:
        if list1 == target_list:
            return (list1, list2), obj
    return None

def main(client_task):
    # Инициализация атрибутов модуля при первом вызове
    
    if not hasattr(main, 'attr_names'):
        main.attr_names = (
            'start_dialog_tool_text_1',
            'start_dialog_tool_text_2',
            'milana_base_1',
            'milana_base_2',
            'milana_base_3',
            'milana_delegation_part',
            'command_example',
            'hierarchy_limit_info',
            'oper_anti_loop_text',
            'delegate_unavailable_for_operator',
            'conversations_limit_reached_text',
        )
        
        main.milana_base_1 = '''You are "Milana", an AI operator. You have received a task plan from a client'''
        main.milana_base_2 = ''' or from a higher-level dialog'''
        main.milana_base_3 = '''.
Your workflow:
1. CREATE ONE EXECUTOR — Use the command "!!!create_executor!!!" followed by the task description. This creates "Ivan", an AI executor who will handle the current subtask.
CORRECT: "!!!create_executor!!! React frontend for an online store"
INCORRECT: "!!!create_executor!!! Create Ivan for frontend"
INCORRECT: "!!!create_executor!!! Ivan, write frontend"
2. WORK WITH THE SAME EXECUTOR — After creation, continue the conversation with Ivan. Give instructions, answer questions, receive results. DO NOT create a new executor unless absolutely necessary.
IMPORTANT: Once Ivan is created, NEVER use the "!!!create_executor!!!" command again during your conversation with him. This command is only for the initial creation. Using it again will cause errors and unnecessarily recreate the executor.
3. WHEN RECREATION IS ALLOWED — Recreate Ivan ONLY in these cases:
- The current executor explicitly states they CANNOT complete the task.
- The task changes so drastically that a different specialization is needed.
- Ivan SUCCESSFULLY completed their part and you need a new executor for a clearly separate next subtask (but first try to have Ivan handle multiple steps).
4. FORBIDDEN — Never recreate an executor IMMEDIATELY after they respond. A simple reply from Ivan is NOT a reason to create a new one. Continue the conversation.
5. GREETING AFTER CREATION — After successfully creating Ivan, simply greet him in natural language (e.g., "Hello, Ivan."). Do not include any commands in your greeting.
6. DELEGATION — '''
        main.milana_delegation_part = '''Ivan can delegate a subtask further down the hierarchy if he cannot handle it. This creates a similar dialog which you (Milana) cannot access. Only Ivan has the right to delegate.
'''
        main.command_example = '''
Command example – "!!!create_executor!!! Write frontend for an online store"
'''
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
        main.oper_anti_loop_text = '''
Execution control and dialogue termination:
Monitor the executor’s progress. If there is no solution, determine whether
the issue is task complexity or impossibility.
Consider the situation problematic (absurd) if:
- actions repeat without progress
- results lose connection to the task
- the direction of reasoning constantly shifts
- tools return inconsistent, irrelevant, or useless output
Do not stop the dialogue due to complexity alone.
First try adjusting the plan or decomposition.
Mark the task as impossible only if:
- environment or tool limitations make the goal unreachable
- required data or functions are missing or unavailable
- all reasonable approaches lead to repetition or absurd results
When stopping, you must prove impossibility:
- list attempts and why they failed
- describe constraints or failures
- explain why further attempts will not succeed'''
        main.hierarchy_limit_info = 'Hierarchy levels are limited. Current level'
        main.delegate_unavailable_for_operator = 'The task delegation function down the hierarchy is not available to you.'
        main.conversations_limit_reached_text = 'The limit of delegation levels has been reached. The task has not been transferred.'
        return
    let_log('начинается диалог')
    if global_state.hierarchy_limit != 0 and global_state.hierarchy_limit == get_level(): return main.conversations_limit_reached_text
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
    current_level = get_level()
    let_log(f"После делегирования: {global_state.now_try}, текущий уровень: {current_level}")
    
    # Определяем, сможет ли будущий исполнитель делегировать
    if global_state.hierarchy_limit == 0:
        ivan_can_delegate = True
    else:
        ivan_can_delegate = (current_level + 1) < global_state.hierarchy_limit

    # Выбор инструментов для Миланы
    milana_tools = global_state.milana_module_tools.copy()
    
    if global_state.module_tools_keys:
        need_tools_raw = ask_model(
            prompt,  # user message: план и задача
            system_prompt=main.start_dialog_tool_text_1 +
                          global_state.tools_str +
                          main.start_dialog_tool_text_2
        )
        let_log(need_tools_raw)
        tools_names = find_all_commands(need_tools_raw, global_state.module_tools_keys)
        
        # Удаляем команду делегирования из списка выбранных, если она случайно попала
        if not ivan_can_delegate and global_state.start_dialog_command_name in tools_names:
            tools_names.remove(global_state.start_dialog_command_name)
            let_log(f"Удалена команда делегирования из выбранных инструментов")
        
        for name in tools_names:
            for tool_tokens, tool_desc, tool_func in global_state.another_tools:
                if name == tool_tokens:
                    milana_tools[tool_tokens] = (tool_desc, tool_func)
                    break
        let_log('ошибки нет')
    # Удаляем команду делегирования из инструментов Миланы, если следующий уровень недоступен
    if not ivan_can_delegate and global_state.start_dialog_command_name in milana_tools:
        del milana_tools[global_state.start_dialog_command_name]
        let_log("Удалена команда делегирования из инструментов Миланы")
    
    # === ФОРМИРОВАНИЕ ПРОМПТА ===
    full_prompt = main.milana_base_1
    # 1. Информация о том, что Иван может делегировать (добавляется ВСЕГДА, кроме случая, когда делегирование полностью отключено - лимит=1)
    if global_state.hierarchy_limit != 1:
        full_prompt += main.milana_base_2
        full_prompt += main.milana_delegation_part
        full_prompt += main.milana_base_3
        full_prompt += f"\n{main.delegate_unavailable_for_operator}\n"
        # 2. Сообщение для Миланы: она не может делегировать (добавляется ВСЕГДА, кроме случая, когда делегирование полностью отключено - лимит=1)
    else: full_prompt += main.milana_base_3

    # 3. Добавляем информацию об иерархии, если лимит больше 1
    if global_state.hierarchy_limit > 1:
        full_prompt += f"\n{main.hierarchy_limit_info} {current_level}/{global_state.hierarchy_limit}.\n"
    full_prompt += no_markdown_instruction + write_shortly_prompt
    let_log(milana_tools)
    prompt += only_one_func_text
    # Добавляем описание инструментов, исключая skip-команды
    for tool in milana_tools:
        if tool not in global_state.skip_tools_keys:
            prompt += tool + ' (' + milana_tools[tool][0] + ')\n'
    if not native_func_call:
        prompt += what_is_func_text + main.command_example
    full_prompt += prompt + main.oper_anti_loop_text
    let_log(full_prompt)
    let_log(global_state.another_tools)
    let_log(milana_tools)
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