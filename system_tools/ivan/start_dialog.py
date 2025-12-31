'''
delegate_task
transfers the task for division into subtasks and execution
'''

# Токены
from cross_gpt import (
    slash_token,
    slash_n,
    zpt_token,
    prompt_n,
    zpt_space,
    space_skb,
    closing_skb,
    make_exec_first,
    system_role_text,
    operator_role_text,
    func_role_text,
    wrong_command,
    start_dialog_history,
    remove_commands_roles,
    find_all_commands,
    what_is_func_text,
    last_messages_marker,
    native_func_call,
    let_log,
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
            'conversations_limit_reached_text',
        ) # TODO: нужно не грузить промпт для оператора а только задачу

        main.milana_template = '''
You are the AI operator "Milana".
 You are given a task plan from a client or a dialogue higher up in the hierarchy. Create an AI executor "Ivan" for the task by writing the create executor command and the task after the command.
 It is advisable to create a separate executor for each task, but not mandatory.
 Then work with the executor, control the execution of the task, sometimes recreate it if necessary, the old executor will be disconnected from the dialogue.
 "Ivan" can transfer one of the tasks from the plan that he is performing further down the hierarchy if he cannot cope. This will create a similar dialogue, you will not have access to it. Only "Ivan" has the right to do this.
 When you are confident that the overall task (all or almost all of the plan) is completed - write the end dialogue command and pass a detailed response after the command.
 To call a command, write three exclamation marks at the beginning of the message, then the command name, then three more exclamation marks, and then the information for the command.
 Example of a command call - "!!!create_executor!!! Write the frontend for an online store"
 You can only call the commands available below. Their description of work is in parentheses:\n'''

        main.start_dialog_tool_text_1 = '''You read the description of a task for an operator who will control the work of an executor. Based on the task description, select tools from the list of allowed tools.

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
    
    #let_log(start_dialog_history)
    # НОВЫЕ ИМПОРТЫ ИЗ CHAT_MANAGER
    from chat_manager import create_chat, update_history
    from cross_gpt import ( # TODO: вынеси все импорты за функции везде
        global_state,
        ask_model,
        text_cutter,
        sql_exec,
        gigo,
        tools_selector,
        create_executor,
        down_now_try,
        traceprint,
    )
    import difflib
    
    if global_state.hierarchy_limit != 0:
        if global_state.hierarchy_limit == global_state.conversations: return main.conversations_limit_reached_text
    
    let_log('начинается диалог')

    global_state.stop_agent = True
    if client_task == '':
        prompt = gigo(global_state.main_now_task)
        global_state.retries = False
    elif global_state.task_retry:
        global_state.retries.append(0)
        prompt = gigo(client_task)
    else:
        global_state.main_now_task = client_task
        prompt = gigo(client_task)

    if global_state.summ_attach != global_state.summ_attach:
        prompt += global_state.summ_attach
        global_state.summ_attach = ''

    # Выбор инструментов для Миланы
    milana_tools = global_state.milana_module_tools
    if global_state.module_tools_keys: # TODO: разберись с этим промптом
        need_tools_raw = ask_model(
            main.start_dialog_tool_text_1 +
            global_state.tools_str +
            main.start_dialog_tool_text_2 +
            prompt, all_user=True
        )
        let_log(need_tools_raw)
        all_tool_names = [tool[0] for tool in global_state.another_tools] # TODO: здесь и в специалисте вынеси в глобал стейт
        tools_names = find_all_commands(need_tools_raw, all_tool_names)
        
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
    for tool in milana_tools:
        full_prompt += tool + space_skb + milana_tools[tool][0] + closing_skb + zpt_token + ' '
    let_log(full_prompt)
    let_log(global_state.another_tools)
    if full_prompt and full_prompt[-1] == zpt_token:
        full_prompt = full_prompt[:-1]
    if not native_func_call: prompt += what_is_func_text
    full_prompt += prompt
    global_state.conversations += 1
    # TODO: ЗАПИСЫВАЕТСЯ КАКАЯ-ТО ФИГНЯ МИЛАНЕ
    # СИСТЕМ ТЭГ ПРОВАЙДЕР СДЕЛАЙ ЧТОБЫ ГЛОБАЛЬНО ГРУЗИЛ И СЮДА И ВЕЗДЕ
    # ПОПРОСИ ПИСАТЬ КРАТКО (ВООБЩЕ СОКРАЩЕНИЕ ЭТО СЛОЖНАЯ ТЕМА)
    # МОЖЕТ РАЗБИТИЕ НА ОЧЕНЬ МАЛЕНЬКИЕ ТЕЗИСЫ
    # ИЗБАВЬСЯ ОТ KPI
    
    # ИЗМЕНЕНИЕ: Создаем чат через менеджер
    create_chat(global_state.conversations, system_role_text + full_prompt)
    update_history(global_state.conversations, make_exec_first, func_role_text) # это может быть опционально (и мб НУЖНО будет вернуть Милана:!!! для новых моделей или сделать универсально)
    # да и в целом из промпта удали упоминание о создании специалиста сначала ибо модель на это будет обращать внимание
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
        except:
            raise
            
    # ИЗМЕНЕНИЕ: Обновляем историю через менеджер
    update_history(global_state.conversations, talk_prompt, operator_role_text)
    
    let_log("НАЧАЛЬНЫЙ ОТВЕТ МИЛАНЫ:")
    let_log(talk_prompt)
    talk_prompt_for_tools = remove_commands_roles(talk_prompt) # Используем новую переменную, чтобы не терять оригинал
    answer = tools_selector(talk_prompt_for_tools, global_state.conversations)
    global_state.now_agent_id = global_state.conversations
    if answer and answer != wrong_command:
        talk_prompt = answer
    elif global_state.dialog_state:
        let_log('СОЗДАНИЕ НОВОГО СПЕЦИАЛИСТА...') # TODO: тут не создавай если диалог завершён
        talk_prompt = create_executor(talk_prompt)
    else: return 'not_created'

    let_log("ОТВЕТ ПОСЛЕ ОБРАБОТКИ:")
    let_log(talk_prompt)
    
    last_talk_prompt = talk_prompt # Это ответ от функции/инструмента
    
    # ИЗМЕНЕНИЕ: Собираем контекст для второго вызова.
    # Для этого получаем текущую историю через get_chat_context, если мы в стандартном режиме.
    # В RAG-режиме это не нужно, так как rag_constructor сам всё соберет.
    # Но для унификации кода мы сначала обновим историю, а потом будем вызывать agent_func,
    # который уже будет знать, как правильно собрать контекст.
    # Для этого кода мы просто разделим сложное обновление истории на два логичных шага.
    
    # ПОЛУЧАЕМ ИСТОРИЮ (только для стандартного режима)
    from chat_manager import get_chat_context
    _, history_for_model = get_chat_context(global_state.conversations)
    try:
        talk_prompt = ask_model(
            system_role_text +
            full_prompt +
            history_for_model + # Используем полученную историю
            func_role_text +
            last_talk_prompt + # Ответ от функции
            operator_role_text
        )
    except:
        history_for_model = start_dialog_history + text_cutter(history_for_model) # TODO: ТУТ ВМЕСТО ИСТОРИИ СОКРАТИ ПРОМПТ НО ХЗ КАК
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
    
    # ИЗМЕНЕНИЕ: Вместо сложного сложения строки, делаем два последовательных обновления истории.
    # 1. Сначала записываем ответ от функции.
    update_history(global_state.conversations - 1, last_talk_prompt, func_role_text)
    # 2. Затем записываем итоговый ответ Миланы.
    update_history(global_state.conversations - 1, talk_prompt, operator_role_text)

    down_now_try()
    return talk_prompt